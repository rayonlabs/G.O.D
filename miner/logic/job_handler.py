import os
from dataclasses import dataclass
from typing import Type
from typing import Union

import docker
import toml
import yaml
from docker.errors import DockerException
from fiber.logging_utils import get_logger
from huggingface_hub import HfApi

from core import constants as cst
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import save_config_toml
from core.config.config_handler import update_flash_attention
from core.config.config_handler import update_model_info
from core.dataset.prepare_dataset import prepare_diffusion_dataset
from core.docker_utils import stream_logs
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import DatasetType
from core.models.utility_models import DiffusionJob
from core.models.utility_models import FileFormat
from core.models.utility_models import Job
from core.models.utility_models import TextJob


logger = get_logger(__name__)


@dataclass
class DockerEnvironmentDiffusion:
    huggingface_token: str
    wandb_token: str
    job_id: str

    def to_dict(self) -> dict[str, str]:
        return {"HUGGINGFACE_TOKEN": self.huggingface_token, "WANDB_TOKEN": self.wandb_token, "JOB_ID": self.job_id}


@dataclass
class DockerEnvironment:
    huggingface_token: str
    wandb_token: str
    job_id: str
    dataset_type: str
    dataset_filename: str

    def to_dict(self) -> dict[str, str]:
        return {
            "HUGGINGFACE_TOKEN": self.huggingface_token,
            "WANDB_TOKEN": self.wandb_token,
            "JOB_ID": self.job_id,
            "DATASET_TYPE": self.dataset_type,
            "DATASET_FILENAME": self.dataset_filename,
        }


def _load_and_modify_config(
    dataset: str,
    model: str,
    dataset_type: DatasetType | CustomDatasetType,
    file_format: FileFormat,
    task_id: str,
) -> dict:
    """
    Loads the config template and modifies it to create a new job config.
    """
    logger.info("Loading config template")
    with open(cst.CONFIG_TEMPLATE_PATH, "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = []

    dataset_entry = create_dataset_entry(dataset, dataset_type, file_format)
    config["datasets"].append(dataset_entry)

    config = update_flash_attention(config, model)
    config = update_model_info(config, model, task_id)
    config["mlflow_experiment_name"] = dataset

    return config


def _load_and_modify_config_diffusion(
    config_path: str,
    model: str,
    hf_repo_to_upload: str,
    hf_folder_name_in_repo:str,
    task_id: str
    ) -> dict:
    """
    Loads the config template and modifies it to create a new job config.
    """
    logger.info("Loading config template")
    logger.info(cst.CONFIG_TEMPLATE_PATH_DIFFUSION)
    with open(cst.CONFIG_TEMPLATE_PATH_DIFFUSION, "r") as file:
        config = toml.load(file)
    config["pretrained_model_name_or_path"] = model
    config["train_data_dir"] = f"/dataset/images/{task_id}/img/"
    config["huggingface_token"] = cst.HUGGINGFACE_TOKEN
    config["huggingface_repo_id"] = hf_repo_to_upload
    config["huggingface_path_in_repo"] = hf_folder_name_in_repo
    return config


def create_job(
    job_class: Type[Union[TextJob, DiffusionJob]],
    **kwargs,
) -> Job:
    return job_class(**kwargs)


def start_tuning_container_diffusion(job: DiffusionJob):
    logger.info("=" * 80)
    logger.info("STARTING THE DIFFUSION TUNING CONTAINER")
    logger.info("=" * 80)

    config_filename = "base_diffusion"
    config_path = os.path.join(cst.CONFIG_DIR, f"{job.job_id}.toml")

    config = _load_and_modify_config_diffusion(config_path, job.model, job.hf_repo, job.hf_folder, job.job_id)
    save_config_toml(config, config_path)

    logger.info(config)

    prepare_diffusion_dataset(
        training_images_zip_path=job.dataset_zip,
        training_images_repeat=cst.DIFFUSION_DEFAULT_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=job.job_id,
    )

    docker_env = DockerEnvironmentDiffusion(
        huggingface_token=cst.HUGGINGFACE_TOKEN, wandb_token=cst.WANDB_TOKEN, job_id=job.job_id
    ).to_dict()
    logger.info(f"Docker environment: {docker_env}")

    try:
        docker_client = docker.from_env()

        volume_bindings = {
            os.path.abspath(cst.CONFIG_DIR): {
                "bind": "/dataset/configs",
                "mode": "rw",
            },
            os.path.abspath(cst.OUTPUT_DIR): {
                "bind": "/dataset/outputs",
                "mode": "rw",
            },
            os.path.abspath(cst.DIFFUSION_DATASET_DIR): {
                "bind": "/dataset/images",
                "mode": "rw",
            },
        }

        container = docker_client.containers.run(
            image=cst.MINER_DOCKER_IMAGE_DIFFUSION,
            environment=docker_env,
            volumes=volume_bindings,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])],
            detach=True,
            tty=True,
        )

        # Use the shared stream_logs function
        stream_logs(container)

        result = container.wait()

        if result["StatusCode"] != 0:
            raise DockerException(f"Container exited with non-zero status code: {result['StatusCode']}")

    except Exception as e:
        logger.error(f"Error processing job: {str(e)}")
        raise

    finally:
        if "container" in locals():
            container.remove(force=True)


def start_tuning_container(job: TextJob):
    logger.info("=" * 80)
    logger.info("STARTING THE TUNING CONTAINER")
    logger.info("=" * 80)

    config_filename = f"{job.job_id}.yml"
    config_path = os.path.join(cst.CONFIG_DIR, config_filename)

    config = _load_and_modify_config(job.dataset, job.model, job.dataset_type, job.file_format, job.job_id)
    save_config(config, config_path)

    logger.info(config)

    logger.info(os.path.basename(job.dataset) if job.file_format != FileFormat.HF else "")

    docker_env = DockerEnvironment(
        huggingface_token=cst.HUGGINGFACE_TOKEN,
        wandb_token=cst.WANDB_TOKEN,
        job_id=job.job_id,
        dataset_type=job.dataset_type.value if isinstance(job.dataset_type, DatasetType) else cst.CUSTOM_DATASET_TYPE,
        dataset_filename=os.path.basename(job.dataset) if job.file_format != FileFormat.HF else "",
    ).to_dict()
    logger.info(f"Docker environment: {docker_env}")

    try:
        docker_client = docker.from_env()

        volume_bindings = {
            os.path.abspath(cst.CONFIG_DIR): {
                "bind": "/workspace/axolotl/configs",
                "mode": "rw",
            },
            os.path.abspath(cst.OUTPUT_DIR): {
                "bind": "/workspace/axolotl/outputs",
                "mode": "rw",
            },
        }

        if job.file_format != FileFormat.HF:
            dataset_dir = os.path.dirname(os.path.abspath(job.dataset))
            logger.info(dataset_dir)
            volume_bindings[dataset_dir] = {
                "bind": "/workspace/input_data",
                "mode": "ro",
            }

        container = docker_client.containers.run(
            image=cst.MINER_DOCKER_IMAGE,
            environment=docker_env,
            volumes=volume_bindings,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])],
            detach=True,
            tty=True,
        )

        # Use the shared stream_logs function
        stream_logs(container)

        result = container.wait()

        if result["StatusCode"] != 0:
            raise DockerException(f"Container exited with non-zero status code: {result['StatusCode']}")

    except Exception as e:
        logger.error(f"Error processing job: {str(e)}")
        raise

    finally:
        repo = config.get("hub_model_id", None)
        if repo:
            hf_api = HfApi(token=cst.HUGGINGFACE_TOKEN)
            hf_api.update_repo_visibility(repo_id=repo, private=False, token=cst.HUGGINGFACE_TOKEN)
            logger.info(f"Successfully made repository {repo} public")

        if "container" in locals():
            try:
                logger.info("Cleaning up HuggingFace cache...")
                container.exec_run("rm -rf /root/.cache/huggingface/hub/*", user="root")
            except Exception as e:
                logger.warning(f"Failed to clean HuggingFace cache: {e}")
            finally:
                container.remove(force=True)
