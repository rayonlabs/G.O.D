import json
import os
import shutil
import uuid
from dataclasses import dataclass

import docker
import pandas as pd
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
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.docker_utils import stream_logs
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DiffusionJob
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import ImageModelType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TextDatasetType
from core.models.utility_models import TextJob
from miner.utils import download_flux_unet


logger = get_logger(__name__)


@dataclass
class DockerEnvironmentDiffusion:
    huggingface_token: str
    wandb_token: str
    job_id: str
    base_model: str

    def to_dict(self) -> dict[str, str]:
        return {
            "HUGGINGFACE_TOKEN": self.huggingface_token,
            "WANDB_TOKEN": self.wandb_token,
            "JOB_ID": self.job_id,
            "BASE_MODEL": self.base_model,
        }


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
    dataset_type: TextDatasetType,
    file_format: FileFormat,
    task_id: str,
    expected_repo_name: str | None,
) -> dict:
    """
    Loads the config template and modifies it to create a new job config.
    """
    if isinstance(dataset_type, InstructTextDatasetType | DpoDatasetType | ChatTemplateDatasetType):
        config_path = cst.CONFIG_TEMPLATE_PATH
    elif isinstance(dataset_type, GrpoDatasetType):
        config_path = cst.CONFIG_TEMPLATE_PATH_GRPO

    logger.info("Loading config template")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = []

    dataset_entry = create_dataset_entry(dataset, dataset_type, file_format)
    config["datasets"].append(dataset_entry)

    if isinstance(dataset_type, DpoDatasetType):
        config["rl"] = "dpo"
    elif isinstance(dataset_type, GrpoDatasetType):
        filename, reward_funcs_names = create_reward_funcs_file(
            [reward_function.reward_func for reward_function in dataset_type.reward_functions], task_id
            )
        config["trl"]["reward_funcs"] = [f"{filename}.{func_name}" for func_name in reward_funcs_names]
        config["trl"]["reward_weights"] = [reward_function.reward_weight for reward_function in dataset_type.reward_functions]

    config = update_flash_attention(config, model)
    config = update_model_info(config, model, task_id, expected_repo_name, dataset_type=dataset_type)
    config["mlflow_experiment_name"] = dataset
    config["output_dir"] = cst.GRPO_MINER_OUTPUT_DIR

    return config


def create_reward_funcs_file(reward_funcs: list[str], task_id: str, destination_dir: str = cst.CONFIG_DIR) -> list[str]:
    """
    Create a Python file with reward functions for GRPO training.

    Args:
        reward_funcs: List of strings containing Python reward function implementations
        task_id: Unique task identifier
    """
    filename = f"rewards_{task_id}"
    filepath = os.path.join(destination_dir, f"{filename}.py")

    func_names = []
    for reward_func in reward_funcs:
        if "def " in reward_func:
            func_name = reward_func.split("def ")[1].split("(")[0].strip()
            func_names.append(func_name)

    with open(filepath, "w") as f:
        f.write("# Auto-generated reward functions file\n\n")
        for reward_func in reward_funcs:
            f.write(f"{reward_func}\n\n")

    return filename, func_names


def _load_and_modify_config_diffusion(job: DiffusionJob) -> dict:
    """
    Loads the config template and modifies it to create a new job config.
    """
    logger.info("Loading config template")
    if job.model_type == ImageModelType.SDXL:
        with open(cst.CONFIG_TEMPLATE_PATH_DIFFUSION_SDXL, "r") as file:
            config = toml.load(file)
        config["pretrained_model_name_or_path"] = job.model
        config["train_data_dir"] = f"/dataset/images/{job.job_id}/img/"
        config["huggingface_token"] = cst.HUGGINGFACE_TOKEN
        config["huggingface_repo_id"] = f"{cst.HUGGINGFACE_USERNAME}/{job.expected_repo_name or str(uuid.uuid4())}"
    elif job.model_type == ImageModelType.FLUX:
        with open(cst.CONFIG_TEMPLATE_PATH_DIFFUSION_FLUX, "r") as file:
            config = toml.load(file)
        config["pretrained_model_name_or_path"] = f"{cst.CONTAINER_FLUX_PATH}/flux_unet_{job.model.replace('/', '_')}.safetensors"
        config["train_data_dir"] = f"/dataset/images/{job.job_id}/img/"
        config["huggingface_token"] = cst.HUGGINGFACE_TOKEN
        config["huggingface_repo_id"] = f"{cst.HUGGINGFACE_USERNAME}/{job.expected_repo_name or str(uuid.uuid4())}"
    else:
        logger.error(f"Unknown model type: {job.model_type}")
    return config


def create_job_diffusion(
    job_id: str,
    model: str,
    dataset_zip: str,
    model_type: ImageModelType,
    expected_repo_name: str | None
):
    return DiffusionJob(
        job_id=job_id,
        model=model,
        dataset_zip=dataset_zip,
        model_type=model_type,
        expected_repo_name=expected_repo_name,
    )


def create_job_text(
    job_id: str,
    dataset: str,
    model: str,
    dataset_type: TextDatasetType,
    file_format: FileFormat,
    expected_repo_name: str | None,
):
    return TextJob(
        job_id=job_id,
        dataset=dataset,
        model=model,
        dataset_type=dataset_type,
        file_format=file_format,
        expected_repo_name=expected_repo_name,
    )


def start_tuning_container_diffusion(job: DiffusionJob):
    logger.info("=" * 80)
    logger.info("STARTING THE DIFFUSION TUNING CONTAINER")
    logger.info("=" * 80)

    config_path = os.path.join(cst.CONFIG_DIR, f"{job.job_id}.toml")

    config = _load_and_modify_config_diffusion(job)
    save_config_toml(config, config_path)

    logger.info(config)
    if job.model_type == ImageModelType.FLUX:
        logger.info(f"Downloading flux unet from {job.model}")
        flux_unet_path = download_flux_unet(job.model)

    prepare_dataset(
        training_images_zip_path=job.dataset_zip,
        training_images_repeat=(
            cst.DIFFUSION_SDXL_REPEATS if job.model_type == ImageModelType.SDXL
            else cst.DIFFUSION_FLUX_REPEATS
        ),
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=job.job_id,
    )

    docker_env = DockerEnvironmentDiffusion(
        huggingface_token=cst.HUGGINGFACE_TOKEN, wandb_token=cst.WANDB_TOKEN, job_id=job.job_id, base_model=job.model_type.value
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

        if job.model_type == ImageModelType.FLUX:
            volume_bindings[os.path.dirname(flux_unet_path)] =  {
                "bind": cst.CONTAINER_FLUX_PATH,
                "mode": "rw",
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

        train_data_path = f"{cst.DIFFUSION_DATASET_DIR}/{job.job_id}"

        if os.path.exists(train_data_path):
            shutil.rmtree(train_data_path)


def _dpo_format_prompt(row, format_str):
    result = format_str
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _dpo_format_chosen(row, format_str):
    result = format_str
    if "{chosen}" in format_str and cst.DPO_DEFAULT_FIELD_CHOSEN in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_CHOSEN]):
        result = result.replace("{chosen}", str(row[cst.DPO_DEFAULT_FIELD_CHOSEN]))
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _dpo_format_rejected(row, format_str):
    result = format_str
    if "{rejected}" in format_str and cst.DPO_DEFAULT_FIELD_REJECTED in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_REJECTED]):
        result = result.replace("{rejected}", str(row[cst.DPO_DEFAULT_FIELD_REJECTED]))
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _adapt_columns_for_dpo_dataset(dataset_path: str, dataset_type: DpoDatasetType, apply_formatting: bool = False):
    """
    Transform a DPO JSON dataset file to match axolotl's `chatml.argilla` expected column names.

    Args:
        dataset_path: Path to the JSON dataset file
        dataset_type: DpoDatasetType with field mappings
        apply_formatting: If True, apply formatting templates to the content
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    column_mapping = {
        dataset_type.field_prompt: cst.DPO_DEFAULT_FIELD_PROMPT,
        dataset_type.field_system: cst.DPO_DEFAULT_FIELD_SYSTEM,
        dataset_type.field_chosen: cst.DPO_DEFAULT_FIELD_CHOSEN,
        dataset_type.field_rejected: cst.DPO_DEFAULT_FIELD_REJECTED
    }
    df = df.rename(columns=column_mapping)

    if apply_formatting:
        if dataset_type.prompt_format and dataset_type.prompt_format != "{prompt}":
            format_str = dataset_type.prompt_format
            df[cst.DPO_DEFAULT_FIELD_PROMPT] = df.apply(lambda row: _dpo_format_prompt(row, format_str), axis=1)
        if dataset_type.chosen_format and dataset_type.chosen_format != "{chosen}":
            format_str = dataset_type.chosen_format
            df[cst.DPO_DEFAULT_FIELD_CHOSEN] = df.apply(lambda row: _dpo_format_chosen(row, format_str), axis=1)
        if dataset_type.rejected_format and dataset_type.rejected_format != "{rejected}":
            format_str = dataset_type.rejected_format
            df[cst.DPO_DEFAULT_FIELD_REJECTED] = df.apply(lambda row: _dpo_format_rejected(row, format_str), axis=1)

    output_data = df.to_dict(orient='records')
    with open(dataset_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def _adapt_columns_for_grpo_dataset(dataset_path: str, dataset_type: GrpoDatasetType):
    """
    Transform a GRPO JSON dataset file to match axolotl's `prompt` expected column name.

    Args:
        dataset_path: Path to the JSON dataset file
        dataset_type: GrpoDatasetType with field mappings
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.rename(columns={dataset_type.field_prompt: cst.GRPO_DEFAULT_FIELD_PROMPT})
    output_data = df.to_dict(orient='records')
    with open(dataset_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def _create_hf_cache_volume(docker_client, volume_name: str):
    """
    Create a Docker volume for HuggingFace model cache.
    """
    try:
        volume = docker_client.volumes.get(volume_name)
        logger.info(f"Using existing volume: {volume_name}")
        return volume
    except docker.errors.NotFound:
        logger.info(f"Creating new volume: {volume_name}")
        return docker_client.volumes.create(name=volume_name)


def _populate_hf_cache_volume(docker_client, volume_name: str, model: str):
    """
    Download model and populate the HF cache volume using a temporary container.
    """
    logger.info(f"Downloading model to volume: {model}")

    temp_container = docker_client.containers.run(
        image=cst.MINER_DOCKER_IMAGE,
        command=["/bin/bash", "-c", f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{model}')\""],
        volumes={volume_name: {"bind": "/root/.cache/huggingface", "mode": "rw"}},
        detach=True,
        remove=True
    )

    result = temp_container.wait()
    if result["StatusCode"] != 0:
        logs = temp_container.logs().decode()
        raise DockerException(f"Failed to download model {model}: {logs}")


def _run_grpo_upload_container(docker_client, job: TextJob, volume_bindings: dict):
    """
    Run a separate container with network access to upload the GRPO trained model to HuggingFace.
    """
    logger.info("Running upload container for GRPO model...")

    upload_script = f"{cst.MINER_CONTAINER_SCRIPTS_PATH}/upload_grpo_model.sh"

    docker_env = {
        "HUGGINGFACE_TOKEN": cst.HUGGINGFACE_TOKEN,
        "HUB_MODEL_ID": f"{cst.HUGGINGFACE_USERNAME}/{job.expected_repo_name or str(uuid.uuid4())}",
        "GRPO_MINER_OUTPUT_DIR": cst.GRPO_MINER_OUTPUT_DIR
    }

    # Add scripts directory to volume bindings for upload script access
    upload_volume_bindings = volume_bindings.copy()
    upload_volume_bindings[os.path.abspath("scripts")] = {
        "bind": cst.MINER_CONTAINER_SCRIPTS_PATH,
        "mode": "ro",
    }

    upload_container = docker_client.containers.run(
        image=cst.MINER_DOCKER_IMAGE,
        environment=docker_env,
        volumes=upload_volume_bindings,
        detach=True,
        tty=True,
        command=["/bin/bash", "-c", upload_script]
    )

    try:
        stream_logs(upload_container)
        result = upload_container.wait()

        if result["StatusCode"] != 0:
            logger.warning(f"Upload container exited with status {result['StatusCode']}")
        else:
            logger.info("Model upload completed successfully")

    finally:
        upload_container.remove(force=True)

def _create_docker_entrypoint(job):
    # For GRPO, skip network-dependent operations (HF/W&B login) since running offline
    if isinstance(job.dataset_type, GrpoDatasetType):
        setup_commands = """
    echo 'Preparing data for GRPO (offline mode)...' && \\
    if [ "$DATASET_TYPE" != "hf" ] && [ -f "/workspace/input_data/${DATASET_FILENAME}" ]; then \\
    cp /workspace/input_data/${DATASET_FILENAME} /workspace/axolotl/${DATASET_FILENAME}; \\
    fi"""

        reward_file = f"rewards_{job.job_id}.py"
        grpo_command = f"""
    echo "Moving specific reward function file to src directory..." && \\
    cp ${{CONFIG_DIR}}/{reward_file} /workspace/axolotl/src/"""
        setup_commands += " && \\" + grpo_command
    else:
        # Standard setup with network access for non-GRPO jobs
        setup_commands = """
    echo 'Preparing data...' && \\
    if [ -n "$HUGGINGFACE_TOKEN" ]; then \\
    echo "Attempting to log in to Hugging Face" && \\
    huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential; \\
    else \\
    echo "HUGGINGFACE_TOKEN is not set. Skipping Hugging Face login."; \\
    fi && \\
    if [ -n "$WANDB_TOKEN" ]; then \\
    echo "Attempting to log in to W&B" && \\
    wandb login "$WANDB_TOKEN"; \\
    else \\
    echo "WANDB_TOKEN is not set. Skipping W&B login."; \\
    fi && \\
    if [ "$DATASET_TYPE" != "hf" ] && [ -f "/workspace/input_data/${DATASET_FILENAME}" ]; then \\
    cp /workspace/input_data/${DATASET_FILENAME} /workspace/axolotl/${DATASET_FILENAME}; \\
    fi"""

    training_command = """
    echo 'Starting training command' && \\
    accelerate launch -m axolotl.cli.train ${CONFIG_DIR}/${JOB_ID}.yml
    """

    return setup_commands + " && \\" + training_command

def _adapt_columns_for_dataset(job: TextJob):
    """
    Adapt column names in the dataset based on job type.
    Only processes JSON files that require column name adaptation.
    """
    if job.file_format != FileFormat.JSON:
        return

    if isinstance(job.dataset_type, DpoDatasetType):
        _adapt_columns_for_dpo_dataset(job.dataset, job.dataset_type, True)
    elif isinstance(job.dataset_type, GrpoDatasetType):
        _adapt_columns_for_grpo_dataset(job.dataset, job.dataset_type)


def start_tuning_container(job: TextJob):
    logger.info("=" * 80)
    logger.info("STARTING THE TUNING CONTAINER")
    logger.info("=" * 80)

    config_filename = f"{job.job_id}.yml"
    config_path = os.path.join(cst.CONFIG_DIR, config_filename)

    docker_entrypoint = _create_docker_entrypoint(job)

    config = _load_and_modify_config(
        job.dataset,
        job.model,
        job.dataset_type,
        job.file_format,
        job.job_id,
        job.expected_repo_name,
    )
    save_config(config, config_path)

    logger.info(config)

    logger.info(os.path.basename(job.dataset) if job.file_format != FileFormat.HF else "")

    docker_env = DockerEnvironment(
        huggingface_token=cst.HUGGINGFACE_TOKEN,
        wandb_token=cst.WANDB_TOKEN,
        job_id=job.job_id,
        dataset_type=cst.CUSTOM_DATASET_TYPE,
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

        _adapt_columns_for_dataset(job)

        network_mode = None
        if isinstance(job.dataset_type, GrpoDatasetType):
            logger.info(f"GRPO job detected, downloading model beforehand: {job.model}")

            volume_name = f"hf_cache_{job.job_id}"
            _create_hf_cache_volume(docker_client, volume_name)
            _populate_hf_cache_volume(docker_client, volume_name, job.model)

            volume_bindings[volume_name] = {
                "bind": "/root/.cache/huggingface",
                "mode": "rw",
            }
            network_mode = "none"

        container_kwargs = {
            "image": cst.MINER_DOCKER_IMAGE,
            "environment": docker_env,
            "volumes": volume_bindings,
            "runtime": "nvidia",
            "device_requests": [docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])],
            "detach": True,
            "tty": True,
            "command": ["/bin/bash", "-c", docker_entrypoint]
        }

        if network_mode:
            container_kwargs["network_mode"] = network_mode

        container = docker_client.containers.run(**container_kwargs)

        last_logs = stream_logs(container)

        result = container.wait()

        if result["StatusCode"] != 0:
            raise DockerException(f"Container exited with non-zero status code: {result['StatusCode']}")

        if isinstance(job.dataset_type, GrpoDatasetType):
            try:
                _run_grpo_upload_container(docker_client, job, volume_bindings)
            except Exception as upload_error:
                logger.error(f"Failed to upload GRPO model: {str(upload_error)}")

    except Exception as e:
        # Waiting for axolotl to fix the issue
        if "TypeError: DPOTrainer.create_model_card() got an unexpected keyword argument 'dataset_tags'" in last_logs:
            logger.warning("This is probably just an axolotl issue only affecting HF repo model card, continuing...")
        else:
            logger.error(f"Error processing job: {str(e)}")
            raise

    finally:
        if not isinstance(job.dataset_type, GrpoDatasetType):
            repo = config.get("hub_model_id", None)
            if repo:
                hf_api = HfApi(token=cst.HUGGINGFACE_TOKEN)
                hf_api.update_repo_visibility(repo_id=repo, private=False, token=cst.HUGGINGFACE_TOKEN)
                logger.info(f"Successfully made repository {repo} public")

        if "container" in locals():
            try:
                container.remove(force=True)
                logger.info("Container removed")
            except Exception as e:
                logger.warning(f"Failed to remove container: {e}")

        if isinstance(job.dataset_type, GrpoDatasetType):
            volume_name = f"hf_cache_{job.job_id}"
            try:
                volume = docker_client.volumes.get(volume_name)
                volume.remove(force=True)
                logger.info(f"Volume {volume_name} removed")
            except Exception as e:
                logger.warning(f"Failed to remove volume {volume_name}: {e}")
