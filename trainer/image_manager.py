import os
import docker
import uuid
import asyncio
from docker.models.images import Image
from docker.models.containers import Container
from docker.errors import BuildError

from trainer import constants as cst
from validator.utils.logging import get_all_context_tags
from validator.utils.logging import get_logger
from validator.utils.logging import stream_image_build_logs, stream_container_logs
from trainer.tasks import log_task, complete_task
from core.models.payload_models import TrainerProxyRequest


logger = get_logger(__name__)

def build_docker_image(
    dockerfile_path: str = cst.DEFAULT_IMAGE_DOCKERFILE_PATH,
    context_path: str = ".",
    tag: str = None,
    no_cache: bool = True
) -> str:
    client: docker.DockerClient = docker.from_env()

    if tag is None:
        tag = f"standalone-image-trainer:{uuid.uuid4()}"

    logger.info(f"Building Docker image '{tag}', Dockerfile path: {dockerfile_path}, Context Path: {context_path}...")
    try:
        image, logs = client.images.build(
            path=context_path,
            dockerfile=dockerfile_path,
            tag=tag,
            nocache=no_cache
        )
        stream_image_build_logs(logs, get_all_context_tags())
        logger.info("Docker image built successfully.")
        return tag
    except BuildError as e:
        stream_image_build_logs(logs, get_all_context_tags())
        logger.error("Docker build failed.")
        return e


async def run_trainer_container(
    task_id: str,
    tag: str,
    model: str,
    dataset_zip: str,
    model_type: str,
    expected_repo_name: str,
    hours_to_complete: int=1,
    gpu_ids: list[int]=[0]
) -> Container:
    client: docker.DockerClient = docker.from_env()

    command: list[str] = [
        "--task-id", task_id,
        "--model", model,
        "--dataset-zip", dataset_zip,
        "--model-type", model_type,
        "--hours-to-complete", str(hours_to_complete),
        "--expected-repo-name", expected_repo_name
    ]
    
    container_name = f"image-trainer-{uuid.uuid4().hex}"

    try:
        container: Container = client.containers.run(
            image=tag,
            command=command,
            volumes={
                cst.CHECKPOINTS_VOLUME_NAME: {"bind": cst.CONTAINER_SAVE_PATH, "mode": "rw"}
            },
            remove=True,
            name=container_name,
            mem_limit=cst.DEFAULT_TRAINING_CONTAINER_MEM_LIMIT,
            nano_cpus=cst.DEFAULT_TRAINING_CONTAINER_NANO_CPUS * 1_000_000_000,
            device_requests=[docker.types.DeviceRequest(device_ids=[str(i) for i in gpu_ids],  capabilities=[["gpu"]])],
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            detach=True,
        )

        log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, get_all_context_tags()))
        return container
    except Exception as e:
        logger.error(e)
        return e


async def create_volume_if_doesnt_exist():
    client: docker.DockerClient = docker.from_env()
    volume_name = cst.CHECKPOINTS_VOLUME_NAME
    try:
        volume = client.volumes.get(volume_name)
    except docker.errors.NotFound:
        volume = client.volumes.create(name=volume_name)
        logger.info(f"Volume '{volume_name}' created.")


async def upload_repo_to_hf(
    task_id: str,
    expected_repo_name: str,
    huggingface_token: str,
    huggingface_username: str,
    wandb_token: str = None,
    path_in_repo: str = None
):
    try:
        client = docker.from_env()

        environment = {
            "HUGGINGFACE_TOKEN": huggingface_token,
            "HUGGINGFACE_USERNAME": huggingface_username,
            "WANDB_TOKEN": wandb_token or "",
            "TASK_ID": task_id,
            "EXPECTED_REPO_NAME": expected_repo_name,
            "HF_REPO_SUBFOLDER": path_in_repo
        }

        volumes = {
            cst.CHECKPOINTS_VOLUME_NAME: {
                "bind": cst.CONTAINER_SAVE_PATH,
                "mode": "rw"
            }
        }

        container_name = f"hf-upload-{uuid.uuid4().hex}"

        logger.info(f"Starting upload container {container_name} for task {task_id}...")

        container = client.containers.run(
            image=cst.HF_UPLOAD_DOCKER_IMAGE,
            environment=environment,
            volumes=volumes,
            detach=True,
            remove=True,
            name=container_name,
        )

        log_task = asyncio.create_task(
            asyncio.to_thread(stream_container_logs, container, get_all_context_tags())
        )

    except Exception as e:
        logger.exception(f"Unexpected error during upload_repo_to_hf for task {task_id}: {e}")
        raise


async def start_training_task(task: TrainerProxyRequest):
    training_data = task.training_data
    success = False 

    try:
        await create_volume_if_doesnt_exist()

        tag = await asyncio.to_thread(
            build_docker_image,
            dockerfile_path=f"{task.local_repo_path}/{cst.DEFAULT_IMAGE_DOCKERFILE_PATH}",
            context_path=task.local_repo_path,
        )
        log_task(training_data.task_id, task.hotkey, f"Docker image built with tag: {tag}")

        container = await asyncio.wait_for(
            run_trainer_container(
                task_id=training_data.task_id,
                tag=tag,
                model=training_data.model,
                dataset_zip=training_data.dataset_zip,
                model_type=training_data.model_type,
                expected_repo_name=training_data.expected_repo_name,
                hours_to_complete=training_data.hours_to_complete,
                gpu_ids=task.gpu_ids,
            ),
            timeout=60
        )

        timeout_seconds = training_data.hours_to_complete * 3600

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(container.wait),
                timeout=timeout_seconds
            )
            status_code = result.get("StatusCode", -1)
            log_task(training_data.task_id, task.hotkey, f"Container exited with status code {status_code}")

            if status_code != 0:
                log_task(training_data.task_id, task.hotkey, f"Training failed with status code {status_code}")
                return

            log_task(training_data.task_id, task.hotkey, "Training completed successfully.")

        except asyncio.TimeoutError:
            complete_task(training_data.task_id, task.hotkey, success=success)
            log_task(training_data.task_id, task.hotkey, f"Timeout reached ({timeout_seconds}s). Killing container...")
            container.kill()
            container.remove(force=True)
            log_task(training_data.task_id, task.hotkey, f"Container {container.name} killed due to timeout. Proceeding to upload.")

        except Exception as e:
            complete_task(training_data.task_id, task.hotkey, success=success)
            log_task(training_data.task_id, task.hotkey, f"Unexpected error during training: {e}")
            logger.exception(f"Error in training job {training_data.task_id}")
            try:
                container.kill()
                container.remove(force=True)
            except Exception as cleanup_error:
                complete_task(training_data.task_id, task.hotkey, success=success)
                log_task(training_data.task_id, task.hotkey, f"Error during container cleanup: {cleanup_error}")
            return

        await upload_repo_to_hf(
            task_id=training_data.task_id,
            expected_repo_name=training_data.expected_repo_name,
            huggingface_username=os.getenv("HUGGINGFACE_USERNAME"),
            huggingface_token=os.getenv("HUGGINGFACE_TOKEN"),
            path_in_repo=cst.IMAGE_TASKS_HF_SUBFOLDER_PATH
        )

        success = True

    except Exception as outer_e:
        complete_task(training_data.task_id, task.hotkey, success=success)
        log_task(training_data.task_id, task.hotkey, f"Fatal error during training pipeline: {outer_e}")
        logger.exception(f"Fatal error in training job {training_data.task_id}")

    finally:
        complete_task(training_data.task_id, task.hotkey, success=success)



