import os
import docker
import uuid
import json
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
from core.models.payload_models import TrainRequestImage, TrainRequestText
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskType


logger = get_logger(__name__)

def build_docker_image(
    dockerfile_path: str = cst.DEFAULT_IMAGE_DOCKERFILE_PATH,
    context_path: str = ".",
    is_image_task: bool = False,
    tag: str = None,
    no_cache: bool = True
) -> str:
    client: docker.DockerClient = docker.from_env()

    if tag is None:
        tag = f"standalone-image-trainer:{uuid.uuid4()}" if is_image_task else f"standalone-text-trainer:{uuid.uuid4()}"

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


async def run_trainer_container_image(
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
                cst.CHECKPOINTS_VOLUME_NAME: {"bind": cst.IMAGE_CONTAINER_SAVE_PATH, "mode": "rw"}
            },
            remove=False,
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
    

async def run_trainer_container_text(
    task_id: str,
    tag: str,
    model: str,
    dataset: str,
    dataset_type: InstructTextDatasetType | DpoDatasetType,
    task_type: TaskType,
    file_format: FileFormat,
    expected_repo_name: str,
    hours_to_complete: int=1,
    gpu_ids: list[int]=[0]
) -> Container:
    client: docker.DockerClient = docker.from_env()

    environment = {
            "WANDB_TOKEN": os.getenv("WANDB_TOKEN"),
            "WANDB_API_KEY": os.getenv("WANDB_TOKEN")
        }

    command: list[str] = [
        "--task-id", task_id,
        "--model", model,
        "--dataset", dataset,
        "--dataset-type", json.dumps(dataset_type.model_dump()),
        "--task-type", task_type,
        "--file-format", file_format,
        "--hours-to-complete", str(hours_to_complete),
        "--expected-repo-name", expected_repo_name
    ]
    
    container_name = f"text-trainer-{uuid.uuid4().hex}"

    try:
        container: Container = client.containers.run(
            image=tag,
            command=command,
            volumes={
                cst.CHECKPOINTS_VOLUME_NAME: {"bind": cst.TEXT_CONTAINER_SAVE_PATH, "mode": "rw"}
            },
            remove=False,
            name=container_name,
            mem_limit=cst.DEFAULT_TRAINING_CONTAINER_MEM_LIMIT,
            nano_cpus=cst.DEFAULT_TRAINING_CONTAINER_NANO_CPUS * 1_000_000_000,
            device_requests=[docker.types.DeviceRequest(device_ids=[str(i) for i in gpu_ids],  capabilities=[["gpu"]])],
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            detach=True,
            environment=environment
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
    task_type: TaskType,
    wandb_token: str | None = None,
    path_in_repo: str | None = None
):
    try:
        client = docker.from_env()

        local_container_folder = cst.IMAGE_CONTAINER_SAVE_PATH if task_type == TaskType.IMAGETASK else cst.TEXT_CONTAINER_SAVE_PATH

        environment = {
            "HUGGINGFACE_TOKEN": huggingface_token,
            "HUGGINGFACE_USERNAME": huggingface_username,
            "WANDB_TOKEN": wandb_token or "",
            "LOCAL_FOLDER": local_container_folder,
            "TASK_ID": task_id,
            "EXPECTED_REPO_NAME": expected_repo_name,
            "HF_REPO_SUBFOLDER": path_in_repo
        }

        container_path = cst.IMAGE_CONTAINER_SAVE_PATH if task_type == TaskType.IMAGETASK else cst.TEXT_CONTAINER_SAVE_PATH

        volumes = {
            cst.CHECKPOINTS_VOLUME_NAME: {
                "bind": container_path,
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


def get_task_type(request: TrainerProxyRequest) -> TaskType:
    training_data = request.training_data

    if isinstance(training_data, TrainRequestImage):
        return TaskType.IMAGETASK

    elif isinstance(training_data, TrainRequestText):
        if isinstance(training_data.dataset_type, DpoDatasetType):
            return TaskType.DPOTASK
        elif isinstance(training_data.dataset_type, InstructTextDatasetType):
            return TaskType.INSTRUCTTEXTTASK
        else:
            raise ValueError(f"Unsupported dataset_type for text task: {type(training_data.dataset_type)}")

    raise ValueError(f"Unsupported training_data type: {type(training_data)}")
    

async def start_training_task(task: TrainerProxyRequest):
    training_data = task.training_data
    success = False
    container = None
    timeout_seconds = training_data.hours_to_complete * 3600
    task_type = get_task_type(task)
    logger.info(f"Task Type: {task_type}")

    try:
        await create_volume_if_doesnt_exist()

        dockerfile_path = f"{task.local_repo_path}/{cst.DEFAULT_IMAGE_DOCKERFILE_PATH}" if task_type == TaskType.IMAGETASK else f"{task.local_repo_path}/{cst.DEFAULT_TEXT_DOCKERFILE_PATH}"

        tag = await asyncio.to_thread(
            build_docker_image,
            dockerfile_path=dockerfile_path,
            is_image_task=(task_type == TaskType.IMAGETASK),
            context_path=task.local_repo_path,
        )
        log_task(training_data.task_id, task.hotkey, f"Docker image built with tag: {tag}")

        if task_type == TaskType.IMAGETASK:
            container = await asyncio.wait_for(
                run_trainer_container_image(
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
        else:
            container = await asyncio.wait_for(
                run_trainer_container_text(
                    task_id=training_data.task_id,
                    tag=tag,
                    model=training_data.model,
                    dataset=training_data.dataset,
                    dataset_type=training_data.dataset_type,
                    task_type=task_type,
                    file_format=training_data.file_format,
                    expected_repo_name=training_data.expected_repo_name,
                    hours_to_complete=training_data.hours_to_complete,
                    gpu_ids=task.gpu_ids,
                ),
                timeout=60
            )

        log_task(training_data.task_id, task.hotkey, f"Container started: {container.name}")

        log_task(training_data.task_id, task.hotkey, f"Waiting for container to finish (timeout={timeout_seconds})...")
        wait_task = asyncio.create_task(asyncio.to_thread(container.wait))
        done, pending = await asyncio.wait({wait_task}, timeout=timeout_seconds)
        log_task(training_data.task_id, task.hotkey, "Container wait completed or timed out.")

        if wait_task in done:
            result = await wait_task
            logger.info(f"Container.wait() returned: {result}")
            status_code = result.get("StatusCode", -1)
            if status_code == 0:
                log_task(training_data.task_id, task.hotkey, "Training completed successfully.")
                success = True
            else:
                complete_task(training_data.task_id, task.hotkey, success=success)
                log_task(training_data.task_id, task.hotkey, f"Training failed with status code {status_code}")
        else:
            log_task(training_data.task_id, task.hotkey, f"Timeout reached ({timeout_seconds}s). Killing container...")
            success = True  

    except Exception as e:
        log_task(training_data.task_id, task.hotkey, f"Fatal error during training: {e}")
        logger.exception(f"Training job failed: {training_data.task_id}")

    finally:
        if container:
            try:
                container.kill()
                container.remove(force=True)
                log_task(training_data.task_id, task.hotkey, f"Container {container.name} cleaned up.")
            except Exception as cleanup_err:
                log_task(training_data.task_id, task.hotkey, f"Error during container cleanup: {cleanup_err}")

        if success:
            try:
                path_in_repo= cst.IMAGE_TASKS_HF_SUBFOLDER_PATH if task_type == TaskType.IMAGETASK else None
                await upload_repo_to_hf(
                    task_id=training_data.task_id,
                    expected_repo_name=training_data.expected_repo_name,
                    huggingface_username=os.getenv("HUGGINGFACE_USERNAME"),
                    huggingface_token=os.getenv("HUGGINGFACE_TOKEN"),
                    task_type=task_type,
                    wandb_token=os.getenv("WANDB_TOKEN", None),
                    path_in_repo=path_in_repo
                )
                log_task(training_data.task_id, task.hotkey, "Repo uploaded successfully.")
            except Exception as upload_err:
                log_task(training_data.task_id, task.hotkey, f"Upload to HuggingFace failed: {upload_err}")
                success = False

        complete_task(training_data.task_id, task.hotkey, success=success)



