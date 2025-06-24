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
from core.models.payload_models import TrainerProxyJobImage


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

    checkpoints_dir: str = f"{cst.TRAINER_CHECKPOINTS_PATH}/{task_id}/"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.chmod(checkpoints_dir, 0o700)

    command: list[str] = [
        "--task-id", task_id,
        "--model", model,
        "--dataset-zip", dataset_zip,
        "--model-type", model_type,
        "--hours-to-complete", str(hours_to_complete),
        "--expected-repo-name", expected_repo_name
    ]
    
    try:
        container: Container = client.containers.run(
            image=tag,
            command=command,
            volumes={
                checkpoints_dir: {"bind": cst.CONTAINER_SAVE_PATH, "mode": "rw"}
            },
            remove=True,
            name="image-trainer-example",
            mem_limit="16g",
            nano_cpus=8 * 1_000_000_000,
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

async def start_training_task(task: TrainerProxyJobImage):
    tag = await asyncio.to_thread(
        build_docker_image,
        dockerfile_path=f"{task.local_repo_path}/{cst.DEFAULT_IMAGE_DOCKERFILE_PATH}",
        context_path=task.local_repo_path,
    )

    log_task(task.task_id, f"Docker image built with tag: {tag}")

    container = await asyncio.wait_for(
        run_trainer_container(
            task_id=task.task_id,
            tag=tag,
            model=task.model,
            dataset_zip=task.dataset_zip,
            model_type=task.model_type,
            expected_repo_name=task.expected_repo_name,
            hours_to_complete=task.hours_to_complete,
            gpu_ids=task.gpu_ids,
        ),
        timeout=60
    )

    timeout_seconds = task.hours_to_complete * 3600

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(container.wait),
            timeout=timeout_seconds
        )
        status_code = result.get("StatusCode", -1)
        log_task(task.task_id, f"Container exited with status code {status_code}")

        if status_code == 0:
            log_task(task.task_id, "Training completed successfully.")
            complete_task(task.task_id, success=True)
        else:
            log_task(task.task_id, f"Training failed with status code {status_code}.")
            complete_task(task.task_id, success=False)

    except asyncio.TimeoutError:
        log_task(task.task_id, f"Timeout reached ({timeout_seconds}s). Killing container...")
        logger.info(f"Timeout reached for job {task.task_id}. Killing container...")
        container.kill()
        container.remove(force=True)
        log_task(task.task_id, f"Container {container.name} killed due to timeout. Marking task as complete.")
        complete_task(task.task_id, success=True)

    except Exception as e:
        log_task(task.task_id, f"Unexpected error during training: {e}")
        logger.exception(f"Error in training job {task.task_id}")
        try:
            container.kill()
            container.remove(force=True)
            log_task(task.task_id, f"Container {container.name} forcibly removed due to error.")
        except Exception as cleanup_error:
            log_task(task.task_id, f"Error during container cleanup: {cleanup_error}")
        complete_task(task.task_id, success=False)

