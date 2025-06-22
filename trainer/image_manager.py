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
        raise e


async def run_trainer_container(
    task_id: str,
    tag: str,
    model: str,
    dataset_zip: str,
    model_type: str,
    hours_to_complete: int=1,
    gpu_ids: list[int]=[0]
) -> Container:
    client: docker.DockerClient = docker.from_env()

    checkpoints_dir: str = cst.TRAINER_CHECKPOINTS_PATH
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.chmod(checkpoints_dir, 0o700)

    command: list[str] = [
        "--task-id", task_id,
        "--model", model,
        "--dataset-zip", dataset_zip,
        "--model-type", model_type,
        "--hours-to-complete", str(hours_to_complete),
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
