import io
import json
import os
import tarfile
import asyncio
from typing import Union, AsyncGenerator
import docker
from fiber.logging_utils import get_logger
from core import constants as cst
from core.models.payload_models import EvaluationResult
from core.models.utility_models import CustomDatasetType, DatasetType, FileFormat

logger = get_logger(__name__)


async def stream_logs_async(container) -> AsyncGenerator[str, None]:
    for line in container.logs(stream=True, follow=True):
        yield line.decode('utf-8').strip()
        await asyncio.sleep(0)


async def process_container_archive(container) -> dict:
    tar_stream, _ = await asyncio.to_thread(container.get_archive, cst.CONTAINER_EVAL_RESULTS_PATH)

    file_like_object = io.BytesIO()
    async for chunk in tar_stream:
        file_like_object.write(chunk)
        await asyncio.sleep(0)

    file_like_object.seek(0)

    def extract_results():
        with tarfile.open(fileobj=file_like_object) as tar:
            for member_info in tar.getmembers():
                if member_info.name.endswith("evaluation_results.json"):
                    eval_results_file = tar.extractfile(member_info)
                    if eval_results_file:
                        return json.loads(eval_results_file.read().decode("utf-8"))
            raise Exception("Evaluation results file not found in tar archive")

    return await asyncio.to_thread(extract_results)


async def run_evaluation_docker(
    dataset: str,
    model: str,
    original_model: str,
    dataset_type: Union[DatasetType, CustomDatasetType],
    file_format: FileFormat,
) -> EvaluationResult:

    client = await asyncio.to_thread(docker.from_env)

    if isinstance(dataset_type, DatasetType):
        dataset_type_str = dataset_type.value
    elif isinstance(dataset_type, CustomDatasetType):
        dataset_type_str = dataset_type.model_dump_json()
    else:
        raise ValueError("Invalid dataset_type provided.")

    environment = {
        "DATASET": dataset,
        "MODEL": model,
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type_str,
        "FILE_FORMAT": file_format.value,
    }

    dataset_dir = os.path.dirname(os.path.abspath(dataset))
    volume_bindings = {
        dataset_dir: {
            "bind": "/workspace/input_data",
            "mode": "ro",
        }
    }

    container = None
    try:
        container = await asyncio.to_thread(
            client.containers.run,
            cst.VALIDATOR_DOCKER_IMAGE,
            environment=environment,
            volumes=volume_bindings,
            runtime="nvidia",
            device_requests=[
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ],
            detach=True,
        )

        log_task = asyncio.create_task(handle_logs(container))

        result = await asyncio.to_thread(container.wait)

        await log_task

        if result["StatusCode"] != 0:
            raise Exception(
                f"Container exited with status {result['StatusCode']}")

        eval_results = await process_container_archive(container)

        await asyncio.to_thread(container.remove)

        return EvaluationResult(**eval_results)

    except Exception as e:
        logger.error(f"Failed to retrieve evaluation results: {str(e)}")
        raise Exception(f"Failed to retrieve evaluation results: {str(e)}")

    finally:
        try:
            if container:
                await asyncio.to_thread(container.remove, force=True)
            await asyncio.to_thread(client.images.prune)
        except Exception as cleanup_error:
            logger.error(f"Cleanup error: {cleanup_error}")
        await asyncio.to_thread(client.close)


async def handle_logs(container) -> None:
    async for log_line in stream_logs_async(container):
        logger.info(log_line)
