import io
import json
import os
import tarfile
import asyncio
from typing import Union
import docker
from fiber.logging_utils import get_logger
from core import constants as cst
from core.docker_utils import stream_logs
from core.models.payload_models import EvaluationResult
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import DatasetType
from core.models.utility_models import FileFormat

logger = get_logger(__name__)


async def get_evaluation_results(container):
    archive_data = await asyncio.to_thread(
        container.get_archive,
        cst.CONTAINER_EVAL_RESULTS_PATH
    )
    tar_stream = archive_data[0]

    file_like_object = io.BytesIO()
    for chunk in tar_stream:
        file_like_object.write(chunk)
    file_like_object.seek(0)

    with tarfile.open(fileobj=file_like_object) as tar:
        members = tar.getnames()
        logger.debug(f"Tar archive members: {members}")
        eval_results_file = None
        for member_info in tar.getmembers():
            if member_info.name.endswith("evaluation_results.json"):
                eval_results_file = tar.extractfile(member_info)
                break

        if eval_results_file is None:
            raise Exception("Evaluation results file not found in tar archive")

        eval_results_content = eval_results_file.read().decode("utf-8")
        return json.loads(eval_results_content)


async def run_evaluation_docker(
    dataset: str,
    model: str,
    original_model: str,
    dataset_type: Union[DatasetType, CustomDatasetType],
    file_format: FileFormat,
) -> EvaluationResult:
    client = docker.from_env()

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
        log_task = asyncio.create_task(
            asyncio.to_thread(stream_logs, container))
        result = await asyncio.to_thread(container.wait)
        log_task.cancel()

        if result["StatusCode"] != 0:
            raise Exception(
                f"Container exited with status {result['StatusCode']}")

        eval_results = await get_evaluation_results(container)

        await asyncio.to_thread(container.remove)

        return EvaluationResult(**eval_results)

    except Exception as e:
        logger.error(f"Failed to retrieve evaluation results: {str(e)}")
        raise Exception(f"Failed to retrieve evaluation results: {str(e)}")

    finally:
        try:
            await asyncio.to_thread(container.remove, force=True)
            await asyncio.to_thread(client.images.prune)
        except:
            pass
        client.close()
