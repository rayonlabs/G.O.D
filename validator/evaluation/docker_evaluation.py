import asyncio
import io
import json
import os
import tarfile
import threading
from typing import Union

import docker
from fiber.logging_utils import get_logger
from pydantic import TypeAdapter

from core import constants as cst
from core.docker_utils import stream_logs
from core.models.payload_models import EvaluationResult
from core.models.payload_models import EvaluationResultDiffusion
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import DatasetType
from core.models.utility_models import FileFormat


logger = get_logger(__name__)


async def get_evaluation_results(container):
    archive_data = await asyncio.to_thread(container.get_archive, cst.CONTAINER_EVAL_RESULTS_PATH_DIFFUSION)
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
            if member_info.name.endswith(("evaluation_results_diffusion.json")):
                eval_results_file = tar.extractfile(member_info)
                break

        if eval_results_file is None:
            raise Exception("Evaluation results file not found in tar archive")

        eval_results_content = eval_results_file.read().decode("utf-8")
        return json.loads(eval_results_content)


async def run_evaluation_docker(
    dataset: str,
    models: list[str],
    original_model: str,
    dataset_type: Union[DatasetType, CustomDatasetType],
    file_format: FileFormat,
    gpu_ids: list[int],
) -> dict[str, Union[EvaluationResult, Exception]]:
    client = docker.from_env()

    if isinstance(dataset_type, DatasetType):
        dataset_type_str = dataset_type.value
    elif isinstance(dataset_type, CustomDatasetType):
        dataset_type_str = dataset_type.model_dump_json()
    else:
        raise ValueError("Invalid dataset_type provided.")

    environment = {
        "DATASET": dataset,
        "MODELS": ",".join(models),
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

    async def cleanup_resources():
        try:
            await asyncio.to_thread(client.containers.prune)
            await asyncio.to_thread(client.images.prune, filters={"dangling": True})
            await asyncio.to_thread(client.volumes.prune)
            logger.debug("Completed Docker resource cleanup")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

    try:
        container = await asyncio.to_thread(
            client.containers.run,
            cst.VALIDATOR_DOCKER_IMAGE,
            environment=environment,
            volumes=volume_bindings,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=[str(gid) for gid in gpu_ids])],
            detach=True,
        )
        log_task = asyncio.create_task(asyncio.to_thread(stream_logs, container))
        result = await asyncio.to_thread(container.wait)
        log_task.cancel()

        if result["StatusCode"] != 0:
            raise Exception(f"Container exited with status {result['StatusCode']}")

        eval_results_dict = await get_evaluation_results(container)

        processed_results = {}
        for repo, result in eval_results_dict.items():
            if isinstance(result, str) and not isinstance(result, dict):
                processed_results[repo] = Exception(result)
            else:
                processed_results[repo] = TypeAdapter(EvaluationResult).validate_python(result)

        return processed_results

    except Exception as e:
        logger.error(f"Failed to retrieve evaluation results: {str(e)}")
        raise Exception(f"Failed to retrieve evaluation results: {str(e)}")

    finally:
        try:
            await asyncio.to_thread(container.remove, force=True)
            await cleanup_resources()
        except Exception as e:
            logger.info(f"A problem with cleaning up {e}")
        client.close()


# TODO: CLEAN this up
async def run_evaluation_docker_diffusion(
    test_split_path: str,
    base_model_repo: str,
    base_model_filename: str,
    lora_repo_list: str,
    lora_filename_list: str
) -> EvaluationResultDiffusion:
    client = docker.from_env()

    dataset_dir = os.path.abspath(test_split_path)
    container_dataset_path = "/workspace/input_data"
    volume_bindings = {}
    volume_bindings[dataset_dir] = {
        "bind": container_dataset_path,
        "mode": "ro",
    }

    environment = {
        "TEST_DATASET_PATH": container_dataset_path,
        "BASE_MODEL_REPO": base_model_repo,
        "BASE_MODEL_FILENAME": base_model_filename,
        "TRAINED_LORA_MODEL_REPOS": lora_repo_list,
        "LORA_MODEL_FILENAMES": lora_filename_list,
    }

    async def cleanup_resources():
        try:
            await asyncio.to_thread(client.containers.prune)
            await asyncio.to_thread(client.images.prune, filters={"dangling": True})
            await asyncio.to_thread(client.volumes.prune)
            logger.debug("Completed Docker resource cleanup")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

    try:
        container = await asyncio.to_thread(
            client.containers.run,
            cst.VALIDATOR_DOCKER_IMAGE_DIFFUSION,
            environment=environment,
            volumes=volume_bindings,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
            detach=True,
        )
        log_task = asyncio.create_task(asyncio.to_thread(stream_logs, container))
        result = await asyncio.to_thread(container.wait)
        log_task.cancel()

        if result["StatusCode"] != 0:
            raise Exception(f"Container exited with status {result['StatusCode']}")

        eval_results_dict = await get_evaluation_results(container)

        return eval_results_dict

    except Exception as e:
        logger.error(f"Failed to retrieve evaluation results: {str(e)}")
        raise Exception(f"Failed to retrieve evaluation results: {str(e)}")

    finally:
        try:
            await asyncio.to_thread(container.remove, force=True)
            await cleanup_resources()
        except Exception as e:
            logger.info(f"A problem with cleaning up {e}")
        client.close()
