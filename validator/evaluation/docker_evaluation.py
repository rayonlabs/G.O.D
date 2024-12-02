import io
import json
import os
import tarfile
import threading
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
    volume_bindings = {}
    volume_bindings[dataset_dir] = {
        "bind": "/workspace/input_data",
        "mode": "ro",
    }

    try:
        container = client.containers.run(
            cst.VALIDATOR_DOCKER_IMAGE,
            environment=environment,
            volumes=volume_bindings,
            runtime="nvidia",
            device_requests=[
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ],
            detach=True,
        )
        # NOTE: replace with asyncio.to_thread

        log_thread = threading.Thread(target=stream_logs, args=(container,))
        log_thread.start()

        result = container.wait()

        log_thread.join()

        if result["StatusCode"] != 0:
            raise Exception(f"Container exited with status {result['StatusCode']}")

        # Confession, this is a bit of an llm hack, I had issues pulling from the path directly and
        # llm said this was a better solution and faster ... it works so llm knows best :D
        # TODO: come back to this and make it more readable / actually understand it
        tar_stream, _ = container.get_archive(cst.CONTAINER_EVAL_RESULTS_PATH)

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
            eval_results = json.loads(eval_results_content)

        container.remove()
        return EvaluationResult(**eval_results)
    except Exception as e:
        logger.error(f"Failed to retrieve evaluation results: {str(e)}")
        raise Exception(f"Failed to retrieve evaluation results: {str(e)}")
    finally:
        try:
            container.remove(force=True)
        except:
            pass
        client.close()
