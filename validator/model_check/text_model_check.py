import asyncio
import json
import os
import tarfile
import tempfile
from io import BytesIO
from typing import Any
from typing import Dict

import docker
from docker.errors import NotFound as DockerNotFound

from core import constants as cst
from validator.core import constants as vali_cst
from validator.utils.logging import get_all_context_tags
from validator.utils.logging import get_logger
from validator.utils.logging import stream_container_logs


logger = get_logger(__name__)


async def _extract_results_from_container(container) -> Dict[str, Any] | None:
    try:
        stream, _ = await asyncio.to_thread(container.get_archive, vali_cst.CHECK_RESULTS_PATH)

        buffer = BytesIO()
        for chunk in stream:
            buffer.write(chunk)
        buffer.seek(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(fileobj=buffer, mode="r:") as tar:
                members = [m for m in tar.getmembers() if m.name == os.path.basename(vali_cst.CHECK_RESULTS_PATH)]
                if not members:
                    logger.warning(
                        f"Results file '{os.path.basename(vali_cst.CHECK_RESULTS_PATH)}' not found in archive"
                        f" from container {container.short_id}."
                    )
                    return None
                tar.extractall(path=tmpdir, members=[members[0]])
            result_file_path = os.path.join(tmpdir, os.path.basename(vali_cst.CHECK_RESULTS_PATH))
            if os.path.exists(result_file_path):
                with open(result_file_path, "r") as f:
                    results = json.load(f)
                logger.info(
                    f"Successfully extracted and loaded results from {vali_cst.CHECK_RESULTS_PATH}"
                    f" in container {container.short_id}."
                )
                return results
            else:
                logger.warning(
                    f"Results file path '{result_file_path}' not found after extraction for container {container.short_id}."
                )
                return None

    except DockerNotFound:
        logger.warning(f"Results file {vali_cst.CHECK_RESULTS_PATH} not found in container {container.short_id}.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from results file in container {container.short_id}: {e}")
        return None
    except (tarfile.TarError, OSError, Exception) as e:
        logger.error(
            f"Error extracting results file from container {container.short_id}: {type(e).__name__} - {e}", exc_info=True
        )
        return None


async def run_model_loading_and_inference_check(model_id: str, gpu_ids: list[int]) -> Dict[str, Any]:
    client = docker.from_env()

    environment = {
        "MODEL_ID": model_id,
        "NVIDIA_VISIBLE_DEVICES": ",".join(str(gid) for gid in gpu_ids),
    }

    command = ["python", "-m", "validator.model_check.check_model_inference"]

    container = None
    default_error_result = {
        "model_id": model_id,
        "status": "Failure",
        "parameter_count": None,
        "error_message": "Unknown error during container execution.",
    }

    try:
        container = await asyncio.to_thread(
            client.containers.run,
            cst.VALIDATOR_DOCKER_IMAGE,
            command=command,
            environment=environment,
            volumes=volume_bindings,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=[str(gid) for gid in gpu_ids])],
            detach=True,
        )

        log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, get_all_context_tags()))
        result = await asyncio.to_thread(container.wait)
        await log_task

        extracted_results = await _extract_results_from_container(container)

        if result["StatusCode"] == 0:
            if extracted_results:
                logger.info(f"Model check successful (exit code 0) for {model_id}. Results extracted.")
                if extracted_results.get("status") != "Success":
                    logger.warning(f"Container exit code 0, but extracted results status is '{extracted_results.get('status')}'")
                    extracted_results["status"] = "Success"
                return extracted_results
            else:
                logger.error(
                    f"Model check container exited successfully (0) for {model_id}, but failed to extract/parse results JSON."
                )
                default_error_result["error_message"] = "Container exited successfully, but results JSON extraction failed."
                return default_error_result
        else:
            # Container failed
            error_message = f"Container exited with code {result['StatusCode']} for model {model_id}."
            logs = ""
            try:
                logs = await asyncio.to_thread(container.logs)
                logs = logs.decode("utf-8", errors="replace")
                error_message += f"\nLogs:\n{logs}"
            except Exception as log_err:
                error_message += f"\n(Failed to retrieve container logs: {log_err})"

            logger.error(error_message)

            if extracted_results and extracted_results.get("error_message"):
                logger.info("Using error message from extracted results JSON.")
                extracted_results["status"] = "Failure"
                return extracted_results
            else:
                default_error_result["error_message"] = error_message
                return default_error_result

    except docker.errors.ContainerError as e:
        error_message = f"ContainerError during model check for {model_id}: {e}"
        if container:
            try:
                logs = await asyncio.to_thread(container.logs)
                error_message += f"\nLogs:\n{logs.decode('utf-8', errors='replace')}"
            except Exception as log_err:
                logger.error(f"Failed to retrieve logs after ContainerError: {log_err}")
        logger.error(error_message, exc_info=True)
        default_error_result["error_message"] = error_message
        return default_error_result
    except Exception as e:
        error_message = f"Unexpected error during model check for {model_id}: {type(e).__name__} - {str(e)}"
        logger.error(error_message, exc_info=True)
        default_error_result["error_message"] = error_message
        return default_error_result

    finally:
        try:
            if container:
                await asyncio.to_thread(container.remove, force=True)
                logger.debug(f"Removed container {container.short_id} for model {model_id}")
        except Exception as e:
            logger.warning(f"Problem cleaning up container for model {model_id}: {e}")
        try:
            await asyncio.to_thread(client.close)
        except Exception as e:
            logger.warning(f"Problem closing Docker client: {e}")
