import asyncio
import io
import json
import os
import re
import shutil
import tarfile

import docker
from docker.models.containers import Container
from docker.types import Mount
from huggingface_hub import snapshot_download

from core import constants as cst
from core.models.payload_models import DockerEvaluationResults
from core.models.payload_models import EvaluationResultImage
from core.models.payload_models import EvaluationResultText
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import ImageModelType
from core.models.utility_models import InstructTextDatasetType
from core.utils import download_s3_file
from validator.tasks.task_prep import unzip_to_temp_path
from validator.utils.logging import get_all_context_tags
from validator.utils.logging import get_logger
from validator.utils.logging import stream_container_logs


logger = get_logger(__name__)


async def cleanup_resources(client):
    """Clean up Docker resources including containers, images, and volumes."""
    try:
        await asyncio.to_thread(client.containers.prune)
        await asyncio.to_thread(client.images.prune, filters={"dangling": True})
        await asyncio.to_thread(client.volumes.prune)
        logger.debug("Completed Docker resource cleanup")
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")


async def get_evaluation_results(container):
    archive_data = await asyncio.to_thread(container.get_archive, cst.CONTAINER_EVAL_RESULTS_PATH)
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
            if member_info.name.endswith(("evaluation_results.json")):
                eval_results_file = tar.extractfile(member_info)
                break

        if eval_results_file is None:
            raise Exception("Evaluation results file not found in tar archive")

        eval_results_content = eval_results_file.read().decode("utf-8")
        return json.loads(eval_results_content)


def process_evaluation_results(results: dict, is_image: bool = False) -> DockerEvaluationResults:
    model_params_count = results.pop("model_params_count", 0)

    processed_results = {}
    for repo, result in results.items():
        if isinstance(result, str) and not isinstance(result, dict):
            processed_results[repo] = Exception(result)
        else:
            if is_image:
                result["is_finetune"] = True
                processed_results[repo] = EvaluationResultImage.model_validate(result)
            else:
                processed_results[repo] = EvaluationResultText.model_validate(result)

    return DockerEvaluationResults(
        results=processed_results,
        base_model_params_count=model_params_count
    )


async def run_evaluation_docker_text(
    dataset: str,
    models: list[str],
    original_model: str,
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType | ChatTemplateDatasetType,
    file_format: FileFormat,
    gpu_ids: list[int],
) -> DockerEvaluationResults:

    if isinstance(dataset_type, (InstructTextDatasetType, ChatTemplateDatasetType)):
        command = ["python", "-m", "validator.evaluation.eval_instruct_text"]
    elif isinstance(dataset_type, DpoDatasetType):
        command = ["python", "-m", "validator.evaluation.eval_dpo"]
    elif isinstance(dataset_type, GrpoDatasetType):
        return await run_evaluation_docker_grpo(dataset, models, original_model, dataset_type, file_format, gpu_ids)
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset_type)}")
    task_type = type(dataset_type).__name__

    client = docker.from_env()
    dataset_type_str = dataset_type.model_dump_json()
    dataset_filename = os.path.basename(dataset)
    dataset_dir = os.path.dirname(os.path.abspath(dataset))

    environment = {
        "DATASET": f"/workspace/input_data/{dataset_filename}",
        "MODELS": ",".join(models),
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type_str,
        "FILE_FORMAT": file_format.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
    }
    logger.info(f"Running {task_type} evaluation for models: {models}")

    volume_bindings = {
        dataset_dir: {
            "bind": "/workspace/input_data",
            "mode": "ro",
        },
        os.path.expanduser(cst.CACHE_DIR_HUB): {
            "bind": "/root/.cache/huggingface/hub",
            "mode": "rw",
        }
    }

    try:
        container: Container = await asyncio.to_thread(
            client.containers.run,
            cst.VALIDATOR_DOCKER_IMAGE,
            command=command,
            environment=environment,
            volumes=volume_bindings,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=[str(gid) for gid in gpu_ids])],
            detach=True,
        )
        log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, None, get_all_context_tags()))
        result = await asyncio.to_thread(container.wait)
        log_task.cancel()

        if result["StatusCode"] != 0:
            raise Exception(f"Container exited with status {result['StatusCode']}")

        eval_results = await get_evaluation_results(container)
        return process_evaluation_results(eval_results, is_image=False)

    except Exception as e:
        logger.error(f"Failed to retrieve {task_type} evaluation results: {str(e)}", exc_info=True)
        raise Exception(f"Failed to retrieve {task_type} evaluation results: {str(e)}")

    finally:
        try:
            await asyncio.to_thread(container.remove, force=True)
            await cleanup_resources(client)
        except Exception as e:
            logger.info(f"A problem with cleaning up {e}")
        client.close()


async def run_evaluation_docker_grpo(
    dataset: str,
    models: list[str],
    original_model: str,
    dataset_type: GrpoDatasetType,
    file_format: FileFormat,
    gpu_ids: list[int],
) -> DockerEvaluationResults:
    """
    Run GRPO evaluation with separate containers for each model repo.
    This approach launches one container per repo and merges results.
    """
    logger.info(f"Downloading original GRPO model: {original_model}")
    cache_dir = os.path.expanduser(cst.CACHE_DIR_HUB)
    original_model_path = await asyncio.to_thread(
        snapshot_download, 
        repo_id=original_model, 
        cache_dir=cache_dir,
        ignore_patterns=None
    )
    logger.info(f"Original model downloaded to: {original_model_path}")
    
    # Log what files were downloaded for the original model
    if os.path.exists(original_model_path):
        files = os.listdir(original_model_path)
        logger.info(f"Original model files downloaded: {files}")
        tokenizer_files = [f for f in files if 'tokenizer' in f.lower() or f.endswith('.model')]
        if tokenizer_files:
            logger.info(f"Tokenizer files found: {tokenizer_files}")
        else:
            logger.warning(f"WARNING: No tokenizer files found in original model download!")

    command = ["python", "-m", "validator.evaluation.eval_grpo"]
    dataset_type_str = dataset_type.model_dump_json()
    dataset_filename = os.path.basename(dataset)
    dataset_dir = os.path.dirname(os.path.abspath(dataset))

    # Shared environment settings
    base_environment = {
        "DATASET": f"/workspace/input_data/{dataset_filename}",
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type_str,
        "FILE_FORMAT": file_format.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
    }

    volume_bindings = {
        dataset_dir: {
            "bind": "/workspace/input_data",
            "mode": "ro",
        },
        os.path.expanduser(cst.CACHE_DIR_HUB): {
            "bind": "/root/.cache/huggingface/hub",
            "mode": "rw",
        }
    }

    logger.info(f"Starting sequential GRPO evaluation for {len(models)} repos: {models}")

    evaluation_results = {}
    for repo in models:
        client = docker.from_env()
        environment = base_environment.copy()
        environment["MODELS"] = repo
        try:
            logger.info(f"Starting download of model {repo}...")
            model_path = await asyncio.to_thread(
                snapshot_download, 
                repo_id=repo, 
                cache_dir=cache_dir,
                ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.pkl", "*.pth"]
            )
            logger.info(f"Model {repo} downloaded to: {model_path}")
            
            # Log what files are actually in the downloaded model
            if os.path.exists(model_path):
                files = os.listdir(model_path)
                logger.info(f"Downloaded files for {repo}: {files}")
                
                # Check file sizes to ensure they're not just LFS pointers
                for file in files:
                    if file.endswith(('.safetensors', '.bin')):
                        file_path = os.path.join(model_path, file)
                        file_size = os.path.getsize(file_path)
                        logger.info(f"  {file}: {file_size / (1024*1024*1024):.2f} GB")
                        
                        # LFS pointer files are typically < 1KB
                        if file_size < 1000:
                            logger.warning(f"WARNING: {file} appears to be an LFS pointer (only {file_size} bytes)")
                
                # Check for essential files
                has_config = 'config.json' in files
                has_weights = any(f.endswith(('.safetensors', '.bin')) for f in files)
                logger.info(f"Model validation - has config.json: {has_config}, has model weights: {has_weights}")
            else:
                logger.error(f"Model path does not exist after download: {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to download {repo}: {str(e)}")
            evaluation_results[repo] = f"Failed to download model: {str(e)}"
            continue

        container = None  # Initialize container variable
        try:
            logger.info(f"Creating container for {repo} with GPUs: {gpu_ids}")
            logger.info(f"Docker image: {cst.VALIDATOR_DOCKER_IMAGE}")
            logger.info(f"Command: {command}")
            logger.info(f"Environment variables set: {list(environment.keys())}")
            logger.info(f"Volume bindings: {list(volume_bindings.keys())}")
            
            container: Container = await asyncio.to_thread(
                client.containers.run,
                cst.VALIDATOR_DOCKER_IMAGE,
                command=command,
                environment=environment,
                volumes=volume_bindings,
                runtime="nvidia",
                device_requests=[docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=[str(gid) for gid in gpu_ids])],
                detach=True,
                network_mode="none",
            )
            logger.info(f"Container created successfully for {repo} - ID: {container.id[:12]}")

            log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, None, get_all_context_tags()))
            result = await asyncio.to_thread(container.wait)
            log_task.cancel()

            if result["StatusCode"] != 0:

                logger.error(f"Container for {repo} exited with non-zero status: {result['StatusCode']}")
                # Try to get container logs to understand the failure
                try:
                    logs = await asyncio.to_thread(container.logs, tail=200)
                    error_logs = logs.decode('utf-8') if isinstance(logs, bytes) else str(logs)
                    logger.error(f"Last 200 lines of container logs for {repo}:\n{error_logs}")
                    
                    # Look for specific error patterns
                    if "ModuleNotFoundError" in error_logs or "ImportError" in error_logs:
                        import_match = re.search(r"(ModuleNotFoundError|ImportError): (.+)", error_logs)
                        if import_match:
                            error_msg = f"Missing module: {import_match.group(2)}"
                            logger.error(f"Container failed due to missing module: {import_match.group(2)}")
                            evaluation_results[repo] = f"Container failed - {error_msg}"
                        else:
                            evaluation_results[repo] = f"Container failed - Import error detected"
                    else:
                        # Include first error line in the result
                        error_lines = [line for line in error_logs.split('\n') if line.strip()]
                        if error_lines:
                            evaluation_results[repo] = f"Container failed - {error_lines[-1][:100]}"
                        else:
                            evaluation_results[repo] = f"Container for {repo} exited with status {result['StatusCode']}"
                except Exception as log_error:
                    logger.error(f"Failed to retrieve logs for {repo}: {str(log_error)}")
                    evaluation_results[repo] = f"Container for {repo} exited with status {result['StatusCode']}"

            else:
                eval_results = await get_evaluation_results(container)
                evaluation_results[repo] = eval_results[repo]
                if "model_params_count" in eval_results and "model_params_count" not in evaluation_results:
                    evaluation_results["model_params_count"] = eval_results["model_params_count"]

        except Exception as e:
            logger.error(f"Failed to evaluate repo {repo}: {str(e)}", exc_info=True)
            evaluation_results[repo] = str(e)

        finally:
            try:
                if container is not None:
                    await asyncio.to_thread(container.remove, force=True)
                await cleanup_resources(client)
            except Exception as e:
                logger.info(f"Problem with cleaning up container for {repo}: {e}")
            client.close()

    return process_evaluation_results(evaluation_results, is_image=False)


async def run_evaluation_docker_image(
    test_split_url: str,
    original_model_repo: str,
    models: list[str],
    model_type: ImageModelType,
    gpu_ids: list[int]
) -> DockerEvaluationResults:
    raw_data = await download_s3_file(test_split_url)
    test_split_path = unzip_to_temp_path(raw_data)
    dataset_dir = os.path.abspath(test_split_path)
    container_dataset_path = "/workspace/input_data"

    client = docker.from_env()

    base_path = "/app/validator/evaluation/ComfyUI/models"
    mounts = [
        Mount(
            target=container_dataset_path,
            source=dataset_dir,
            type='bind',
            read_only=True
        ),
        Mount(
            target=f"{base_path}/checkpoints",
            source=cst.CACHE_DIR_HUB,
            type='bind',
            read_only=False
        ),
        Mount(
            target=f"{base_path}/diffusers",
            source=cst.CACHE_DIR_HUB,
            type='bind',
            read_only=False
        )
    ]

    environment = {
        "DATASET": container_dataset_path,
        "MODELS": ",".join(models),
        "ORIGINAL_MODEL_REPO": original_model_repo,
        "MODEL_TYPE": model_type.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
    }

    try:
        container = await asyncio.to_thread(
            client.containers.run,
            cst.VALIDATOR_DOCKER_IMAGE_DIFFUSION,
            mounts=mounts,
            environment=environment,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=[str(gid) for gid in gpu_ids])],
            detach=True,
        )
        log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, None, get_all_context_tags()))
        result = await asyncio.to_thread(container.wait)
        log_task.cancel()

        if result["StatusCode"] != 0:
            raise Exception(f"Container exited with status {result['StatusCode']}")

        eval_results_dict = await get_evaluation_results(container)
        return process_evaluation_results(eval_results_dict, is_image=True)

    except Exception as e:
        logger.error(f"Failed to retrieve evaluation results: {str(e)}")
        raise Exception(f"Failed to retrieve evaluation results: {str(e)}")

    finally:
        try:
            await asyncio.to_thread(container.remove, force=True)
            await cleanup_resources(client)
            if os.path.exists(dataset_dir):
                shutil.rmtree(dataset_dir)
        except Exception as e:
            logger.info(f"A problem with cleaning up {e}")
        client.close()
