import asyncio
import io
import json
import os
import shutil
import tarfile

import docker
from docker.models.containers import Container
from docker.types import Mount
from huggingface_hub import snapshot_download
from scipy import stats

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
from validator.core import constants as vcst
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


def normalize_rewards_and_compute_loss(evaluation_results: dict) -> dict:
    """
    Normalize rewards using Gaussian percentiles, then apply weights and compute adjusted eval_loss.

    New flow:
    1. Normalize RAW rewards using Gaussian percentiles (0-1 range)
    2. Apply weights to normalized percentiles
    3. Sum weighted normalized rewards
    4. Scale to 0-1 range across all repos
    5. Subtract KL: final_score - (BETA_GRPO * kl_divergence)

    Args:
        evaluation_results: Dict with model repos as keys and evaluation data as values

    Returns:
        Modified evaluation_results dict with updated eval_loss values
    """
    # Filter out non-repo keys (like model_params_count)
    repo_keys = [key for key in evaluation_results.keys() if key != "model_params_count"]

    if len(repo_keys) < 2:
        # Need at least 2 repos for meaningful normalization
        return evaluation_results

    reward_collections = {}
    for repo_key in repo_keys:
        repo_data = evaluation_results[repo_key]
        if isinstance(repo_data, str):  # Skip error entries
            continue

        final_raw_rewards = repo_data.get('final_raw_rewards', {})

        for reward_name, reward_value in final_raw_rewards.items():
            if reward_name not in reward_collections:
                reward_collections[reward_name] = []
            reward_collections[reward_name].append(reward_value)

    # Fit Gaussian distributions for each raw reward type
    reward_distributions = {}

    for reward_name, values in reward_collections.items():
        if len(values) < 2:
            # Skip normalization if not enough values, use default parameters
            reward_distributions[reward_name] = {'mu': values[0] if values else 0.0, 'sigma': 0.0}
            continue

        # Fit Gaussian distribution
        mu, sigma = stats.norm.fit(values)
        reward_distributions[reward_name] = {'mu': mu, 'sigma': sigma}

    # Step 1-3: Calculate weighted normalized scores for each repo
    weighted_scores = []

    for repo_key in repo_keys:
        repo_data = evaluation_results[repo_key]
        if isinstance(repo_data, str):  # Skip error entries
            continue

        final_raw_rewards = repo_data.get('final_raw_rewards', {})
        weights = repo_data.get('weights', {})

        # Calculate normalized percentiles and apply weights
        weighted_percentiles = []
        for reward_name, raw_reward_value in final_raw_rewards.items():
            if reward_name in reward_distributions:
                dist_params = reward_distributions[reward_name]
                mu, sigma = dist_params['mu'], dist_params['sigma']

                # Calculate percentile (0-1 range)
                if sigma > 0:
                    percentile = stats.norm.cdf(raw_reward_value, mu, sigma)
                else:
                    percentile = 0.5  # All values are the same, use median

                # Apply weight to normalized percentile
                weight = weights.get(reward_name, 1.0)
                weighted_percentile = percentile * weight
                weighted_percentiles.append(weighted_percentile)

        # Sum weighted normalized rewards
        if weighted_percentiles:
            total_weighted_score = sum(weighted_percentiles)
            weighted_scores.append(total_weighted_score)
        else:
            weighted_scores.append(0.0)

    # Step 4: Shift to positive range, then normalize by max
    if weighted_scores:
        min_score = min(weighted_scores)
        
        # Shift all scores to be positive
        if min_score < 0:
            shifted_scores = [score - min_score for score in weighted_scores]
        else:
            shifted_scores = weighted_scores
        
        # Now normalize by max
        max_shifted = max(shifted_scores)
        if max_shifted > 0:
            scaled_scores = [score / max_shifted for score in shifted_scores]
        else:
            # All scores are identical
            scaled_scores = [0.5] * len(shifted_scores)
    else:
        scaled_scores = []

    # Step 5: Apply final formula and update eval_loss
    for i, repo_key in enumerate(repo_keys):
        repo_data = evaluation_results[repo_key]
        if isinstance(repo_data, str):  # Skip error entries
            continue

        if i < len(scaled_scores):
            kl_divergence = repo_data.get('kl_divergence', 0.0)
            # Final score: scaled_score - BETA_GRPO * kl_divergence
            new_eval_loss = scaled_scores[i] - (vcst.BETA_GRPO * kl_divergence)
            repo_data['eval_loss'] = new_eval_loss

    return evaluation_results


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
            model_path = await asyncio.to_thread(
                snapshot_download,
                repo_id=repo,
                cache_dir=cache_dir,
                ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.pkl", "*.pth"]
            )

        except Exception as e:
            logger.error(f"Failed to download {repo}: {str(e)}")
            evaluation_results[repo] = f"Failed to download model: {str(e)}"
            continue

        container = None  # Initialize container variable
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
                network_mode="none",
            )

            log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, None, get_all_context_tags()))
            result = await asyncio.to_thread(container.wait)
            log_task.cancel()

            if result["StatusCode"] != 0:

                logger.error(f"Container for {repo} exited with non-zero status: {result['StatusCode']}")
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

    evaluation_results = normalize_rewards_and_compute_loss(evaluation_results)
    logger.info(f"Grpo evaluation results post normalization: {evaluation_results}")
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
