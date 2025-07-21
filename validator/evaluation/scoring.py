import math
import os
import re
import requests
from datetime import datetime

import numpy as np
from fiber.chain.models import Node
from huggingface_hub import HfApi

import validator.core.constants as cts
from core.models.payload_models import DiffusionLosses
from core.models.payload_models import EvaluationResultImage
from core.models.payload_models import EvaluationResultText
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import TextDatasetType
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import MinerSubmission
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import MinerSubmission

from validator.utils.hash_verification import verify_model_hash

from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from core.models.utility_models import TextDatasetType
from core.utils import download_s3_file
from validator.core.config import Config
from validator.core.models import AnyTypeRawTask
from validator.core.models import MinerResults
from validator.core.models import MinerResultsImage
from validator.core.models import MinerResultsText
from validator.core.models import MiniTaskWithScoringOnly
from validator.core.models import NodeAggregationResult
from validator.core.models import PeriodScore
from validator.core.models import Submission
from validator.core.models import TaskNode
from validator.core.models import TaskResults
from validator.db.sql.submissions_and_scoring import add_submission
from validator.db.sql.submissions_and_scoring import set_task_node_quality_score
from validator.db.sql.tasks import get_expected_repo_name
from validator.db.sql.tasks import get_nodes_assigned_to_task
from validator.db.sql.tournaments import is_task_in_tournament
from validator.evaluation.docker_evaluation import run_evaluation_docker_image
from validator.evaluation.docker_evaluation import run_evaluation_docker_text
from validator.utils.call_endpoint import process_non_stream_fiber_get
from validator.utils.call_endpoint import process_non_stream_get
from validator.utils.hash_verification import verify_model_hash
from validator.utils.logging import LogContext
from validator.utils.logging import add_context_tag
from validator.utils.logging import get_logger
from validator.utils.minio import async_minio_client


logger = get_logger(__name__)


def get_task_work_score(task: MiniTaskWithScoringOnly) -> float:
    assert task.hours_to_complete > 0, "Hours to complete must be positive"
    assert task.model_id, "Model ID must be present"

    hours = task.hours_to_complete

    if getattr(task, "model_params_count", 0) > 0:
        model_size_billions = min(40, max(1, task.model_params_count // 1_000_000_000))
    else:
        model = task.model_id
        model_size = re.search(r"(\d+)(?=[bB])", model)
        model_size_billions = min(8, int(model_size.group(1)) if model_size else 1)

    if hours * model_size_billions == 0:
        logger.error(
            f"Hours to complete: {hours} and model size in billions: {model_size_billions} for task {task.task_id} "
            f"and model id: {task.model_id}\nReturning 1 regardless as a failsafe, but please look into this"
        )
        return 1
    return max(1, 2 * np.sqrt(float(hours * model_size_billions)))


def calculate_adjusted_task_score(quality_score: float, task_work_score: float) -> float:
    assert not np.isnan(quality_score), "Quality score cannot be NaN"
    assert not np.isnan(task_work_score), "Task work score cannot be NaN"
    return quality_score * task_work_score


def update_node_aggregation(
    node_aggregations: dict[str, NodeAggregationResult], node_score: TaskNode, task_work_score: float
) -> None:
    assert isinstance(node_score.hotkey, str)
    assert not np.isnan(task_work_score), "Task work score cannot be NaN"

    if node_score.hotkey not in node_aggregations:
        node_aggregations[node_score.hotkey] = NodeAggregationResult(hotkey=node_score.hotkey)

    node_result = node_aggregations[node_score.hotkey]
    adjusted_score = calculate_adjusted_task_score(node_score.quality_score, task_work_score)

    node_result.summed_adjusted_task_scores += adjusted_score
    node_result.task_raw_scores.append(node_score.quality_score)
    node_result.task_work_scores.append(task_work_score)


def calculate_node_quality_scores(
    node_aggregations: dict[str, NodeAggregationResult],
    weight_multiplier: float,
) -> list[PeriodScore]:
    assert node_aggregations, "Node aggregations dictionary cannot be empty"

    final_scores: list[PeriodScore] = []

    for hotkey, node_agg in node_aggregations.items():
        assert node_agg.task_raw_scores, f"No raw scores available for node {hotkey}"

        node_agg.average_raw_score = float(np.mean(node_agg.task_raw_scores))
        std_score = float(np.std(node_agg.task_raw_scores))

        if node_agg.average_raw_score < 0:
            score = 0.0
        else:
            score = node_agg.summed_adjusted_task_scores * node_agg.average_raw_score

        node_agg.quality_score = score

        final_scores.append(
            PeriodScore(
                hotkey=hotkey,
                quality_score=score,
                average_score=node_agg.average_raw_score,
                std_score=std_score,
                summed_task_score=node_agg.summed_adjusted_task_scores,
                weight_multiplier=weight_multiplier,
            )
        )

    return final_scores


def _normalise_scores(period_scores: list[PeriodScore]) -> list[PeriodScore]:
    assert period_scores, "Period scores list cannot be empty"
    valid_scores = [ps.quality_score for ps in period_scores if ps.quality_score is not None]
    if not valid_scores:
        raise ValueError("No valid quality scores found in period_scores")
    max_score = max(valid_scores)
    if max_score <= 0:
        for node_period_score in period_scores:
            node_period_score.normalised_score = 0.0
        return period_scores

    for node_period_score in period_scores:
        if node_period_score.quality_score is None or node_period_score.quality_score <= 0:
            node_period_score.normalised_score = 0.0
        else:
            normalised_input = node_period_score.quality_score / max_score
            sigmoid_part = 1 / (1 + np.exp(-cts.SIGMOID_STEEPNESS * (normalised_input - cts.SIGMOID_SHIFT)))
            sigmoid_score = pow(sigmoid_part, cts.SIGMOID_POWER)
            linear_score = normalised_input
            node_period_score.normalised_score = (cts.SIGMOID_WEIGHT * sigmoid_score) + (cts.LINEAR_WEIGHT * linear_score)

    total_score = sum(ps.normalised_score for ps in period_scores)
    if total_score > 0:
        for node_period_score in period_scores:
            node_period_score.normalised_score = node_period_score.normalised_score / total_score

    return period_scores


def get_period_scores_from_results(task_results: list[TaskResults], weight_multiplier: float) -> list[PeriodScore]:
    if not task_results:
        return []

    node_aggregations: dict[str, NodeAggregationResult] = {}

    for task_res in task_results:
        task_work_score = get_task_work_score(task_res.task)
        for node_score in task_res.node_scores:
            update_node_aggregation(node_aggregations, node_score, task_work_score)

    final_scores = calculate_node_quality_scores(node_aggregations, weight_multiplier=weight_multiplier)
    final_scores = _normalise_scores(final_scores)

    return final_scores


def calculate_weighted_loss(test_loss: float, synth_loss: float, use_max_of_synth_test: bool = False) -> float:
    assert not np.isnan(test_loss), "Test loss cannot be NaN"
    assert not np.isnan(synth_loss), "Synthetic loss cannot be NaN"

    if use_max_of_synth_test:
        adjusted_loss = max(test_loss, synth_loss)

        if test_loss >= synth_loss:
            logger.info(f"Using test_loss: test={test_loss:.6f} >= synth={synth_loss:.6f}")
        else:
            logger.info(f"Using synth_loss: test={test_loss:.6f} < synth={synth_loss:.6f}, adjusted={adjusted_loss:.6f}")

        return adjusted_loss
    else:
        return cts.TEST_SCORE_WEIGHTING * test_loss + (1 - cts.TEST_SCORE_WEIGHTING) * synth_loss


def _is_synth_loss_valid_for_group(valid_results: list[MinerResults], max_ratio: float = 1.5, threshold: float = 0.4) -> bool:
    if all(np.isnan(result.synth_loss) for result in valid_results):
        logger.info("All synth losses are NaN, using test_loss only for ranking")
        return False

    if not valid_results:
        return False

    real_synth_miners = [
        result
        for result in valid_results
        if result.is_finetune and not np.isnan(result.test_loss) and not np.isnan(result.synth_loss) and result.synth_loss < 999.0
    ]

    if not real_synth_miners:
        logger.info("No miners with real synthetic loss values")
        return False

    valid_miners = len(real_synth_miners)
    valid_ratios = 0

    miners_with_ratios = [(result, result.synth_loss / result.test_loss) for result in real_synth_miners if result.test_loss > 0]

    valid_ratios = sum(1 for _, ratio in miners_with_ratios if ratio <= max_ratio)
    ratio = valid_ratios / valid_miners if valid_miners > 0 else 0
    logger.info(f"Valid ratios: {valid_ratios}/{valid_miners} = {ratio:.3f}, threshold is {threshold}")
    return ratio >= threshold


def calculate_miner_ranking_and_scores(
    miner_results: list[MinerResultsText | MinerResultsImage],
) -> list[MinerResultsText | MinerResultsImage]:
    logger.info("Beginning score calculation...")

    valid_results = []
    # Initialize all scores to 0.0 and set appropriate reasons
    for result in miner_results:
        with LogContext(miner_hotkey=result.hotkey):
            result.score = 0.0
            # atp, we only set score_reason in these cases (all are invalid and is_finetune == False):
            # "Invalid/No repo submitted", "Evaluation failed", "Duplicated submission"
            if result.score_reason:
                continue
            elif not result.is_finetune:
                result.score_reason = "Non-finetuned submission"
                logger.info(f"Miner {result.hotkey}: Non-finetuned, score initialized to 0.0")
            elif np.isnan(result.test_loss):
                result.score_reason = "Invalid test loss"
                logger.info(f"Miner {result.hotkey}: Invalid test loss, score initialized to 0.0")
            elif result.synth_loss == 1000.0:
                result.score_reason = "Outside of top-4 test doesn't get scored."
                logger.info(f"Miner {result.hotkey}: Outside of top-4")
            else:
                valid_results.append(result)

    if not valid_results:
        logger.warning("No valid finetuned submissions found. All scores set to 0.0")
        return miner_results

    is_dpo_task = False
    is_grpo_task = False
    is_instruct_task = False
    use_max_approach = False

    if valid_results and isinstance(valid_results[0], MinerResultsText):
        is_dpo_task = valid_results[0].task_type == TaskType.DPOTASK
        is_grpo_task = valid_results[0].task_type == TaskType.GRPOTASK
        is_instruct_task = valid_results[0].task_type == TaskType.INSTRUCTTEXTTASK
        is_chat_task = valid_results[0].task_type == TaskType.CHATTASK

        # For both DPO and Instruct Text tasks, use max(synth, test)
        use_max_approach = is_dpo_task or is_instruct_task or is_chat_task

        if use_max_approach:
            logger.info(f"Processing {valid_results[0].task_type} - using max(test, synth) loss for ranking")
        else:
            logger.info("Processing GRPO task - higher loss is better")

    use_weighted_loss = use_max_approach or _is_synth_loss_valid_for_group(valid_results)
    if use_weighted_loss:
        if use_max_approach:
            logger.info(f"Using max loss for ranking {valid_results[0].task_type}")
        else:
            logger.info("Using weighted loss for ranking (at least one miner has valid synth loss)")

        ranked_results = []
        for result in valid_results:
            adjusted_loss = calculate_weighted_loss(result.test_loss, result.synth_loss, use_max_of_synth_test=use_max_approach)
            result.adjusted_loss = adjusted_loss
            ranked_results.append((result, adjusted_loss))
            logger.info(f"Miner {result.hotkey}: calculated ranking loss {adjusted_loss:.6f}")

        if is_grpo_task:
            # For GRPO, sort in reverse order (higher value is better)
            ranked_results.sort(key=lambda x: float("-inf") if math.isnan(x[1]) else -x[1])
            ranking_type = "GRPO score (bigger is better)"
        else:
            # For other tasks, sort normally (lower loss is better)
            ranked_results.sort(key=lambda x: float("inf") if math.isnan(x[1]) else x[1])
            if is_dpo_task:
                ranking_type = "DPO loss (max test, synth + train)"
            elif is_instruct_task:
                ranking_type = "INSTRUCT loss (max test, synth + train)"
            else:
                ranking_type = "weighted_loss"
    else:
        logger.info("Using test loss only for ranking (all synth losses are invalid)")
        ranked_results = []
        for result in valid_results:
            result.adjusted_loss = result.test_loss  # Store the adjusted loss
            ranked_results.append((result, result.test_loss))

        if is_grpo_task:
            # For GRPO, sort in reverse order (higher value is better)
            ranked_results.sort(key=lambda x: float("-inf") if math.isnan(x[1]) else -x[1])
            ranking_type = "GRPO score (bigger is better)"
        else:
            # For other tasks, sort normally (lower loss is better)
            ranked_results.sort(key=lambda x: float("inf") if math.isnan(x[1]) else x[1])
            ranking_type = "test_loss_only"

    if ranked_results:
        top_result, top_metric = ranked_results[0]
        with LogContext(miner_hotkey=top_result.hotkey):
            top_result.score = cts.FIRST_PLACE_SCORE
            top_result.score_reason = f"Ranked 1st by {ranking_type}"
            logger.info(
                f"Miner {top_result.hotkey} (finetuned):"
                f" test_loss={top_result.test_loss:.4f}"
                f" synth_loss={top_result.synth_loss:.4f}"
                f" {ranking_type}={top_metric:.4f}"
                f" score={top_result.score:.4f}"
                f" score_reason={top_result.score_reason}"
            )

    total_valid_miners = len(valid_results)
    if total_valid_miners > cts.MIN_IDEAL_NUM_MINERS_IN_POOL:
        penalty_count = max(1, int(total_valid_miners * 0.25))
        penalty_start_idx = total_valid_miners - penalty_count

        for result, metric in ranked_results[1:penalty_start_idx]:
            with LogContext(miner_hotkey=result.hotkey):
                result.score_reason = f"Ranked below top 1 by {ranking_type}"
                logger.info(
                    f"Miner {result.hotkey} (finetuned):"
                    f" test_loss={result.test_loss:.4f}"
                    f" synth_loss={result.synth_loss:.4f}"
                    f" {ranking_type}={metric:.4f}"
                    f" score=0.0"
                    f" score_reason={result.score_reason}"
                )

        for result, metric in ranked_results[penalty_start_idx:]:
            with LogContext(miner_hotkey=result.hotkey):
                result.score = cts.SCORE_PENALTY
                result.score_reason = f"Bottom 25% ranked by {ranking_type}"
                logger.info(
                    f"Miner {result.hotkey} (finetuned):"
                    f" test_loss={result.test_loss:.4f}"
                    f" synth_loss={result.synth_loss:.4f}"
                    f" {ranking_type}={metric:.4f}"
                    f" score={result.score:.4f}"
                    f" score_reason={result.score_reason}"
                )
    else:
        for result, metric in ranked_results[1:]:
            with LogContext(miner_hotkey=result.hotkey):
                result.score_reason = f"Ranked below top 1 by {ranking_type}"
                logger.info(
                    f"Miner {result.hotkey} (finetuned):"
                    f" test_loss={result.test_loss:.4f}"
                    f" synth_loss={result.synth_loss:.4f}"
                    f" {ranking_type}={metric:.4f}"
                    f" score=0.0"
                    f" score_reason={result.score_reason}"
                )

    # Apply penalty scores to failed submissions when valid submissions exist
    if valid_results:
        for result in miner_results:
            # Find failed submissions that haven't been scored yet
            if (not result.is_finetune or np.isnan(result.test_loss)) and result.score == 0.0:
                result.score = cts.SCORE_PENALTY
                logger.info(
                    f"Miner {result.hotkey}: Failed submission ({result.score_reason}), "
                    f"applying penalty score {cts.SCORE_PENALTY}"
                )

    return miner_results


def _get_dataset_type(task: AnyTypeRawTask) -> TextDatasetType | None:
    if task.task_type == TaskType.INSTRUCTTEXTTASK:
        return InstructTextDatasetType(
            field_system=task.field_system,
            field_instruction=task.field_instruction,
            field_input=task.field_input,
            field_output=task.field_output,
            format=task.format,
            no_input_format=task.no_input_format,
        )
    elif task.task_type == TaskType.IMAGETASK:
        return None
    elif task.task_type == TaskType.DPOTASK:
        return DpoDatasetType(
            field_prompt=task.field_prompt,
            field_system=task.field_system,
            field_chosen=task.field_chosen,
            field_rejected=task.field_rejected,
            prompt_format=task.prompt_format,
            chosen_format=task.chosen_format,
            rejected_format=task.rejected_format,
        )
    elif task.task_type == TaskType.GRPOTASK:
        return GrpoDatasetType(
            field_prompt=task.field_prompt,
            reward_functions=task.reward_functions,
        )
    elif task.task_type == TaskType.CHATTASK:
        return ChatTemplateDatasetType(
            chat_template=task.chat_template,
            chat_column=task.chat_column,
            chat_role_field=task.chat_role_field,
            chat_content_field=task.chat_content_field,
            chat_user_reference=task.chat_user_reference,
            chat_assistant_reference=task.chat_assistant_reference,
        )
    else:
        raise ValueError(f"Unknown task type: {task.task_type}")


def _create_failed_miner_result(hotkey: str, score_reason: str, task_type: TaskType) -> MinerResults:
    """Create a result object for failed miner submissions with initial score of 0.0.
    The score may later be adjusted to a penalty if valid submissions exist."""
    if task_type in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK]:
        return MinerResultsText(
            hotkey=hotkey,
            test_loss=np.nan,
            synth_loss=np.nan,
            is_finetune=False,
            score=0.0,
            score_reason=score_reason,
            task_type=task_type,
        )
    else:
        return MinerResultsImage(
            hotkey=hotkey, test_loss=np.nan, synth_loss=np.nan, is_finetune=False, score=0.0, score_reason=score_reason
        )


def _calculate_weighted_loss_for_image_eval(eval_result: EvaluationResultImage) -> float:
    if isinstance(eval_result.eval_loss, DiffusionLosses):
        text_guided_avg = (
            sum(eval_result.eval_loss.text_guided_losses) / len(eval_result.eval_loss.text_guided_losses)
            if eval_result.eval_loss.text_guided_losses
            else 0
        )

        no_text_avg = (
            sum(eval_result.eval_loss.no_text_losses) / len(eval_result.eval_loss.no_text_losses)
            if eval_result.eval_loss.no_text_losses
            else 0
        )

        weighted_loss = (
            cts.DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT * text_guided_avg + (1 - cts.DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT) * no_text_avg
        )
        return weighted_loss

    return None


async def _get_submission_repo(miner: Node, task_id: str, config: Config) -> MinerSubmission | None:
    url = f"{cts.SUBMISSION_ENDPOINT}{task_id}"
    try:
        response = await process_non_stream_fiber_get(url, config, miner)

        if isinstance(response, dict):
            return MinerSubmission(repo=response["repo"], model_hash=response.get("model_hash"))
        else:
            repo = str(response)
            if repo == "None":
                return None
            return MinerSubmission(repo=repo, model_hash=None)

    except Exception as e:
        logger.error(f"Failed to get submission for miner {miner.hotkey}: {e}")
        return None


async def _evaluate_submissions(
    task: AnyTypeRawTask,
    submission_repos: list[str],
    gpu_ids: list[int],
    dataset_type: TextDatasetType | None = None,
) -> dict[str, tuple[EvaluationResultText, EvaluationResultText] | EvaluationResultImage | Exception]:
    unique_repos = list(set(submission_repos))
    if len(unique_repos) != len(submission_repos):
        logger.warning(f"Found duplicate repos. Deduplicating {len(submission_repos)} repos to {len(unique_repos)} unique repos")

    if task.task_type in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.CHATTASK]:
        results: dict[str, tuple[EvaluationResultText, EvaluationResultText] | Exception] = {}
        repos_to_evaluate = []
        for repo in unique_repos:
            if repo == task.model_id:
                logger.warning(f"Repository {repo} matches original model ID - marking as non-finetuned")
                results[repo] = (
                    EvaluationResultText(is_finetune=False, eval_loss=0.0),
                    EvaluationResultText(is_finetune=False, eval_loss=0.0),
                )
            else:
                repos_to_evaluate.append(repo)

        if not repos_to_evaluate:
            return results

        is_grpo_task = task.task_type == TaskType.GRPOTASK

        assert task.synthetic_data is not None, "Synthetic data shouldn't be none for text tasks"
        assert task.test_data is not None, "Test data shouldn't be none for text tasks"

        evaluation_params = {
            "file_format": FileFormat.JSON,
            "original_model": task.model_id,
            "models": repos_to_evaluate,
            "dataset_type": dataset_type,
            "gpu_ids": gpu_ids,
        }

        logger.info("Starting test evaluation")
        test_data_filepath = await download_s3_file(task.test_data)
        test_results = await run_evaluation_docker_text(dataset=test_data_filepath, **evaluation_params)

        try:
            os.remove(test_data_filepath)
        except Exception as e:
            logger.warning(f"Failed to remove test data file {test_data_filepath}: {e}")

        test_eval_results = test_results.results
        task.model_params_count = test_results.base_model_params_count

        test_losses = []
        for repo in repos_to_evaluate:
            if isinstance(test_eval_results.get(repo), Exception):
                results[repo] = test_eval_results[repo]
                continue

            test_result = test_eval_results[repo]
            if not test_result.is_finetune:
                results[repo] = (
                    EvaluationResultText(is_finetune=False, eval_loss=0.0),
                    EvaluationResultText(is_finetune=False, eval_loss=0.0),
                )
            else:
                test_losses.append((repo, test_result.eval_loss))

        if is_grpo_task:
            test_losses.sort(key=lambda x: float("-inf") if math.isnan(x[1]) else x[1], reverse=True)
        else:
            test_losses.sort(key=lambda x: float("inf") if math.isnan(x[1]) else x[1])
        top_4_repos = [repo for repo, _ in test_losses[:4]]

        for repo, _ in test_losses[4:]:
            results[repo] = (
                EvaluationResultText(is_finetune=True, eval_loss=1000.0),
                test_eval_results[repo],
            )

        if top_4_repos:
            logger.info(f"Evaluating synthetic data for top {len(top_4_repos)} models")
            synthetic_data_filepath = await download_s3_file(task.synthetic_data)
            synth_results = await run_evaluation_docker_text(
                dataset=synthetic_data_filepath,
                models=top_4_repos,
                **{k: v for k, v in evaluation_params.items() if k != "models"},
            )

            try:
                os.remove(synthetic_data_filepath)
            except Exception as e:
                logger.warning(f"Failed to remove synthetic data file {synthetic_data_filepath}: {e}")

            synth_eval_results = synth_results.results

            for repo in top_4_repos:
                if isinstance(synth_eval_results.get(repo), Exception):
                    results[repo] = synth_eval_results[repo]
                else:
                    results[repo] = (synth_eval_results[repo], test_eval_results[repo])

    elif task.task_type == TaskType.IMAGETASK:
        results: dict[str, EvaluationResultImage | Exception] = {}
        repos_to_evaluate = []
        for repo in unique_repos:
            if repo == task.model_id:
                logger.warning(f"Repository {repo} matches original model ID - marking as non-finetuned")
                results[repo] = EvaluationResultImage(
                    eval_losses=DiffusionLosses(text_guided_losses=[0], no_text_losses=[0]), is_finetune=False
                )
            else:
                repos_to_evaluate.append(repo)

        if not repos_to_evaluate:
            return results

        evaluation_params = {
            "test_split_url": task.test_data,
            "original_model_repo": task.model_id,
            "models": repos_to_evaluate,
            "model_type": task.model_type,
            "gpu_ids": gpu_ids,
        }

        assert task.test_data is not None, "Test data shouldn't be none for image tasks"
        logger.info("Starting image model evaluation")
        image_results = await run_evaluation_docker_image(**evaluation_params)
        image_eval_results = image_results.results
        task.model_params_count = image_results.base_model_params_count
        for repo in repos_to_evaluate:
            results[repo] = image_eval_results[repo]

    for repo in unique_repos:
        if repo not in results:
            results[repo] = Exception("Evaluation failed to complete")

    return results


async def _clear_up_s3(file_paths: list[str]) -> None:
    for file_path in file_paths:
        try:
            logger.info(f"files = {file_paths} and bucket is {cts.BUCKET_NAME}")
            object_name = file_path.split(cts.BUCKET_NAME + "/")[-1]
            logger.info(f"Deleting file {object_name} from MinIO bucket {cts.BUCKET_NAME}")
            await async_minio_client.delete_file(cts.BUCKET_NAME, object_name)
        except Exception as e:
            logger.error(f"Failed to delete file {file_path} from MinIO: {e}")


async def _update_scores(task: AnyTypeRawTask, task_results: list[MinerResultsText | MinerResultsImage], psql_db) -> None:
    assert task.task_id is not None, "task id needs to be set to update scores"
    for result in task_results:
        with LogContext(miner_hotkey=result.hotkey):
            if result.score is None:
                continue

            await set_task_node_quality_score(
                task_id=task.task_id,
                hotkey=result.hotkey,
                quality_score=float(result.score),
                test_loss=result.test_loss,
                synth_loss=result.synth_loss,
                score_reason=result.score_reason,
                psql_db=psql_db,
            )

            if result.submission:
                result.submission.score = result.score
                await add_submission(result.submission, psql_db)


async def get_repo_creation_time(repo_name: str) -> datetime:
    try:
        clean_name = repo_name.replace("https://huggingface.co/", "")
        parts = clean_name.split("/")

        if len(parts) >= 2:
            org, model = parts[-2], parts[-1]
            url = f"https://huggingface.co/api/models/{org}/{model}"

            logger.debug(f"Fetching creation time from: {url}")
            response = await process_non_stream_get(url, None)
            if response:
                return datetime.fromisoformat(response["createdAt"].replace("Z", "+00:00"))
    except Exception as e:
        logger.error(f"Error fetching repo creation time for {repo_name}: {e}")
    return datetime.max


def group_by_losses(task_results: list[MinerResults]) -> dict[tuple[float, float], list[tuple[str, str]]]:
    loss_groups: dict[tuple[float, float], list[tuple[str, str]]] = {}

    for result in task_results:
        if result.submission and not np.isnan(result.test_loss) and not np.isnan(result.synth_loss):
            losses = (float(result.test_loss), float(result.synth_loss))
            if losses not in loss_groups:
                loss_groups[losses] = []
            loss_groups[losses].append((result.hotkey, result.submission.repo))

    return loss_groups


def get_hf_upload_timestamp(repo_url: str) -> datetime | None:
    try:
        repo_path = repo_url.replace("https://huggingface.co/", "").split("/tree/")[0]
        api = HfApi()

        model_info = api.model_info(repo_path, timeout=5.0)
        if model_info and model_info.lastModified:
            return model_info.lastModified

    except Exception as e:
        logger.error(f"Failed to get upload timestamp for {repo_url}: {e}")
    return None


def get_hf_upload_timestamp(repo_url: str) -> datetime | None:
    try:
        repo_path = repo_url.replace("https://huggingface.co/", "").split("/tree/")[0]
        api = HfApi()
        
        model_info = api.model_info(repo_path, timeout=5.0)
        if model_info and model_info.lastModified:
            return model_info.lastModified
            
    except Exception as e:
        logger.error(f"Failed to get upload timestamp for {repo_url}: {e}")
    return None


async def handle_duplicate_submissions(task_results: list[MinerResultsText | MinerResultsImage]) -> dict[str, bool]:
    keep_submission = {result.hotkey: True for result in task_results}
    loss_groups = group_by_losses(task_results)

    for losses, submissions in loss_groups.items():
        if len(submissions) > 1:
            logger.warning(f"Found {len(submissions)} submissions with identical losses {losses}")


            submissions_with_hashes = []
            submissions_without_hashes = []


            for hotkey, repo in submissions:
                result = next(r for r in task_results if r.hotkey == hotkey)
                if result.submission and result.submission.model_hash:
                    submissions_with_hashes.append((hotkey, repo, result.submission.model_hash))
                else:
                    submissions_without_hashes.append((hotkey, repo))

            # If we have both hashed and non-hashed submissions, prioritize hashed ones
            if submissions_with_hashes and submissions_without_hashes:
                logger.warning("Mixed hash/no-hash submissions with identical losses - prioritizing hashed submissions")
                for hotkey, repo in submissions_without_hashes:
                    keep_submission[hotkey] = False
                    logger.warning(f"Marking duplicate {hotkey} (no hash provided, hashed submission exists)")

            # Handle multiple submissions with hashes - group by hash
            if len(submissions_with_hashes) > 1:
                hash_groups = {}
                for hotkey, repo, model_hash in submissions_with_hashes:
                    if model_hash not in hash_groups:
                        hash_groups[model_hash] = []
                    hash_groups[model_hash].append((hotkey, repo))

                for model_hash, hash_submissions in hash_groups.items():
                    if len(hash_submissions) > 1:
                        logger.warning(f"Found {len(hash_submissions)} submissions with identical hash {model_hash[:16]}...")
                        for hotkey, repo in hash_submissions[1:]:
                            keep_submission[hotkey] = False
                            logger.warning(f"Marking duplicate {hotkey} (identical model hash)")

            # Handle multiple submissions without hashes (only if no hashed submissions exist)
            if len(submissions_without_hashes) > 1 and not submissions_with_hashes:
                logger.warning("Multiple submissions without hashes, using timestamp fallback")
                submissions_with_timestamps = [
                    (hotkey, repo, get_hf_upload_timestamp(repo)) for hotkey, repo in submissions_without_hashes
                ]
                valid_timestamps = [(h, r, t) for h, r, t in submissions_with_timestamps if t]

                if valid_timestamps:
                    earliest_hotkey = min(valid_timestamps, key=lambda x: x[2])[0]
                    for hotkey, repo in submissions_without_hashes:
                        if hotkey != earliest_hotkey:
                            keep_submission[hotkey] = False
                            logger.warning(f"Marking duplicate {hotkey} (later commit)")
                else:
                    for hotkey, repo in submissions_without_hashes:
                        keep_submission[hotkey] = False
                        logger.warning(f"Marking duplicate {hotkey} (no timestamps)")

    return keep_submission


def zero_duplicate_scores(
    task_results: list[MinerResultsText | MinerResultsImage], keep_submission: dict[str, bool]
) -> list[MinerResultsText | MinerResultsImage]:
    # Count remaining valid submissions after filtering duplicates
    remaining_valid_count = sum(
        1
        for result in task_results
        if result.is_finetune and not np.isnan(result.test_loss) and keep_submission.get(result.hotkey, False)
    )

    for result in task_results:
        if not keep_submission[result.hotkey]:
            result.test_loss = np.nan
            result.synth_loss = np.nan
            result.is_finetune = False
            result.score_reason = result.score_reason or "Duplicated submission"

            # Apply penalty only if valid submissions remain
            if remaining_valid_count > 0:
                result.score = cts.SCORE_PENALTY
                logger.info(f"Miner {result.hotkey}: Duplicate submission, applying penalty score {cts.SCORE_PENALTY}")
            else:
                result.score = 0.0
                logger.info(f"Miner {result.hotkey}: Duplicate submission but no valid submissions remain, score set to 0.0")

    return task_results


async def process_miners_pool(
    miners: list[Node],
    task: AnyTypeRawTask,
    config: Config,
    gpu_ids: list[int],
    dataset_type: TextDatasetType | None = None,
) -> list[MinerResultsText | MinerResultsImage]:
    assert task.task_id is not None, "We should have a task id when processing miners"

    is_tournament_task = await is_task_in_tournament(str(task.task_id), config.psql_db)
    miner_repos: dict[str, str] = {}

    failed_results = []

    for miner in miners:
        with LogContext(miner_hotkey=miner.hotkey):
            expected_name = await get_expected_repo_name(task.task_id, miner.hotkey, config.psql_db)

            if is_tournament_task and expected_name:
                repo = f"{cts.RAYONLABS_HF_USERNAME}/{expected_name}"
                logger.info(f"Tournament task: constructed repo {repo} for miner {miner.hotkey}")
                miner_repos[miner.hotkey] = repo
            else:
                submission = await _get_submission_repo(miner, str(task.task_id), config)
                if submission is not None and submission.repo is not None:
                    repo_parts = submission.repo.split("/")
                    if len(repo_parts) >= 2:
                        submitted_name = repo_parts[-1]

                        if expected_name and submitted_name != expected_name:
                            logger.warning(
                                f"Miner {miner.hotkey} submitted a repo with name {submitted_name} "
                                f"but expected {expected_name}. Marking as failed."
                            )
                            failed_results.append(
                                _create_failed_miner_result(
                                    miner.hotkey, score_reason="Repository name mismatch", task_type=task.task_type
                                )
                            )
                            continue

                        # Hash verification
                        if submission.model_hash is not None:
                            if verify_model_hash(submission.repo, submission.model_hash):
                                logger.info(f"Hash verification passed for miner {miner.hotkey}")
                            else:
                                logger.warning(f"Hash verification failed for miner {miner.hotkey}. Marking as failed.")
                                failed_results.append(
                                    _create_failed_miner_result(
                                        miner.hotkey, score_reason="Hash verification failed", task_type=task.task_type
                                    )
                                )
                                continue
                        else:
                            logger.info(f"No hash provided by miner {miner.hotkey}, skipping verification")

                        miner_repos[miner.hotkey] = submission.repo

                logger.info(f"Found repo {submission.repo if submission else None} for miner {miner.hotkey}")

    results = failed_results + [
        _create_failed_miner_result(miner.hotkey, score_reason="Invalid/No repo submitted", task_type=task.task_type)
        for miner in miners
        if miner.hotkey not in miner_repos and miner.hotkey not in [r.hotkey for r in failed_results]
    ]

    if miner_repos:
        try:
            eval_results = await _evaluate_submissions(
                task=task, submission_repos=list(miner_repos.values()), gpu_ids=gpu_ids, dataset_type=dataset_type or None
            )

            for miner in miners:
                with LogContext(miner_hotkey=miner.hotkey):
                    if miner.hotkey not in miner_repos:
                        continue

                    repo = miner_repos[miner.hotkey]
                    eval_result = eval_results.get(repo)

                    if isinstance(eval_result, Exception):
                        logger.error(f"Evaluation failed for miner {miner.hotkey}: {eval_result}")
                        results.append(
                            _create_failed_miner_result(
                                miner.hotkey,
                                score_reason=f"Evaluation failed: {str(eval_result)[:350]}",
                                task_type=task.task_type,
                            )
                        )
                        continue
                    elif task.task_type in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.CHATTASK]:
                        synth_result, test_result = eval_result
                    elif task.task_type == TaskType.IMAGETASK:
                        test_result = eval_result
                        test_result.eval_loss = _calculate_weighted_loss_for_image_eval(test_result)
                        synth_result = test_result
                    else:
                        raise ValueError(f"Unknown task type: {task.task_type}")

                    submission = Submission(
                        task_id=task.task_id,
                        hotkey=miner.hotkey,
                        repo=repo,
                        created_on=datetime.now(),
                        updated_on=datetime.now(),
                    )

                if task.task_type in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.CHATTASK]:
                    results.append(
                        MinerResultsText(
                            hotkey=miner.hotkey,
                            test_loss=float(test_result.eval_loss),
                            synth_loss=float(synth_result.eval_loss),
                            is_finetune=test_result.is_finetune,
                            submission=submission,
                            task_type=task.task_type,
                        )
                    )
                elif task.task_type == TaskType.IMAGETASK:
                    results.append(
                        MinerResultsImage(
                            hotkey=miner.hotkey,
                            test_loss=float(test_result.eval_loss),
                            synth_loss=float(synth_result.eval_loss),
                            is_finetune=test_result.is_finetune,
                            submission=submission,
                        )
                    )
                else:
                    raise ValueError(f"Unknown task type: {task.task_type}")

        except Exception as e:
            logger.error(f"Error during batch evaluation: {e}", exc_info=True)
            results.extend(
                [
                    _create_failed_miner_result(
                        miner.hotkey, score_reason=f"Evaluation failed: {str(e)[:350]}", task_type=task.task_type
                    )
                    for miner in miners
                    if miner.hotkey not in [r.hotkey for r in results]
                ]
            )

    return results


async def evaluate_and_score(task: AnyTypeRawTask, gpu_ids: list[int], config: Config) -> AnyTypeRawTask:
    assert task.task_id is not None, "Task ID must be present"
    assert task.test_data is not None, "Test data must be present"

    miner_pool = await get_nodes_assigned_to_task(str(task.task_id), config.psql_db)
    dataset_type = _get_dataset_type(task)

    logger.info(f"Beginning evaluation for task {task.task_id} with {len(miner_pool)} miners")
    task_results = await process_miners_pool(miner_pool, task, config, gpu_ids, dataset_type)

    logger.info("Checking for duplicates ...")
    keep_submission = await handle_duplicate_submissions(task_results)
    task_results = zero_duplicate_scores(task_results, keep_submission)

    logger.info("Calculating final scores...")
    task_results = calculate_miner_ranking_and_scores(task_results)
    await _update_scores(task, task_results, config.psql_db)
    all_scores_zero = all(result.score == 0.0 for result in task_results)

    if cts.DELETE_S3_AFTER_COMPLETE:
        if task.task_type in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.CHATTASK]:
            files_to_delete = [task.training_data, task.test_data, task.synthetic_data]
        elif task.task_type == TaskType.IMAGETASK:
            files_to_delete = [task.training_data, task.test_data]
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")

    if all_scores_zero:
        if task.n_eval_attempts < cts.MAX_EVAL_ATTEMPTS - 1:
            task.status = TaskStatus.PREEVALUATION
            add_context_tag("status", task.status.value)
            logger.info(f"All scores are zero for task {task.task_id}, setting status to PREEVALUATION to re-evaluate")
        else:
            task.status = TaskStatus.FAILURE
            add_context_tag("status", task.status.value)
            logger.info(f"Task {task.task_id} marked as failure")
            await _clear_up_s3(files_to_delete)
    else:
        await _clear_up_s3(files_to_delete)
        task.status = TaskStatus.SUCCESS
        add_context_tag("status", task.status.value)
        logger.info(f"Task {task.task_id} completed successfully with non-zero scores")
    task.n_eval_attempts = (task.n_eval_attempts or 0) + 1
    return task
