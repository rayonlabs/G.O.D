import os
import re
from datetime import datetime

import numpy as np
from fiber.chain.models import Node

import validator.core.constants as cts
from core.models.payload_models import DiffusionLosses
from core.models.payload_models import EvaluationResultImage
from core.models.payload_models import EvaluationResultText
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from core.utils import download_s3_file
from validator.core.config import Config
from validator.core.models import ImageRawTask
from validator.core.models import MinerResults
from validator.core.models import MinerResultsImage
from validator.core.models import MinerResultsText
from validator.core.models import MiniTaskWithScoringOnly
from validator.core.models import NodeAggregationResult
from validator.core.models import PeriodScore
from validator.core.models import Submission
from validator.core.models import TaskNode
from validator.core.models import TaskResults
from validator.core.models import TextRawTask
from validator.db.sql.submissions_and_scoring import add_submission
from validator.db.sql.submissions_and_scoring import set_task_node_quality_score
from validator.db.sql.tasks import get_expected_repo_name
from validator.db.sql.tasks import get_nodes_assigned_to_task
from validator.evaluation.docker_evaluation import run_evaluation_docker_image
from validator.evaluation.docker_evaluation import run_evaluation_docker_text
from validator.utils.call_endpoint import process_non_stream_fiber_get
from validator.utils.call_endpoint import process_non_stream_get
from validator.utils.logging import LogContext
from validator.utils.logging import add_context_tag
from validator.utils.logging import get_logger
from validator.utils.minio import async_minio_client


logger = get_logger(__name__)


def get_task_work_score(task: MiniTaskWithScoringOnly) -> float:
    """Calculate work score for a task based on hours and model size."""
    assert task.hours_to_complete > 0, "Hours to complete must be positive"
    assert task.model_id, "Model ID must be present"

    hours = task.hours_to_complete

    if getattr(task, "model_params_count", 0) > 0:
        model_size_billions = min(14, max(1, task.model_params_count // 1_000_000_000))
    else:
        # Fallback to parsing from model id
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
    """Calculate adjusted task score based on quality score and work score."""
    assert not np.isnan(quality_score), "Quality score cannot be NaN"
    assert not np.isnan(task_work_score), "Task work score cannot be NaN"
    return quality_score * task_work_score


def update_node_aggregation(
    node_aggregations: dict[str, NodeAggregationResult], node_score: TaskNode, task_work_score: float
) -> None:
    """Update node aggregation results with new scores for a particular task."""
    assert isinstance(node_score.hotkey, str), "hotkey is string"
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
    """Calculate quality scores for each node."""
    assert node_aggregations, "Node aggregations dictionary cannot be empty"

    final_scores: list[PeriodScore] = []

    for hotkey, node_agg in node_aggregations.items():
        assert node_agg.task_raw_scores, f"No raw scores available for node {hotkey}"

        node_agg.average_raw_score = float(np.mean(node_agg.task_raw_scores))
        score = node_agg.summed_adjusted_task_scores * node_agg.average_raw_score
        node_agg.quality_score = score

        final_scores.append(
            PeriodScore(
                hotkey=hotkey,
                quality_score=score,
                average_score=node_agg.average_raw_score,
                summed_task_score=node_agg.summed_adjusted_task_scores,
                weight_multiplier=weight_multiplier,
            )
        )

    return final_scores


def _normalise_scores(period_scores: list[PeriodScore]) -> list[PeriodScore]:
    """Normalise scores using a combination of sigmoid and linear functions."""

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

    return period_scores


async def get_period_scores_from_results(task_results: list[TaskResults], weight_multiplier: float) -> list[PeriodScore]:
    """Aggregate and normalise scores across all nodes."""
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


def calculate_weighted_loss(test_loss: float, synth_loss: float) -> float:
    """Calculate weighted average of losses with more weight on test loss."""
    assert not np.isnan(test_loss), "Test loss cannot be NaN"
    assert not np.isnan(synth_loss), "Synthetic loss cannot be NaN"
    return cts.TEST_SCORE_WEIGHTING * test_loss + (1 - cts.TEST_SCORE_WEIGHTING) * synth_loss


def _is_synth_loss_valid_for_group(miner_results: list[MinerResults], max_ratio: float = 2.0, threshold: float = 0.75) -> bool:
    """
    Check if the synthetic loss to test loss ratio is valid for a sufficient percentage of miners.
    """
    if not miner_results:
        return False

    valid_miners = 0
    valid_ratios = 0

    for result in miner_results:
        if result.is_finetune and not np.isnan(result.test_loss) and not np.isnan(result.synth_loss):
            valid_miners += 1
            if result.test_loss > 0 and (result.synth_loss / result.test_loss) <= max_ratio:
                logger.info(f"ratio between is {result.synth_loss / result.test_loss}")
                valid_ratios += 1

    if valid_miners == 0:
        return False

    return (valid_ratios / valid_miners) >= threshold


def calculate_miner_ranking_and_scores(miner_results: list[MinerResults]) -> list[MinerResults]:
    """Calculate scores based on either test_loss or weighted_loss.
    Top ranked gets score=2, second gets score=1, others get 0.
    Bottom 25% get a penalty (cts.SCORE_PENALTY) if there are more than cts.MIN_IDEAL_NUM_MINERS_IN_POOL miners."""
    logger.info("Beginning score calculation...")
    for result in miner_results:
        with LogContext(miner_hotkey=result.hotkey):
            result.score = 0.0
            if not result.is_finetune:
                result.score_reason = "Non-finetuned submission"
                logger.info(f"Miner {result.hotkey}: Non-finetuned, score set to 0.0")
            elif np.isnan(result.test_loss) or np.isnan(result.synth_loss):
                result.score_reason = "Invalid loss"
                logger.info(f"Miner {result.hotkey}: Invalid loss, score set to 0.0")
            elif result.synth_loss == 1000.0:
                result.score_reason = "Outside of top-4 test doesn't get scored."
                logger.info(f"Miner {result.hotkey}: Outside of top-4")

    valid_results = [
        result
        for result in miner_results
        if result.is_finetune and not np.isnan(result.test_loss) and not np.isnan(result.synth_loss)
    ]
    if not valid_results:
        logger.warning("No valid finetuned submissions found. All scores set to 0.0")
        return miner_results
    # Check if synth losses are valid across all the miners, if it isn't then we just use the test loss
    use_weighted_loss = _is_synth_loss_valid_for_group(valid_results)
    if use_weighted_loss:
        logger.info("Using weighted loss for ranking (at least one miner has valid synth loss)")
        ranked_results = [(result, calculate_weighted_loss(result.test_loss, result.synth_loss)) for result in valid_results]
        ranked_results.sort(key=lambda x: x[1])
        ranking_type = "weighted_loss"
    else:
        logger.info("Using test loss only for ranking (all synth losses are invalid)")
        ranked_results = [(result, result.test_loss) for result in valid_results]
        ranked_results.sort(key=lambda x: x[1])
        ranking_type = "test_loss_only"

    # Assign scores for top 2 miners
    for i, (result, metric) in enumerate(ranked_results[:2]):
        with LogContext(miner_hotkey=result.hotkey):
            result.score = cts.FIRST_PLACE_SCORE if i == 0 else cts.SECOND_PLACE_SCORE
            rank = "1st" if i == 0 else "2nd"
            result.score_reason = f"Ranked {rank} by {ranking_type}"
            logger.info(
                f"Miner {result.hotkey} (finetuned):"
                f" test_loss={result.test_loss:.4f}"
                f" synth_loss={result.synth_loss:.4f}"
                f" {ranking_type}={metric:.4f}"
                f" score={result.score:.4f}"
                f" score_reason={result.score_reason}"
            )

    # Apply penalties to bottom 25% if enough miners are in the competition
    total_valid_miners = len(valid_results)
    if total_valid_miners > cts.MIN_IDEAL_NUM_MINERS_IN_POOL:
        penalty_count = max(1, int(total_valid_miners * 0.25))
        penalty_start_idx = total_valid_miners - penalty_count

        # Log miners ranked below top 2 but not in bottom 25%
        for result, metric in ranked_results[2:penalty_start_idx]:
            with LogContext(miner_hotkey=result.hotkey):
                result.score_reason = f"Ranked below top 2 by {ranking_type}"
                logger.info(
                    f"Miner {result.hotkey} (finetuned):"
                    f" test_loss={result.test_loss:.4f}"
                    f" synth_loss={result.synth_loss:.4f}"
                    f" {ranking_type}={metric:.4f}"
                    f" score=0.0"
                    f" score_reason={result.score_reason}"
                )

        # Apply penalty to bottom 25%
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
        # No penalties if not enough miners
        for result, metric in ranked_results[2:]:
            with LogContext(miner_hotkey=result.hotkey):
                result.score_reason = f"Ranked below top 2 by {ranking_type}"
                logger.info(
                    f"Miner {result.hotkey} (finetuned):"
                    f" test_loss={result.test_loss:.4f}"
                    f" synth_loss={result.synth_loss:.4f}"
                    f" {ranking_type}={metric:.4f}"
                    f" score=0.0"
                    f" score_reason={result.score_reason}"
                )

    return miner_results


def _get_dataset_type(task: TextRawTask) -> CustomDatasetType:
    return CustomDatasetType(
        field_system=task.field_system,
        field_instruction=task.field_instruction,
        field_input=task.field_input,
        field_output=task.field_output,
        format=task.format,
        no_input_format=task.no_input_format,
    )


def _create_failed_miner_result(hotkey: str, reason: str, task_type: TaskType) -> MinerResults:
    """Create a failed miner result with zero score."""
    if task_type == TaskType.TEXTTASK:
        return MinerResultsText(
            hotkey=hotkey, test_loss=np.nan, synth_loss=np.nan, is_finetune=False, score=0.0, score_reason=reason
        )
    else:
        return MinerResultsImage(
            hotkey=hotkey, test_loss=np.nan, synth_loss=np.nan, is_finetune=False, score=0.0, score_reason=reason
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


async def _get_submission_repo(miner: Node, task_id: str, config: Config) -> str | None:
    url = f"{cts.SUBMISSION_ENDPOINT}{task_id}"
    try:
        repo = str(await process_non_stream_fiber_get(url, config, miner))
        return None if repo == "None" else repo
    except Exception as e:
        logger.error(f"Failed to get submission for miner {miner.hotkey}: {e}")
        return None


async def _evaluate_submissions(
    task: TextRawTask | ImageRawTask,
    submission_repos: list[str],
    gpu_ids: list[int],
    dataset_type: CustomDatasetType | None = None,
) -> dict[str, tuple[EvaluationResultText, EvaluationResultText] | EvaluationResultImage | Exception]:
    """Evaluate same task submissions within same docker container.
    Docker evaluations with an exception will return the Exception for the repo."""
    unique_repos = list(set(submission_repos))
    if len(unique_repos) != len(submission_repos):
        logger.warning(f"Found duplicate repos. Deduplicating {len(submission_repos)} repos to {len(unique_repos)} unique repos")

    if isinstance(task, TextRawTask):
        results: dict[str, tuple[EvaluationResultText, EvaluationResultText] | Exception] = {}
        repos_to_evaluate = []
        for repo in unique_repos:
            if repo == task.model_id:
                logger.warning(f"Repository {repo} matches original model ID - marking as non-finetuned")
                results[repo] = (
                    EvaluationResultText(is_finetune=False, eval_loss=0.0, perplexity=0.0),
                    EvaluationResultText(is_finetune=False, eval_loss=0.0, perplexity=0.0),
                )
            else:
                repos_to_evaluate.append(repo)

        if not repos_to_evaluate:
            return results

        evaluation_params = {
            "file_format": FileFormat.JSON,
            "original_model": task.model_id,
            "models": repos_to_evaluate,
            "dataset_type": dataset_type,
            "gpu_ids": gpu_ids,
        }

        assert task.synthetic_data is not None, "Synthetic data shouldn't be none for text tasks"
        assert task.test_data is not None, "Test data shouldn't be none for text tasks"

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
                    EvaluationResultText(is_finetune=False, eval_loss=0.0, perplexity=0.0),
                    EvaluationResultText(is_finetune=False, eval_loss=0.0, perplexity=0.0),
                )
            else:
                test_losses.append((repo, test_result.eval_loss))

        test_losses.sort(key=lambda x: x[1])
        top_4_repos = [repo for repo, _ in test_losses[:4]]

        for repo, _ in test_losses[4:]:
            results[repo] = (
                # setting to 1k for now
                EvaluationResultText(is_finetune=True, eval_loss=1000.0, perplexity=1000.0),
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

    elif isinstance(task, ImageRawTask):
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
            #  assert cts.BUCKET_NAME is not None 'bucket name needs setting to delete'
            object_name = file_path.split(cts.BUCKET_NAME + "/")[-1]
            logger.info(f"Deleting file {object_name} from MinIO bucket {cts.BUCKET_NAME}")
            await async_minio_client.delete_file(cts.BUCKET_NAME, object_name)
        except Exception as e:
            logger.error(f"Failed to delete file {file_path} from MinIO: {e}")


async def _update_scores(
    task: TextRawTask | ImageRawTask, task_results: list[MinerResultsText | MinerResultsImage], psql_db
) -> None:
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
    """Get the creation timestamp of a Hugging Face repository."""
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
    """Group submissions by their loss values."""
    loss_groups: dict[tuple[float, float], list[tuple[str, str]]] = {}

    for result in task_results:
        if result.submission and not np.isnan(result.test_loss) and not np.isnan(result.synth_loss):
            losses = (float(result.test_loss), float(result.synth_loss))
            if losses not in loss_groups:
                loss_groups[losses] = []
            loss_groups[losses].append((result.hotkey, result.submission.repo))

    return loss_groups


async def get_earliest_submission(submissions: list[tuple[str, str]]) -> tuple[str, str, list[tuple[str, str]]]:
    """Determine earliest submission and list of duplicates."""
    timestamps = []
    for hotkey, repo in submissions:
        creation_time = await get_repo_creation_time(repo)
        timestamps.append((hotkey, repo, creation_time))

    timestamps.sort(key=lambda x: x[2])
    earliest_hotkey, earliest_repo, _ = timestamps[0]
    duplicates = [(hotkey, repo) for hotkey, repo, _ in timestamps]

    return earliest_hotkey, earliest_repo, duplicates


async def handle_duplicate_submissions(task_results: list[MinerResultsText | MinerResultsImage]) -> dict[str, bool]:
    """Process submissions and identify duplicates."""
    keep_submission = {result.hotkey: True for result in task_results}
    loss_groups = group_by_losses(task_results)

    for losses, submissions in loss_groups.items():
        if len(submissions) > 1:
            logger.warning(f"Found {len(submissions)} submissions with identical losses {losses}")
            earliest_hotkey, earliest_repo, duplicates = await get_earliest_submission(submissions)

            for hotkey, repo in duplicates:
                with LogContext(miner_hotkey=hotkey):
                    keep_submission[hotkey] = False
                    logger.warning(
                        f"Setting score to 0 for node {hotkey} (repo: {repo}) "
                        f"as it has identical losses to earlier submission "
                        f"from node {earliest_hotkey} (repo: {earliest_repo})"
                    )

    return keep_submission


def zero_duplicate_scores(
    task_results: list[MinerResultsText | MinerResultsImage], keep_submission: dict[str, bool]
) -> list[MinerResultsText | MinerResultsImage]:
    """Zero out scores for duplicate submissions."""
    for result in task_results:
        if not keep_submission[result.hotkey]:
            result.test_loss = np.nan
            result.synth_loss = np.nan
            result.is_finetune = False
            result.score_reason = result.score_reason or "Duplicated submission"
    return task_results


async def process_miners_pool(
    miners: list[Node],
    task: ImageRawTask | TextRawTask,
    config: Config,
    gpu_ids: list[int],
    dataset_type: CustomDatasetType | None = None,
) -> list[MinerResultsText | MinerResultsImage]:
    """Process same task miners"""
    assert task.task_id is not None, "We should have a task id when processing miners"

    miner_repos: dict[str, str] = {}
    for miner in miners:
        with LogContext(miner_hotkey=miner.hotkey):
            expected_name = await get_expected_repo_name(task.task_id, miner.hotkey, config.psql_db)
            repo = await _get_submission_repo(miner, str(task.task_id), config)
            if repo is not None:
                repo_parts = repo.split("/")
                if len(repo_parts) >= 2:
                    submitted_name = repo_parts[-1]

                    if expected_name and submitted_name != expected_name:
                        logger.warning(
                            f"Miner {miner.hotkey} submitted a repo with name {submitted_name} "
                            f"but expected {expected_name}. Marking as failed."
                        )
                        continue

                miner_repos[miner.hotkey] = repo
            logger.info(f"Found repo {repo} for miner {miner.hotkey}")

    results = [
        _create_failed_miner_result(miner.hotkey, reason="No repo submitted", task_type=task.task_type)
        for miner in miners
        if miner.hotkey not in miner_repos
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
                            _create_failed_miner_result(miner.hotkey, reason="Evaluation failed", task_type=task.task_type)
                        )
                        continue
                    elif isinstance(task, TextRawTask):
                        synth_result, test_result = eval_result
                    else:
                        test_result = eval_result
                        test_result.eval_loss = _calculate_weighted_loss_for_image_eval(test_result)
                        synth_result = test_result

                    submission = Submission(
                        task_id=task.task_id,
                        hotkey=miner.hotkey,
                        repo=repo,
                        created_on=datetime.now(),
                        updated_on=datetime.now(),
                    )

                if isinstance(task, TextRawTask):
                    results.append(
                        MinerResultsText(
                            hotkey=miner.hotkey,
                            test_loss=float(test_result.eval_loss),
                            synth_loss=float(synth_result.eval_loss),
                            is_finetune=test_result.is_finetune,
                            submission=submission,
                        )
                    )
                else:
                    results.append(
                        MinerResultsImage(
                            hotkey=miner.hotkey,
                            test_loss=float(test_result.eval_loss),
                            synth_loss=float(synth_result.eval_loss),
                            is_finetune=test_result.is_finetune,
                            submission=submission,
                        )
                    )

        except Exception as e:
            logger.error(f"Error during batch evaluation: {e}", exc_info=True)
            results.extend(
                [
                    _create_failed_miner_result(miner.hotkey, reason="Evaluation failed", task_type=task.task_type)
                    for miner in miners
                    if miner.hotkey not in [r.hotkey for r in results]
                ]
            )

    return results


async def evaluate_and_score(task: TextRawTask | ImageRawTask, gpu_ids: list[int], config: Config) -> TextRawTask | ImageRawTask:
    """Main function to evaluate and score task submissions."""
    assert task.task_id is not None, "Task ID must be present"
    assert task.test_data is not None, "Test data must be present"

    miner_pool = await get_nodes_assigned_to_task(str(task.task_id), config.psql_db)
    if isinstance(task, TextRawTask):
        dataset_type = _get_dataset_type(task)
    else:
        dataset_type = None

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
        if task.task_type == TaskType.TEXTTASK:
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
