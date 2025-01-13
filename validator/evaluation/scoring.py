import os
import re
from datetime import datetime
from datetime import timedelta

import numpy as np
from fiber.chain.models import Node
from scipy.stats import gmean

import validator.core.constants as cts
from core.models.payload_models import EvaluationResult
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskStatus
from core.utils import download_s3_file
from validator.core.config import Config
from validator.core.models import MinerResults
from validator.core.models import NodeAggregationResult
from validator.core.models import PeriodScore
from validator.core.models import RawTask
from validator.core.models import Submission
from validator.core.models import TaskNode
from validator.core.models import TaskResults
from validator.db.sql.submissions_and_scoring import add_submission
from validator.db.sql.submissions_and_scoring import get_aggregate_scores_since
from validator.db.sql.submissions_and_scoring import set_task_node_quality_score
from validator.db.sql.tasks import get_expected_repo_name
from validator.db.sql.tasks import get_nodes_assigned_to_task
from validator.evaluation.docker_evaluation import run_evaluation_docker
from validator.utils.call_endpoint import process_non_stream_fiber_get
from validator.utils.call_endpoint import process_non_stream_get
from validator.utils.logging import LogContext
from validator.utils.logging import add_context_tag
from validator.utils.logging import get_logger
from validator.utils.minio import async_minio_client


logger = get_logger(__name__)


def get_task_work_score(task: RawTask) -> float:
    """Calculate work score for a task based on hours and model size."""
    assert task.hours_to_complete > 0, "Hours to complete must be positive"
    assert task.model_id, "Model ID must be present"

    hours = task.hours_to_complete
    model = task.model_id
    model_size = re.search(r"(\d+)(?=[bB])", model)
    model_size_value = max(8, int(model_size.group(1)) if model_size else 1)

    return max(1, 2 * np.log(float(hours * model_size_value)))


def calculate_adjusted_task_score(quality_score: float, task_work_score: float) -> float:
    """Calculate adjusted task score based on quality score and work score."""
    assert not np.isnan(quality_score), "Quality score cannot be NaN"
    assert not np.isnan(task_work_score), "Task work score cannot be NaN"
    return max(cts.MIN_TASK_SCORE, quality_score - cts.TASK_SCORE_THRESHOLD) * task_work_score


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
            )
        )

    return final_scores


def normalise_scores(period_scores: list[PeriodScore]) -> list[PeriodScore]:
    """Normalise scores and update node emission values. Now < 0 maps to zero"""
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
            node_period_score.normalised_score = (node_period_score.quality_score / max_score) ** cts.REWEIGHTING_EXP

    return period_scores


async def scoring_aggregation_from_date(psql_db: str) -> list[PeriodScore]:
    """Aggregate and normalise scores across all nodes."""
    date = datetime.now() - timedelta(days=cts.SCORING_WINDOW)
    task_results: list[TaskResults] = await get_aggregate_scores_since(date, psql_db)
    logger.info(f"Got task results {task_results}")
    if not task_results:
        logger.info("There were not results to be scored")
        return []

    node_aggregations: dict[str, NodeAggregationResult] = {}

    for task_res in task_results:
        task_work_score = get_task_work_score(task_res.task)
        for node_score in task_res.node_scores:
            update_node_aggregation(node_aggregations, node_score, task_work_score)

    final_scores = calculate_node_quality_scores(node_aggregations)
    final_scores = normalise_scores(final_scores)
    return final_scores


def calculate_weighted_loss(test_loss: float, synth_loss: float) -> float:
    """Calculate weighted average of losses with more weight on test loss."""
    assert not np.isnan(test_loss), "Test loss cannot be NaN"
    assert not np.isnan(synth_loss), "Synthetic loss cannot be NaN"
    return cts.TEST_SCORE_WEIGHTING * test_loss + (1 - cts.TEST_SCORE_WEIGHTING) * synth_loss


def calculate_scaled_score(weighted_loss: float, scale_factor: float) -> float:
    """Calculate score using exponential decay."""
    assert not np.isnan(weighted_loss), "Weighted loss cannot be NaN"
    if scale_factor <= 0:
        scale_factor = 1.0
    return float(np.exp(-weighted_loss * scale_factor))


def compute_adaptive_scale_factor(miner_results: list[MinerResults]) -> float:
    """Compute scale factor based only on finetuned submissions."""
    finetuned_results = [
        res for res in miner_results if res.is_finetune and not np.isnan(res.test_loss) and not np.isnan(res.synth_loss)
    ]

    if not finetuned_results or len(finetuned_results) == 1:
        logger.info("No finetuned results found for scale factor calculation")
        return 1.0

    weighted_losses = [calculate_weighted_loss(res.test_loss, res.synth_loss) for res in finetuned_results]

    min_loss, max_loss = min(weighted_losses), max(weighted_losses)
    logger.info(f"Loss range for finetuned submissions - min: {min_loss:.4f}, max: {max_loss:.4f}")

    if min_loss == max_loss:
        logger.info("All finetuned submissions have identical losses, using default scale factor")
        return 2.0

    scale = float(np.log(cts.TARGET_SCORE_RATIO) / (max_loss - min_loss))
    logger.info(f"Computed scale factor: {scale:.4f}")
    return scale


def adjust_miner_scores_to_be_relative_to_other_comps(miner_results: list[MinerResults]) -> list[MinerResults]:
    """Adjusts scores to have geometric mean of 1.0 for finetuned submissions only."""
    valid_scores = [
        res.score
        for res in miner_results
        if res.is_finetune and res.score is not None and not np.isnan(res.score) and res.score > 0
    ]

    if not valid_scores:
        logger.warning("No valid finetuned submissions found for score adjustment")
        return miner_results

    logger.info(f"Adjusting scores for {len(valid_scores)} finetuned submissions")
    logger.info(f"Pre-adjustment scores: {valid_scores}")

    geometric_mean = float(gmean(np.array(valid_scores)))

    if np.isnan(geometric_mean) or np.isinf(geometric_mean) or geometric_mean <= 0:
        logger.warning(f"Invalid geometric mean: {geometric_mean}. Scores unchanged.")
        geometric_mean = 1.0

    logger.info(f"Geometric mean: {geometric_mean:.4f}")

    for res in miner_results:
        with LogContext(miner_hotkey=res.hotkey):
            if res.is_finetune and res.score is not None and not np.isnan(res.score):
                original_score = res.score
                res.score = min(float(res.score / geometric_mean), cts.MAX_TASK_SCORE)
                logger.info(f"Miner {res.hotkey}: {original_score:.4f} -> {res.score:.4f}")
            else:
                res.score = 0.0
                logger.info(f"Miner {res.hotkey}: score set to 0.0 (non-finetuned or invalid)")

    return miner_results


def add_raw_scores_to_miner_results(miner_results: list[MinerResults]) -> list[MinerResults]:
    """Calculate scores using only finetuned submissions."""
    logger.info("Beginning score calculation...")

    for result in miner_results:
        with LogContext(miner_hotkey=result.hotkey):
            if not result.is_finetune:
                result.score = 0.0
                logger.info(f"Miner {result.hotkey}: Non-finetuned, score set to 0.0")

    finetuned_results = [
        res for res in miner_results if res.is_finetune and not np.isnan(res.test_loss) and not np.isnan(res.synth_loss)
    ]

    if not finetuned_results:
        logger.warning("No valid finetuned submissions found. All scores set to 0.0")
        for result in miner_results:
            result.score = 0.0
        return miner_results

    scale_factor = compute_adaptive_scale_factor(finetuned_results)
    logger.info(f"Using scale factor: {scale_factor} (calculated from {len(finetuned_results)} finetuned submissions)")

    for result in miner_results:
        with LogContext(miner_hotkey=result.hotkey):
            if result.is_finetune and not np.isnan(result.test_loss) and not np.isnan(result.synth_loss):
                weighted_loss = calculate_weighted_loss(result.test_loss, result.synth_loss)
                result.score = calculate_scaled_score(weighted_loss, scale_factor)
                logger.info(
                    f"Miner {result.hotkey} (finetuned):"
                    f" test_loss={result.test_loss:.4f}"
                    f" synth_loss={result.synth_loss:.4f}"
                    f" weighted_loss={weighted_loss:.4f}"
                    f" score={result.score:.4f}"
                )
            else:
                result.score = 0.0
                logger.info(f"Miner {result.hotkey}: score=0.0 (non-finetuned or invalid losses)")

    return miner_results


def _get_dataset_type(task: RawTask) -> CustomDatasetType:
    return CustomDatasetType(
        field_system=task.field_system,
        field_instruction=task.field_instruction,
        field_input=task.field_input,
        field_output=task.field_output,
        format=task.format,
        no_input_format=task.no_input_format,
    )


def _create_failed_miner_result(hotkey: str) -> MinerResults:
    return MinerResults(hotkey=hotkey, test_loss=np.nan, synth_loss=np.nan, is_finetune=False, submission=None)


async def _get_submission_repo(miner: Node, task_id: str, config: Config) -> str | None:
    url = f"{cts.SUBMISSION_ENDPOINT}{task_id}"
    try:
        repo = str(await process_non_stream_fiber_get(url, config, miner))
        return None if repo == "None" else repo
    except Exception as e:
        logger.error(f"Failed to get submission for miner {miner.hotkey}: {e}")
        return None


async def _evaluate_submissions(
    task: RawTask, submission_repos: list[str], dataset_type: CustomDatasetType, gpu_ids: list[int]
) -> dict[str, tuple[EvaluationResult, EvaluationResult] | Exception]:
    """Evaluate same task submissions within same docker container.
    Docker evaluations with an exception will return the Exception for the repo."""
    unique_repos = list(set(submission_repos))
    if len(unique_repos) != len(submission_repos):
        logger.warning(f"Found duplicate repos. Deduplicating {len(submission_repos)} repos to {len(unique_repos)} unique repos")

    results: dict[str, tuple[EvaluationResult, EvaluationResult] | Exception] = {}
    repos_to_evaluate = []
    for repo in unique_repos:
        if repo == task.model_id:
            logger.warning(f"Repository {repo} matches original model ID - marking as non-finetuned")
            results[repo] = (
                EvaluationResult(is_finetune=False, eval_loss=0.0, perplexity=0.0),
                EvaluationResult(is_finetune=False, eval_loss=0.0, perplexity=0.0),
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

    assert task.synthetic_data is not None, "Synthetic data shouldn't be none"
    assert task.test_data is not None, "Test data shouldn't be none"

    logger.info("Starting synth evaluation")
    synthetic_data_filepath = await download_s3_file(task.synthetic_data)
    synth_eval_results = await run_evaluation_docker(dataset=synthetic_data_filepath, **evaluation_params)
    os.remove(synthetic_data_filepath)

    finetuned_repos = []
    for repo in repos_to_evaluate:
        if isinstance(synth_eval_results.get(repo), Exception):
            results[repo] = synth_eval_results[repo]
            continue

        synth_result = synth_eval_results[repo]
        if not synth_result.is_finetune:
            results[repo] = (
                EvaluationResult(is_finetune=False, eval_loss=0.0, perplexity=0.0),
                EvaluationResult(is_finetune=False, eval_loss=0.0, perplexity=0.0),
            )
        else:
            finetuned_repos.append(repo)

    if finetuned_repos:
        test_data_filepath = await download_s3_file(task.test_data)
        test_eval_results = await run_evaluation_docker(
            dataset=test_data_filepath, models=finetuned_repos, **{k: v for k, v in evaluation_params.items() if k != "models"}
        )
        os.remove(test_data_filepath)

        for repo in finetuned_repos:
            if isinstance(test_eval_results.get(repo), Exception):
                results[repo] = test_eval_results[repo]
            else:
                results[repo] = (synth_eval_results[repo], test_eval_results[repo])

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


async def _update_scores(task: RawTask, task_results: list[MinerResults], psql_db) -> None:
    assert task.task_id is not None, "task id needs to be seet to update scores"
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
                psql_db=psql_db
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


async def handle_duplicate_submissions(task_results: list[MinerResults]) -> dict[str, bool]:
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


def zero_duplicate_scores(task_results: list[MinerResults], keep_submission: dict[str, bool]) -> list[MinerResults]:
    """Zero out scores for duplicate submissions."""
    for result in task_results:
        if not keep_submission[result.hotkey]:
            result.test_loss = np.nan
            result.synth_loss = np.nan
            result.is_finetune = False
    return task_results


async def process_miners_pool(
    miners: list[Node], task: RawTask, dataset_type: CustomDatasetType, config: Config, gpu_ids: list[int]
) -> list[MinerResults]:
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

    results = [_create_failed_miner_result(miner.hotkey) for miner in miners if miner.hotkey not in miner_repos]

    if miner_repos:
        try:
            eval_results = await _evaluate_submissions(
                task=task, submission_repos=list(miner_repos.values()), dataset_type=dataset_type, gpu_ids=gpu_ids
            )

            for miner in miners:
                with LogContext(miner_hotkey=miner.hotkey):
                    if miner.hotkey not in miner_repos:
                        continue

                    repo = miner_repos[miner.hotkey]
                    eval_result = eval_results.get(repo)

                    if isinstance(eval_result, Exception):
                        logger.error(f"Evaluation failed for miner {miner.hotkey}: {eval_result}")
                        results.append(_create_failed_miner_result(miner.hotkey))
                        continue

                    synth_result, test_result = eval_result
                    submission = Submission(
                        task_id=task.task_id,
                        hotkey=miner.hotkey,
                        repo=repo,
                        created_on=datetime.now(),
                        updated_on=datetime.now(),
                    )

                    results.append(
                        MinerResults(
                            hotkey=miner.hotkey,
                            test_loss=float(test_result.eval_loss),
                            synth_loss=float(synth_result.eval_loss),
                            is_finetune=test_result.is_finetune,
                            submission=submission,
                        )
                    )

        except Exception as e:
            logger.error(f"Error during batch evaluation: {e}")
            results.extend(
                [_create_failed_miner_result(miner.hotkey) for miner in miners if miner.hotkey not in [r.hotkey for r in results]]
            )

    return results


async def evaluate_and_score(task: RawTask, gpu_ids: list[int], config: Config) -> RawTask:
    """Main function to evaluate and score task submissions."""
    assert task.task_id is not None, "Task ID must be present"
    assert task.synthetic_data is not None, "Synthetic data must be present"
    assert task.test_data is not None, "Test data must be present"

    miner_pool = await get_nodes_assigned_to_task(str(task.task_id), config.psql_db)
    dataset_type = _get_dataset_type(task)

    logger.info(f"Beginning evaluation for task {task.task_id} with {len(miner_pool)} miners")
    task_results = await process_miners_pool(miner_pool, task, dataset_type, config, gpu_ids)

    logger.info("Checking for duplicates ...")
    keep_submission = await handle_duplicate_submissions(task_results)
    task_results = zero_duplicate_scores(task_results, keep_submission)

    logger.info("Calculating final scores...")
    task_results = add_raw_scores_to_miner_results(task_results)
    task_results = adjust_miner_scores_to_be_relative_to_other_comps(task_results)
    await _update_scores(task, task_results, config.psql_db)
    # all_scores_zero = all(result.score == 0.0 for result in task_results)
    # for now we just let them fail, need to come back to decide whether we wanna restart the job
    all_scores_zero = False
    if all_scores_zero:
        task.status = TaskStatus.NODE_TRAINING_FAILURE
        add_context_tag("status", task.status.value)
        logger.info(
            f"All scores are zero for task {task.task_id}, setting status to LOOKING FOR NODES to find new miner since"
            "we are going to try again."
        )
    else:
        if cts.DELETE_S3_AFTER_COMPLETE:
            await _clear_up_s3([task.training_data, task.test_data, task.synthetic_data])
        task.status = TaskStatus.SUCCESS
        add_context_tag("status", task.status.value)
        logger.info(f"Task {task.task_id} completed successfully with non-zero scores")
    return task
