from datetime import datetime
from scipy.stats import gmean

import numpy as np
from fiber.logging_utils import get_logger

import validator.core.constants as cts
from core.utils import download_s3_file
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.models import Submission
from validator.core.models import Task
from validator.core.models import MinerResults
from validator.db.sql import add_submission
from validator.db.sql import get_miners_assigned_to_task
from validator.db.sql import set_task_node_quality_score
from validator.evaluation.docker_evaluation import run_evaluation_docker
from validator.utils.call_endpoint import process_non_stream_get


logger = get_logger(__name__)



def calculate_weighted_loss(test_loss: float, synth_loss: float) -> float:
    """
    Calculate a weighted average of test and synthetic losses.

    This function combines the test loss and synthetic loss into a single metric,
    giving more weight to the test loss as defined by TEST_SCORE_WEIGHTING.
    """
    return cts.TEST_SCORE_WEIGHTING * test_loss + (1 - cts.TEST_SCORE_WEIGHTING) * synth_loss


def calculate_scaled_score(weighted_loss: float, is_finetune: bool, scale_factor: float) -> float:
    """
    Calculate a score for a miner based on their weighted loss, using an exponential decay function.

    This function converts a loss value into a score, where lower losses result in higher scores.
    """
    return np.exp(-weighted_loss * scale_factor) if is_finetune else 0.0


def adjust_miner_scores_to_be_relative_to_other_comps(miner_results: list[MinerResults]) -> list[MinerResults]:
    """
    This function adjusts all scores so that their geometric mean becomes 1.

    By dividing each miner's score by the geometric mean of all scores for that task,
    we're essentially measuring each miner's performance relative to the overall performance on that specific task.

    This normalisation makes the scores scale-independent.
    If Task A is inherently more difficult and results in lower scores overall,
    dividing by the geometric mean will adjust for this.
    """
    geometric_mean = gmean(np.array([res.score for res in miner_results]))

    for res in miner_results:
        res.score /= geometric_mean

    return miner_results


def compute_adaptive_scale_factor(miner_results: list[MinerResults]) -> float:
    """
    We want to calculate a scaling factor that can be applied to the loss results to make them more meaningful
    especially when the scores are closely clustered.
    For instance, if all scores fall between 0.8 and 0.85, it's difficult to distinguish performance differences.
    The function determines how much to "stretch" this range by computing a scale factor.
    This factor is calculated based on the lowest and highest scores in the set,
    aiming to create a consistent ratio between the best and worst scores (defined by a target ratio, typically 2:1).
    If the scores are tightly grouped, the scaling factor will be larger to amplify small differences.
    Conversely, if the scores are already well spread out, the scaling factor will be smaller.

    Examples:
    1. Closely clustered scores:
       miner_results = [
           ("Miner1", 0.82, 0.81, True),
           ("Miner2", 0.83, 0.82, True),
           ("Miner3", 0.81, 0.80, True),
           ("Miner4", 0.84, 0.83, True)
       ]
       Result: scale_factor ≈ 13.8
       (High scale factor due to tightly clustered scores)

    2. More spread out scores:
       miner_results = [
           ("Miner1", 0.5, 0.5, True),
           ("Miner2", 0.7, 0.7, True),
           ("Miner3", 0.9, 0.9, True),
           ("Miner4", 1.1, 1.1, True)
       ]
       Result: scale_factor ≈ 1.2
       (Lower scale factor due to already spread out scores)

    """
    # NOTE: can we make miner_results be a dataclass or a namedtuple? Rather than needing to just *know*
    # WW : YES!
    weighted_losses = [calculate_weighted_loss(result.test_loss, result.synth_loss) for result in miner_results]
    min_loss, max_loss = min(weighted_losses), max(weighted_losses)

    if min_loss == max_loss:
        return 1.0  # Default to 1 if all losses are the same

    return np.log(cts.TARGET_SCORE_RATIO) / (max_loss - min_loss)


def add_raw_scores_to_miner_results(miner_results: list[MinerResults]) -> list[MinerResults]:
    scale_factor = compute_adaptive_scale_factor(miner_results)  # see function def for details
    for result in miner_results:
        weighted_loss = calculate_weighted_loss(result.test_loss, result.synth_loss)
        result.score = calculate_scaled_score(weighted_loss, result.is_finetune, scale_factor)
    return miner_results

async def evaluate_and_score(task: Task, config: Config) -> Task:
    miner_pool = await get_miners_assigned_to_task(str(task.task_id), config.psql_db)
    assert task.task_id is not None
    task_results = []
    dataset_type = CustomDatasetType(
        field_system=task.system, field_instruction=task.instruction, field_input=task.input, field_output=task.output
    )

    for miner in miner_pool:
        try:
            url = f"{miner.ip}:{miner.port}/get_latest_model_submission/{task.task_id}"
            submission_repo = str(await process_non_stream_get(url, None))
            current_time = datetime.now()
            submission = Submission(
                task_id=task.task_id,
                node_id=miner.node_id,
                repo=submission_repo,
                created_on=current_time,
                updated_on=current_time,
            )
            evaluation_params = {
                "file_format": FileFormat.JSON,
                "original_model": task.model_id,
                "model": submission_repo,
                "dataset_type": dataset_type,
            }

            assert task.synthetic_data is not None
            assert task.test_data is not None

            synthetic_data_filepath = await download_s3_file(task.synthetic_data)
            test_data_filepath = await download_s3_file(task.test_data)

            synth_eval_result = await run_evaluation_docker(
                dataset=synthetic_data_filepath, **evaluation_params
            )
            test_eval_result = await run_evaluation_docker(dataset=test_data_filepath, **evaluation_params)

            logger.info(f"The losses that we have out from {miner.node_id} are synth: {synth_eval_result.eval_loss} and test {test_eval_result.eval_loss}")
            logger.info(
                f"The perplexities that we have out from {miner.node_id} are synth: {synth_eval_result.perplexity} and test {test_eval_result.perplexity}"
            )

            miner_result = MinerResults(node_id = miner.node_id,
                                        test_loss = test_eval_result.eval_loss,
                                        synth_loss = synth_eval_result.eval_loss,
                                        is_finetune= test_eval_result.is_finetune, submission=submission)
            task_results.append(miner_result)

        except Exception as e:
            logger.info(f"There was an issue with scoring {e}")

    task_results = add_raw_scores_to_miner_results(task_results)
    task_results = adjust_miner_scores_to_be_relative_to_other_comps(task_results)

    for result in task_results:
        assert result.score is not None
        await set_task_node_quality_score(task.task_id, result.node_id, result.score, config.psql_db)
        result.submission.score  = result.score * task.hours_to_complete
        logger.info(f"Adding submission {result.submission}")
        await add_submission(result.submission, config.psql_db)
    logger.info(f"The final results are {task_results}")
    task.status = TaskStatus.SUCCESS

    return task