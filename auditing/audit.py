import asyncio
import json

from fiber.chain.weights import _normalize_and_quantize_weights

from core.utils import download_s3_file
from validator.core.config import Config
from validator.core.config import load_config
from validator.core.models import NodeAggregationResult
from validator.core.models import PeriodScore
from validator.core.models import TaskResults
from validator.core.weight_setting import get_node_weights_from_results
from validator.evaluation.scoring import _normalise_scores
from validator.evaluation.scoring import calculate_node_quality_scores
from validator.evaluation.scoring import get_task_work_score
from validator.evaluation.scoring import update_node_aggregation
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def _get_task_results_for_rayon_validator(config: Config) -> list[TaskResults]:
    """Get task results for a rayon validator."""
    url_to_get_latest_scores_url = "https://api.gradients.io/auditing/scores-url"
    response = await config.httpx_client.get(url_to_get_latest_scores_url)

    if response.status_code != 200:
        logger.error(f"Failed to get latest scores url: {response.status_code} {response.text} :(")
        raise Exception(f"Failed to get latest scores url: {response.status_code} {response.text} :(")

    url = response.json()["url"]

    logger.info(f"Getting task results from {url}")
    result_filepath = await download_s3_file(url)
    with open(result_filepath, "r") as f:
        task_results_dicts = json.loads(json.load(f))

    logger.info(f"Task results dicts: {list(task_results_dicts[0].keys())}")

    return [TaskResults(**task_results_dict) for task_results_dict in task_results_dicts]


async def get_scores_from_rayon_vali(config: Config) -> list[PeriodScore]:
    """Check that scores are calculated correctly by the validator.

    Receive details of every task that occurred in the past 7 days.
    Check that the scores for these tasks correctly add up to the weights set by the validator
    for all miners on chain.

    This helps to audit validators by:
        - Ensuring the weights set are backed up by scores for tasks
        - Preventing the validator from being able to manipulate scores to set weights on chain, without a task being completed
        - Ensuring miners are rewarded for fair work

    This is one tool, used in conjunction with others.
    Miners (well, everyone) can see every task that runs through the subnet,
    and see their scores using the dashboards on gradients.io. Details about the models trained by
    miners are available on there too, so anyone can check the evaluation was fair.

    To make this more robust: we can add a function which sends tasks through to the api, to ensure all organic
    jobs are indeed included in the scoring. For the short term, miners can of course always check this, by comparing the work
    they have done to the tasks on the dashboards - it's only a minor improvement.

    """

    task_results = await _get_task_results_for_rayon_validator(config)

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
    final_scores = _normalise_scores(final_scores)
    return final_scores


async def get_weights_for_rayon_validator() -> list[float]:
    """Get the weights for a rayon validator."""
    config = load_config()
    results = await get_scores_from_rayon_vali(config)

    node_ids, node_weights = await get_node_weights_from_results(config.substrate, config.netuid, results)

    node_ids_formatted, node_weights_formatted = _normalize_and_quantize_weights(node_ids, node_weights)
    return node_ids_formatted, node_weights_formatted

    print(node_ids_formatted, node_weights_formatted)

    # Get on chain weights set by that validator

    # Compare and ensure they are close by ensuring dot product is close to 1


if __name__ == "__main__":
    asyncio.run(get_weights_for_rayon_validator())
