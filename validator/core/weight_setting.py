"""
Calculates and schedules weights every SCORING_PERIOD
"""

import asyncio
import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone

from dotenv import load_dotenv

from validator.db.sql.auditing import store_latest_scores_url
from validator.db.sql.submissions_and_scoring import get_aggregate_scores_since


load_dotenv(os.getenv("ENV_FILE", ".vali.env"))

import json
from uuid import UUID

from fiber.chain import fetch_nodes
from fiber.chain import weights
from fiber.chain.chain_utils import query_substrate
from fiber.chain.models import Node
from substrateinterface import SubstrateInterface

import validator.core.constants as cts
from core import constants as ccst
from core.constants import BUCKET_NAME
from validator.core.config import Config
from validator.core.config import load_config
from validator.core.models import PeriodScore
from validator.core.models import TaskResults
from validator.db.database import PSQLDB
from validator.db.sql.nodes import get_vali_node_id
from validator.evaluation.scoring import get_period_scores_from_results
from validator.utils.logging import get_logger
from validator.utils.util import save_json_to_temp_file
from validator.utils.util import try_db_connections
from validator.utils.util import upload_json_to_minio


logger = get_logger(__name__)


TIME_PER_BLOCK: int = 500


async def get_period_scores_from_task_results(task_results: list[TaskResults], psql_db: PSQLDB) -> list[PeriodScore]:
    if not task_results:
        logger.info("There were not results to be scored")
        return []

    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)
    one_day_ago = datetime.now(timezone.utc) - timedelta(days=1)

    seven_day_task_results = [i for i in task_results if i.task.created_at > seven_days_ago]
    three_day_task_results = [i for i in task_results if i.task.created_at > three_days_ago]
    one_day_task_results = [i for i in task_results if i.task.created_at > one_day_ago]

    seven_day_scores = await get_period_scores_from_results(seven_day_task_results, weight_multiplier=0.25, psql_db=psql_db)
    three_day_scores = await get_period_scores_from_results(three_day_task_results, weight_multiplier=0.4, psql_db=psql_db)
    one_day_scores = await get_period_scores_from_results(one_day_task_results, weight_multiplier=0.35, psql_db=psql_db)

    all_period_scores = seven_day_scores + three_day_scores + one_day_scores

    return all_period_scores


async def _get_weights_to_set(config: Config) -> tuple[list[PeriodScore], list[TaskResults]]:
    """
    Retrieve task results from the database and score multiple periods independently.
    This ensures a fairer ramp-up for new miners.

    In the future, as miners become more stable, we aim to encourage long-term stability.
    This means not giving new miners more weight than necessary, while still allowing them
    the potential to reach the top position without being deregistered.

    Period scores are calculated completely independently
    """
    date = datetime.now() - timedelta(days=cts.SCORING_WINDOW)
    task_results: list[TaskResults] = await get_aggregate_scores_since(date, config.psql_db)

    all_period_scores = await get_period_scores_from_task_results(task_results, config.psql_db)

    return all_period_scores, task_results


async def _upload_results_to_s3(config: Config, task_results: list[TaskResults]) -> None:
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, UUID):
                return str(obj)
            return super().default(obj)

    scores_json = json.dumps([result.model_dump() for result in task_results], indent=2, cls=DateTimeEncoder)

    temp_file, _ = await save_json_to_temp_file(scores_json, "latest_scores", dump_json=False)
    datetime_of_upload = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    presigned_url = await upload_json_to_minio(temp_file, BUCKET_NAME, f"latest_scores_{datetime_of_upload}.json")
    os.remove(temp_file)
    await store_latest_scores_url(presigned_url, config)
    return presigned_url


async def get_node_weights_from_period_scores(
    substrate: SubstrateInterface, netuid: int, node_results: list[PeriodScore]
) -> tuple[list[int], list[float]]:
    """
    Get the node ids and weights from the node results.
    """
    all_nodes: list[Node] = fetch_nodes.get_nodes_for_netuid(substrate, netuid)

    hotkey_to_node_id = {node.hotkey: node.node_id for node in all_nodes}

    all_node_ids = [node.node_id for node in all_nodes]
    all_node_weights = [0.0 for _ in all_nodes]
    for node_result in node_results:
        if node_result.normalised_score is not None:
            node_id = hotkey_to_node_id.get(node_result.hotkey)
            if node_id is not None:
                all_node_weights[node_id] = (
                    all_node_weights[node_id] + node_result.normalised_score * node_result.weight_multiplier
                )

    logger.info(f"Node ids: {all_node_ids}")
    logger.info(f"Node weights: {all_node_weights}")
    logger.info(f"Number of non zero node weights: {sum(1 for weight in all_node_weights if weight != 0)}")
    logger.info(f"Everything going in is {all_node_ids} {all_node_weights} {netuid} {ccst.VERSION_KEY}")
    return all_node_ids, all_node_weights


async def set_weights(config: Config, all_node_ids: list[int], all_node_weights: list[float], validator_node_id: int) -> bool:
    try:
        success = await asyncio.to_thread(
            weights.set_node_weights,
            substrate=config.substrate,
            keypair=config.keypair,
            node_ids=all_node_ids,
            node_weights=all_node_weights,
            netuid=config.netuid,
            version_key=ccst.VERSION_KEY,
            validator_node_id=int(validator_node_id),
            wait_for_inclusion=False,
            wait_for_finalization=False,
            max_attempts=3,
        )
    except Exception as e:
        logger.error(f"Failed to set weights: {e}")
        return False

    if success:
        logger.info("Weights set successfully.")

        return True
    else:
        logger.error("Failed to set weights :(")
        return False


async def _get_and_set_weights(config: Config, validator_node_id: int) -> bool:
    node_results, task_results = await _get_weights_to_set(config)
    if node_results is None:
        logger.info("No weights to set. Skipping weight setting.")
        return False
    if len(node_results) == 0:
        logger.info("No nodes to set weights for. Skipping weight setting.")
        return False

    all_node_ids, all_node_weights = await get_node_weights_from_period_scores(config.substrate, config.netuid, node_results)
    logger.info("Weights calculated, about to set...")

    success = await set_weights(config, all_node_ids, all_node_weights, validator_node_id)
    if success:
        url = await _upload_results_to_s3(config, task_results)
        logger.info(f"Uploaded the scores to s3 for auditing - url: {url}")

    return success


async def _set_metagraph_weights(config: Config) -> None:
    nodes: list[Node] = fetch_nodes.get_nodes_for_netuid(config.substrate, config.netuid)
    node_ids = [node.node_id for node in nodes]
    node_weights = [node.incentive for node in nodes]
    validator_node_id = await get_vali_node_id(config.substrate, config.keypair.ss58_address)
    if validator_node_id is None:
        raise ValueError("Validator node id not found")

    await asyncio.to_thread(
        weights.set_node_weights,
        substrate=config.substrate,
        keypair=config.keypair,
        node_ids=node_ids,
        node_weights=node_weights,
        netuid=config.netuid,
        version_key=ccst.VERSION_KEY,
        validator_node_id=int(validator_node_id),
        wait_for_inclusion=False,
        wait_for_finalization=False,
        max_attempts=3,
    )


#


# To improve: use activity cutoff & The epoch length to set weights at the perfect times
async def set_weights_periodically(config: Config, just_once: bool = False) -> None:
    substrate = config.substrate
    substrate, uid = query_substrate(
        substrate,
        "SubtensorModule",
        "Uids",
        [config.netuid, config.keypair.ss58_address],
        return_value=True,
    )

    if uid is None:
        raise ValueError(f"Can't find hotkey {config.keypair.ss58_address} for our keypair on netuid: {config.netuid}.")

    consecutive_failures = 0
    while True:
        substrate, current_block = query_substrate(substrate, "System", "Number", [], return_value=True)
        substrate, last_updated_value = query_substrate(
            substrate, "SubtensorModule", "LastUpdate", [config.netuid], return_value=False
        )
        updated: int = current_block - last_updated_value[uid].value
        substrate, weights_set_rate_limit = query_substrate(
            substrate, "SubtensorModule", "WeightsSetRateLimit", [config.netuid], return_value=True
        )
        logger.info(
            f"My Validator Node ID: {uid}. Last updated {updated} blocks ago. Weights set rate limit: {weights_set_rate_limit}."
        )

        if updated < weights_set_rate_limit:
            logger.info("Sleeping for a bit as we set recently...")
            await asyncio.sleep((weights_set_rate_limit - updated + 1) * 12)
            continue

        if os.getenv("ENV", "prod").lower() == "dev":
            success = await _get_and_set_weights(config, uid)
        else:
            try:
                success = await _get_and_set_weights(config, uid)
            except Exception as e:
                logger.error(f"Failed to set weights with error: {e}")
                logger.exception(e)
                success = False

        if success:
            consecutive_failures = 0
            logger.info("Successfully set weights! Sleeping for 25 blocks before next check...")
            if just_once:
                return
            await asyncio.sleep(12 * 25)
            continue

        consecutive_failures += 1
        if just_once:
            logger.info("Failed to set weights, will try again...")
            await asyncio.sleep(12 * 1)
        else:
            logger.info(f"Failed to set weights {consecutive_failures} times in a row - sleeping for a bit...")
            await asyncio.sleep(12 * 25)  # Try again in 25 blocks

        if consecutive_failures == 1 or updated < 3000:
            continue

        if just_once or config.set_metagraph_weights_with_high_updated_to_not_dereg:
            logger.warning("Setting metagraph weights as our updated value is getting too high!")
            if just_once:
                logger.warning("Please exit if you do not want to do this!!!")
                await asyncio.sleep(4)
            try:
                success = await _set_metagraph_weights(config)
            except Exception as e:
                logger.error(f"Failed to set metagraph weights: {e}")
                success = False

            if just_once:
                return

            if success:
                consecutive_failures = 0
                continue


async def main():
    config = load_config()
    await try_db_connections(config)
    await set_weights_periodically(config)


if __name__ == "__main__":
    asyncio.run(main())
