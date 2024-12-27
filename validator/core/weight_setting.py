"""
Calculates and schedules weights every SCORING_PERIOD
"""

import asyncio
import os

from dotenv import load_dotenv
from fiber.chain import fetch_nodes
from fiber.chain import weights
from fiber.chain.interface import get_substrate
from fiber.chain.models import Node
from fiber.logging_utils import get_logger

from core import constants as ccst
from validator.core.config import Config
from validator.core.config import load_config
from validator.core.models import PeriodScore
from validator.db.sql.nodes import get_node_id_by_hotkey
from validator.db.sql.nodes import get_vali_node_id
from validator.evaluation.scoring import scoring_aggregation_from_date
from validator.utils.query_substrate import query_substrate
from validator.utils.util import try_db_connections


load_dotenv(os.getenv("ENV_FILE", ".vali.env"))


logger = get_logger(__name__)


TIME_PER_BLOCK: int = 500


async def _get_weights_to_set(config: Config) -> list[PeriodScore] | None:
    return await scoring_aggregation_from_date(config.psql_db)


async def _get_and_set_weights(config: Config, validator_node_id: int) -> bool:
    node_results = await _get_weights_to_set(config)
    if node_results is None:
        logger.info("No weights to set. Skipping weight setting.")
        return False
    if len(node_results) == 0:
        logger.info("No nodes to set weights for. Skipping weight setting.")
        return False

    logger.info("Weights calculated, about to set...")

    all_nodes: list[Node] = fetch_nodes.get_nodes_for_netuid(config.substrate, config.netuid)
    all_node_ids = [node.node_id for node in all_nodes]
    all_node_weights = [0.0 for _ in all_nodes]
    for node_result in node_results:
        if node_result.normalised_score is not None:
            node_id = await get_node_id_by_hotkey(node_result.hotkey, config.psql_db)
            if node_id is not None:
                all_node_weights[node_id] = node_result.normalised_score

    logger.info(f"Node ids: {all_node_ids}")
    logger.info(f"Node weights: {all_node_weights}")
    logger.info(f"Number of non zero node weights: {sum(1 for weight in all_node_weights if weight != 0)}")
    logger.info(
        f"Everything going in is {weights.set_node_weights} {config.substrate} {config.keypair}"
        f" {all_node_ids} {all_node_weights} {config.netuid} {ccst.VERSION_KEY} {validator_node_id}"
    )

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
    substrate = get_substrate(subtensor_address=config.substrate.url)
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
