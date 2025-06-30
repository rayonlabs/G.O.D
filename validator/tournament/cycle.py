import asyncio

from validator.core.config import load_config
from validator.tournament.tournament_manager import process_active_tournaments
from validator.tournament.tournament_manager import process_pending_tournaments
from validator.tournament.tournament_manager import process_prepped_tournament_tasks
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def cycle():
    config = load_config()

    if config.netuid == 56:
        raise Exception("This is not meant to be run on mainnet yet. It may affect real miner weights.")

    await config.psql_db.connect()
    logger.info("Connected to database")

    while True:
        # this gets the submissions and populates the tournament participants
        await process_pending_tournaments(config)
        # this skips the LOOKING_FOR_NODES status and sets the tasks to ready
        await process_prepped_tournament_tasks(config)
        # this advances the tournament till completion
        await process_active_tournaments(config)

        logger.info("Sleeping for 15 seconds")
        await asyncio.sleep(15)  # 15 seconds


if __name__ == "__main__":
    asyncio.run(cycle())
