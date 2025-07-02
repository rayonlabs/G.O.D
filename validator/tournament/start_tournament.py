import asyncio

from core.models.tournament_models import TournamentType
from validator.core.config import load_config
from validator.db.sql.tournaments import cancel_all_active_tournaments
from validator.tournament.tournament_manager import create_basic_tournament
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def main():
    logger.info("Starting tournament...")

    config = load_config()

    if config.netuid == 56:
        raise Exception("This is not meant to be run on mainnet yet. It may affect real miner weights.")

    await config.psql_db.connect()
    logger.info("Connected to database")

    try:
        # Step 0: Cancel all active or pending tournaments
        logger.info("Step 0: Cancelling all active or pending tournaments...")
        cancelled_count = await cancel_all_active_tournaments(config.psql_db)
        logger.info(f"Cancelled {cancelled_count} tournaments")

        # Step 1: Create basic tournament in DB
        logger.info("Step 1: Creating basic tournament...")
        tournament_id = await create_basic_tournament(TournamentType.TEXT, config.psql_db, config)
        logger.info(f"Created basic tournament: {tournament_id}")

    except Exception as e:
        logger.error(f"Error starting tournament: {e}")
        raise
    finally:
        await config.psql_db.close()


if __name__ == "__main__":
    asyncio.run(main())
