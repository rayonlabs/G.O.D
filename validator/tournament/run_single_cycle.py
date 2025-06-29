#!/usr/bin/env python3

import asyncio
import random

from core.models.tournament_models import RoundStatus
from core.models.tournament_models import TournamentStatus
from core.models.tournament_models import TournamentType
from core.models.utility_models import TaskStatus
from validator.core.config import load_config
from validator.db.database import PSQLDB
from validator.db.sql.submissions_and_scoring import set_task_node_quality_score
from validator.db.sql.tasks import update_task_status
from validator.db.sql.tournaments import cancel_all_active_tournaments
from validator.db.sql.tournaments import get_active_tournaments
from validator.db.sql.tournaments import get_miners_for_tournament
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import get_tournament_tasks
from validator.db.sql.tournaments import update_round_status
from validator.tournament.tournament_manager import advance_tournament
from validator.tournament.tournament_manager import create_basic_tournament
from validator.tournament.tournament_manager import create_first_round_for_active_tournament
from validator.tournament.tournament_manager import populate_tournament_participants
from validator.tournament.tournament_manager import print_tournament_summary
from validator.tournament.tournament_manager import process_pending_tournaments
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def mock_task_evaluation(task_id: str, psql_db: PSQLDB):
    """Mock a task completion by updating its status and adding mock scores."""

    # Get nodes assigned to this task from the database
    assigned_nodes = await get_miners_for_tournament(task_id, psql_db)

    if assigned_nodes:
        for node in assigned_nodes:
            # Generate random score (lower is better for loss)
            mock_score = random.uniform(0.1, 2.0)
            mock_test_loss = random.uniform(0.1, 2.0)
            mock_synth_loss = random.uniform(0.1, 2.0)

            await set_task_node_quality_score(
                task_id, node.hotkey, mock_score, mock_test_loss, mock_synth_loss, psql_db, "Mock evaluation"
            )
            logger.info(f"Added mock score {mock_score:.3f} for {node.hotkey} on task {task_id}")
    else:
        logger.warning(f"No assigned miners found for task {task_id}")

    # Update task status to completed
    logger.info(f"Updating task {task_id} status to completed")
    await update_task_status(task_id, TaskStatus.SUCCESS, psql_db)


async def mock_training_and_submission(round_id: str, psql_db: PSQLDB):
    """Mock the training and submission process for a specific round."""
    logger.info(f"Mocking training and submission for round {round_id}")

    tasks = await get_tournament_tasks(round_id, psql_db)

    for task in tasks:
        await mock_task_evaluation(task.task_id, psql_db)
        logger.info(f"Mocked completion for task {task.task_id}")


async def run_tournament_to_completion(tournament_id: str, config, psql_db: PSQLDB):
    logger.info(f"Running tournament {tournament_id} to completion...")

    while True:
        tournaments = await get_active_tournaments(psql_db)
        tournament = next((t for t in tournaments if t.tournament_id == tournament_id), None)

        if not tournament:
            completed_tournament = await get_tournament(tournament_id, psql_db)
            if completed_tournament and completed_tournament.status == TournamentStatus.COMPLETED:
                logger.info(f"Tournament {tournament_id} completed successfully!")
                break
            else:
                logger.error(f"Tournament {tournament_id} not found")
                break

        if tournament.status == TournamentStatus.COMPLETED:
            logger.info(f"Tournament {tournament_id} completed!")
            break

        rounds = await get_tournament_rounds(tournament_id, psql_db)
        if not rounds:
            logger.info(f"No rounds found for tournament {tournament_id}, creating first round...")
            round_created = await create_first_round_for_active_tournament(tournament_id, config, psql_db)
            if not round_created:
                logger.error(f"Failed to create first round for tournament {tournament_id}")
                break
            rounds = await get_tournament_rounds(tournament_id, psql_db)

        current_round = rounds[-1]
        logger.info(f"Processing round {current_round.round_id} (status: {current_round.status})")

        if current_round.status in [RoundStatus.PENDING, RoundStatus.ACTIVE]:
            await mock_training_and_submission(current_round.round_id, psql_db)

            await update_round_status(current_round.round_id, RoundStatus.COMPLETED, psql_db)
            logger.info(f"Completed round {current_round.round_id}")

        elif current_round.status == RoundStatus.COMPLETED:
            await advance_tournament(tournament, current_round, config, psql_db)


async def main():
    """Main function to run the tournament to test"""
    logger.info("Starting tournament runner...")

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
        tournament_id = await create_basic_tournament(TournamentType.TEXT, config.psql_db)
        logger.info(f"Created basic tournament: {tournament_id}")

        # Step 2: Populate participants by fetching training repos (real miners, but submissions are mocked)
        logger.info("Step 2: Populating tournament participants...")
        num_participants = await populate_tournament_participants(tournament_id, config, config.psql_db, max_nodes=18)
        logger.info(f"Populated {num_participants} participants")

        if num_participants == 0:
            logger.error("No participants found. Cannot proceed with tournament.")
            return

        # Step 3: Process pending tournaments (this will activate our tournament)
        logger.info("Step 3: Processing pending tournaments...")
        activated_tournaments = await process_pending_tournaments(config, config.psql_db)
        logger.info(f"Activated tournaments: {activated_tournaments}")

        # Step 4: Run tournament to completion
        logger.info("Step 4: Running tournament to completion...")
        await run_tournament_to_completion(tournament_id, config, config.psql_db)

        await print_tournament_summary(tournament_id, config.psql_db)

        logger.info("Tournament completed successfully!")

    except Exception as e:
        logger.error(f"Error running tournament: {e}")
        raise
    finally:
        # Clean up
        await config.psql_db.close()


if __name__ == "__main__":
    asyncio.run(main())
