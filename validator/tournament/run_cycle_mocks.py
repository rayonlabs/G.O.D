#!/usr/bin/env python3

import asyncio
import random

from core.models.tournament_models import RoundStatus
from core.models.tournament_models import RoundType
from core.models.tournament_models import TournamentStatus
from core.models.tournament_models import TournamentType
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.config import load_config
from validator.db.database import PSQLDB
from validator.db.sql.submissions_and_scoring import set_task_node_quality_score
from validator.db.sql.tasks import get_nodes_assigned_to_task
from validator.db.sql.tasks import get_tasks_with_status
from validator.db.sql.tasks import update_task_status
from validator.db.sql.tournaments import cancel_all_active_tournaments
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import get_tournament_group_members
from validator.db.sql.tournaments import get_tournament_pairs
from validator.db.sql.tournaments import get_tournament_rounds
from validator.tournament.tournament_manager import create_basic_tournament
from validator.tournament.tournament_manager import get_round_winners
from validator.tournament.utils import draw_group_stage_table
from validator.tournament.utils import draw_knockout_bracket
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def print_tournament_summary(tournament_id: str, psql_db: PSQLDB):
    """Print a visual summary of the tournament progression and winners.
    Helps visualize the tournament. AI generated, we can remove once we ship"""

    logger.info("\n" + "=" * 60)
    logger.info("TOURNAMENT SUMMARY")
    logger.info("=" * 60)

    tournament = await get_tournament(tournament_id, psql_db)
    if not tournament:
        logger.error(f"Tournament {tournament_id} not found")
        return

    logger.info(f"Tournament ID: {tournament_id}")
    logger.info(f"Type: {tournament.tournament_type.value}")
    logger.info(f"Status: {tournament.status.value}")
    logger.info("")

    rounds = await get_tournament_rounds(tournament_id, psql_db)
    rounds.sort(key=lambda r: r.round_number)

    rounds_data = []
    winners_by_round = {}

    logger.info("TOURNAMENT PROGRESSION:")
    logger.info("-" * 40)

    for i, round_data in enumerate(rounds):
        round_header = f"Round {round_data.round_number}: {round_data.round_type.value}"
        if round_data.is_final_round:
            round_header += " 🔥 BOSS ROUND 🔥"

        logger.info(round_header)
        logger.info(f"  Round ID: {round_data.round_id}")
        logger.info(f"  Status: {round_data.status.value}")
        logger.info(f"  Final Round: {round_data.is_final_round}")

        round_participants = []

        if round_data.round_type == RoundType.GROUP:
            groups = await get_tournament_group_members(round_data.round_id, psql_db)
            if groups:
                # Group by group_id
                group_dict = {}
                for member in groups:
                    if member.group_id not in group_dict:
                        group_dict[member.group_id] = []
                    group_dict[member.group_id].append(member.hotkey)

                logger.info("  Groups:")
                for group_id, hotkeys in group_dict.items():
                    logger.info(f"    {group_id}: {', '.join(hotkeys)}")
                    round_participants.extend(hotkeys)
        else:
            pairs = await get_tournament_pairs(round_data.round_id, psql_db)
            if pairs:
                logger.info("  Pairs:")
                for pair in pairs:
                    logger.info(f"    {pair.hotkey1} vs {pair.hotkey2}")
                    round_participants.extend([pair.hotkey1, pair.hotkey2])

        if round_data.status == RoundStatus.COMPLETED:
            winners = await get_round_winners(round_data, psql_db)
            if winners:
                logger.info(f"  Winners: {', '.join(winners)}")
                winners_by_round[i] = winners
            else:
                logger.info("  Winners: None")
                winners_by_round[i] = []

        rounds_data.append(
            {
                "participants": round_participants,
                "type": round_data.round_type,
                "status": round_data.status,
                "round_id": round_data.round_id,
                "is_final_round": round_data.is_final_round,
            }
        )

        logger.info("")

    await draw_group_stage_table(rounds_data, winners_by_round, psql_db)

    draw_knockout_bracket(rounds_data, winners_by_round)

    if tournament.status == TournamentStatus.COMPLETED:
        final_round = rounds[-1] if rounds else None
        if final_round and final_round.status == RoundStatus.COMPLETED:
            final_winners = await get_round_winners(final_round, psql_db)
            if final_winners:
                logger.info("🏆 TOURNAMENT CHAMPION 🏆")
                logger.info(f"Winner: {final_winners[0]}")
                logger.info("=" * 60)


async def mock_task_evaluation(task_id: str, config: Config):
    """Mock a task completion by updating its status and adding mock scores."""

    assigned_nodes = await get_nodes_assigned_to_task(task_id, config.psql_db)

    if assigned_nodes:
        for node in assigned_nodes:
            mock_score = random.uniform(0.1, 3.0)
            mock_test_loss = random.uniform(0.1, 2.0)
            mock_synth_loss = random.uniform(0.1, 2.0)

            await set_task_node_quality_score(
                task_id, node.hotkey, mock_score, mock_test_loss, mock_synth_loss, config.psql_db, "Mock evaluation"
            )
            logger.info(f"Added mock score {mock_score:.3f} for {node.hotkey} on task {task_id}")
    else:
        logger.warning(f"No assigned miners found for task {task_id}")

    logger.info(f"Updating task {task_id} status to completed")
    await update_task_status(task_id, TaskStatus.SUCCESS, config.psql_db)


async def mock_training_and_submission(config: Config):
    logger.info("Mocking training and submission for all tournament tasks...")

    tasks = await get_tasks_with_status(TaskStatus.READY, config.psql_db, tournament_filter="only")

    for task in tasks:
        await mock_task_evaluation(task.task_id, config)
        logger.info(f"Mocked completion for task {task.task_id}")


async def mock_tournament_task_prep(config: Config):
    pending_tasks = await get_tasks_with_status(TaskStatus.PENDING, config.psql_db, tournament_filter="only")

    for task in pending_tasks:
        await update_task_status(task.task_id, TaskStatus.LOOKING_FOR_NODES, config.psql_db)
        logger.info(f"Set task {task.task_id} status to looking for nodes")


async def main():
    """Main function to run the tournament to test"""
    logger.info("Starting tournament mocks...")

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

        # Step 2: Run mocks until tournament is completed
        logger.info("Step 2: Running mocks until tournament is completed...")
        while True:
            await mock_tournament_task_prep(config)
            await mock_training_and_submission(config)

            tournament = await get_tournament(tournament_id, config.psql_db)
            if tournament.status == TournamentStatus.COMPLETED:
                break

            logger.info("Sleeping for 15 seconds")
            await asyncio.sleep(15)  # 15 seconds

        logger.info("Tournament completed successfully!")

        # Step 3: Print tournament summary
        logger.info("Step 3: Printing tournament summary...")
        await print_tournament_summary(tournament_id, config.psql_db)

    except Exception as e:
        logger.error(f"Error running tournament mocks: {e}")
        raise
    finally:
        await config.psql_db.close()


if __name__ == "__main__":
    asyncio.run(main())
