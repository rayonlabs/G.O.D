#!/usr/bin/env python3

from datetime import datetime
from uuid import uuid4

import validator.core.constants as cst
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentTask
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.core.config import Config
from validator.core.models import AnyTypeTask
from validator.db.database import PSQLDB
from validator.db.sql import grpo as grpo_sql
from validator.db.sql.tasks import add_task
from validator.db.sql.tasks import get_task
from validator.db.sql.tournaments import add_tournament_tasks
from validator.db.sql.tournaments import get_tournament_tasks
from validator.utils.logging import get_logger
from validator.utils.minio import async_minio_client


logger = get_logger(__name__)


async def _postprocess_task_for_tournament(tournament_task: AnyTypeTask, psql_db: PSQLDB) -> AnyTypeTask:
    """
    For GRPO tasks: filter out non-manual reward functions.
    If no manual reward functions remain, add one fallback from the database.
    """
    tournament_task.is_organic = False
    tournament_task.task_id = uuid4()
    tournament_task.status = TaskStatus.LOOKING_FOR_NODES
    tournament_task.account_id = cst.NULL_ACCOUNT_ID
    tournament_task.times_delayed = 0
    tournament_task.assigned_miners = None
    tournament_task.miner_scores = None
    tournament_task.training_repo_backup = None
    tournament_task.result_model_name = None
    tournament_task.created_at = datetime.utcnow()
    tournament_task.next_delay_at = None
    tournament_task.updated_at = None
    tournament_task.started_at = None
    tournament_task.termination_at = None
    tournament_task.completed_at = None
    tournament_task.n_eval_attempts = 0

    # Resign S3 URLs to ensure they're not expired
    if tournament_task.training_data:
        logger.info("Regenerating presigned URL for training_data")
        tournament_task.training_data = await async_minio_client.get_new_presigned_url(tournament_task.training_data)

    if tournament_task.test_data:
        logger.info("Regenerating presigned URL for test_data")
        tournament_task.test_data = await async_minio_client.get_new_presigned_url(tournament_task.test_data)

    if tournament_task.task_type == TaskType.GRPOTASK:
        manual_reward_functions = [rf for rf in tournament_task.reward_functions if rf.is_manual]
        if not manual_reward_functions:
            logger.warning(f"GRPO task {tournament_task.task_id} has no manual reward functions after filtering, adding fallback")
            # Add a fallback reward function from database
            fallback_functions = await grpo_sql.get_generic_reward_functions_from_db(psql_db, 1)
            tournament_task.reward_functions = fallback_functions
            logger.info(f"Added fallback reward function for GRPO task {tournament_task.task_id}")
        else:
            original_count = len(tournament_task.reward_functions)
            tournament_task.reward_functions = manual_reward_functions
            if len(manual_reward_functions) < original_count:
                logger.info(
                    f"Filtered GRPO task {tournament_task.task_id} reward functions: {original_count} -> {len(manual_reward_functions)} (manual only)"
                )

    return tournament_task


async def sync_boss_round_tasks_to_general(
    tournament_id: str, completed_round: TournamentRoundData, psql_db: PSQLDB, config: Config
):
    logger.info(f"Starting boss round task sync for tournament {tournament_id}, round {completed_round.round_id}")

    boss_round_tasks = await get_tournament_tasks(completed_round.round_id, psql_db)
    if not boss_round_tasks:
        logger.warning(f"No tasks found for boss round {completed_round.round_id}")
        return

    logger.info(f"Found {len(boss_round_tasks)} boss round tasks to sync")

    # Sync all tasks immediately instead of with random delays
    for i, tournament_task in enumerate(boss_round_tasks):
        try:
            await copy_tournament_task_into_general_miner_pool(tournament_task.task_id, psql_db)
            logger.info(f"Successfully synced task {tournament_task.task_id} (task {i + 1} of {len(boss_round_tasks)})")
        except Exception as e:
            logger.error(f"Failed to sync task {tournament_task.task_id}: {e}")
            raise  # Re-raise to prevent tournament from completing with incomplete sync


async def copy_historical_task_into_boss_round_tournament(
    historical_task_id: str,
    tournament_id: str,
    round_id: str,
    pair_id: str,
    psql_db: PSQLDB,
) -> AnyTypeTask | None:
    original_task = await get_task(historical_task_id, psql_db)
    if not original_task:
        logger.error(f"Could not find historical task {historical_task_id}")
        return

    logger.info(f"Copying historical task {historical_task_id} for boss round tournament")

    tournament_task = original_task.model_copy()
    tournament_task = await _postprocess_task_for_tournament(tournament_task, psql_db)

    await add_task(tournament_task, psql_db)

    tournament_task_entry = TournamentTask(
        tournament_id=tournament_id,
        round_id=round_id,
        task_id=tournament_task.task_id,
        group_id=None,
        pair_id=pair_id,
    )
    await add_tournament_tasks([tournament_task_entry], psql_db)

    await _record_task_sync_link(tournament_task.task_id, historical_task_id, psql_db)

    logger.info(f"Successfully copied historical task {historical_task_id} -> tournament task {tournament_task.task_id}")
    return tournament_task


async def copy_tournament_task_into_general_miner_pool(
    tournament_task_id: str,
    psql_db: PSQLDB,
):
    original_task = await get_task(tournament_task_id, psql_db)
    if not original_task:
        logger.error(f"Could not find tournament task {tournament_task_id}")
        return

    logger.info(f"Copying tournament task {tournament_task_id} to general miner pool")

    general_task = original_task.model_copy()
    general_task.is_organic = False
    general_task.task_id = uuid4()
    general_task.status = TaskStatus.LOOKING_FOR_NODES
    general_task.account_id = cst.NULL_ACCOUNT_ID
    general_task.times_delayed = 0
    general_task.assigned_miners = None
    general_task.miner_scores = None
    general_task.training_repo_backup = None
    general_task.result_model_name = None
    general_task.created_at = datetime.utcnow()
    general_task.next_delay_at = None
    general_task.updated_at = None
    general_task.started_at = None
    general_task.termination_at = None
    general_task.completed_at = None
    general_task.n_eval_attempts = 0

    await add_task(general_task, psql_db)
    await _record_task_sync_link(tournament_task_id, general_task.task_id, psql_db)

    logger.info(f"Successfully copied tournament task {tournament_task_id} -> general task {general_task.task_id}")
    return general_task


async def _record_task_sync_link(tournament_task_id: str, general_task_id: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        query = """
            INSERT INTO boss_round_synced_tasks
            (tournament_task_id, general_task_id)
            VALUES ($1, $2)
            ON CONFLICT (tournament_task_id, general_task_id) DO NOTHING
        """
        await connection.execute(query, tournament_task_id, general_task_id)
        logger.info(f"Recorded sync link: {tournament_task_id} -> {general_task_id}")


async def get_synced_task_id(tournament_task_id: str, psql_db: PSQLDB) -> str | None:
    # returns the general task id if the task is synced, otherwise None
    async with await psql_db.connection() as connection:
        query = """
            SELECT general_task_id FROM boss_round_synced_tasks WHERE tournament_task_id = $1
        """
        general_task_id = await connection.fetchval(query, tournament_task_id)
        return general_task_id


async def get_synced_task_ids(tournament_task_ids: list[str], psql_db: PSQLDB) -> list[str]:
    async with await psql_db.connection() as connection:
        query = """
            SELECT general_task_id FROM boss_round_synced_tasks WHERE tournament_task_id = ANY($1)
        """
        general_task_ids = await connection.fetch(query, tournament_task_ids)
        return [general_task_id for (general_task_id,) in general_task_ids]
