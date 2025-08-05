#!/usr/bin/env python3

import asyncio
import random
from datetime import datetime
from uuid import uuid4

import validator.core.constants as cst
from core.models.tournament_models import TournamentRoundData
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.db.database import PSQLDB
from validator.db.sql.tasks import add_task
from validator.db.sql.tasks import get_task
from validator.db.sql.tournaments import get_tournament_tasks
from validator.utils.logging import get_logger


logger = get_logger(__name__)


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
            await _copy_task_to_general(tournament_task.task_id, psql_db)
            logger.info(f"Successfully synced task {tournament_task.task_id} (task {i + 1} of {len(boss_round_tasks)})")
        except Exception as e:
            logger.error(f"Failed to sync task {tournament_task.task_id}: {e}")
            raise  # Re-raise to prevent tournament from completing with incomplete sync


async def _schedule_task_sync(tournament_task_id: str, delay_hours: int, psql_db: PSQLDB, config: Config):
    delay_seconds = delay_hours * 3600
    logger.info(f"Waiting {delay_hours} hours before syncing task {tournament_task_id}")

    await asyncio.sleep(delay_seconds)

    try:
        await _copy_task_to_general(tournament_task_id, psql_db)
    except Exception as e:
        logger.error(f"Failed to sync task {tournament_task_id}: {e}")


async def _copy_task_to_general(
    tournament_task_id: str,
    psql_db: PSQLDB,
):
    original_task = await get_task(tournament_task_id, psql_db)
    if not original_task:
        logger.error(f"Could not find original task {tournament_task_id}")
        return

    logger.info(f"Copying task {tournament_task_id} to general side")

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

    logger.info(f"Successfully synced task {tournament_task_id} -> {general_task.task_id}")


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
