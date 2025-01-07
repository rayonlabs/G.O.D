from typing import Dict
from typing import List
from typing import Optional
from uuid import UUID

from asyncpg.connection import Connection
from fiber.chain.models import Node

import validator.db.constants as cst
from core.constants import NETUID
from core.models.utility_models import TaskStatus

from validator.core.models import NetworkStats
from validator.core.models import RawTask

from validator.core.models import Task
from validator.db.database import PSQLDB
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def add_task(task: RawTask, psql_db: PSQLDB) -> RawTask:
    """Add a new task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {cst.TASKS_TABLE}
            ({cst.ACCOUNT_ID}, {cst.MODEL_ID}, {cst.DS_ID}, {cst.FIELD_SYSTEM},
            {cst.FIELD_INSTRUCTION}, {cst.FIELD_INPUT}, {cst.STATUS},
             {cst.HOURS_TO_COMPLETE}, {cst.FIELD_OUTPUT}, {cst.FORMAT},
             {cst.NO_INPUT_FORMAT}, {cst.IS_ORGANIC}, {cst.CREATED_AT})
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING *
        """
        task = await connection.fetchrow(
            query,
            task.account_id,
            task.model_id,
            task.ds_id,
            task.field_system,
            task.field_instruction,
            task.field_input,
            task.status,
            task.hours_to_complete,
            task.field_output,
            task.format,
            task.no_input_format,
            task.is_organic,
            task.created_at,
        )
    return RawTask(**task)


async def get_nodes_assigned_to_task(task_id: str, psql_db: PSQLDB) -> List[Node]:
    """Get all nodes assigned to a task for the current NETUID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            f"""
            SELECT nodes.* FROM {cst.NODES_TABLE} nodes
            JOIN {cst.TASK_NODES_TABLE} ON nodes.hotkey = task_nodes.hotkey
            WHERE task_nodes.task_id = $1
            AND nodes.netuid = $2
            AND task_nodes.netuid = $2
            """,
            task_id,
            NETUID,
        )
        return [Node(**dict(row)) for row in rows]


async def get_tasks_with_status(status: TaskStatus, psql_db: PSQLDB, include_not_ready_tasks=False) -> List[RawTask]:
    """Get all tasks with a specific status and delay_timestamp before current time if even_not_ready is False"""

    delay_timestamp_clause = (
        "" if include_not_ready_tasks else f"AND ({cst.NEXT_DELAY_AT} IS NULL OR {cst.NEXT_DELAY_AT} <= NOW())"
    )

    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {cst.TASKS_TABLE}
            WHERE {cst.STATUS} = $1
            {delay_timestamp_clause}
        """
        rows = await connection.fetch(query, status.value)
    logger.info(f"We got {len(rows)} tasks which are current in status {status.value}")
    return [RawTask(**dict(row)) for row in rows]


async def assign_node_to_task(task_id: str, node: Node, psql_db: PSQLDB) -> None:
    """Assign a node to a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {cst.TASK_NODES_TABLE} ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID})
            VALUES ($1, $2, $3)
        """
        await connection.execute(query, task_id, node.hotkey, NETUID)


async def update_task(updated_task: RawTask, psql_db: PSQLDB) -> RawTask:
    existing_task = await get_task(updated_task.task_id, psql_db)
    if not existing_task:
        raise ValueError(f"Task {updated_task.task_id} not found in the database?")

    updates = {}
    for field, value in updated_task.dict(exclude_unset=True, exclude={cst.ASSIGNED_MINERS, cst.UPDATED_AT}).items():
        if getattr(existing_task, field) != value:
            updates[field] = value

    async with await psql_db.connection() as connection:
        connection: Connection
        async with connection.transaction():
            if updates:
                set_clause = ", ".join([f"{column} = ${i+2}" for i, column in enumerate(updates.keys())])
                values = list(updates.values())
                query = f"""
                    UPDATE {cst.TASKS_TABLE}
                    SET {set_clause}, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
                    WHERE {cst.TASK_ID} = $1
                """
                await connection.execute(query, updated_task.task_id, *values)
            else:
                query = f"""
                    UPDATE {cst.TASKS_TABLE}
                    SET {cst.UPDATED_AT} = CURRENT_TIMESTAMP
                    WHERE {cst.TASK_ID} = $1
                """
                await connection.execute(query, updated_task.task_id)

            if updated_task.assigned_miners is not None:
                await connection.execute(
                    f"DELETE FROM {cst.TASK_NODES_TABLE} WHERE {cst.TASK_ID} = $1 AND {cst.NETUID} = $2",
                    updated_task.task_id,
                    NETUID,
                )
                if updated_task.assigned_miners:
                    query = f"""
                        INSERT INTO {cst.TASK_NODES_TABLE} ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID})
                        SELECT $1, nodes.{cst.HOTKEY}, $3
                        FROM {cst.NODES_TABLE} nodes
                        WHERE nodes.{cst.HOTKEY} = ANY($2)
                        AND nodes.{cst.NETUID} = $3
                    """
                    await connection.execute(query, updated_task.task_id, updated_task.assigned_miners, NETUID)

    return await get_task(updated_task.task_id, psql_db)


async def get_test_set_for_task(task_id: str, psql_db: PSQLDB):
    """Get test data for a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {cst.TEST_DATA} FROM {cst.TASKS_TABLE}
            WHERE {cst.TASK_ID} = $1
        """
        return await connection.fetchval(query, task_id)


async def get_synthetic_set_for_task(task_id: str, psql_db: PSQLDB):
    """Get synthetic data for a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {cst.SYNTHETIC_DATA} FROM {cst.TASKS_TABLE}
            WHERE {cst.TASK_ID} = $1
        """
        return await connection.fetchval(query, task_id)



async def get_current_task_stats(psql_db: PSQLDB) -> NetworkStats:

    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT

                COUNT(*) FILTER (WHERE {cst.STATUS} = $1) as number_of_jobs_training,
                COUNT(*) FILTER (WHERE {cst.STATUS} = $2) as number_of_jobs_preevaluation,
                COUNT(*) FILTER (WHERE {cst.STATUS} = $3) as number_of_jobs_evaluating,
                COUNT(*) FILTER (WHERE {cst.STATUS} = $4) as number_of_jobs_success,
                MIN(termination_at) FILTER (WHERE {cst.STATUS} = $1) as next_training_end
            FROM {cst.TASKS_TABLE}
        """
        row = await connection.fetchrow(
            query,
            TaskStatus.TRAINING.value,
            TaskStatus.PREEVALUATION.value,
            TaskStatus.EVALUATING.value,
            TaskStatus.SUCCESS.value,
        )
        return NetworkStats(
            number_of_jobs_training=row["number_of_jobs_training"],
            number_of_jobs_preevaluation=row["number_of_jobs_preevaluation"],
            number_of_jobs_evaluating=row["number_of_jobs_evaluating"],
            number_of_jobs_success=row["number_of_jobs_success"],
            next_training_end=row["next_training_end"],
        )


async def get_tasks_ready_to_evaluate(psql_db: PSQLDB) -> List[RawTask]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = """
            SELECT * FROM tasks t
            WHERE status IN ($1, $2)
            AND NOW() > termination_at
            AND EXISTS (
                SELECT 1 FROM task_nodes tn
                WHERE tn.task_id = t.task_id
                AND tn.netuid = $3
            )
            ORDER BY termination_at ASC
        """
        rows = await connection.fetch(query, TaskStatus.TRAINING.value, TaskStatus.PREEVALUATION.value, NETUID)
        return [RawTask(**dict(row)) for row in rows]


async def delete_task(task_id: UUID, psql_db: PSQLDB) -> None:
    """Delete a task and its associated node assignments"""
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            # First delete task_nodes entries for this netuid
            await connection.execute(
                f"""
                DELETE FROM {cst.TASK_NODES_TABLE}
                WHERE {cst.TASK_ID} = $1 AND {cst.NETUID} = $2
                """,
                task_id,
                NETUID,
            )

            # Then delete the task if it has no more node assignments
            await connection.execute(
                f"""
                DELETE FROM {cst.TASKS_TABLE}
                WHERE {cst.TASK_ID} = $1
                AND NOT EXISTS (
                    SELECT 1 FROM {cst.TASK_NODES_TABLE}
                    WHERE {cst.TASK_ID} = $1
                    AND {cst.NETUID} = $2
                )
                """,
                task_id,
                NETUID,
            )


async def get_miners_for_task(task_id: UUID, psql_db: PSQLDB) -> List[Node]:
    """Retrieve all miners assigned to a specific task."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT nodes.* FROM {cst.NODES_TABLE} nodes
            JOIN {cst.TASK_NODES_TABLE} task_nodes
            ON nodes.hotkey = task_nodes.hotkey AND nodes.netuid = task_nodes.netuid
            WHERE task_nodes.task_id = $1
        """
        rows = await connection.fetch(query, task_id)
        return [Node(**dict(row)) for row in rows]


async def get_task(task_id: UUID, psql_db: PSQLDB) -> Optional[RawTask]:
    """Get a task by ID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {cst.TASKS_TABLE} WHERE {cst.TASK_ID} = $1
        """
        row = await connection.fetchrow(query, task_id)
        if row:
            return RawTask(**dict(row))
        return None


async def get_winning_submissions_for_task(task_id: UUID, psql_db: PSQLDB) -> List[Dict]:
    """Retrieve the winning submission for a task based on the highest quality score in task_nodes."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT submissions.repo
            FROM {cst.SUBMISSIONS_TABLE} submissions
            JOIN {cst.TASK_NODES_TABLE} task_nodes
            ON submissions.task_id = task_nodes.task_id
            AND submissions.hotkey = task_nodes.hotkey
            AND submissions.netuid = task_nodes.netuid
            WHERE submissions.task_id = $1
            ORDER BY task_nodes.quality_score DESC
            LIMIT 1
        """
        rows = await connection.fetch(query, task_id)
        return [dict(row) for row in rows]


async def get_task_by_id(task_id: UUID, psql_db: PSQLDB) -> Task:
    """Get a task by ID along with its winning submissions"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            WITH victorious_repo AS (
                SELECT submissions.task_id, submissions.repo
                FROM {cst.SUBMISSIONS_TABLE} submissions
                JOIN {cst.TASK_NODES_TABLE} task_nodes
                ON submissions.task_id = task_nodes.task_id
                AND submissions.hotkey = task_nodes.hotkey
                AND submissions.netuid = task_nodes.netuid
                WHERE submissions.task_id = $1
                AND task_nodes.quality_score IS NOT NULL
                ORDER BY task_nodes.quality_score DESC
                LIMIT 1
            )
            SELECT
                tasks.*,
                victorious_repo.repo as trained_model_repository
            FROM {cst.TASKS_TABLE} tasks
            LEFT JOIN victorious_repo ON tasks.task_id = victorious_repo.task_id
            WHERE tasks.{cst.TASK_ID} = $1
        """
        row = await connection.fetchrow(query, task_id)
        if not row:
            return None

        return Task(**dict(row))


async def get_tasks(psql_db: PSQLDB, limit: int = 100, offset: int = 0) -> List[Task]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            WITH victorious_repo AS (
                SELECT submissions.{cst.TASK_ID}, submissions.{cst.REPO}
                FROM {cst.SUBMISSIONS_TABLE} submissions
                JOIN {cst.TASK_NODES_TABLE} task_nodes
                ON submissions.{cst.TASK_ID} = task_nodes.{cst.TASK_ID}
                AND submissions.{cst.HOTKEY} = task_nodes.{cst.HOTKEY}
                AND submissions.{cst.NETUID} = task_nodes.{cst.NETUID}
                AND task_nodes.{cst.QUALITY_SCORE} IS NOT NULL
                ORDER BY task_nodes.{cst.QUALITY_SCORE} DESC
                LIMIT 1
            )
            SELECT
                tasks.*,
                victorious_repo.{cst.REPO} as trained_model_repository
            FROM {cst.TASKS_TABLE} tasks
            LEFT JOIN victorious_repo ON tasks.{cst.TASK_ID} = victorious_repo.{cst.TASK_ID}
            ORDER BY tasks.{cst.CREATED_AT} DESC
            LIMIT $1 OFFSET $2
        """

        rows = await connection.fetch(query, limit, offset)
        return [Task(**dict(row)) for row in rows]


async def get_tasks_by_account_id(psql_db: PSQLDB, account_id: UUID, limit: int = 100, offset: int = 0) -> List[Task]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            WITH victorious_repo AS (
                SELECT
                    submissions.{cst.TASK_ID},
                    submissions.{cst.REPO},
                    ROW_NUMBER() OVER (
                        PARTITION BY submissions.{cst.TASK_ID}
                        ORDER BY task_nodes.{cst.QUALITY_SCORE} DESC
                    ) AS rn
                FROM {cst.SUBMISSIONS_TABLE} submissions
                JOIN {cst.TASK_NODES_TABLE} task_nodes
                    ON submissions.{cst.TASK_ID} = task_nodes.{cst.TASK_ID}
                   AND submissions.{cst.HOTKEY} = task_nodes.{cst.HOTKEY}
                   AND submissions.{cst.NETUID} = task_nodes.{cst.NETUID}
                WHERE task_nodes.{cst.QUALITY_SCORE} IS NOT NULL
            )
            SELECT
                tasks.*,
                victorious_repo.{cst.REPO} AS trained_model_repository
            FROM {cst.TASKS_TABLE} tasks
            LEFT JOIN victorious_repo
                ON tasks.{cst.TASK_ID} = victorious_repo.{cst.TASK_ID}
               AND victorious_repo.rn = 1
            WHERE tasks.{cst.ACCOUNT_ID} = $1
            ORDER BY tasks.{cst.CREATED_AT} DESC
            LIMIT $2 OFFSET $3
        """

        rows = await connection.fetch(query, account_id, limit, offset)
        return [Task(**dict(row)) for row in rows]


async def get_completed_organic_tasks(psql_db: PSQLDB, hours: int = 5) -> List[Task]:
    """Get completed organic tasks from the specified timeframe

    Args:
        psql_db: Database connection
        hours: Number of hours to look back (default: 5)
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            WITH victorious_repo AS (
                SELECT
                    submissions.{cst.TASK_ID},
                    submissions.{cst.REPO},
                    ROW_NUMBER() OVER (
                        PARTITION BY submissions.{cst.TASK_ID}
                        ORDER BY task_nodes.{cst.QUALITY_SCORE} DESC
                    ) AS rn
                FROM {cst.SUBMISSIONS_TABLE} submissions
                JOIN {cst.TASK_NODES_TABLE} task_nodes
                    ON submissions.{cst.TASK_ID} = task_nodes.{cst.TASK_ID}
                    AND submissions.{cst.HOTKEY} = task_nodes.{cst.HOTKEY}
                    AND submissions.{cst.NETUID} = task_nodes.{cst.NETUID}
                WHERE task_nodes.{cst.QUALITY_SCORE} IS NOT NULL
                ORDER BY task_nodes.{cst.QUALITY_SCORE} DESC
            )
            SELECT
                tasks.*,
                victorious_repo.{cst.REPO} as trained_model_repository
            FROM {cst.TASKS_TABLE} tasks
            LEFT JOIN victorious_repo
                ON tasks.{cst.TASK_ID} = victorious_repo.{cst.TASK_ID}
                AND victorious_repo.rn = 1
            WHERE tasks.{cst.STATUS} = $1
            AND tasks.{cst.IS_ORGANIC} = true
            AND tasks.{cst.TERMINATION_AT} >= NOW() - $2 * INTERVAL '1 hour'
            ORDER BY tasks.{cst.TERMINATION_AT} DESC
        """

        rows = await connection.fetch(query, TaskStatus.SUCCESS.value, hours)
        return [Task(**dict(row)) for row in rows]
