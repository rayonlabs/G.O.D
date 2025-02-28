from typing import Dict
from typing import List
from typing import Optional
from uuid import UUID

from asyncpg.connection import Connection
from fiber.chain.models import Node

import validator.db.constants as cst
from core.constants import NETUID
from core.models.utility_models import ImageTextPair
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.core.models import ImageRawTask
from validator.core.models import ImageTask
from validator.core.models import NetworkStats
from validator.core.models import RawTask
from validator.core.models import Task
from validator.core.models import TextRawTask
from validator.core.models import TextTask
from validator.db.database import PSQLDB
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def add_task(task: TextRawTask | ImageRawTask, psql_db: PSQLDB) -> TextRawTask | ImageRawTask:
    """Add a new task"""
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            query_tasks = f"""
                INSERT INTO {cst.TASKS_TABLE}
                ({cst.ACCOUNT_ID},
                {cst.MODEL_ID},
                {cst.DS},
                {cst.STATUS},
                {cst.IS_ORGANIC},
                {cst.HOURS_TO_COMPLETE},
                {cst.TEST_DATA},
                {cst.TRAINING_DATA},
                {cst.CREATED_AT},
                {cst.TASK_TYPE},
                {cst.RESULT_MODEL_NAME})
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING *
            """
            task_record = await connection.fetchrow(
                query_tasks,
                task.account_id,
                task.model_id,
                task.ds,
                task.status,
                task.is_organic,
                task.hours_to_complete,
                task.test_data,
                task.training_data,
                task.created_at,
                task.task_type.value,
                task.result_model_name,
            )

            if isinstance(task, TextRawTask):
                query_text_tasks = f"""
                    INSERT INTO {cst.TEXT_TASKS_TABLE}
                    ({cst.TASK_ID}, {cst.FIELD_SYSTEM}, {cst.FIELD_INSTRUCTION},
                    {cst.FIELD_INPUT}, {cst.FIELD_OUTPUT}, {cst.FORMAT},
                    {cst.NO_INPUT_FORMAT}, {cst.SYNTHETIC_DATA}, {cst.FILE_FORMAT})
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """
                await connection.execute(
                    query_text_tasks,
                    task_record[cst.TASK_ID],
                    task.field_system,
                    task.field_instruction,
                    task.field_input,
                    task.field_output,
                    task.format,
                    task.no_input_format,
                    task.synthetic_data,
                    task.file_format,
                )
            elif isinstance(task, ImageRawTask):
                query_image_tasks = f"""
                    INSERT INTO {cst.IMAGE_TASKS_TABLE}
                    ({cst.TASK_ID})
                    VALUES ($1)
                """
                await connection.execute(query_image_tasks, task_record[cst.TASK_ID])

                if task.image_text_pairs:
                    query_pairs = f"""
                        INSERT INTO {cst.IMAGE_TEXT_PAIRS_TABLE}
                        ({cst.TASK_ID}, {cst.IMAGE_URL}, {cst.TEXT_URL})
                        VALUES ($1, $2, $3)
                    """
                    for pair in task.image_text_pairs:
                        await connection.execute(query_pairs, task_record[cst.TASK_ID], pair.image_url, pair.text_url)

            task.task_id = task_record[cst.TASK_ID]
            return task


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


async def get_tasks_with_status(
    status: TaskStatus, psql_db: PSQLDB, include_not_ready_tasks=False
) -> List[TextRawTask | ImageRawTask]:
    delay_timestamp_clause = (
        "" if include_not_ready_tasks else f"AND ({cst.NEXT_DELAY_AT} IS NULL OR {cst.NEXT_DELAY_AT} <= NOW())"
    )

    async with await psql_db.connection() as connection:
        connection: Connection
        base_query = f"""
            SELECT * FROM {cst.TASKS_TABLE}
            WHERE {cst.STATUS} = $1
            {delay_timestamp_clause}
        """
        base_rows = await connection.fetch(base_query, status.value)

        tasks = []
        for row in base_rows:
            task_type = row[cst.TASK_TYPE]
            if task_type == TaskType.TEXTTASK.value:
                specific_query = f"""
                    SELECT t.*, tt.field_system,
                           tt.field_instruction, tt.field_input, tt.field_output,
                           tt.format, tt.no_input_format, tt.synthetic_data
                    FROM {cst.TASKS_TABLE} t
                    LEFT JOIN {cst.TEXT_TASKS_TABLE} tt ON t.{cst.TASK_ID} = tt.{cst.TASK_ID}
                    WHERE t.{cst.TASK_ID} = $1
                """
            elif task_type == TaskType.IMAGETASK.value:
                specific_query = f"""
                    SELECT t.*
                    FROM {cst.TASKS_TABLE} t
                    LEFT JOIN {cst.IMAGE_TASKS_TABLE} it ON t.{cst.TASK_ID} = it.{cst.TASK_ID}
                    WHERE t.{cst.TASK_ID} = $1
                """
            else:
                logger.warning(f"Unknown task type {task_type} for task_id {row[cst.TASK_ID]}")
                continue

            specific_row = await connection.fetchrow(specific_query, row[cst.TASK_ID])
            if specific_row:
                task_data = dict(specific_row)
                if task_type == TaskType.TEXTTASK.value:
                    tasks.append(TextRawTask(**task_data))
                elif task_type == TaskType.IMAGETASK.value:
                    image_text_pairs = await get_image_text_pairs(row[cst.TASK_ID], psql_db)
                    tasks.append(ImageRawTask(**task_data, image_text_pairs=image_text_pairs))

        logger.info(f"Retrieved {len(tasks)} tasks with status {status.value}")
        return tasks


async def assign_node_to_task(task_id: str, node: Node, psql_db: PSQLDB) -> None:
    """Assign a node to a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {cst.TASK_NODES_TABLE}
            ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID})
            VALUES ($1, $2, $3)
        """
        await connection.execute(query, task_id, node.hotkey, NETUID)


async def set_expected_repo_name(task_id: str, node: Node, psql_db: PSQLDB, expected_repo_name: str) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            UPDATE {cst.TASK_NODES_TABLE}
            SET {cst.EXPECTED_REPO_NAME} = $1
            WHERE {cst.TASK_ID} = $2
            AND {cst.HOTKEY} = $3
            AND {cst.NETUID} = $4
        """
        await connection.execute(query, expected_repo_name, task_id, node.hotkey, NETUID)


async def get_table_fields(table_name: str, connection: Connection) -> set[str]:
    """Get all column names for a given table"""
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = $1
    """
    rows = await connection.fetch(query, table_name)
    return {row['column_name'] for row in rows}


async def update_task(updated_task: TextRawTask | ImageRawTask, psql_db: PSQLDB) -> TextRawTask | ImageRawTask:
    existing_task = await get_task(updated_task.task_id, psql_db)

    if not existing_task:
        raise ValueError(f"Task {updated_task.task_id} not found in the database?")

    existing_task_dict = existing_task.model_dump()
    updates = {}
    for field, value in updated_task.dict(exclude_unset=True, exclude={cst.ASSIGNED_MINERS, cst.UPDATED_AT}).items():
        if existing_task_dict.get(field, None) != value:
            updates[field] = value

    async with await psql_db.connection() as connection:
        connection: Connection
        async with connection.transaction():
            base_task_fields = await get_table_fields(cst.TASKS_TABLE, connection)
            text_fields = await get_table_fields(cst.TEXT_TASKS_TABLE, connection)
            text_specific_fields = [f for f in text_fields if f != cst.TASK_ID]

            base_updates = {k: v for k, v in updates.items() if k in base_task_fields}
            if base_updates:
                set_clause = ", ".join([f"{column} = ${i + 2}" for i, column in enumerate(base_updates.keys())])
                values = list(base_updates.values())
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

            if updated_task.task_type == TaskType.TEXTTASK:
                specific_updates = {k: v for k, v in updates.items() if k in text_specific_fields}
                if specific_updates:
                    specific_clause = ", ".join([f"{column} = ${i + 2}" for i, column in enumerate(specific_updates.keys())])
                    specific_values = list(specific_updates.values())
                    query = f"""
                        UPDATE {cst.TEXT_TASKS_TABLE}
                        SET {specific_clause}
                        WHERE {cst.TASK_ID} = $1
                    """
                    await connection.execute(query, updated_task.task_id, *specific_values)
            elif updated_task.task_type == TaskType.IMAGETASK:
                if "image_text_pairs" in updates:
                    await delete_image_text_pairs(updated_task.task_id, psql_db)
                    pairs = [ImageTextPair(**pair) for pair in updates["image_text_pairs"]]
                    await add_image_text_pairs(updated_task.task_id, pairs, psql_db)

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


async def get_task(task_id: UUID, psql_db: PSQLDB) -> Optional[TextRawTask | ImageRawTask]:
    """Get a full task by ID"""
    async with await psql_db.connection() as connection:
        connection: Connection

        base_query = f"""
            SELECT * FROM {cst.TASKS_TABLE} WHERE {cst.TASK_ID} = $1
        """
        base_row = await connection.fetchrow(base_query, task_id)

        if not base_row:
            return None

        task_type = base_row[cst.TASK_TYPE]

        if task_type == TaskType.TEXTTASK.value:
            specific_query = f"""
                SELECT t.*, tt.field_system,
                       tt.field_instruction, tt.field_input, tt.field_output,
                       tt.format, tt.no_input_format, tt.synthetic_data
                FROM {cst.TASKS_TABLE} t
                LEFT JOIN {cst.TEXT_TASKS_TABLE} tt ON t.{cst.TASK_ID} = tt.{cst.TASK_ID}
                WHERE t.{cst.TASK_ID} = $1
            """
        elif task_type == TaskType.IMAGETASK.value:
            specific_query = f"""
                SELECT t.*
                FROM {cst.TASKS_TABLE} t
                LEFT JOIN {cst.IMAGE_TASKS_TABLE} it ON t.{cst.TASK_ID} = it.{cst.TASK_ID}
                WHERE t.{cst.TASK_ID} = $1
            """
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        full_row = await connection.fetchrow(specific_query, task_id)

        if not full_row:
            return None

        full_task_data = dict(full_row)
        if task_type == TaskType.TEXTTASK.value:
            return TextRawTask(**full_task_data)
        elif task_type == TaskType.IMAGETASK.value:
            image_text_pairs = await get_image_text_pairs(task_id, psql_db)
            return ImageRawTask(**full_task_data, image_text_pairs=image_text_pairs)


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


async def get_task_by_id(task_id: UUID, psql_db: PSQLDB) -> TextTask | ImageTask:
    """Get a task by ID along with its winning submissions and task-specific details"""
    async with await psql_db.connection() as connection:
        connection: Connection

        base_query = f"""
            SELECT * FROM {cst.TASKS_TABLE} WHERE {cst.TASK_ID} = $1
        """
        base_row = await connection.fetchrow(base_query, task_id)

        if not base_row:
            return None

        task_type = base_row[cst.TASK_TYPE]

        if task_type == TaskType.TEXTTASK.value:
            specific_query = f"""
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
                    tt.field_system,
                    tt.field_instruction, tt.field_input, tt.field_output,
                    tt.format, tt.no_input_format, tt.synthetic_data,
                    COALESCE(tasks.training_repo_backup, victorious_repo.repo) as trained_model_repository
                FROM {cst.TASKS_TABLE} tasks
                LEFT JOIN {cst.TEXT_TASKS_TABLE} tt ON tasks.{cst.TASK_ID} = tt.{cst.TASK_ID}
                LEFT JOIN victorious_repo ON tasks.task_id = victorious_repo.task_id
                WHERE tasks.{cst.TASK_ID} = $1
            """
        elif task_type == TaskType.IMAGETASK.value:
            specific_query = f"""
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
                LEFT JOIN {cst.IMAGE_TASKS_TABLE} it ON tasks.{cst.TASK_ID} = it.{cst.TASK_ID}
                LEFT JOIN victorious_repo ON tasks.task_id = victorious_repo.task_id
                WHERE tasks.{cst.TASK_ID} = $1
            """
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        row = await connection.fetchrow(specific_query, task_id)
        if not row:
            return None

        full_task_data = dict(row)
        if task_type == TaskType.TEXTTASK.value:
            return TextTask(**full_task_data)
        elif task_type == TaskType.IMAGETASK.value:
            image_text_pairs = await get_image_text_pairs(task_id, psql_db)
            return ImageTask(**full_task_data, image_text_pairs=image_text_pairs)


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
                COALESCE(tasks.training_repo_backup, victorious_repo.{cst.REPO}) as trained_model_repository
            FROM {cst.TASKS_TABLE} tasks
            LEFT JOIN victorious_repo ON tasks.{cst.TASK_ID} = victorious_repo.{cst.TASK_ID}
            ORDER BY tasks.{cst.CREATED_AT} DESC
            LIMIT $1 OFFSET $2
        """

        rows = await connection.fetch(query, limit, offset)
        return [Task(**dict(row)) for row in rows]


async def get_tasks_by_account_id(
    psql_db: PSQLDB, account_id: UUID, limit: int = 100, offset: int = 0
) -> List[TextTask | ImageTask]:
    async with await psql_db.connection() as connection:
        connection: Connection
        base_query = f"""
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
                COALESCE(tasks.training_repo_backup, victorious_repo.{cst.REPO}) AS trained_model_repository
            FROM {cst.TASKS_TABLE} tasks
            LEFT JOIN victorious_repo
                ON tasks.{cst.TASK_ID} = victorious_repo.{cst.TASK_ID}
               AND victorious_repo.rn = 1
            WHERE tasks.{cst.ACCOUNT_ID} = $1
            ORDER BY tasks.{cst.CREATED_AT} DESC
            LIMIT $2 OFFSET $3
        """

        rows = await connection.fetch(base_query, account_id, limit, offset)
        tasks = []

        for row in rows:
            task_data = dict(row)
            task_type = task_data[cst.TASK_TYPE]

            if task_type == TaskType.TEXTTASK.value:
                text_query = f"""
                    SELECT field_system, field_instruction, field_input, field_output,
                           format, no_input_format, synthetic_data
                    FROM {cst.TEXT_TASKS_TABLE}
                    WHERE {cst.TASK_ID} = $1
                """
                text_row = await connection.fetchrow(text_query, task_data[cst.TASK_ID])
                if text_row:
                    task_data.update(dict(text_row))
                tasks.append(TextTask(**task_data))

            elif task_type == TaskType.IMAGETASK.value:
                image_text_pairs = await get_image_text_pairs(task_data[cst.TASK_ID], psql_db)
                tasks.append(ImageTask(**task_data, image_text_pairs=image_text_pairs))

        return tasks


async def get_completed_organic_tasks(
    psql_db: PSQLDB,
    hours: int | None = None,
    task_type: TaskType | None = None,
    search_model_name: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> List[TextTask | ImageTask]:
    """Get completed organic tasks with optional filters

    Args:
        psql_db: Database connection
        hours: Optional number of hours to look back
        task_type: Optional task type filter
        search_model_name: Optional search term to filter models by name
        limit: Number of tasks per page
        offset: Offset for pagination
    """
    async with await psql_db.connection() as connection:
        connection: Connection

        where_clauses = [f"tasks.{cst.STATUS} = $1", f"tasks.{cst.IS_ORGANIC} = true"]
        params = [TaskStatus.SUCCESS.value]
        param_count = 1

        if hours is not None:
            param_count += 1
            where_clauses.append(f"tasks.{cst.TERMINATION_AT} >= NOW() - ${param_count} * INTERVAL '1 hour'")
            params.append(hours)

        if task_type is not None:
            param_count += 1
            where_clauses.append(f"tasks.{cst.TASK_TYPE} = ${param_count}")
            params.append(task_type.value)

        if search_model_name is not None:
            search_terms = search_model_name.lower().split()
            for term in search_terms:
                param_count += 1
                where_clauses.append(f"tasks.result_model_name_lower LIKE ${param_count}")
                params.append(f"%{term}%")

        where_clause = " AND ".join(where_clauses)

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
                tasks.{cst.TASK_ID},
                COALESCE(tasks.training_repo_backup, victorious_repo.{cst.REPO}) as trained_model_repository
            FROM {cst.TASKS_TABLE} tasks
            LEFT JOIN victorious_repo
                ON tasks.{cst.TASK_ID} = victorious_repo.{cst.TASK_ID}
                AND victorious_repo.rn = 1
            WHERE {where_clause}
            ORDER BY tasks.{cst.TERMINATION_AT} DESC
            LIMIT ${param_count + 1} OFFSET ${param_count + 2}
        """

        params.extend([limit, offset])
        task_ids = await connection.fetch(query, *params)
        tasks_list = []
        for task_row in task_ids:
            task = await get_task_by_id(task_row[cst.TASK_ID], psql_db)
            tasks_list.append(task)

        return tasks_list


async def get_expected_repo_name(task_id: UUID, hotkey: str, psql_db: PSQLDB) -> str | None:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.EXPECTED_REPO_NAME}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2 AND {cst.NETUID} = $3
        """
        return await connection.fetchval(query, task_id, hotkey, NETUID)


async def store_offer_response(task_id: UUID, hotkey: str, offer_response: str, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {cst.OFFER_RESPONSES_TABLE} ({cst.TASK_ID}, {cst.HOTKEY}, {cst.OFFER_RESPONSE}) VALUES ($1, $2, $3)
        """
        await connection.execute(query, task_id, hotkey, offer_response)


async def add_image_text_pairs(task_id: UUID, pairs: list[ImageTextPair], psql_db: PSQLDB) -> None:
    query = f"""
        INSERT INTO {cst.IMAGE_TEXT_PAIRS_TABLE} ({cst.TASK_ID}, {cst.IMAGE_URL}, {cst.TEXT_URL})
        VALUES ($1, $2, $3)
    """

    async with await psql_db.connection() as conn:
        async with conn.transaction():
            for pair in pairs:
                await conn.execute(query, task_id, pair.image_url, pair.text_url)


async def get_image_text_pairs(task_id: UUID, psql_db: PSQLDB) -> list[ImageTextPair]:
    query = f"""
        SELECT {cst.IMAGE_URL}, {cst.TEXT_URL}
        FROM {cst.IMAGE_TEXT_PAIRS_TABLE}
        WHERE {cst.TASK_ID} = $1
        ORDER BY {cst.ID}
    """

    async with await psql_db.connection() as conn:
        rows = await conn.fetch(query, task_id)
        return [ImageTextPair(image_url=row["image_url"], text_url=row["text_url"]) for row in rows]


async def delete_image_text_pairs(task_id: UUID, psql_db: PSQLDB) -> None:
    query = f"""
        DELETE FROM {cst.IMAGE_TEXT_PAIRS_TABLE}
        WHERE {cst.TASK_ID} = $1
    """

    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(query, task_id)
