import json
import math

from asyncpg import Connection
from fastapi import Depends
from fastapi import HTTPException
from loguru import logger  # noqa

from core.models.utility_models import TaskType
from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.core.models import AnyTypeTask
from validator.core.models import AnyTypeTaskWithHotkeyDetails
from validator.core.models import DpoTask
from validator.core.models import DpoTaskWithHotkeyDetails
from validator.core.models import GrpoTask
from validator.core.models import GrpoTaskWithHotkeyDetails
from validator.core.models import HotkeyDetails
from validator.core.models import ImageTask
from validator.core.models import ImageTaskWithHotkeyDetails
from validator.core.models import InstructTextTask
from validator.core.models import InstructTextTaskWithHotkeyDetails
from validator.db import constants as cst
from validator.db.sql import tasks as tasks_sql
from validator.utils.util import hide_sensitive_data_till_finished


def normalise_float(float: float | None) -> float | None:
    if float is None:
        return 0.0

    if math.isnan(float):
        return None

    if math.isinf(float):
        float = 1e100 if float > 0 else -1e100
    return float


async def get_recent_tasks(
    hotkeys: list[str] | None = None, limit: int = 100, page: int = 1, config: Config = Depends(get_config)
) -> list[AnyTypeTask]:
    full_tasks_list = []
    if hotkeys is not None:
        query = f"""
            SELECT {cst.TASK_ID} FROM {cst.SUBMISSIONS_TABLE}
            WHERE {cst.HOTKEY} = ANY($1)
            ORDER BY {cst.CREATED_ON} DESC
            LIMIT $2
            OFFSET $3
        """
        async with await config.psql_db.connection() as connection:
            connection: Connection
            task_ids = await connection.fetch(query, hotkeys, limit, (page - 1) * limit)

        for task_row in task_ids:
            task = await tasks_sql.get_task_by_id(task_row[cst.TASK_ID], config.psql_db)
            full_tasks_list.append(task)

    else:
        query = f"""
            SELECT {cst.TASK_ID} FROM {cst.TASKS_TABLE}
            ORDER BY {cst.CREATED_AT} DESC
            LIMIT $1
            OFFSET $2
        """
        async with await config.psql_db.connection() as connection:
            connection: Connection
            task_ids = await connection.fetch(query, limit, (page - 1) * limit)

        for task_row in task_ids:
            task = await tasks_sql.get_task_by_id(task_row[cst.TASK_ID], config.psql_db)
            full_tasks_list.append(task)

    tasks_processed = []
    for task in full_tasks_list:
        task = hide_sensitive_data_till_finished(task)
        tasks_processed.append(task)

    return tasks_processed


async def _process_task_batch(
    connection, hotkey: str, task_ids: list[str]
) -> list[AnyTypeTaskWithHotkeyDetails]:
    """
    Helper function to process a batch of task IDs.
    """
    tasks_with_details = []

    tasks_by_id = {}
    if task_ids:
        task_placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(task_ids)))
        tasks_query = f"""
            SELECT
                t.*
            FROM
                {cst.TASKS_TABLE} t
            WHERE
                t.{cst.TASK_ID} IN ({task_placeholders})
        """

        tasks_rows = await connection.fetch(tasks_query, *task_ids)

        tasks_by_id = {str(row[cst.TASK_ID]): dict(row) for row in tasks_rows}
    else:
        return []

    # Step 3: Get all hotkey-specific details for these tasks in a single query
    details_rows = []
    if task_ids:
        details_placeholders = ", ".join("$%d::uuid" % (i + 2) for i in range(len(task_ids)))
        details_query = f"""
            SELECT
                t.{cst.TASK_ID}::text AS task_id,
                s.{cst.SUBMISSION_ID} AS submission_id,
                tn.{cst.QUALITY_SCORE} AS quality_score,
                tn.{cst.TEST_LOSS} AS test_loss,
                tn.{cst.SYNTH_LOSS} AS synth_loss,
                tn.{cst.SCORE_REASON} AS score_reason,
                RANK() OVER (PARTITION BY t.{cst.TASK_ID} ORDER BY tn.{cst.QUALITY_SCORE} DESC) AS rank,
                s.{cst.REPO} AS repo,
                o.{cst.OFFER_RESPONSE} AS offer_response,
                t.{cst.TASK_TYPE} AS task_type
            FROM
                {cst.TASKS_TABLE} t
            LEFT JOIN
                {cst.TASK_NODES_TABLE} tn ON t.{cst.TASK_ID} = tn.{cst.TASK_ID} AND tn.{cst.HOTKEY} = $1
            LEFT JOIN
                {cst.SUBMISSIONS_TABLE} s ON t.{cst.TASK_ID} = s.{cst.TASK_ID} AND s.{cst.HOTKEY} = $1
            LEFT JOIN
                {cst.OFFER_RESPONSES_TABLE} o ON t.{cst.TASK_ID} = o.{cst.TASK_ID} AND o.{cst.HOTKEY} = $1
            WHERE
                t.{cst.TASK_ID} IN ({details_placeholders})
        """

        details_rows = await connection.fetch(details_query, hotkey, *task_ids)

    # Step 4: Group details by task_id
    details_by_task_id = {}
    for row in details_rows:
        task_id = row["task_id"]
        if task_id not in details_by_task_id:
            details_by_task_id[task_id] = []

        detail = dict(row)

        if detail.get("offer_response"):
            try:
                detail["offer_response"] = json.loads(detail["offer_response"])
            except (json.JSONDecodeError, TypeError):
                detail["offer_response"] = None

        for field in ["quality_score", "test_loss", "synth_loss"]:
            if detail.get(field) is not None:
                detail[field] = normalise_float(detail[field])

        details_by_task_id[task_id].append(detail)

    # Step 5: Get type-specific data for each task type
    instruct_text_task_ids = []
    image_task_ids = []
    dpo_task_ids = []
    grpo_task_ids = []

    for task_id, task_data in tasks_by_id.items():
        task_type = task_data.get(cst.TASK_TYPE)
        if task_type == TaskType.INSTRUCTTEXTTASK.value:
            instruct_text_task_ids.append(task_id)
        elif task_type == TaskType.IMAGETASK.value:
            image_task_ids.append(task_id)
        elif task_type == TaskType.DPOTASK.value:
            dpo_task_ids.append(task_id)
        elif task_type == TaskType.GRPOTASK.value:
            grpo_task_ids.append(task_id)

    # Get all InstructTextTask specific data in one query
    instruct_text_task_data = {}
    if instruct_text_task_ids:
        placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(instruct_text_task_ids)))
        query = f"""
            SELECT * FROM {cst.INSTRUCT_TEXT_TASKS_TABLE}
            WHERE {cst.TASK_ID} IN ({placeholders})
        """
        rows = await connection.fetch(query, *instruct_text_task_ids)
        instruct_text_task_data = {str(row[cst.TASK_ID]): dict(row) for row in rows}

    # Get all ImageTask specific data in one query
    image_task_data = {}
    if image_task_ids:
        placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(image_task_ids)))
        query = f"""
            SELECT * FROM {cst.IMAGE_TASKS_TABLE}
            WHERE {cst.TASK_ID} IN ({placeholders})
        """
        rows = await connection.fetch(query, *image_task_ids)
        image_task_data = {str(row[cst.TASK_ID]): dict(row) for row in rows}

    # Get all DpoTask specific data in one query
    dpo_task_data = {}
    if dpo_task_ids:
        placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(dpo_task_ids)))
        query = f"""
            SELECT * FROM {cst.DPO_TASKS_TABLE}
            WHERE {cst.TASK_ID} IN ({placeholders})
        """
        rows = await connection.fetch(query, *dpo_task_ids)
        dpo_task_data = {str(row[cst.TASK_ID]): dict(row) for row in rows}

    # Get all GrpoTask specific data in one query
    grpo_task_data = {}
    if grpo_task_ids:
        placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(grpo_task_ids)))
        query = f"""
            SELECT * FROM {cst.GRPO_TASKS_TABLE}
            WHERE {cst.TASK_ID} IN ({placeholders})
        """
        rows = await connection.fetch(query, *grpo_task_ids)
        grpo_task_data = {str(row[cst.TASK_ID]): dict(row) for row in rows}


    # Step 6: Assemble final results
    for task_id in task_ids:
        if task_id not in tasks_by_id:
            continue

        task_data = tasks_by_id[task_id].copy()
        task_type = task_data.get(cst.TASK_TYPE)

        if task_type == TaskType.INSTRUCTTEXTTASK.value and task_id in instruct_text_task_data:
            task_data.update(instruct_text_task_data[task_id])
        elif task_type == TaskType.IMAGETASK.value and task_id in image_task_data:
            task_data.update(image_task_data[task_id])
        elif task_type == TaskType.DPOTASK.value and task_id in dpo_task_data:
            task_data.update(dpo_task_data[task_id])
        elif task_type == TaskType.GRPOTASK.value and task_id in grpo_task_data:
            task_data.update(grpo_task_data[task_id])

        hotkey_details = []
        if task_id in details_by_task_id:
            for detail in details_by_task_id[task_id]:
                hotkey_details.append(
                    HotkeyDetails(
                        hotkey=hotkey,
                        submission_id=detail.get("submission_id"),
                        quality_score=detail.get("quality_score"),
                        test_loss=detail.get("test_loss"),
                        synth_loss=detail.get("synth_loss"),
                        score_reason=detail.get("score_reason"),
                        rank=detail.get("rank"),
                        repo=detail.get("repo"),
                        offer_response=detail.get("offer_response"),
                    )
                )

        if task_type == TaskType.INSTRUCTTEXTTASK.value:
            task_fields = {k: v for k, v in task_data.items() if k in InstructTextTask.model_fields}
            task = InstructTextTask(**task_fields)
            task = hide_sensitive_data_till_finished(task)
            tasks_with_details.append(InstructTextTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details))
        elif task_type == TaskType.IMAGETASK.value:
            task_fields = {k: v for k, v in task_data.items() if k in ImageTask.model_fields}
            task = ImageTask(**task_fields)
            task = hide_sensitive_data_till_finished(task)
            tasks_with_details.append(ImageTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details))
        elif task_type == TaskType.DPOTASK.value:
            task_fields = {k: v for k, v in task_data.items() if k in DpoTask.model_fields}
            task = DpoTask(**task_fields)
            task = hide_sensitive_data_till_finished(task)
            tasks_with_details.append(DpoTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details))
        elif task_type == TaskType.GRPOTASK.value:
            task_fields = {k: v for k, v in task_data.items() if k in GrpoTask.model_fields}
            task = GrpoTask(**task_fields)
            task = _check_if_task_has_finished(task)
            tasks_with_details.append(GrpoTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details))

    return tasks_with_details


async def get_recent_tasks_for_hotkey(
    hotkey: str, limit: int = 100, page: int = 1, config: Config = Depends(get_config)
) -> list[AnyTypeTaskWithHotkeyDetails]:
    """
    Retrieves recent tasks for a specific hotkey with detailed information.
    """
    MAX_BATCH_SIZE = 500

    async with await config.psql_db.connection() as connection:
        task_ids_query = f"""
            SELECT
                s.{cst.TASK_ID}::text AS task_id
            FROM
                {cst.SUBMISSIONS_TABLE} s
            WHERE
                s.{cst.HOTKEY} = $1
            ORDER BY
                s.{cst.CREATED_ON} DESC
            LIMIT $2 OFFSET $3
        """
        offset = (page - 1) * limit
        task_ids_rows = await connection.fetch(task_ids_query, hotkey, limit, offset)

        if not task_ids_rows:
            return []

        task_ids = [row["task_id"] for row in task_ids_rows]

        if len(task_ids) > MAX_BATCH_SIZE:
            all_results = []
            for i in range(0, len(task_ids), MAX_BATCH_SIZE):
                batch_ids = task_ids[i : i + MAX_BATCH_SIZE]
                batch_results = await _process_task_batch(connection, hotkey, batch_ids)
                all_results.extend(batch_results)
            return all_results

        return await _process_task_batch(connection, hotkey, task_ids)

async def get_task_with_hotkey_details(task_id: str, config: Config = Depends(get_config)) -> AnyTypeTaskWithHotkeyDetails:
    # First get all the task details like normal
    task_raw = await tasks_sql.get_task_by_id(task_id, config.psql_db)
    if task_raw is None:
        raise HTTPException(status_code=404, detail="Task not found")

    logger.info("Got a task!!")

    # NOTE: If the task is not finished, remove details about synthetic data & test data?
    task = hide_sensitive_data_till_finished(task_raw)

    query = f"""
        SELECT
            tn.{cst.HOTKEY},
            s.{cst.SUBMISSION_ID},
            tn.{cst.QUALITY_SCORE},
            tn.{cst.TEST_LOSS},
            tn.{cst.SYNTH_LOSS},
            tn.{cst.SCORE_REASON},
            RANK() OVER (ORDER BY tn.{cst.QUALITY_SCORE} DESC) as rank,
            s.{cst.REPO},
            o.{cst.OFFER_RESPONSE}
        FROM {cst.TASK_NODES_TABLE} tn
        LEFT JOIN {cst.SUBMISSIONS_TABLE} s
            ON tn.{cst.TASK_ID} = s.{cst.TASK_ID}
            AND tn.{cst.HOTKEY} = s.{cst.HOTKEY}
        LEFT JOIN {cst.OFFER_RESPONSES_TABLE} o
            ON tn.{cst.TASK_ID} = o.{cst.TASK_ID}
            AND tn.{cst.HOTKEY} = o.{cst.HOTKEY}
        WHERE tn.{cst.TASK_ID} = $1
    """
    async with await config.psql_db.connection() as connection:
        connection: Connection
        results = await connection.fetch(query, task_id)

    logger.info(f"Got {len(results)} results for task {task_id}")

    hotkey_details = []
    for result in results:
        result_dict = dict(result)
        if result_dict[cst.OFFER_RESPONSE] is not None:
            result_dict[cst.OFFER_RESPONSE] = json.loads(result_dict[cst.OFFER_RESPONSE])

        float_fields = [cst.QUALITY_SCORE, cst.TEST_LOSS, cst.SYNTH_LOSS]
        for field in float_fields:
            result_dict[field] = normalise_float(result_dict[field])

        hotkey_details.append(HotkeyDetails(**result_dict))

    if task.task_type == TaskType.INSTRUCTTEXTTASK:
        return InstructTextTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details)
    elif task.task_type == TaskType.IMAGETASK:
        return ImageTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details)
    elif task.task_type == TaskType.DPOTASK:
        return DpoTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details)
    elif task.task_type == TaskType.GRPOTASK:
        return GrpoTaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details)


async def store_latest_scores_url(url: str, config: Config = Depends(get_config)) -> None:
    async with await config.psql_db.connection() as connection:
        connection: Connection

        # First expire all existing URLs
        expire_query = f"""
            UPDATE {cst.LATEST_SCORES_URL_TABLE}
            SET expired_at = NOW()
            WHERE expired_at IS NULL
        """
        await connection.execute(expire_query)

        # Then insert the new URL
        insert_query = f"""
            INSERT INTO {cst.LATEST_SCORES_URL_TABLE} (url)
            VALUES ($1)
        """
        await connection.execute(insert_query, url)


async def get_latest_scores_url(config: Config = Depends(get_config)) -> str | None:
    async with await config.psql_db.connection() as connection:
        connection: Connection

        query = f"""
            SELECT url FROM {cst.LATEST_SCORES_URL_TABLE} WHERE expired_at IS NULL ORDER BY created_at DESC LIMIT 1
        """
        return await connection.fetchval(query)
