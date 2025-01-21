import json
import math

from asyncpg import Connection
from fastapi import Depends
from fastapi import HTTPException
from loguru import logger  # noqa

from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.core.models import HotkeyDetails
from validator.core.models import Task
from validator.core.models import TaskWithHotkeyDetails
from validator.db import constants as cst


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
) -> list[Task]:
    async with await config.psql_db.connection() as connection:
        connection: Connection

        if hotkeys is not None:
            query = f"""
                SELECT {cst.TASK_ID} FROM {cst.SUBMISSIONS_TABLE}
                WHERE {cst.HOTKEY} = ANY($1)
                ORDER BY {cst.CREATED_ON} DESC
                LIMIT $2
                OFFSET $3
            """
            task_ids = await connection.fetch(query, hotkeys, limit, (page - 1) * limit)

            query_for_tasks = f"""
                SELECT * FROM {cst.TASKS_TABLE}
                WHERE {cst.TASK_ID} = ANY($1)
            """
            tasks = await connection.fetch(query_for_tasks, task_ids)
        else:
            query = f"""
                SELECT * FROM {cst.TASKS_TABLE}
                ORDER BY {cst.CREATED_AT} DESC
                LIMIT $1
                OFFSET $2
            """
            tasks = await connection.fetch(query, limit, (page - 1) * limit)

    tasks_processed = []
    for task in tasks:
        task = Task(**task)
        if task.status not in [
            TaskStatus.SUCCESS,
            TaskStatus.FAILURE,
            TaskStatus.FAILURE_FINDING_NODES,
            TaskStatus.PREP_TASK_FAILURE,
            TaskStatus.NODE_TRAINING_FAILURE,
        ]:
            continue
            task.ds_id = "Hidden"
            task.test_data = None
            task.synthetic_data = None
            task.training_data = None
        tasks_processed.append(task)
    return tasks_processed


async def get_recent_tasks_for_hotkey(
    hotkey: str, limit: int = 100, page: int = 1, config: Config = Depends(get_config)
) -> list[TaskWithHotkeyDetails]:
    async with await config.psql_db.connection() as connection:
        connection: Connection

        query = f"""
            SELECT {cst.TASK_ID} FROM {cst.SUBMISSIONS_TABLE}
            WHERE {cst.HOTKEY} = $1
            ORDER BY {cst.CREATED_ON} DESC
            LIMIT $2
            OFFSET $3
        """
        task_ids = await connection.fetch(query, hotkey, limit, (page - 1) * limit)

        query = f"""
            SELECT
                t.*,
                s.{cst.SUBMISSION_ID},
                tn.{cst.QUALITY_SCORE},
                tn.{cst.TEST_LOSS},
                tn.{cst.SYNTH_LOSS},
                tn.{cst.SCORE_REASON},
                RANK() OVER (PARTITION BY t.{cst.TASK_ID} ORDER BY tn.{cst.QUALITY_SCORE} DESC) as rank,
                s.{cst.REPO},
                o.{cst.OFFER_RESPONSE}
            FROM {cst.TASKS_TABLE} t
            LEFT JOIN {cst.TASK_NODES_TABLE} tn
                ON t.{cst.TASK_ID} = tn.{cst.TASK_ID}
                AND tn.{cst.HOTKEY} = $2
            LEFT JOIN {cst.SUBMISSIONS_TABLE} s
                ON t.{cst.TASK_ID} = s.{cst.TASK_ID}
                AND s.{cst.HOTKEY} = $2
            LEFT JOIN {cst.OFFER_RESPONSES_TABLE} o
                ON t.{cst.TASK_ID} = o.{cst.TASK_ID}
                AND o.{cst.HOTKEY} = $2
            WHERE t.{cst.TASK_ID} = ANY($1)
            ORDER BY t.{cst.CREATED_AT} DESC
        """
        results = await connection.fetch(query, task_ids, hotkey)

        tasks_with_details = []
        for result in results:
            result_dict = dict(result)

            if result_dict.get(cst.OFFER_RESPONSE):
                result_dict[cst.OFFER_RESPONSE] = json.loads(result_dict[cst.OFFER_RESPONSE])

            float_fields = [cst.QUALITY_SCORE, cst.TEST_LOSS, cst.SYNTH_LOSS]
            for field in float_fields:
                if result_dict.get(field):
                    result_dict[field] = normalise_float(result_dict[field])

            hotkey_details = [
                HotkeyDetails(
                    hotkey=hotkey,
                    submission_id=result_dict.get(cst.SUBMISSION_ID),
                    quality_score=result_dict.get(cst.QUALITY_SCORE),
                    test_loss=result_dict.get(cst.TEST_LOSS),
                    synth_loss=result_dict.get(cst.SYNTH_LOSS),
                    score_reason=result_dict.get(cst.SCORE_REASON),
                    rank=result_dict.get("rank"),
                    repo=result_dict.get(cst.REPO),
                    offer_response=result_dict.get(cst.OFFER_RESPONSE),
                )
            ]

            task_fields = {k: v for k, v in result_dict.items() if k in Task.model_fields}
            task = Task(**task_fields)

            if task.status not in [
                TaskStatus.SUCCESS,
                TaskStatus.FAILURE,
                TaskStatus.FAILURE_FINDING_NODES,
                TaskStatus.PREP_TASK_FAILURE,
                TaskStatus.NODE_TRAINING_FAILURE,
            ]:
                continue
                task.synthetic_data = None
                task.test_data = None
                task.training_data = None
                task.ds_id = "Hidden"

            tasks_with_details.append(TaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details))

    return tasks_with_details


async def get_task_with_hotkey_details(task_id: str, config: Config = Depends(get_config)) -> TaskWithHotkeyDetails:
    # First get all the task details like normal
    async with await config.psql_db.connection() as connection:
        connection: Connection

        query = f"""
            SELECT * FROM {cst.TASKS_TABLE}
            WHERE {cst.TASK_ID} = $1
        """
        task_raw = await connection.fetchrow(query, task_id)
        if task_raw is None:
            raise HTTPException(status_code=404, detail="Task not found")
        task = Task(**task_raw)

        logger.info("Got a task!!")

        # NOTE: If the task is not finished, remove details about synthetic data & test data?
        if task.status not in [
            TaskStatus.SUCCESS,
            TaskStatus.FAILURE,
            TaskStatus.FAILURE_FINDING_NODES,
            TaskStatus.PREP_TASK_FAILURE,
            TaskStatus.NODE_TRAINING_FAILURE,
        ]:
            raise HTTPException(status_code=400, detail="Task not finished!")
            task.synthetic_data = None
            task.test_data = None
            task.training_data = None
            task.ds_id = "Hidden"

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

        return TaskWithHotkeyDetails(**task.model_dump(), hotkey_details=hotkey_details)


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
