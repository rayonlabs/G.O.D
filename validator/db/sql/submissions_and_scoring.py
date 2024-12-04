import asyncio
import json
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from uuid import UUID

from asyncpg.connection import Connection

import validator.db.constants as cst
from core.constants import NETUID
from validator.core.models import AllNodeStats
from validator.core.models import ModelMetrics
from validator.core.models import NodeStats
from validator.core.models import QualityMetrics
from validator.core.models import Submission
from validator.core.models import Task
from validator.core.models import TaskNode
from validator.core.models import TaskResults
from validator.core.models import WorkloadMetrics
from validator.db.database import PSQLDB


async def add_submission(submission: Submission, psql_db: PSQLDB) -> Submission:
    """Add a new submission for the current NETUID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {cst.SUBMISSIONS_TABLE} (
                {cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}, {cst.REPO}
            )
            VALUES ($1, $2, $3, $4)
            RETURNING {cst.SUBMISSION_ID}
        """
        submission_id = await connection.fetchval(
            query,
            submission.task_id,
            submission.hotkey,
            NETUID,
            submission.repo,
        )
        return await get_submission(submission_id, psql_db)


async def get_submission(submission_id: UUID, psql_db: PSQLDB) -> Optional[Submission]:
    """Get a submission by its ID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {cst.SUBMISSIONS_TABLE} WHERE {cst.SUBMISSION_ID} = $1
        """
        row = await connection.fetchrow(query, submission_id)
        if row:
            return Submission(**dict(row))
        return None


async def get_submissions_by_task(task_id: UUID, psql_db: PSQLDB) -> List[Submission]:
    """Get all submissions for a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {cst.SUBMISSIONS_TABLE}
            WHERE {cst.TASK_ID} = $1 AND {cst.NETUID} = $2
        """
        rows = await connection.fetch(query, task_id, NETUID)
        return [Submission(**dict(row)) for row in rows]


async def get_node_latest_submission(task_id: str, hotkey: str, psql_db: PSQLDB) -> Optional[Submission]:
    """Get the latest submission for a node on a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {cst.SUBMISSIONS_TABLE}
            WHERE {cst.TASK_ID} = $1
            AND {cst.HOTKEY} = $2
            AND {cst.NETUID} = $3
            ORDER BY {cst.CREATED_ON} DESC
            LIMIT 1
        """
        row = await connection.fetchrow(query, task_id, hotkey, NETUID)
        if row:
            return Submission(**dict(row))
        return None


async def submission_repo_is_unique(repo: str, psql_db: PSQLDB) -> bool:
    """Check if a repository URL is unique"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT 1 FROM {cst.SUBMISSIONS_TABLE}
            WHERE {cst.REPO} = $1 AND {cst.NETUID} = $2
            LIMIT 1
        """
        result = await connection.fetchval(query, repo, NETUID)
        return result is None


async def set_task_node_quality_score(task_id: UUID, hotkey: str, quality_score: float, psql_db: PSQLDB) -> None:
    """Set quality score for a node's task submission"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {cst.TASK_NODES_TABLE} (
                {cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}, {cst.TASK_NODE_QUALITY_SCORE}
            )
            VALUES ($1, $2, $3, $4)
            ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}) DO UPDATE
            SET {cst.TASK_NODE_QUALITY_SCORE} = $4
        """
        await connection.execute(query, task_id, hotkey, NETUID, quality_score)


async def get_task_node_quality_score(task_id: UUID, hotkey: str, psql_db: PSQLDB) -> Optional[float]:
    """Get quality score for a node's task submission"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {cst.TASK_NODE_QUALITY_SCORE}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1
            AND {cst.HOTKEY} = $2
            AND {cst.NETUID} = $3
        """
        return await connection.fetchval(query, task_id, hotkey, NETUID)


async def get_all_quality_scores_for_task(task_id: UUID, psql_db: PSQLDB) -> Dict[str, float]:
    """Get all quality scores for a task, keyed by hotkey"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {cst.HOTKEY}, {cst.TASK_NODE_QUALITY_SCORE}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1
            AND {cst.NETUID} = $2
            AND {cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL
        """
        rows = await connection.fetch(query, task_id, NETUID)
        return {row[cst.HOTKEY]: row[cst.TASK_NODE_QUALITY_SCORE] for row in rows}


async def set_multiple_task_node_quality_scores(task_id: UUID, quality_scores: Dict[str, float], psql_db: PSQLDB) -> None:
    """Set multiple quality scores for task nodes"""
    async with await psql_db.connection() as connection:
        connection: Connection
        async with connection.transaction():
            query = f"""
                INSERT INTO {cst.TASK_NODES_TABLE} (
                    {cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}, {cst.TASK_NODE_QUALITY_SCORE}
                )
                VALUES ($1, $2, $3, $4)
                ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}) DO UPDATE
                SET {cst.TASK_NODE_QUALITY_SCORE} = EXCLUDED.{cst.TASK_NODE_QUALITY_SCORE}
            """
            await connection.executemany(query, [(task_id, hotkey, NETUID, score) for hotkey, score in quality_scores.items()])


async def get_all_scores_for_hotkey(hotkey: str, psql_db: PSQLDB) -> List[Dict]:
    """
    Get all quality scores for a specific hotkey across all tasks.
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT
                {cst.TASK_ID},
                {cst.TASK_NODE_QUALITY_SCORE} as quality_score
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.HOTKEY} = $1
            AND {cst.NETUID} = $2
            AND {cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL
        """
        rows = await connection.fetch(query, hotkey, NETUID)
        return [dict(row) for row in rows]


async def get_aggregate_scores_since(start_time: datetime, psql_db: PSQLDB) -> List[TaskResults]:
    """
    Get aggregate scores for all completed tasks since the given start time.
    Only includes tasks that have at least one node with score > 0.
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT
                t.*,
                COALESCE(
                    json_agg(
                        json_build_object(
                            '{cst.TASK_ID}', t.{cst.TASK_ID}::text,
                            '{cst.HOTKEY}', tn.{cst.HOTKEY},
                            '{cst.QUALITY_SCORE}', tn.{cst.TASK_NODE_QUALITY_SCORE}
                        )
                        ORDER BY tn.{cst.TASK_NODE_QUALITY_SCORE} DESC NULLS LAST
                    ) FILTER (WHERE tn.{cst.HOTKEY} IS NOT NULL AND tn.{cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL),
                    '[]'::json
                ) as node_scores
            FROM {cst.TASKS_TABLE} t
            LEFT JOIN {cst.TASK_NODES_TABLE} tn ON t.{cst.TASK_ID} = tn.{cst.TASK_ID}
            WHERE t.{cst.STATUS} = 'success'
            AND t.created_timestamp >= $1
            AND tn.{cst.NETUID} = $2
            AND EXISTS (
                SELECT 1
                FROM {cst.TASK_NODES_TABLE} tn2
                WHERE tn2.{cst.TASK_ID} = t.{cst.TASK_ID}
                AND tn2.{cst.TASK_NODE_QUALITY_SCORE} >= 0
                AND tn2.{cst.NETUID} = $2
            )
            GROUP BY t.{cst.TASK_ID}
            ORDER BY t.created_timestamp DESC
        """
        rows = await connection.fetch(query, start_time, NETUID)

        results = []
        for row in rows:
            row_dict = dict(row)
            task_dict = {k: v for k, v in row_dict.items() if k !=
                         "node_scores"}
            task = Task(**task_dict)

            node_scores_data = row_dict["node_scores"]
            if isinstance(node_scores_data, str):
                node_scores_data = json.loads(node_scores_data)

            node_scores = [
                TaskNode(
                    task_id=str(node[cst.TASK_ID]),
                    hotkey=node[cst.HOTKEY],
                    quality_score=float(
                        node[cst.QUALITY_SCORE]) if node[cst.QUALITY_SCORE] is not None else None,
                )
                for node in node_scores_data
            ]

            results.append(TaskResults(task=task, node_scores=node_scores))

        return results


async def get_node_quality_metrics(hotkey: str, interval: str, psql_db: PSQLDB) -> QualityMetrics:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT
                COALESCE(AVG(tn.{cst.QUALITY_SCORE}), 0) as avg_quality_score,
                COALESCE(COUNT(CASE WHEN tn.{cst.QUALITY_SCORE} > 0 THEN 1 END)::FLOAT / NULLIF(COUNT(*), 0), 0) as success_rate,
                COALESCE(COUNT(CASE WHEN tn.{cst.QUALITY_SCORE} > 0.8 THEN 1 END)::FLOAT / NULLIF(COUNT(*), 0), 0) as quality_rate,
                COALESCE(COUNT(*), 0) as total_count,
                COALESCE(SUM(tn.{cst.QUALITY_SCORE}), 0) as total_score,
                COALESCE(COUNT(CASE WHEN tn.{cst.QUALITY_SCORE} > 0 THEN 1 END), 0) as total_success,
                COALESCE(COUNT(CASE WHEN tn.{cst.QUALITY_SCORE} > 0.8 THEN 1 END), 0) as total_quality
            FROM {cst.TASK_NODES_TABLE} tn
            JOIN {cst.TASKS_TABLE} t ON tn.{cst.TASK_ID} = t.{cst.TASK_ID}
            WHERE tn.{cst.HOTKEY} = $1
            AND tn.{cst.NETUID} = $2
            AND tn.{cst.QUALITY_SCORE} IS NOT NULL
            AND t.created_timestamp >= CASE
                WHEN $3 = 'all' THEN '1970-01-01'::TIMESTAMP
                ELSE NOW() - $3::INTERVAL
            END
        """
        row = await connection.fetchrow(query, hotkey, NETUID, interval)
        return QualityMetrics.model_validate(dict(row) if row else {})


# llm wrote this - someone that's more experienced should read through - tests work ok but still
async def get_node_workload_metrics(hotkey: str, interval: str, psql_db: PSQLDB) -> WorkloadMetrics:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            WITH param_extract AS (
                SELECT
                    t.{cst.TASK_ID},
                    CASE
                        -- Match patterns like: number followed by B/b or M/m
                        -- Will match: 0.5B, 7B, 1.5b, 70M, etc. anywhere in the string
                        WHEN LOWER(t.{cst.MODEL_ID}) ~ '.*?([0-9]+\.?[0-9]*)[mb]' THEN
                            CASE
                                WHEN LOWER(t.{cst.MODEL_ID}) ~ '.*?([0-9]+\.?[0-9]*)b' THEN
                                    -- Extract just the number before 'b'/'B'
                                    SUBSTRING(LOWER(t.{cst.MODEL_ID}) FROM '.*?([0-9]+\.?[0-9]*)b')::FLOAT
                                WHEN LOWER(t.{cst.MODEL_ID}) ~ '.*?([0-9]+\.?[0-9]*)m' THEN
                                    -- Extract just the number before 'm'/'M' and convert to billions
                                    SUBSTRING(LOWER(t.{cst.MODEL_ID}) FROM '.*?([0-9]+\.?[0-9]*)m')::FLOAT / 1000.0
                            END
                        ELSE 1.0
                    END as params_billions
                FROM {cst.TASKS_TABLE} t
            )
            SELECT
                COALESCE(SUM(t.{cst.HOURS_TO_COMPLETE}), 0)::INTEGER as competition_hours,
                COALESCE(SUM(pe.params_billions), 0) as total_params_billions
            FROM {cst.TASK_NODES_TABLE} tn
            JOIN {cst.TASKS_TABLE} t ON tn.{cst.TASK_ID} = t.{cst.TASK_ID}
            LEFT JOIN param_extract pe ON t.{cst.TASK_ID} = pe.{cst.TASK_ID}
            WHERE tn.{cst.HOTKEY} = $1
            AND tn.{cst.QUALITY_SCORE} IS NOT NULL
            AND tn.{cst.NETUID} = $2
            AND t.created_timestamp >= CASE
                WHEN $3 = 'all' THEN '1970-01-01'::TIMESTAMP
                ELSE NOW() - $3::INTERVAL
            END
        """
        row = await connection.fetchrow(query, hotkey, NETUID, interval)
        return WorkloadMetrics.model_validate(dict(row) if row else {})


async def get_node_model_metrics(hotkey: str, interval: str, psql_db: PSQLDB) -> ModelMetrics:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
        WITH model_counts AS (
            SELECT
                t.{cst.MODEL_ID},
                COUNT(*) as model_count
            FROM {cst.TASK_NODES_TABLE} tn
            JOIN {cst.TASKS_TABLE} t ON tn.{cst.TASK_ID} = t.{cst.TASK_ID}
            WHERE tn.{cst.HOTKEY} = $1
            AND tn.{cst.NETUID} = $2
            AND t.created_timestamp >= CASE
                WHEN $3 = 'all' THEN '1970-01-01'::TIMESTAMP
                ELSE NOW() - $3::INTERVAL
            END
            AND tn.{cst.QUALITY_SCORE} IS NOT NULL
            GROUP BY t.{cst.MODEL_ID}
            ORDER BY model_count DESC
            LIMIT 1
        )
        SELECT
            COALESCE((
                SELECT {cst.MODEL_ID}
                FROM model_counts
                ORDER BY model_count DESC
                LIMIT 1
            ), 'none') as modal_model,
            COUNT(DISTINCT CASE WHEN tn.{cst.QUALITY_SCORE} IS NOT NULL THEN t.{cst.MODEL_ID} END) as unique_models,
            COUNT(DISTINCT CASE WHEN tn.{cst.QUALITY_SCORE} IS NOT NULL THEN t.{cst.DS_ID} END) as unique_datasets
        FROM {cst.TASK_NODES_TABLE} tn
        JOIN {cst.TASKS_TABLE} t ON tn.{cst.TASK_ID} = t.{cst.TASK_ID}
        WHERE tn.{cst.HOTKEY} = $1
        AND tn.{cst.NETUID} = $2
        AND t.created_timestamp >= CASE
            WHEN $3 = 'all' THEN '1970-01-01'::TIMESTAMP
            ELSE NOW() - $3::INTERVAL
        END
        """
        row = await connection.fetchrow(query, hotkey, NETUID, interval)
        return ModelMetrics.model_validate(dict(row) if row else {})


async def get_node_stats(hotkey: str, interval: str, psql_db: PSQLDB) -> NodeStats:
    quality, workload, models = await asyncio.gather(
        get_node_quality_metrics(hotkey, interval, psql_db),
        get_node_workload_metrics(hotkey, interval, psql_db),
        get_node_model_metrics(hotkey, interval, psql_db),
    )

    return NodeStats(quality_metrics=quality, workload_metrics=workload, model_metrics=models)


async def get_all_node_stats(hotkey: str, psql_db: PSQLDB) -> AllNodeStats:
    daily, three_day, weekly, monthly, all_time = await asyncio.gather(
        get_node_stats(hotkey, "24 hours", psql_db),
        get_node_stats(hotkey, "3 days", psql_db),
        get_node_stats(hotkey, "7 days", psql_db),
        get_node_stats(hotkey, "30 days", psql_db),
        get_node_stats(hotkey, "all", psql_db),
    )

    return AllNodeStats(daily=daily, three_day=three_day, weekly=weekly, monthly=monthly, all_time=all_time)
