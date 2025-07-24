import asyncio
import json
from datetime import datetime
from uuid import UUID

from asyncpg.connection import Connection

import validator.db.constants as cst
from core.constants import NETUID
from core.models.utility_models import TaskStatus
from validator.core.models import AllNodeStats
from validator.core.models import MiniTaskWithScoringOnly
from validator.core.models import ModelMetrics
from validator.core.models import NodeStats
from validator.core.models import QualityMetrics
from validator.core.models import Submission
from validator.core.models import TaskNode
from validator.core.models import TaskResults
from validator.core.models import WorkloadMetrics
from validator.db.database import PSQLDB


async def get_nodes_daily_status(hotkeys: list[str], psql_db: PSQLDB) -> dict[str, dict]:
    """
    Get both daily participation status and average scores for nodes.
    """
    if not hotkeys:
        return {}

    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT
                tn.{cst.HOTKEY},
                AVG(tn.{cst.QUALITY_SCORE}) as avg_quality_score,
                COUNT(*) > 0 as has_participated_today
            FROM {cst.TASK_NODES_TABLE} tn
            JOIN {cst.TASKS_TABLE} t ON tn.{cst.TASK_ID} = t.{cst.TASK_ID}
            WHERE tn.{cst.HOTKEY} = ANY($1)
            AND tn.{cst.NETUID} = $2
            AND tn.{cst.QUALITY_SCORE} IS NOT NULL
            AND t.{cst.CREATED_AT} >= CURRENT_DATE  -- Only tasks from today
            GROUP BY tn.{cst.HOTKEY}
        """
        rows = await connection.fetch(query, hotkeys, NETUID)

        result = {hotkey: {"has_participated_today": False, "avg_quality_score": None} for hotkey in hotkeys}

        for row in rows:
            hotkey = row[cst.HOTKEY]
            result[hotkey]["has_participated_today"] = row["has_participated_today"]
            result[hotkey]["avg_quality_score"] = row["avg_quality_score"]

        return result


async def add_submission(submission: Submission, psql_db: PSQLDB) -> Submission:
    """Add or update a submission for the current NETUID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {cst.SUBMISSIONS_TABLE} (
                {cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}, {cst.REPO}
            )
            VALUES ($1, $2, $3, $4)
            ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID})
            DO UPDATE SET
                {cst.REPO} = EXCLUDED.{cst.REPO},
                updated_on = CURRENT_TIMESTAMP
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


async def get_submission(submission_id: UUID, psql_db: PSQLDB) -> Submission | None:
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


async def get_submissions_by_task(task_id: UUID, psql_db: PSQLDB) -> list[Submission]:
    """Get all submissions for a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {cst.SUBMISSIONS_TABLE}
            WHERE {cst.TASK_ID} = $1 AND {cst.NETUID} = $2
        """
        rows = await connection.fetch(query, task_id, NETUID)
        return [Submission(**dict(row)) for row in rows]


async def get_node_latest_submission(task_id: str, hotkey: str, psql_db: PSQLDB) -> Submission | None:
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


async def set_task_node_quality_score(
    task_id: UUID,
    hotkey: str,
    quality_score: float,
    test_loss: float,
    synth_loss: float,
    psql_db: PSQLDB,
    score_reason: str | None = None,
) -> None:
    """Set quality score, losses and zero score reason for a node's task submission"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {cst.TASK_NODES_TABLE} (
                {cst.TASK_ID},
                {cst.HOTKEY},
                {cst.NETUID},
                {cst.TASK_NODE_QUALITY_SCORE},
                {cst.TEST_LOSS},
                {cst.SYNTH_LOSS},
                {cst.SCORE_REASON}
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}) DO UPDATE
            SET
                {cst.TASK_NODE_QUALITY_SCORE} = $4,
                {cst.TEST_LOSS} = $5,
                {cst.SYNTH_LOSS} = $6,
                {cst.SCORE_REASON} = $7
        """
        await connection.execute(
            query,
            task_id,
            hotkey,
            NETUID,
            quality_score,
            test_loss,
            synth_loss,
            score_reason,
        )


async def get_task_node_quality_score(task_id: UUID, hotkey: str, psql_db: PSQLDB) -> float | None:
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


async def get_all_scores_and_losses_for_task(task_id: UUID, psql_db: PSQLDB) -> list[dict]:
    """Get all quality scores and losses for a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT
                {cst.HOTKEY},
                {cst.TASK_NODE_QUALITY_SCORE},
                {cst.TEST_LOSS},
                {cst.SYNTH_LOSS},
                {cst.SCORE_REASON}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1
            AND {cst.NETUID} = $2
            AND {cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL
        """
        rows = await connection.fetch(query, task_id, NETUID)

        def clean_float(value):
            if value is None:
                return None
            if isinstance(value, float):
                if value in (float("inf"), float("-inf")) or value != value:
                    return None
            return value

        return [
            {
                cst.HOTKEY: row[cst.HOTKEY],
                cst.TASK_NODE_QUALITY_SCORE: clean_float(row[cst.TASK_NODE_QUALITY_SCORE]),
                cst.TEST_LOSS: clean_float(row[cst.TEST_LOSS]),
                cst.SYNTH_LOSS: clean_float(row[cst.SYNTH_LOSS]),
                cst.SCORE_REASON: row[cst.SCORE_REASON],
            }
            for row in rows
        ]


async def get_all_scores_for_hotkey(hotkey: str, psql_db: PSQLDB) -> list[dict]:
    """
    Get all quality scores for a specific hotkey across all completed tasks.
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT
                tn.{cst.TASK_ID},
                tn.{cst.TASK_NODE_QUALITY_SCORE} as quality_score
            FROM {cst.TASK_NODES_TABLE} tn
            JOIN {cst.TASKS_TABLE} t ON tn.{cst.TASK_ID} = t.{cst.TASK_ID}
            WHERE tn.{cst.HOTKEY} = $1
            AND tn.{cst.NETUID} = $2
            AND tn.{cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL
            AND t.{cst.STATUS} = $3
        """
        rows = await connection.fetch(query, hotkey, NETUID, TaskStatus.SUCCESS.value)
        return [dict(row) for row in rows]


async def get_aggregate_scores_since(start_time: datetime, psql_db: PSQLDB) -> list[TaskResults]:
    """
    Get aggregate scores for all completed tasks since the given start time.
    Only includes tasks that have at least one node with score >= 1 or < 0
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
                    ) FILTER (
                        WHERE tn.{cst.HOTKEY} IS NOT NULL 
                        AND tn.{cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL
                    ),
                    '[]'::json
                ) as node_scores
            FROM {cst.TASKS_TABLE} t
            LEFT JOIN {cst.TASK_NODES_TABLE} tn ON t.{cst.TASK_ID} = tn.{cst.TASK_ID}
            WHERE t.{cst.STATUS} = 'success'
            AND t.{cst.CREATED_AT} >= $1
            AND tn.{cst.NETUID} = $2
            AND EXISTS (
                SELECT 1
                FROM {cst.TASK_NODES_TABLE} tn2
                WHERE tn2.{cst.TASK_ID} = t.{cst.TASK_ID}
                AND (tn2.{cst.TASK_NODE_QUALITY_SCORE} >= 1 OR tn2.{cst.TASK_NODE_QUALITY_SCORE} < 0)
                AND tn2.{cst.NETUID} = $2
            )
            GROUP BY t.{cst.TASK_ID}
            ORDER BY t.{cst.CREATED_AT} DESC
        """
        rows = await connection.fetch(query, start_time, NETUID)
        results = []
        for row in rows:
            row_dict = dict(row)
            task_dict = {k: v for k, v in row_dict.items() if k != "node_scores"}
            task = MiniTaskWithScoringOnly(**task_dict)
            node_scores_data = row_dict["node_scores"]
            if isinstance(node_scores_data, str):
                node_scores_data = json.loads(node_scores_data)
            node_scores = [
                TaskNode(
                    task_id=str(node[cst.TASK_ID]),
                    hotkey=node[cst.HOTKEY],
                    quality_score=float(node[cst.QUALITY_SCORE]) if node[cst.QUALITY_SCORE] is not None else None,
                )
                for node in node_scores_data
                if node[cst.QUALITY_SCORE] is not None
                and (float(node[cst.QUALITY_SCORE]) >= 1 or float(node[cst.QUALITY_SCORE]) < 0)
            ]
            results.append(TaskResults(task=task, node_scores=node_scores))
        return results


async def get_aggregate_scores_for_leaderboard_since(start_time: datetime, psql_db: PSQLDB) -> list[TaskResults]:
    """
    Get aggregate scores for all completed tasks since the given start time.
    Includes ALL scores (including zeros) for leaderboard and analytics purposes.
    This is separate from get_aggregate_scores_since which filters for weight calculations.
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
                    ) FILTER (
                        WHERE tn.{cst.HOTKEY} IS NOT NULL 
                        AND tn.{cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL
                    ),
                    '[]'::json
                ) as node_scores
            FROM {cst.TASKS_TABLE} t
            LEFT JOIN {cst.TASK_NODES_TABLE} tn ON t.{cst.TASK_ID} = tn.{cst.TASK_ID}
            WHERE t.{cst.STATUS} = 'success'
            AND t.{cst.CREATED_AT} >= $1
            AND tn.{cst.NETUID} = $2
            GROUP BY t.{cst.TASK_ID}
            ORDER BY t.{cst.CREATED_AT} DESC
        """
        rows = await connection.fetch(query, start_time, NETUID)
        results = []
        for row in rows:
            row_dict = dict(row)
            task_dict = {k: v for k, v in row_dict.items() if k != "node_scores"}
            task = MiniTaskWithScoringOnly(**task_dict)
            node_scores_data = row_dict["node_scores"]
            if isinstance(node_scores_data, str):
                node_scores_data = json.loads(node_scores_data)
            node_scores = [
                TaskNode(
                    task_id=str(node[cst.TASK_ID]),
                    hotkey=node[cst.HOTKEY],
                    quality_score=float(node[cst.QUALITY_SCORE]) if node[cst.QUALITY_SCORE] is not None else None,
                )
                for node in node_scores_data
                if node[cst.QUALITY_SCORE] is not None
            ]
            results.append(TaskResults(task=task, node_scores=node_scores))
        return results



async def get_organic_proportion_since(start_time: datetime, psql_db: PSQLDB, task_type: str | None = None) -> float:

    """
    Get the proportion of organic tasks since the given start time.
    Optionally filter by task_type.
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT
                COALESCE(
                    COUNT(CASE WHEN is_organic = TRUE THEN 1 END)::FLOAT /
                    NULLIF(COUNT(*), 0),
                    0.0
                ) as organic_proportion
            FROM {cst.TASKS_TABLE} t
            WHERE t.{cst.CREATED_AT} >= $1
            AND t.{cst.NETUID} = $2
            {"AND t.task_type = $3" if task_type else ""}
        """
        params = [start_time, NETUID]
        if task_type:
            params.append(task_type)

        row = await connection.fetchrow(query, *params)
        return float(row["organic_proportion"]) if row else 0.0


async def get_node_quality_metrics(hotkey: str, interval: str, psql_db: PSQLDB) -> QualityMetrics:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT
                COALESCE(AVG(tn.{cst.QUALITY_SCORE}), 0) as avg_quality_score,
                COALESCE(COUNT(
                    CASE WHEN tn.{cst.QUALITY_SCORE} > -1 THEN 1 END
                )::FLOAT / NULLIF(COUNT(*), 0), 0) as success_rate,
                COALESCE(COUNT(
                    CASE WHEN tn.{cst.QUALITY_SCORE} > 0 THEN 1 END
                )::FLOAT / NULLIF(COUNT(*), 0), 0) as quality_rate,
                COALESCE(COUNT(*), 0) as total_count,
                COALESCE(SUM(tn.{cst.QUALITY_SCORE}), 0) as total_score,
                COALESCE(COUNT(CASE WHEN tn.{cst.QUALITY_SCORE} > -1 THEN 1 END), 0) as total_success,
                COALESCE(COUNT(CASE WHEN tn.{cst.QUALITY_SCORE} > 0 THEN 1 END), 0) as total_quality

            FROM {cst.TASK_NODES_TABLE} tn
            JOIN {cst.TASKS_TABLE} t ON tn.{cst.TASK_ID} = t.{cst.TASK_ID}
            WHERE tn.{cst.HOTKEY} = $1
            AND tn.{cst.NETUID} = $2
            AND tn.{cst.QUALITY_SCORE} IS NOT NULL
            AND t.{cst.CREATED_AT} >= CASE
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
            AND t.{cst.CREATED_AT} >= CASE
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
            AND t.{cst.CREATED_AT} >= CASE
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
            COUNT(DISTINCT CASE WHEN tn.{cst.QUALITY_SCORE} IS NOT NULL THEN t.{cst.DS} END) as unique_datasets
        FROM {cst.TASK_NODES_TABLE} tn
        JOIN {cst.TASKS_TABLE} t ON tn.{cst.TASK_ID} = t.{cst.TASK_ID}
        WHERE tn.{cst.HOTKEY} = $1
        AND tn.{cst.NETUID} = $2
        AND t.{cst.CREATED_AT} >= CASE
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


async def get_all_node_stats_batched(hotkeys: list[str], psql_db: PSQLDB) -> dict[str, AllNodeStats]:
    period_mapping = AllNodeStats.get_periods_sql_mapping()

    async with await psql_db.connection() as connection:
        query = f"""
        WITH periods AS (
            SELECT unnest($3::text[]) as interval
        ),
        model_counts AS (
            SELECT
                tn.{cst.HOTKEY},
                p.interval,
                t.{cst.MODEL_ID},
                COUNT(*) as model_count
            FROM periods p
            CROSS JOIN {cst.TASK_NODES_TABLE} tn
            JOIN {cst.TASKS_TABLE} t ON tn.{cst.TASK_ID} = t.{cst.TASK_ID}
            WHERE tn.{cst.HOTKEY} = ANY($1)
            AND tn.{cst.NETUID} = $2
            AND tn.{cst.QUALITY_SCORE} IS NOT NULL
            AND t.{cst.CREATED_AT} >= CASE
                WHEN p.interval = 'all' THEN '1970-01-01'::TIMESTAMP
                ELSE NOW() - p.interval::INTERVAL
            END
            GROUP BY tn.{cst.HOTKEY}, p.interval, t.{cst.MODEL_ID}
        ),
        aggregated_metrics AS (
            SELECT
                tn.{cst.HOTKEY},
                p.interval,
                -- Quality metrics
                COALESCE(AVG(tn.{cst.QUALITY_SCORE}), 0) as avg_quality_score,
                COALESCE(COUNT(CASE WHEN tn.{cst.QUALITY_SCORE} > -1 THEN 1 END)::FLOAT / NULLIF(COUNT(*), 0), 0) as success_rate,
                COALESCE(COUNT(CASE WHEN tn.{cst.QUALITY_SCORE} > 0 THEN 1 END)::FLOAT / NULLIF(COUNT(*), 0), 0) as quality_rate,
                COALESCE(COUNT(*), 0) as total_count,
                COALESCE(SUM(tn.{cst.QUALITY_SCORE}), 0) as total_score,
                COALESCE(COUNT(CASE WHEN tn.{cst.QUALITY_SCORE} > -1 THEN 1 END), 0) as total_success,
                COALESCE(COUNT(CASE WHEN tn.{cst.QUALITY_SCORE} > 0 THEN 1 END), 0) as total_quality,

                -- Workload metrics
                COALESCE(SUM(t.{cst.HOURS_TO_COMPLETE}), 0)::INTEGER as competition_hours,
                COALESCE(SUM(
                    CASE
                        WHEN LOWER(t.{cst.MODEL_ID}) ~ '.*?([0-9]+\.?[0-9]*)[mb]' THEN
                            CASE
                                WHEN LOWER(t.{cst.MODEL_ID}) ~ '.*?([0-9]+\.?[0-9]*)b' THEN
                                    SUBSTRING(LOWER(t.{cst.MODEL_ID}) FROM '.*?([0-9]+\.?[0-9]*)b')::FLOAT
                                WHEN LOWER(t.{cst.MODEL_ID}) ~ '.*?([0-9]+\.?[0-9]*)m' THEN
                                    SUBSTRING(LOWER(t.{cst.MODEL_ID}) FROM '.*?([0-9]+\.?[0-9]*)m')::FLOAT / 1000.0
                            END
                        ELSE 1.0
                    END
                ), 0) as total_params_billions,

                -- Model metrics
                COALESCE((
                    SELECT mc.{cst.MODEL_ID}
                    FROM model_counts mc
                    WHERE mc.{cst.HOTKEY} = tn.{cst.HOTKEY}
                    AND mc.interval = p.interval
                    ORDER BY mc.model_count DESC
                    LIMIT 1
                ), 'none') as modal_model,
                COUNT(DISTINCT t.{cst.MODEL_ID}) as unique_models,
                COUNT(DISTINCT t.{cst.DS}) as unique_datasets

            FROM periods p
            CROSS JOIN {cst.TASK_NODES_TABLE} tn
            JOIN {cst.TASKS_TABLE} t ON tn.{cst.TASK_ID} = t.{cst.TASK_ID}
            WHERE tn.{cst.HOTKEY} = ANY($1)
            AND tn.{cst.NETUID} = $2
            AND tn.{cst.QUALITY_SCORE} IS NOT NULL
            AND t.{cst.CREATED_AT} >= CASE
                WHEN p.interval = 'all' THEN '1970-01-01'::TIMESTAMP
                ELSE NOW() - p.interval::INTERVAL
            END
            GROUP BY tn.{cst.HOTKEY}, p.interval
        )
        SELECT * FROM aggregated_metrics
        """

        intervals = list(period_mapping.values())
        rows = await connection.fetch(query, hotkeys, NETUID, intervals)

        results = {hotkey: {} for hotkey in hotkeys}
        for hotkey in hotkeys:
            for period_name in period_mapping.keys():
                results[hotkey][period_name] = NodeStats(
                    quality_metrics=QualityMetrics(
                        avg_quality_score=0,
                        success_rate=0,
                        quality_rate=0,
                        total_count=0,
                        total_score=0,
                        total_success=0,
                        total_quality=0,
                    ),
                    workload_metrics=WorkloadMetrics(competition_hours=0, total_params_billions=0),
                    model_metrics=ModelMetrics(modal_model="none", unique_models=0, unique_datasets=0),
                )

        for row in rows:
            hotkey = row[cst.HOTKEY]
            interval = row["interval"]

            stats = NodeStats(
                quality_metrics=QualityMetrics(
                    avg_quality_score=row["avg_quality_score"],
                    success_rate=row["success_rate"],
                    quality_rate=row["quality_rate"],
                    total_count=row["total_count"],
                    total_score=row["total_score"],
                    total_success=row["total_success"],
                    total_quality=row["total_quality"],
                ),
                workload_metrics=WorkloadMetrics(
                    competition_hours=row["competition_hours"], total_params_billions=row["total_params_billions"]
                ),
                model_metrics=ModelMetrics(
                    modal_model=row["modal_model"] or "none",
                    unique_models=row["unique_models"],
                    unique_datasets=row["unique_datasets"],
                ),
            )

            period_name = next(name for name, value in period_mapping.items() if value == interval)
            results[hotkey][period_name] = stats

        return {hotkey: AllNodeStats(**stats) for hotkey, stats in results.items()}


async def get_task_winner(task_id: UUID, psql_db: PSQLDB) -> str | None:
    """Get the winner of a task based on the best quality score (lowest loss)."""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {cst.HOTKEY}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1
            AND {cst.NETUID} = $2
            AND {cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL AND {cst.TASK_NODE_QUALITY_SCORE} > 0
            ORDER BY {cst.TASK_NODE_QUALITY_SCORE} DESC  -- Higher score is better
            LIMIT 1
        """
        return await connection.fetchval(query, task_id, NETUID)


async def get_task_winners(task_ids: list[UUID], psql_db: PSQLDB) -> dict[str, str]:
    """Get winners for multiple tasks. Returns dict mapping task_id to winner hotkey."""
    if not task_ids:
        return {}

    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            WITH task_winners AS (
                SELECT 
                    {cst.TASK_ID}::text as task_id,
                    {cst.HOTKEY},
                    {cst.TASK_NODE_QUALITY_SCORE},
                    ROW_NUMBER() OVER (
                        PARTITION BY {cst.TASK_ID} 
                        ORDER BY {cst.TASK_NODE_QUALITY_SCORE} DESC  -- Higher score is better
                    ) as rn
                FROM {cst.TASK_NODES_TABLE}
                WHERE {cst.TASK_ID} = ANY($1)
                AND {cst.NETUID} = $2
                AND {cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL AND {cst.TASK_NODE_QUALITY_SCORE} > 0
            )
            SELECT task_id, {cst.HOTKEY}
            FROM task_winners
            WHERE rn = 1
        """

        rows = await connection.fetch(query, task_ids, NETUID)
        return {row["task_id"]: row[cst.HOTKEY] for row in rows}


async def get_task_scores_for_participants(task_id: UUID, hotkeys: list[str], psql_db: PSQLDB) -> dict[str, float]:
    """Get quality scores for specific participants in a task."""
    if not hotkeys:
        return {}

    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {cst.HOTKEY}, {cst.TASK_NODE_QUALITY_SCORE}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1
            AND {cst.NETUID} = $2
            AND {cst.HOTKEY} = ANY($3)
            AND {cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL
        """

        rows = await connection.fetch(query, task_id, NETUID, hotkeys)
        return {row[cst.HOTKEY]: row[cst.TASK_NODE_QUALITY_SCORE] for row in rows}
