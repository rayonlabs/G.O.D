from uuid import UUID

import validator.db.constants as cst
from validator.db.database import PSQLDB
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def get_random_historical_task_by_type(
    task_type: str,
    start_date: str,
    end_date: str,
    min_successful_scores: int,
    psql_db: PSQLDB,
    exclude_task_ids: list[UUID] = None,
) -> UUID | None:
    """
    Get a random historical task of a specific type that has at least the minimum number of valid test loss values.

    Args:
        task_type: The type of task to fetch (string value like 'InstructTextTask', 'DpoTask', 'GrpoTask', or 'ImageTask')
        start_date: Start date for task creation (format: YYYY-MM-DD)
        end_date: End date for task creation (format: YYYY-MM-DD)
        min_successful_scores: Minimum number of valid test loss values required
        psql_db: Database connection
        exclude_task_ids: List of task IDs to exclude from selection (for getting unique tasks)

    Returns:
        Task ID of a random qualifying task, or None if no tasks found
    """
    async with await psql_db.connection() as connection:
        if exclude_task_ids:
            exclude_clause = f"AND t.{cst.TASK_ID} != ALL($5::uuid[])"
            params = [task_type, start_date, end_date, min_successful_scores, exclude_task_ids]
        else:
            exclude_clause = ""
            params = [task_type, start_date, end_date, min_successful_scores]

        query = f"""
            WITH eligible_tasks AS (
                SELECT
                    t.{cst.TASK_ID},
                    COUNT(tn.{cst.TEST_LOSS}) as successful_scores
                FROM {cst.TASKS_TABLE} t
                JOIN {cst.TASK_NODES_TABLE} tn ON t.{cst.TASK_ID} = tn.{cst.TASK_ID}
                WHERE t.{cst.TASK_TYPE} = $1
                AND t.{cst.CREATED_AT} >= $2::timestamptz
                AND t.{cst.CREATED_AT} < $3::timestamptz
                AND t.{cst.IS_ORGANIC} = FALSE
                AND tn.{cst.TEST_LOSS} IS NOT NULL
                AND NOT (tn.{cst.TEST_LOSS} = 'NaN'::numeric)
                {exclude_clause}
                GROUP BY t.{cst.TASK_ID}
                HAVING COUNT(tn.{cst.TEST_LOSS}) >= $4
            )
            SELECT {cst.TASK_ID}
            FROM eligible_tasks
            ORDER BY RANDOM()
            LIMIT 1
        """

        result = await connection.fetchval(query, *params)

        if result:
            logger.info(f"Found random historical synthetic {task_type} task: {result}")
        else:
            logger.warning(
                f"No historical synthetic {task_type} tasks found between {start_date} and {end_date} "
                f"with at least {min_successful_scores} valid test loss values"
            )

        return result
