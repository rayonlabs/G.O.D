from datetime import datetime
from datetime import timedelta
from typing import List
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Response
from fiber.logging_utils import get_logger

from core.models.payload_models import AllOfNodeResults
from core.models.payload_models import MinerTaskResult
from core.models.payload_models import NewTaskRequest
from core.models.payload_models import NewTaskResponse
from core.models.payload_models import TaskMinerResult
from core.models.payload_models import TaskResultResponse
from core.models.payload_models import TaskStatusResponse
from core.models.payload_models import WinningSubmission
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.dependencies import get_api_key
from validator.core.dependencies import get_config
from validator.core.models import LeaderboardRow
from validator.core.models import Task
from validator.db.sql import submissions_and_scoring as submissions_and_scoring_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql.nodes import get_all_nodes


logger = get_logger(__name__)


async def delete_task(
    task_id: UUID,
    user_id: str = Depends(get_api_key),
    config: Config = Depends(get_config),
) -> NewTaskResponse:
    task = await task_sql.get_task(task_id, config.psql_db)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    if task.user_id != user_id:
        raise HTTPException(
            status_code=403, detail="Not authorized to delete this task.")

    await task_sql.delete_task(task_id, config.psql_db)
    return Response(success=True)


async def get_tasks(
    config: Config = Depends(get_config),
) -> List[TaskStatusResponse]:
    tasks_with_miners = await task_sql.get_tasks_with_miners(config.psql_db)
    task_status_responses = []

    for task in tasks_with_miners:
        miners = await task_sql.get_miners_for_task(task["task_id"], config.psql_db)
        winning_submission_data = await task_sql.get_winning_submissions_for_task(task["task_id"], config.psql_db)
        winning_submission = None
        if winning_submission_data:
            winning_submission_data = winning_submission_data[0]
            winning_submission = WinningSubmission(
                hotkey=winning_submission_data["hotkey"],
                score=winning_submission_data["quality_score"],
                model_repo=winning_submission_data["repo"],
            )

        task_status_responses.append(
            TaskStatusResponse(
                success=True,
                id=task["task_id"],
                status=task["status"],
                model_repo=task.get("model_id"),
                ds_repo=task.get("ds_id"),
                input_col=task.get("input"),
                system_col=task.get("system"),
                instruction_col=task.get("instruction"),
                output_col=task.get("output"),
                format_col=task.get("format"),
                no_input_format_col=task.get("no_input_format"),
                miners=[{"hotkey": miner.hotkey, "trust": miner.trust}
                        for miner in miners],
                dataset=task.get("ds_id"),
                started=str(task["started_timestamp"]),
                end=str(task["end_timestamp"]),
                created=str(task["created_timestamp"]),
                hours_to_complete=task.get("hours_to_complete"),
                winning_submission=winning_submission,
            )
        )

    return task_status_responses


async def create_task(
    request: NewTaskRequest,
    config: Config = Depends(get_config),
    api_key: str = Depends(get_api_key),
) -> NewTaskResponse:
    logger.info(f"The request coming in is {request}")
    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=request.hours_to_complete)

    # if there are any queued jobs that are organic we can't accept any more to avoid overloading the network
    queued_tasks = await task_sql.get_tasks_with_status(TaskStatus.DELAYED, config.psql_db, include_not_ready_tasks=True)
    if len(queued_tasks) > 0:
        logger.info(
            "We already have some queued organic jobs, we can't a accept any more")
        return NewTaskResponse(success=False, task_id=None)

    logger.info(f"The request coming in {request}")
    task = Task(
        model_id=request.model_repo,
        ds_id=request.ds_repo,
        system=request.system_col,
        instruction=request.instruction_col,
        input=request.input_col,
        output=request.output_col,
        format=request.format_col,
        is_organic=True,
        no_input_format=request.no_input_format_col,
        status=TaskStatus.PENDING,
        end_timestamp=end_timestamp,
        hours_to_complete=request.hours_to_complete,
    )

    logger.info(f"The Task is {task}")

    task = await task_sql.add_task(task, config.psql_db)

    logger.info(task.task_id)
    return NewTaskResponse(success=True, task_id=task.task_id)


async def get_task_results(
    task_id: UUID,
    config: Config = Depends(get_config),
    api_key: str = Depends(get_api_key),
) -> TaskResultResponse:
    try:
        scores = await submissions_and_scoring_sql.get_all_quality_scores_for_task(task_id, config.psql_db)
        miner_results = [MinerTaskResult(
            hotkey=hotkey, quality_score=scores[hotkey]) for hotkey in scores]
    except Exception as e:
        logger.info(e)
        raise HTTPException(status_code=404, detail="Task not found.")
    return TaskResultResponse(success=True, id=task_id, miner_results=miner_results)


async def get_node_results(
    hotkey: str,
    config: Config = Depends(get_config),
) -> AllOfNodeResults:
    try:
        miner_results = [
            TaskMinerResult(**result)
            for result in await submissions_and_scoring_sql.get_all_scores_for_hotkey(hotkey, config.psql_db)
        ]
    except Exception as e:
        logger.info(e)
        raise HTTPException(status_code=404, detail="Hotkey not found")
    return AllOfNodeResults(success=True, hotkey=hotkey, task_results=miner_results)


async def get_task_status(
    task_id: UUID,
    config: Config = Depends(get_config),
    api_key: str = Depends(get_api_key),
) -> TaskStatusResponse:
    task = await task_sql.get_task(task_id, config.psql_db)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    miners = await task_sql.get_miners_for_task(task_id, config.psql_db)
    logger.info(miners)

    winning_submission_data = await task_sql.get_winning_submissions_for_task(task_id, config.psql_db)
    winning_submission = None
    if winning_submission_data:
        winning_submission_data = winning_submission_data[0]
        winning_submission = WinningSubmission(
            hotkey=winning_submission_data["hotkey"],
            score=winning_submission_data["quality_score"],
            model_repo=winning_submission_data["repo"],
        )

    return TaskStatusResponse(
        success=True,
        id=task_id,
        status=task.status,
        model_repo=task.model_id,
        ds_repo=task.ds_id,
        input_col=task.input,
        system_col=task.system,
        instruction_col=task.instruction,
        output_col=task.output,
        format_col=task.format,
        no_input_format_col=task.no_input_format,
        started=str(task.started_timestamp),
        miners=[{"hotkey": miner.hotkey, "trust": miner.trust}
                for miner in miners],
        end=str(task.end_timestamp),
        created=str(task.created_timestamp),
        hours_to_complete=task.hours_to_complete,
        winning_submission=winning_submission,
    )


async def get_leaderboard(
    config: Config = Depends(get_config),
) -> List[LeaderboardRow]:
    nodes = await get_all_nodes(config.psql_db)
    leaderboard_rows = []

    for node in nodes:
        logger.info(f"Trying node {node}")
        try:
            node_stats = await submissions_and_scoring_sql.get_all_node_stats(node.hotkey, config.psql_db)
            leaderboard_rows.append(LeaderboardRow(
                hotkey=node.hotkey, stats=node_stats))
        except Exception as e:
            logger.error(
                f"Error processing scores for hotkey {node.hotkey}: {e}")
            continue
    return leaderboard_rows


def factory_router() -> APIRouter:
    router = APIRouter(dependencies=[Depends(get_api_key)])

    router.add_api_route(
        "/v1/tasks/create",
        create_task,
        response_model=NewTaskResponse,
        tags=["Training"],
        methods=["POST"],
    )

    router.add_api_route(
        "/v1/tasks/{task_id}",
        get_task_status,
        response_model=TaskStatusResponse,
        tags=["Training"],
        methods=["GET"],
    )

    router.add_api_route(
        "/v1/tasks/delete/{task_id}",
        delete_task,
        response_model=NewTaskResponse,
        tags=["Training"],
        methods=["DELETE"],
    )

    router.add_api_route(
        "/v1/tasks/results/{task_id}",
        get_task_results,
        response_model=TaskResultResponse,
        tags=["Training"],
        methods=["GET"],
    )

    router.add_api_route(
        "/v1/tasks/node_results/{hotkey}",
        get_node_results,
        response_model=AllOfNodeResults,
        tags=["Training"],  # ? why do we have these tags everywhere TT?
        methods=["GET"],
    )
    router.add_api_route(
        "/v1/tasks",
        get_tasks,
        response_model=List[TaskStatusResponse],
        tags=["Training"],
        methods=["GET"],
    )
    router.add_api_route(
        "/v1/leaderboard",
        get_leaderboard,
        response_model=list[LeaderboardRow],
        tags=["Training"],
        methods=["GET"],
    )
    return router
