from datetime import datetime
from datetime import timedelta
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import Response

from core.models.payload_models import AllOfNodeResults
from core.models.payload_models import LeaderboardRow
from core.models.payload_models import NewTaskRequest
from core.models.payload_models import NewTaskResponse
from core.models.payload_models import NewTaskWithFixedDatasetsRequest
from core.models.payload_models import TaskDetails
from core.models.payload_models import TaskResultResponse
from core.models.utility_models import FileFormat
from core.models.utility_models import MinerTaskResult
from core.models.utility_models import TaskMinerResult
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.constants import MAX_CONCURRENT_JOBS
from validator.core.dependencies import get_api_key
from validator.core.dependencies import get_config
from validator.core.models import NetworkStats
from validator.core.models import RawTask
from validator.db.sql import submissions_and_scoring as submissions_and_scoring_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql.nodes import get_all_nodes
from validator.utils.logging import get_logger


logger = get_logger(__name__)


TASKS_CREATE_ENDPOINT = "/v1/tasks/create"
TASKS_CREATE_WITH_FIXED_DATASETS_ENDPOINT = "/v1/tasks/create_with_fixed_datasets"
GET_TASKS_BY_ACCOUNT_ENDPOINT = "/v1/tasks/account/{account_id}"
GET_TASK_DETAILS_ENDPOINT = "/v1/tasks/{task_id}"
GET_TASKS_RESULTS_ENDPOINT = "/v1/tasks/breakdown/{task_id}"
GET_NETWORK_STATUS = "/v1/network/status"
GET_NODE_RESULTS_ENDPOINT = "/v1/tasks/node_results/{hotkey}"
DELETE_TASK_ENDPOINT = "/v1/tasks/delete/{task_id}"
LEADERBOARD_ENDPOINT = "/v1/leaderboard"
COMPLETED_ORGANIC_TASKS_ENDPOINT = "/v1/tasks/organic/completed"
UPDATE_TRAINING_REPO_BACKUP_ENDPOINT = "/v1/tasks/{task_id}/training_repo_backup"


async def delete_task(
    task_id: UUID,
    config: Config = Depends(get_config),
) -> Response:
    task = await task_sql.get_task(task_id, config.psql_db)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    await task_sql.delete_task(task_id, config.psql_db)
    return Response(success=True)


async def create_task(
    request: NewTaskRequest,
    config: Config = Depends(get_config),
) -> NewTaskResponse:
    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=request.hours_to_complete)

    # if there are any queued jobs that are organic we can't accept any more to avoid overloading the network
    #    queued_tasks = await task_sql.get_tasks_with_status(TaskStatus.DELAYED, config.psql_db, include_not_ready_tasks=True)
    #    if len(queued_tasks) > 0:
    #        logger.info("We already have some queued organic jobs, we can't a accept any more")
    #        return NewTaskResponse(success=False, task_id=None)

    task = RawTask(
        model_id=request.model_repo,
        ds_id=request.ds_repo,
        file_format=request.file_format,
        field_system=request.field_system,
        field_instruction=request.field_instruction,
        field_input=request.field_input,
        field_output=request.field_output,
        format=request.format,
        is_organic=True,
        no_input_format=request.no_input_format,
        status=TaskStatus.PENDING,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=request.hours_to_complete,
        account_id=request.account_id,
    )

    task = await task_sql.add_task(task, config.psql_db)

    logger.info(task.task_id)
    return NewTaskResponse(success=True, task_id=task.task_id, created_at=task.created_at, account_id=task.account_id)


async def create_task_with_fixed_datasets(
    request: NewTaskWithFixedDatasetsRequest,
    config: Config = Depends(get_config),
) -> NewTaskResponse:
    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=request.hours_to_complete)

    task = RawTask(
        model_id=request.model_repo,
        ds_id=request.ds_repo or request.training_data,
        file_format=request.file_format if request.ds_repo else FileFormat.S3,
        field_system=request.field_system,
        field_instruction=request.field_instruction,
        field_input=request.field_input,
        field_output=request.field_output,
        format=request.format,
        is_organic=True,
        no_input_format=request.no_input_format,
        status=TaskStatus.LOOKING_FOR_NODES,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=request.hours_to_complete,
        account_id=request.account_id,
    )

    task = await task_sql.add_task(task, config.psql_db)
    task.training_data = request.training_data
    task.synthetic_data = request.synthetic_data
    task.test_data = request.test_data
    await task_sql.update_task(task, config.psql_db)

    logger.info(task.task_id)
    return NewTaskResponse(success=True, task_id=task.task_id, created_at=task.created_at, account_id=task.account_id)


async def get_node_results(
    hotkey: str,
    config: Config = Depends(get_config),
) -> AllOfNodeResults:
    try:
        logger.info(f"The hotkey is {hotkey}")
        miner_results = [
            TaskMinerResult(**result)
            for result in await submissions_and_scoring_sql.get_all_scores_for_hotkey(hotkey, config.psql_db)
        ]
    except Exception as e:
        logger.info(e)
        raise HTTPException(status_code=404, detail="Hotkey not found")
    return AllOfNodeResults(hotkey=hotkey, task_results=miner_results)


async def get_task_details_by_account(
    account_id: UUID,
    limit: int = 100,
    page: int = 1,
    config: Config = Depends(get_config),
) -> list[TaskDetails]:
    offset = (page - 1) * limit
    tasks = await task_sql.get_tasks_by_account_id(config.psql_db, account_id, limit, offset)

    task_status_responses = [
        TaskDetails(
            id=task.task_id,
            account_id=task.account_id,
            status=task.status,
            base_model_repository=task.model_id,
            ds_repo=task.ds_id,
            field_input=task.field_input,
            field_system=task.field_system,
            field_instruction=task.field_instruction,
            field_output=task.field_output,
            format=task.format,
            no_input_format=task.no_input_format,
            system_format=task.system_format,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.termination_at,
            hours_to_complete=task.hours_to_complete,
            trained_model_repository=task.trained_model_repository,
        )
        for task in tasks
    ]

    return task_status_responses


async def get_task_details(
    task_id: UUID,
    config: Config = Depends(get_config),
) -> TaskDetails:
    task = await task_sql.get_task_by_id(task_id, config.psql_db)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    return TaskDetails(
        id=task_id,
        account_id=task.account_id,
        status=task.status,
        base_model_repository=task.model_id,
        ds_repo=task.ds_id,
        field_input=task.field_input,
        field_system=task.field_system,
        field_instruction=task.field_instruction,
        field_output=task.field_output,
        format=task.format,
        no_input_format=task.no_input_format,
        system_format=task.system_format,
        created_at=task.created_at,
        started_at=task.started_at,
        finished_at=task.termination_at,
        hours_to_complete=task.hours_to_complete,
        trained_model_repository=task.trained_model_repository,
    )


async def get_miner_breakdown(
    task_id: UUID,
    config: Config = Depends(get_config),
) -> TaskResultResponse:
    try:
        results = await submissions_and_scoring_sql.get_all_scores_and_losses_for_task(task_id, config.psql_db)
        miner_results = [MinerTaskResult(**result) for result in results]
    except Exception as e:
        logger.info(e)
        raise HTTPException(status_code=404, detail="Task not found.")
    return TaskResultResponse(id=task_id, miner_results=miner_results)


async def get_leaderboard(
    config: Config = Depends(get_config),
) -> list[LeaderboardRow]:
    nodes = await get_all_nodes(config.psql_db)
    leaderboard_rows = []

    for node in nodes:
        logger.info(f"Trying node {node}")
        try:
            node_stats = await submissions_and_scoring_sql.get_all_node_stats(node.hotkey, config.psql_db)
            leaderboard_rows.append(LeaderboardRow(hotkey=node.hotkey, stats=node_stats))
        except Exception as e:
            logger.error(f"Error processing scores for hotkey {node.hotkey}: {e}")
            continue
    return leaderboard_rows


async def get_network_status(
    config: Config = Depends(get_config),
) -> NetworkStats:
    try:
        logger.debug("IN get network status")
        current_task_stats = await task_sql.get_current_task_stats(config.psql_db)
        if current_task_stats.number_of_jobs_training >= MAX_CONCURRENT_JOBS:
            current_task_stats.job_can_be_made = False
        return current_task_stats
    except Exception as e:
        logger.info(f"There was an issue with getting training status {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_completed_organic_tasks(
    hours: int = Query(default=5, description="Number of hours to look back for completed organic tasks", ge=1),
    config: Config = Depends(get_config),
) -> list[TaskDetails]:
    """Get all completed organic tasks from the last N hours"""
    tasks = await task_sql.get_completed_organic_tasks(config.psql_db, hours)

    task_details = [
        TaskDetails(
            id=task.task_id,
            account_id=task.account_id,
            status=task.status,
            base_model_repository=task.model_id,
            ds_repo=task.ds_id,
            field_input=task.field_input,
            field_system=task.field_system,
            field_instruction=task.field_instruction,
            field_output=task.field_output,
            format=task.format,
            no_input_format=task.no_input_format,
            system_format=task.system_format,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.termination_at,
            hours_to_complete=task.hours_to_complete,
            trained_model_repository=task.trained_model_repository,
        )
        for task in tasks
    ]

    return task_details


async def update_training_repo_backup(
    task_id: UUID,
    training_repo: str,
    config: Config = Depends(get_config),
) -> Response:
    task = await task_sql.get_task(task_id, config.psql_db)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    task.training_repo_backup = training_repo
    await task_sql.update_task(task, config.psql_db)
    return Response(status_code=200)


def factory_router() -> APIRouter:
    router = APIRouter(tags=["Gradients On Demand"], dependencies=[Depends(get_api_key)])
    router.add_api_route(TASKS_CREATE_ENDPOINT, create_task, methods=["POST"])
    router.add_api_route(TASKS_CREATE_WITH_FIXED_DATASETS_ENDPOINT, create_task_with_fixed_datasets, methods=["POST"])
    router.add_api_route(GET_TASK_DETAILS_ENDPOINT, get_task_details, methods=["GET"])
    router.add_api_route(DELETE_TASK_ENDPOINT, delete_task, methods=["DELETE"])
    router.add_api_route(GET_TASKS_RESULTS_ENDPOINT, get_miner_breakdown, methods=["GET"])
    router.add_api_route(GET_NODE_RESULTS_ENDPOINT, get_node_results, methods=["GET"])
    router.add_api_route(GET_TASKS_BY_ACCOUNT_ENDPOINT, get_task_details_by_account, methods=["GET"])
    router.add_api_route(LEADERBOARD_ENDPOINT, get_leaderboard, methods=["GET"])
    router.add_api_route(GET_NETWORK_STATUS, get_network_status, methods=["GET"])
    router.add_api_route(COMPLETED_ORGANIC_TASKS_ENDPOINT, get_completed_organic_tasks, methods=["GET"])
    router.add_api_route(UPDATE_TRAINING_REPO_BACKUP_ENDPOINT, update_training_repo_backup, methods=["PUT"])
    return router
