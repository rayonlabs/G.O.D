from datetime import datetime
from datetime import timedelta

from fastapi import APIRouter
from fastapi import Depends
from fiber.chain.chain_utils import query_substrate
from loguru import logger  # noqa

from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.core.models import Task
from validator.core.models import TaskWithHotkeyDetails
from validator.db.sql.auditing import get_recent_tasks
from validator.db.sql.auditing import get_recent_tasks_for_hotkey
from validator.db.sql.auditing import get_task_with_hotkey_details
from validator.db.sql.submissions_and_scoring import get_aggregate_scores_since


router = APIRouter(tags=["auditing"])


@router.get("/auditing/tasks")
async def audit_recent_tasks_endpoint(limit: int = 100, page: int = 1, config: Config = Depends(get_config)) -> list[Task]:
    return await get_recent_tasks(None, limit=limit, page=page, config=config)


@router.get("/auditing/tasks/hotkey/{hotkey}")
async def audit_recent_tasks_for_hotkey_endpoint(
    hotkey: str, limit: int = 100, page: int = 1, config: Config = Depends(get_config)
) -> list[TaskWithHotkeyDetails]:
    return await get_recent_tasks_for_hotkey(hotkey, limit=limit, page=page, config=config)


@router.get("/auditing/tasks/{task_id}")
async def audit_task_details_endpoint(task_id: str, config: Config = Depends(get_config)) -> TaskWithHotkeyDetails:
    logger.info(f"Getting task details for task {task_id}")
    return await get_task_with_hotkey_details(task_id, config)


@router.get("/auditing/scores")
async def get_scores_for_setting_weights(config: Config = Depends(get_config)) -> list[float]:
    """
    Get the scores I had when I last set weights, to prove I did it right
    """
    substrate, uid = query_substrate(
        config.substrate,
        "SubtensorModule",
        "Uids",
        [config.netuid, config.keypair.ss58_address],
        return_value=True,
    )
    config.substrate = substrate
    substrate, current_block = query_substrate(substrate, "System", "Number", [], return_value=True)
    substrate, last_updated_value = query_substrate(
        substrate, "SubtensorModule", "LastUpdate", [config.netuid], return_value=False
    )
    updated: int = current_block - last_updated_value[uid].value

    seconds_since_update = updated * 12  # 12 is block time in seconds

    time_when_last_set_weights = datetime.now() - timedelta(seconds=seconds_since_update)

    return await get_aggregate_scores_since(starttime=time_when_last_set_weights, psql_db=config.psql_db)


def factory_router():
    return router
