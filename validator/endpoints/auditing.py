from fastapi import APIRouter
from fastapi import Depends
from loguru import logger  # noqa

from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.core.models import Task
from validator.core.models import TaskWithHotkeyDetails
from validator.db.sql.auditing import get_recent_tasks
from validator.db.sql.auditing import get_recent_tasks_for_hotkey
from validator.db.sql.auditing import get_task_with_hotkey_details


router = APIRouter(tags=["auditing"])


@router.get("/auditing/tasks")
async def audit_recent_tasks_endpoint(limit: int = 100, page: int = 1, config: Config = Depends(get_config)) -> list[Task]:
    return await get_recent_tasks(None, limit=limit, page=page, config=config)


@router.get("/auditing/tasks/{hotkey}")
async def audit_recent_tasks_for_hotkey_endpoint(
    hotkey: str, limit: int = 100, page: int = 1, config: Config = Depends(get_config)
) -> list[TaskWithHotkeyDetails]:
    return await get_recent_tasks_for_hotkey(hotkey, limit=limit, page=page, config=config)


@router.get("/auditing/tasks/{task_id}")
async def audit_task_details_endpoint(task_id: str, config: Config = Depends(get_config)) -> TaskWithHotkeyDetails:
    logger.info(f"Getting task details for task {task_id}")
    return await get_task_with_hotkey_details(task_id, config)


def factory_router():
    return router
