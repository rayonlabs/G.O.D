from fastapi import APIRouter
from fastapi import Depends

from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.core.models import Task
from validator.core.models import TaskWithHotkeyDetails
from validator.db.sql.auditing import get_recent_tasks
from validator.db.sql.auditing import get_task_details


router = APIRouter()


@router.get("/auditing/tasks")
async def get_recent_tasks_endpoint(
    hotkey: str | None = None, limit: int = 100, page: int = 1, config: Config = Depends(get_config)
) -> list[Task]:
    return await get_recent_tasks(hotkeys=[hotkey] if hotkey is not None else None, limit=limit, page=page, config=config)


@router.get("/auditing/tasks/{task_id}")
async def get_task_details_endpoint(task_id: str, config: Config = Depends(get_config)) -> TaskWithHotkeyDetails:
    return await get_task_details(task_id, config.psql_db)


def factory_router():
    router = APIRouter()

    router.get("/auditing/tasks")(get_recent_tasks_endpoint)
    router.get("/auditing/tasks/{task_id}")(get_task_details_endpoint)

    return router
