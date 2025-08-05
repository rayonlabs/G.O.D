import asyncio

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.utility_models import GPUInfo
from trainer import constants as cst
from trainer.image_manager import start_training_task
from trainer.tasks import complete_task
from trainer.tasks import get_task
from trainer.tasks import load_task_history
from trainer.tasks import log_task
from trainer.tasks import start_task
from trainer.tasks import get_recent_tasks
from trainer.utils.misc import clone_repo
from trainer.utils.misc import get_gpu_info
from validator.core.constants import GET_GPU_AVAILABILITY_ENDPOINT
from validator.core.constants import PROXY_TRAINING_IMAGE_ENDPOINT
from validator.core.constants import TASK_DETAILS_ENDPOINT
from validator.core.constants import GET_RECENT_TASKS_ENDPOINT
from validator.utils.logging import get_logger


logger = get_logger(__name__)


load_task_history()

async def start_training(req: TrainerProxyRequest) -> JSONResponse:
    await start_task(req)

    try:
        local_repo_path = await asyncio.to_thread(
            clone_repo,
            repo_url=req.github_repo,
            parent_dir=cst.TEMP_REPO_PATH,
            branch=req.github_branch,
            commit_hash=req.github_commit_hash,
        )
    except RuntimeError as e:
        await log_task(req.training_data.task_id, req.hotkey, f"Failed to clone repo: {e}")
        await complete_task(req.training_data.task_id, req.hotkey, success=False)
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"Repo {req.github_repo} cloned to {local_repo_path}")

    asyncio.create_task(start_training_task(req, local_repo_path))

    return {"message": "Started Training!", "task_id": req.training_data.task_id}


async def get_available_gpus() -> list[GPUInfo]:
    gpu_info = await get_gpu_info()
    return gpu_info


async def get_task_details(task_id: str, hotkey: str) -> TrainerTaskLog:
    task = get_task(task_id, hotkey)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' and hotkey '{hotkey}' not found.")
    return task


async def get_recent_tasks_list(hours:int) -> list[TrainerTaskLog]:
    tasks = get_recent_tasks(hours)
    if not tasks:
        raise HTTPException(status_code=404, detail=f"Tasks not found in the last {hours} hours.")
    return tasks


def factory_router() -> APIRouter:
    router = APIRouter(tags=["Proxy Trainer"])
    router.add_api_route(PROXY_TRAINING_IMAGE_ENDPOINT, start_training, methods=["POST"])
    router.add_api_route(GET_GPU_AVAILABILITY_ENDPOINT, get_available_gpus, methods=["GET"])
    router.add_api_route(GET_RECENT_TASKS_ENDPOINT, get_recent_tasks_list, methods=["GET"])
    router.add_api_route(TASK_DETAILS_ENDPOINT, get_task_details, methods=["GET"])
    return router
