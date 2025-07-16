from fastapi.responses import JSONResponse
from fastapi import APIRouter
import asyncio
from fastapi import HTTPException

from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from trainer import constants as cst
from validator.utils.logging import get_logger
from trainer.image_manager import start_training_task
from trainer.utils.misc import clone_repo
from trainer.utils.misc import get_gpu_info
from trainer.tasks import start_task, load_task_history, get_task, log_task, complete_task
from core.models.utility_models import GPUInfo


logger = get_logger(__name__)


PROXY_TRAINING_IMAGE_ENDPOINT = "/v1/trainer/start_training"
GET_GPU_AVAILABILITY_ENDPOINT = "/v1/trainer/get_gpu_availability"
TASK_DETAILS_ENDPOINT= "/v1/trainer/{task_id}"

load_task_history()


async def start_training(req: TrainerProxyRequest) -> JSONResponse:
    start_task(req)
    try:
        local_repo_path = clone_repo(
            repo_url=req.github_repo,
            parent_dir=cst.TEMP_REPO_PATH,
            branch=req.github_branch,
            commit_hash=req.github_commit_hash
        )
    except RuntimeError as e:
        log_task(req.training_data.task_id, req.hotkey, f"Failed to clone repo: {e}")
        complete_task(req.training_data.task_id, req.hotkey, success=False)
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


def factory_router() -> APIRouter:
    router = APIRouter(tags=["Proxy Trainer"])
    router.add_api_route(PROXY_TRAINING_IMAGE_ENDPOINT, start_training, methods=["POST"])
    router.add_api_route(GET_GPU_AVAILABILITY_ENDPOINT, get_available_gpus, methods=["GET"])
    router.add_api_route(TASK_DETAILS_ENDPOINT, get_task_details, methods=["GET"])
    return router
