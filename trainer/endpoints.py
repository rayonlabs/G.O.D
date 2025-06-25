from fastapi import FastAPI, HTTPException, Response
from fastapi import APIRouter, Depends
from pydantic import BaseModel
import asyncio

from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from trainer import constants as cst
from validator.core.dependencies import get_api_key
from validator.utils.logging import get_logger
from trainer.image_manager import start_training_task
from trainer.utils.misc import clone_repo
from trainer.utils.misc import get_available_gpu_types
from trainer.tasks import start_task, load_task_history, get_task


logger = get_logger(__name__)


PROXY_TRAINING_IMAGE_ENDPOINT = "/v1/trainer/start_training"
GET_GPU_AVAILABILITY_ENDPOINT = "/v1/trainer/get_gpu_availability"
TASK_DETAILS_ENDPOINT= "/v1/trainer/{task_id}"

load_task_history()


async def start_training(req: TrainerProxyRequest) -> Response:
    local_repo_path = clone_repo(req.github_repo, cst.TEMP_REPO_PATH)
    req.local_repo_path = local_repo_path
    logger.info(f"Repo {req.github_repo} cloned to {local_repo_path}")

    asyncio.create_task(start_training_task(req))
    start_task(req)
    
    return {"message": "Started Training!", "task_id": req.training_data.task_id}


async def get_available_gpus():
    gpu_types = await get_available_gpu_types()
    return {gpu_type.value: count for gpu_type, count in gpu_types.items()}


async def get_task_details(task_id: str) -> TrainerTaskLog:
    return get_task(task_id)


def factory_router() -> APIRouter:
    router = APIRouter(tags=["Proxy Trainer"], dependencies=[Depends(get_api_key)])
    router.add_api_route(PROXY_TRAINING_IMAGE_ENDPOINT, start_training, methods=["POST"])
    router.add_api_route(GET_GPU_AVAILABILITY_ENDPOINT, get_available_gpus, methods=["GET"])
    router.add_api_route(TASK_DETAILS_ENDPOINT, get_task_details, methods=["GET"])
    return router
