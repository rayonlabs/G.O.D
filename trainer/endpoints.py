from fastapi import FastAPI, HTTPException, Response
from fastapi import APIRouter, Depends
from pydantic import BaseModel
import asyncio

from core.models.payload_models import TrainerProxyRequestImage
from trainer import constants as cst
from validator.core.dependencies import get_api_key
from validator.utils.logging import get_logger
from trainer.queue import TrainerQueue
from trainer.utils import clone_repo


logger = get_logger(__name__)


PROXY_TRAINING_IMAGE_ENDPOINT = "/v1/trainer/start_proxy_training_image"

trainer_queue = TrainerQueue()


async def build_and_run_trainer(req: TrainerProxyRequestImage) -> Response:
    local_repo_path = clone_repo(req.github_repo, cst.TEMP_REPO_PATH, branch="proxy-trainer")
    logger.info(f"Repo {req.github_repo} cloned to {local_repo_path}")
    req.local_repo_path = local_repo_path

    await trainer_queue.submit(req)
    
    return {"message": "Training job enqueued.", "task_id": req.task_id}


def factory_router() -> APIRouter:
    router = APIRouter(tags=["Proxy Trainer"], dependencies=[Depends(get_api_key)])
    router.add_api_route(PROXY_TRAINING_IMAGE_ENDPOINT, build_and_run_trainer, methods=["POST"])
    return router
