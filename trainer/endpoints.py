from fastapi import FastAPI, HTTPException, Response
from fastapi import APIRouter, Depends
from pydantic import BaseModel
import asyncio

from core.models.payload_models import TrainerProxyRequestImage
from core.models.payload_models import TrainerProxyJobImage
from core.models.payload_models import TrainerProxyTaskOffer
from core.models.payload_models import TrainerProxyResponse
from trainer import constants as cst
from validator.core.dependencies import get_api_key
from validator.utils.logging import get_logger
from trainer.queue import TrainerQueue
from trainer.utils import clone_repo
from trainer.gpupool import get_available_gpu_types


logger = get_logger(__name__)


PROXY_TRAINING_IMAGE_ENDPOINT = "/v1/trainer/queue_training_job"
GET_GPU_AVAILABILITY_ENDPOINT = "/v1/trainer/get_gpu_availability"
TASK_OFFER_ENDPOINT = "/v1/trainer/task_offer"

trainer_queue = TrainerQueue()


async def queue_training_job(req: TrainerProxyRequestImage) -> Response:
    training_job = TrainerProxyJobImage(**req.dict())
    local_repo_path = clone_repo(req.github_repo, cst.TEMP_REPO_PATH)
    training_job.local_repo_path = local_repo_path
    logger.info(f"Repo {req.github_repo} cloned to {local_repo_path}")

    await trainer_queue.submit(training_job)
    
    return {"message": "Training job enqueued.", "task_id": req.task_id}


async def task_offer(offer: TrainerProxyTaskOffer):
    available_gpu_types = await get_available_gpu_types()
    available = available_gpu_types.get(offer.gpu_type)

    if available is None or available < offer.num_gpus:
        return TrainerProxyResponse(
            message=f"Not enough availability",
            accepted=False
        )

    return TrainerProxyResponse(
        message="I can do this",
        accepted=True
    )


async def get_available_gpus():
    gpu_types = await get_available_gpu_types()
    return {gpu_type.value: count for gpu_type, count in gpu_types.items()}


def factory_router() -> APIRouter:
    router = APIRouter(tags=["Proxy Trainer"], dependencies=[Depends(get_api_key)])
    router.add_api_route(PROXY_TRAINING_IMAGE_ENDPOINT, queue_training_job, methods=["POST"])
    router.add_api_route(TASK_OFFER_ENDPOINT, task_offer, methods=["POST"])
    router.add_api_route(GET_GPU_AVAILABILITY_ENDPOINT, get_available_gpus, methods=["GET"])
    return router
