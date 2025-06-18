from fastapi import FastAPI, HTTPException, Response
from fastapi import APIRouter, Depends
from pydantic import BaseModel
import asyncio

from core.models.payload_models import TrainerProxyRequestImage
from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.core.dependencies import get_api_key
from validator.utils.logging import get_logger
from validator.trainer.image_manager import build_docker_image, run_trainer_container


logger = get_logger(__name__)


PROXY_TRAINING_IMAGE_ENDPOINT = "/v1/trainer/start_proxy_training_image"


async def build_and_run_trainer(
    req: TrainerProxyRequestImage,
    ) -> Response:
    try:
        tag = await build_docker_image(
            dockerfile_path=req.dockerfile_path,
            context_path=req.context_path,
        )

        await run_trainer_container(
            task_id=req.task_id,
            tag=tag,
            model=req.model,
            dataset_zip=req.dataset_zip,
            model_type=req.model_type,
            hours_to_complete=req.hours_to_complete
        )

        return Response(content="Successfully trained")

    except Exception as e:
        logger.exception("Error during image build or container start.")
        raise HTTPException(status_code=500, detail=str(e))

def factory_router() -> APIRouter:
    router = APIRouter(tags=["Proxy Trainer"], dependencies=[Depends(get_api_key)])
    router.add_api_route(PROXY_TRAINING_IMAGE_ENDPOINT, build_and_run_trainer, methods=["POST"])
    return router
