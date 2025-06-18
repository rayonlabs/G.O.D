from fastapi import FastAPI, HTTPException, Response
from fastapi import APIRouter, Depends
from pydantic import BaseModel
import asyncio

from core.models.payload_models import TrainerProxyRequestImage
from trainer import constants as cst
from validator.core.dependencies import get_api_key
from validator.utils.logging import get_logger
from trainer.image_manager import build_docker_image, run_trainer_container
from trainer.utils import clone_repo


logger = get_logger(__name__)


PROXY_TRAINING_IMAGE_ENDPOINT = "/v1/trainer/start_proxy_training_image"


async def build_and_run_trainer(
    req: TrainerProxyRequestImage,
    ) -> Response:
    try:
        local_repo_path = clone_repo(req.github_repo, cst.TEMP_REPO_PATH, branch="proxy-trainer")

        logger.info(f"Repo {req.github_repo} cloned to {local_repo_path}")

        tag = await build_docker_image(
            dockerfile_path=f"{local_repo_path}/{cst.DEFAULT_IMAGE_DOCKERFILE_PATH}",
            context_path=local_repo_path,
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
