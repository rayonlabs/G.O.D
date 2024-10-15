from fastapi import Depends
from fastapi import HTTPException
from fastapi.routing import APIRouter
from fiber.logging_utils import get_logger

from fiber.miner.core.configuration import Config
from fiber.miner.dependencies import get_config

from core.models.payload_models import MinerTaskRequst
from core.models.payload_models import MinerTaskResponse
from core.models.payload_models import TrainRequest

from core.models.payload_models import TrainResponse
from core.models.utility_models import FileFormat
from core.utils import validate_dataset
from miner.config import WorkerConfig
from miner.dependencies import get_worker_config
from miner.logic.job_handler import create_job


logger = get_logger(__name__)

async def tune_model(
    decrypted_payload: TrainRequest,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    if not decrypted_payload.dataset or not decrypted_payload.model:
        raise HTTPException(status_code=400, detail="Dataset and model are required.")

    try:
        if decrypted_payload.file_format != FileFormat.HF:
            is_valid = validate_dataset(
                decrypted_payload.dataset,
                decrypted_payload.dataset_type,
                decrypted_payload.file_format,
            )
            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid dataset format for {decrypted_payload.dataset_type} dataset type.",
                )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job(
        job_id= decrypted_payload.job_id,
        dataset=decrypted_payload.dataset,
        model=decrypted_payload.model,
        dataset_type=decrypted_payload.dataset_type,
        file_format=decrypted_payload.file_format,
    )
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "job_id": job.job_id}


async def task_offer(request: MinerTaskRequst) -> MinerTaskResponse:
    import random
    accepted = random.random() > 0.2
    return MinerTaskResponse(message="Task offer response", accepted=accepted)

def factory_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/task_offer/",
        task_offer,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse
    )
    router.add_api_route(
        "/start_training/",
        tune_model,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
    )
    return router
