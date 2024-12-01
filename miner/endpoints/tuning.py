import os
from datetime import datetime
from datetime import timedelta

import yaml
from fastapi import Depends
from fastapi import HTTPException
from fastapi.routing import APIRouter
from fiber.logging_utils import get_logger
from fiber.miner.core.configuration import Config
from fiber.miner.dependencies import blacklist_low_stake
from fiber.miner.dependencies import get_config
from fiber.miner.dependencies import verify_request
from pydantic import ValidationError

import core.constants as cst
from core.models.payload_models import MinerTaskRequest
from core.models.payload_models import MinerTaskResponse
from core.models.payload_models import TrainRequest
from core.models.payload_models import TrainResponse
from core.models.utility_models import FileFormat
from core.utils import download_s3_file
from miner.config import WorkerConfig
from miner.dependencies import get_worker_config
from miner.logic.job_handler import create_job


logger = get_logger(__name__)

current_job_finish_time = None


async def tune_model(
    train_request: TrainRequest,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    global current_job_finish_time
    logger.info("Starting model tuning.")

    current_job_finish_time = datetime.now() + timedelta(
        hours=train_request.hours_to_complete
    )
    logger.info(f"Job received is {train_request}")

    try:
        logger.info(train_request.file_format)
        if train_request.file_format != FileFormat.HF:
            if train_request.file_format == FileFormat.S3:
                train_request.dataset = await download_s3_file(train_request.dataset)
                logger.info(train_request.dataset)
                train_request.file_format = FileFormat.JSON

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job(
        job_id=str(train_request.task_id),
        dataset=train_request.dataset,
        model=train_request.model,
        dataset_type=train_request.dataset_type,
        file_format=train_request.file_format,
    )
    logger.info(f"Created job {job}")
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "task_id": job.job_id}


# I think we need to be v careful that it's validators that are asking for this,
# is there a way to ensure we only reply to validators?
async def get_latest_model_submission(
    task_id: str,
) -> str:
    try:
        config_filename = f"{task_id}.yml"
        config_path = os.path.join(cst.CONFIG_DIR, config_filename)
        with open(config_path, "r") as file:
            config_data = yaml.safe_load(file)
            return config_data.get("hub_model_id", None)

    except FileNotFoundError as e:
        logger.error(f"No submission found for task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=404, detail=f"No model submission found for task {task_id}"
        )
    except Exception as e:
        logger.error(
            f"Error retrieving latest model submission for task {task_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving latest model submission: {str(e)}",
        )


async def task_offer(
    request: MinerTaskRequest,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
) -> MinerTaskResponse:
    try:
        logger.info("An offer has come through")
        # You will want to optimise this as a miner
        global current_job_finish_time
        current_time = datetime.now()
        if 'llama' not in request.model:
                return MinerTaskResponse(
                    message="I'm not yet optimised and only accept llama-type jobs", accepted=False
                )
        if (
            current_job_finish_time is None
            or current_time + timedelta(hours=1) > current_job_finish_time
        ):
            if request.hours_to_complete < 13:
                logger.info("Accepting the offer - ty snr")
                return MinerTaskResponse(message="Yes", accepted=True)
            else:
                logger.info("Rejecting offer")
                return MinerTaskResponse(
                    message="I only accept small jobs", accepted=False
                )
        else:
            return MinerTaskResponse(
                message=f"Currently busy with another job until {current_job_finish_time.isoformat()}",
                accepted=False,
            )

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in task_offer: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing task offer: {str(e)}"
        )


def factory_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/task_offer/",
        task_offer,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )
    router.add_api_route(
        "/get_latest_model_submission/{task_id}",
        get_latest_model_submission,
        tags=["Subnet"],
        methods=["GET"],
        response_model=str,
        summary="Get Latest Model Submission",
        description="Retrieve the latest model submission for a given task ID",
        dependencies=[Depends(blacklist_low_stake)],
    )
    router.add_api_route(
        "/start_training/",
        tune_model,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )
    return router
