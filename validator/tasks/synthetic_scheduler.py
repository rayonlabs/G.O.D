import asyncio
import random
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import AsyncGenerator

from substrateinterface import Keypair

import validator.core.constants as cst
from core.models.payload_models import DatasetColumnsResponse
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.models import RawTask
from validator.db.sql.tasks import add_task
from validator.db.sql.tasks import get_tasks_with_status
from validator.utils.call_endpoint import call_content_service
from validator.utils.logging import get_logger


logger = get_logger(name="task synth")


async def _get_models(keypair: Keypair) -> AsyncGenerator[str, None]:
    response = await call_content_service(cst.GET_RANDOM_MODELS_ENDPOINT, keypair)
    if not isinstance(response, list):
        raise TypeError("Expected a list of responses from GET_ALL_MODELS_ENDPOINT")
    models: list[dict[str, Any]] = response
    TEMP_MODEL_FAMILIES_ACCEPTED = ["qwen", "llama", "falcon", "mistral", "gemma", "gemini", "phi"]
    model_ids = [
        model.get(cst.GET_ALL_MODELS_ID, "")
        for model in models
        if any(family in model.get(cst.GET_ALL_MODELS_ID, "").lower() for family in TEMP_MODEL_FAMILIES_ACCEPTED)
    ]
    random.shuffle(model_ids)
    for model_id in model_ids:
        yield model_id


async def _get_datasets(keypair: Keypair) -> AsyncGenerator[str, None]:
    response = await call_content_service(cst.GET_RANDOM_DATASETS_ENDPOINT, keypair)
    if not isinstance(response, list):
        raise TypeError("Expected a list of responses from GET_ALL_DATASETS_ENDPOINT")
    datasets: list[dict[str, Any]] = response
    dataset_ids = [ds.get(cst.GET_ALL_DATASETS_ID, "") for ds in datasets]
    random.shuffle(dataset_ids)
    for ds_id in dataset_ids:
        yield ds_id


async def _get_columns_for_dataset(dataset_id: str, keypair: Keypair) -> DatasetColumnsResponse:
    url = cst.GET_COLUMNS_FOR_DATASET_ENDPOINT.replace("{dataset}", dataset_id)
    response = await call_content_service(url, keypair)
    if not isinstance(response, dict):
        raise TypeError(f"Expected dictionary response, got {type(response)}")
    try:
        columns = DatasetColumnsResponse.model_validate(response)
    except Exception as exc:
        logger.error(f"The get columns for dataset endpoint should return a DatasetColumnsResponse type: {exc}")
        raise TypeError(f"The get columns for dataset endpoint should return a DatasetColumnsResponse type: {exc}")
    return columns


async def _create_synthetic_task(
    config: Config,
    models: AsyncGenerator[str, None],
    datasets: AsyncGenerator[str, None],
):
    number_of_hours = random.randint(cst.MIN_COMPETITION_HOURS, cst.MAX_COMPETITION_HOURS)
    model_id = await anext(models)
    dataset_id = await anext(datasets)
    columns = await _get_columns_for_dataset(dataset_id, config.keypair)
    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=number_of_hours)

    task = RawTask(
        model_id=model_id,
        ds_id=dataset_id,
        field_system=None,
        field_instruction=columns.field_instruction,
        field_input=columns.field_input,
        field_output=columns.field_output,
        status=TaskStatus.PENDING,
        is_organic=False,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=number_of_hours,
        account_id=cst.NULL_ACCOUNT_ID,
    )
    logger.info(f"New task created and added to the queue {task}")

    task = await add_task(task, config.psql_db)


async def _add_new_task_to_network_if_not_enough(
    config: Config,
    models: AsyncGenerator[str, None],
    datasets: AsyncGenerator[str, None],
):
    current_training_tasks = await get_tasks_with_status(TaskStatus.TRAINING, config.psql_db)
    current_delayed_tasks = await get_tasks_with_status(TaskStatus.DELAYED, config.psql_db, include_not_ready_tasks=True)
    logger.info(f"We have {(len(current_delayed_tasks))} tasks in the queue")
    logger.info(f"There are {len(current_training_tasks)} running at the moment")
    if len(current_delayed_tasks) == 0 and len(current_training_tasks) < cst.HOW_MANY_TASKS_ALLOWED_AT_ONCE:
        logger.info("This is less than the minimal - creating a new task")
        await _create_synthetic_task(config, models, datasets)


async def schedule_synthetics_periodically(config: Config):
    logger.info("Starting the synthetic schedule loop...")
    datasets = _get_datasets(config.keypair)
    models = _get_models(config.keypair)

    current_try = 0
    while True:
        try:
            logger.info(f"Try {current_try+1}/{cst.NUM_SYNTH_RETRIES+1} - We are attempting to create a new task")
            await _add_new_task_to_network_if_not_enough(config, models, datasets)
            current_try = 0
        except Exception as e:
            if current_try == cst.NUM_SYNTH_RETRIES:
                logger.info(f"Synthetic task creation failed too many times, giving up for now. {e}")
            else:
                logger.info(f"Synthetic task creation try {current_try+1}/{cst.NUM_SYNTH_RETRIES+1} failed, retrying... {e}")
                current_try += 1
        finally:
            await asyncio.sleep(cst.NUMBER_OF_MINUTES_BETWEEN_SYNTH_TASK_CHECK * 60)
