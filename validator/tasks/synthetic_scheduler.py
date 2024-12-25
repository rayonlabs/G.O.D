import asyncio
import random
from datetime import datetime
from datetime import timedelta
from logging import getLogger
from typing import Any
from typing import AsyncGenerator

from substrateinterface import Keypair

import validator.core.constants as cst
from core.models.payload_models import DatasetColumnsResponse
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.models import RawTask
from validator.core.models import SuitableDataset
from validator.db.sql.tasks import add_task
from validator.db.sql.tasks import get_tasks_with_status
from validator.utils.call_endpoint import call_content_service
from validator.utils.column_picker import pick_columns_locally


logger = getLogger(name="task synth")


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


async def _get_datasets(keypair: Keypair) -> AsyncGenerator[SuitableDataset, None]:
    response = await call_content_service(cst.GET_RANDOM_DATASETS_ENDPOINT, keypair)
    if not isinstance(response, list):
        raise TypeError("Expected a list of responses from GET_ALL_DATASETS_ENDPOINT")
    datasets: list[SuitableDataset] = [SuitableDataset.model_validate(ds) for ds in response]
    random.shuffle(datasets)
    for dataset in datasets:
        yield dataset


async def _get_columns_for_dataset(dataset: SuitableDataset, keypair: Keypair, config: Config) -> DatasetColumnsResponse:
    if config.use_local_column_picker:
        columns = await pick_columns_locally(config, dataset)
        return DatasetColumnsResponse.model_validate(columns.model_dump())
    else:
        url = cst.GET_COLUMNS_FOR_DATASET_ENDPOINT.replace("{dataset}", dataset.dataset_id)
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
    datasets: AsyncGenerator[SuitableDataset, None],
):
    number_of_hours = random.randint(cst.MIN_COMPETITION_HOURS, cst.MAX_COMPETITION_HOURS)
    model_id = await anext(models)
    dataset = await anext(datasets)

    columns = await _get_columns_for_dataset(dataset, config.keypair, config)
    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=number_of_hours)

    task = RawTask(
        model_id=model_id,
        ds_id=dataset.dataset_id,
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
    datasets: AsyncGenerator[SuitableDataset, None],
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
    while True:
        try:
            logger.info("We are attempting to create a new task")
            await _add_new_task_to_network_if_not_enough(config, models, datasets)
            await asyncio.sleep(cst.NUMBER_OF_MINUTES_BETWEEN_SYNTH_TASK_CHECK * 60)
        except Exception as e:
            logger.info(f"Ah, that dataset was missing some details, trying another one next time. {e}")

            await asyncio.sleep(5 * 60)
