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
from validator.core.models import Dataset
from validator.core.models import RawTask
from validator.db.sql.tasks import add_task
from validator.db.sql.tasks import get_tasks_with_status
from validator.utils.call_endpoint import call_content_service
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def _get_models(keypair: Keypair) -> AsyncGenerator[str, None]:
    while True:
        response = await call_content_service(cst.GET_RANDOM_MODELS_ENDPOINT, keypair)
        if not isinstance(response, list):
            raise TypeError("Expected a list of responses from GET_ALL_MODELS_ENDPOINT")
        models: list[dict[str, Any]] = response
        model_ids = [model.get(cst.GET_ALL_MODELS_ID, "") for model in models]
        random.shuffle(model_ids)
        for model_id in model_ids:
            yield model_id


async def _get_datasets_for_bin(min_bytes: int, max_bytes: int, keypair: Keypair) -> AsyncGenerator[Dataset, None]:
    """Get datasets for a specific size bin."""
    while True:
        params = {"min_parquet_bytes": min_bytes, "max_parquet_bytes": max_bytes}
        try:
            response = await call_content_service(cst.GET_RANDOM_DATASETS_ENDPOINT, keypair, params)
            if not isinstance(response, list):
                raise TypeError("Expected a list of responses from GET_ALL_DATASETS_ENDPOINT")

            dataset_dicts: list[dict[str, Any]] = response
            datasets = [Dataset.model_validate(ds) for ds in dataset_dicts]
            random.shuffle(datasets)

            for dataset in datasets:
                yield dataset

        except Exception as e:
            logger.warning(f"Failed to fetch datasets for bin {min_bytes}-{max_bytes} bytes: {e}")
            await asyncio.sleep(5)


async def _get_datasets(keypair: Keypair) -> AsyncGenerator[Dataset, None]:
    """Round-robin generator that cycles through all dataset size bins."""

    bin_generators = [_get_datasets_for_bin(min_bytes, max_bytes, keypair) for min_bytes, max_bytes in cst.DATASET_BINS_TO_SAMPLE]

    while True:
        for generator in bin_generators:
            try:
                dataset = await anext(generator)
                yield dataset
            except StopAsyncIteration:
                continue
            except Exception as e:
                logger.warning(f"Error getting next dataset from bin: {e}")
                continue


async def _get_columns_for_dataset(
    dataset_id: str,
    keypair: Keypair,
) -> DatasetColumnsResponse:
    url = cst.GET_COLUMNS_FOR_DATASET_ENDPOINT.replace("{dataset}", dataset_id)
    logger.info(f"Getting columns for dataset {dataset_id}")

    response = await call_content_service(url, keypair)
    if not isinstance(response, dict):
        raise TypeError(f"Expected dictionary response, got {type(response)}")
    try:
        columns = DatasetColumnsResponse.model_validate(response)
    except Exception as exc:
        logger.error(f"The get columns for dataset endpoint should return a DatasetColumnsResponse type: {exc}")
        raise TypeError(f"The get columns for dataset endpoint should return a DatasetColumnsResponse type: {exc}")
    return columns


def _get_training_hours_from_bytes(bytes: int) -> tuple[int, int]:
    """Randomly select training hours for a given dataset size in bytes based on range bins."""
    min_hours, max_hours = 0, 0
    for min_bytes, max_bytes in cst.TEXT_DATASET_BINS_TO_TRAINING_HOURS_RANGE.keys():
        if min_bytes <= bytes <= max_bytes:
            min_hours, max_hours = cst.TEXT_DATASET_BINS_TO_TRAINING_HOURS_RANGE[(min_bytes, max_bytes)]
            break
    if min_hours == 0 and max_hours == 0:
        sorted_bins = list(cst.TEXT_DATASET_BINS_TO_TRAINING_HOURS_RANGE.keys())
        if bytes > sorted_bins[-1][1]:  # if greater than the largest bin
            max_hours_range = list(cst.TEXT_DATASET_BINS_TO_TRAINING_HOURS_RANGE.values())[-1]
            return max_hours_range[1]  # max hours
        else:
            raise ValueError(f"No training hours range found for {bytes} bytes")
    return random.randint(min_hours, max_hours)


async def _create_synthetic_task(
    config: Config,
    models: AsyncGenerator[str, None],
    datasets: AsyncGenerator[Dataset, None],
):
    model_id = await anext(models)
    dataset = await anext(datasets)
    number_of_hours = _get_training_hours_from_bytes(dataset.num_bytes_parquet_files)
    columns = await _get_columns_for_dataset(dataset.dataset_id, config.keypair)
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
    datasets: AsyncGenerator[Dataset, None],
):
    current_training_tasks = await get_tasks_with_status(TaskStatus.TRAINING, config.psql_db)
    current_preeval_tasks = await get_tasks_with_status(TaskStatus.PREEVALUATION, config.psql_db)
    current_delayed_tasks = await get_tasks_with_status(TaskStatus.DELAYED, config.psql_db, include_not_ready_tasks=True)

    total_active_tasks = len(current_training_tasks) + len(current_preeval_tasks)

    logger.info(f"We have {len(current_delayed_tasks)} tasks in the queue")
    logger.info(
        f"There are {total_active_tasks} active tasks"
        + f" ({len(current_training_tasks)} training, {len(current_preeval_tasks)} pre-evaluation)"
    )

    if len(current_delayed_tasks) == 0 and total_active_tasks < cst.MAX_CONCURRENT_SYNTHETIC_JOBS:
        logger.info(
            "Current number of training tasks is less than the maximum amount of concurrent synthetic"
            " jobs we can have. New task incoming..."
        )
        await _create_synthetic_task(config, models, datasets)


async def schedule_synthetics_periodically(config: Config):
    logger.info("Starting the synthetic schedule loop...")
    datasets = _get_datasets(config.keypair)
    models = _get_models(config.keypair)

    current_try = 0
    while True:
        try:
            logger.info(f"Try {current_try + 1}/{cst.NUM_SYNTH_RETRIES} - We are attempting to create a new task")
            await _add_new_task_to_network_if_not_enough(config, models, datasets)
            current_try = 0
            await asyncio.sleep(cst.NUMBER_OF_MINUTES_BETWEEN_SYNTH_TASK_CHECK * 60)
        except Exception as e:
            if current_try < cst.NUM_SYNTH_RETRIES - 1:
                logger.info(f"Synthetic task creation try {current_try + 1}/{cst.NUM_SYNTH_RETRIES} failed, retrying... {e}")
                current_try += 1
            else:
                logger.info(f"Synthetic task creation failed after {cst.NUM_SYNTH_RETRIES} attempts, giving up for now. {e}")
                current_try = 0
                await asyncio.sleep(cst.NUMBER_OF_MINUTES_BETWEEN_SYNTH_TASK_CHECK * 60)
