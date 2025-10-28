import asyncio
import os
import random
from ast import literal_eval
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import AsyncGenerator
from uuid import UUID

from substrateinterface import Keypair

import validator.core.constants as cst
from core.models.payload_models import ImageModelInfo
from core.models.payload_models import ImageModelsResponse
from core.models.payload_models import InstructTextDatasetColumnsResponse
from core.models.utility_models import FileFormat
from core.models.utility_models import Message
from core.models.utility_models import Role
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.augmentation.augmentation import load_prompts
from validator.core.config import Config
from validator.core.constants import END_OF_REASONING_TAG
from validator.core.constants import TEXT_SYNTH_MODEL
from validator.core.constants import TEXT_SYNTH_MODEL_MAX_TOKENS
from validator.core.constants import TEXT_SYNTH_MODEL_TEMPERATURE
from validator.core.models import ChatRawTask
from validator.core.models import Dataset
from validator.core.models import DpoRawTask
from validator.core.models import GrpoRawTask
from validator.core.models import InstructTextRawTask
from validator.core.models import RawTask
from validator.core.models import RewardFunction
from validator.db.sql import grpo as grpo_sql
from validator.db.sql.grpo import get_generic_reward_functions_from_db
from validator.db.sql.tasks import add_task
from validator.db.sql.tasks import get_tasks_with_status
from validator.evaluation.utils import get_default_dataset_config
from validator.evaluation.utils import load_default_split
from validator.tasks.diffusion_synth import create_synthetic_image_task
from validator.utils.call_endpoint import call_content_service
from validator.utils.llm import convert_to_nineteen_payload
from validator.utils.llm import post_to_nineteen_chat_with_reasoning
from validator.utils.logging import get_logger
from validator.utils.reward_functions import validate_reward_function
from validator.utils.util import retry_with_backoff
from validator.utils.util import save_json_to_temp_file
from validator.utils.util import upload_file_to_minio


logger = get_logger(__name__)


async def _get_text_models(
    keypair: Keypair, smallest_size_b: float = 0.1, largest_size_b: float = 12.0
) -> AsyncGenerator[str, None]:
    min_params = int(smallest_size_b * 1_000_000_000)
    max_params = int(largest_size_b * 1_000_000_000)
    params = {"min_params": min_params, "max_params": max_params}

    while True:
        response = await call_content_service(
            cst.GET_RANDOM_MODELS_ENDPOINT,
            keypair,
            params=params,
        )
        if not isinstance(response, list):
            raise TypeError("Expected a list of responses from GET_ALL_MODELS_ENDPOINT")
        models: list[dict[str, Any]] = response
        model_ids = [model.get(cst.GET_ALL_MODELS_ID, "") for model in models]
        random.shuffle(model_ids)
        for model_id in model_ids:
            yield model_id


async def _get_image_models(keypair: Keypair) -> AsyncGenerator[ImageModelInfo, None]:
    while True:
        response_data = await call_content_service(cst.GET_IMAGE_MODELS_ENDPOINT, keypair)
        try:
            response = ImageModelsResponse.model_validate(response_data)
        except Exception as e:
            logger.error(f"Invalid response format from {cst.GET_IMAGE_MODELS_ENDPOINT}: {response_data}. Error: {e}")
            await asyncio.sleep(5)
            continue

        models = response.models
        random.shuffle(models)
        for model_info in models:
            yield model_info


async def _get_datasets_for_bin(min_rows: int, max_rows: int, keypair: Keypair, dpo: bool) -> AsyncGenerator[Dataset, None]:
    """Get datasets for a specific size bin."""
    while True:
        # params = {"min_rows": min_rows, "max_rows": max_rows, "dpo": dpo}
        params = {"dpo": dpo}
        try:
            response = await call_content_service(cst.GET_RANDOM_DATASETS_ENDPOINT, keypair, params)
            if not isinstance(response, list):
                raise TypeError("Expected a list of responses from GET_ALL_DATASETS_ENDPOINT")

            dataset_dicts: list[dict[str, Any]] = response
            logger.info(f"[DATASET_BIN] Got {len(dataset_dicts)} dataset dicts from content service")
            datasets = []
            for idx, ds in enumerate(dataset_dicts):
                try:
                    dataset = Dataset.model_validate(ds)
                    datasets.append(dataset)
                except Exception as exc:
                    logger.warning(f"[DATASET_BIN] Failed to validate dataset {idx + 1}: {exc}")

            logger.info(f"[DATASET_BIN] Successfully validated {len(datasets)} datasets")
            random.shuffle(datasets)

            for dataset in datasets:
                logger.info(
                    f"[DATASET_BIN] Yielding dataset: {dataset.dataset_id} (rows: {dataset.num_rows}, "
                    f"bytes: {dataset.num_bytes_parquet_files}, "
                    f"dpo_available: {dataset.dpo_available})"
                )
                yield dataset

        except Exception as e:
            logger.error(f"[DATASET_BIN] Failed to fetch datasets for bin {min_rows}-{max_rows} rows: {e}")
            logger.info("[DATASET_BIN] Sleeping 5 seconds before retry...")
            await asyncio.sleep(5)


async def _get_instruct_text_datasets(keypair: Keypair) -> AsyncGenerator[Dataset, None]:
    """Round-robin generator that cycles through all dataset size bins."""

    bin_generators = [
        _get_datasets_for_bin(min_rows, max_rows, keypair, False) for min_rows, max_rows in cst.DATASET_BINS_TO_SAMPLE
    ]

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


async def _get_dpo_datasets(keypair: Keypair) -> AsyncGenerator[Dataset, None]:
    """Round-robin generator that cycles through all dataset size bins."""

    logger.info("I AM GETTIG THE DPO DATASETS")
    bin_generators = [
        _get_datasets_for_bin(min_rows, max_rows, keypair, True) for min_rows, max_rows in cst.DATASET_BINS_TO_SAMPLE
    ]

    while True:
        for generator in bin_generators:
            try:
                logger.info(f"We have picked {generator}")
                dataset = await anext(generator)
                yield dataset
            except StopAsyncIteration:
                continue
            except Exception as e:
                logger.warning(f"Error getting next dataset from bin: {e}")
                continue


async def _get_columns_for_instruct_dataset(
    dataset_id: str,
    keypair: Keypair,
) -> InstructTextDatasetColumnsResponse:
    from validator.utils.call_endpoint import call_content_service_fast

    url = cst.GET_COLUMNS_FOR_DATASET_ENDPOINT.replace("{dataset}", dataset_id)
    logger.info(f"Getting columns for dataset {dataset_id} - ACTUAL MAPPING CALL")

    response = await call_content_service_fast(url, keypair)
    if not isinstance(response, dict):
        raise TypeError(f"Expected dictionary response, got {type(response)}")
    try:
        columns = InstructTextDatasetColumnsResponse.model_validate(response)
    except Exception as exc:
        logger.error(f"The get columns for dataset endpoint should return a DatasetColumnsResponse type: {exc}")
        raise TypeError(f"The get columns for dataset endpoint should return a DatasetColumnsResponse type: {exc}")
    return columns


def _get_training_hours_from_num_rows(num_rows: int) -> tuple[int, int]:
    """Randomly select training hours for a given dataset size in bytes based on range bins."""
    min_hours, max_hours = 0, 0
    for min_rows, max_rows in cst.INSTRUCT_TEXT_DATASET_BINS_TO_TRAINING_HOURS_RANGE.keys():
        if min_rows <= num_rows <= max_rows:
            min_hours, max_hours = cst.INSTRUCT_TEXT_DATASET_BINS_TO_TRAINING_HOURS_RANGE[(min_rows, max_rows)]
            break
    if min_hours == 0 and max_hours == 0:
        raise ValueError(f"No training hours range found for {num_rows} rows")
    return random.randint(min_hours, max_hours)


async def get_dataset(
    datasets_generator: AsyncGenerator[Dataset, None],
    task_type: TaskType | None = None,
    keypair: Keypair | None = None,
) -> Dataset:
    """Get a single dataset from the generator, validating column availability."""
    while True:
        dataset = await anext(datasets_generator)

        if task_type and keypair and task_type != TaskType.DPOTASK:
            try:
                from validator.utils.call_endpoint import call_content_service_fast

                url = cst.GET_COLUMNS_FOR_DATASET_ENDPOINT.replace("{dataset}", dataset.dataset_id)
                logger.info(f"PRE-VALIDATION: Checking column mapping for dataset {dataset.dataset_id}")
                await call_content_service_fast(url, keypair)
                logger.info(f"PRE-VALIDATION: Dataset {dataset.dataset_id} column mapping validated successfully")
                logger.info(f"Selected dataset: {dataset.dataset_id}")
                return dataset
            except Exception as e:
                logger.warning(f"Dataset {dataset.dataset_id} failed column validation, skipping: {e}")
                continue
        else:
            logger.info(f"Selected dataset: {dataset.dataset_id}")
            return dataset


def is_chat_format(example: dict[str, Any]) -> bool:
    return isinstance(example.get("conversations"), list) and all(
        "from" in msg and "value" in msg for msg in example["conversations"]
    )


def is_evol_format(example: dict[str, Any]) -> bool:
    return isinstance(example.get("conversations"), list) and all(
        "role" in msg and "content" in msg for msg in example["conversations"]
    )


async def convert_instruct_dataset_to_chat_format(dataset_id: str, input_field: str, output_field: str) -> list[dict[str, Any]]:
    ds_config = get_default_dataset_config(dataset_id)
    dataset = load_default_split(dataset_id, config=ds_config)
    chat_data = []

    for ex in dataset:
        if is_chat_format(ex):
            chat_data.append(ex)
        elif is_evol_format(ex):
            convs = []
            for msg in ex["conversations"]:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    convs.append(
                        {cst.STANDARD_CHAT_ROLE_FIELD: cst.STANDARD_CHAT_ROLE_USER, cst.STANDARD_CHAT_CONTENT_FIELD: content}
                    )
                elif role == "assistant":
                    convs.append(
                        {cst.STANDARD_CHAT_ROLE_FIELD: cst.STANDARD_CHAT_ROLE_ASSISTANT, cst.STANDARD_CHAT_CONTENT_FIELD: content}
                    )
            if convs:
                chat_data.append({cst.STANDARD_CHAT_MESSAGES_COLUMN: convs})
        else:
            inp = ex.get(input_field, "").strip()
            out = ex.get(output_field, "").strip()
            if inp and out:
                chat_data.append(
                    {
                        cst.STANDARD_CHAT_MESSAGES_COLUMN: [
                            {cst.STANDARD_CHAT_ROLE_FIELD: cst.STANDARD_CHAT_ROLE_USER, cst.STANDARD_CHAT_CONTENT_FIELD: inp},
                            {
                                cst.STANDARD_CHAT_ROLE_FIELD: cst.STANDARD_CHAT_ROLE_ASSISTANT,
                                cst.STANDARD_CHAT_CONTENT_FIELD: out,
                            },
                        ]
                    }
                )

    return chat_data


async def convert_and_upload_chat_dataset(dataset_id: str, input_field: str, output_field: str) -> str:
    chat_data = await convert_instruct_dataset_to_chat_format(dataset_id, input_field, output_field)
    if not chat_data:
        raise ValueError("Error creating chat dataset - no data")
    dataset_json, _ = await save_json_to_temp_file(chat_data, prefix="chat_dataset_")
    uploaded_url = await upload_file_to_minio(dataset_json, cst.BUCKET_NAME, f"{os.urandom(8).hex()}_chat_data.json")
    if os.path.exists(dataset_json):
        os.remove(dataset_json)
    return uploaded_url


@retry_with_backoff
async def create_synthetic_dpo_task(
    config: Config,
    models: AsyncGenerator[str, None],
    datasets: AsyncGenerator[Dataset, None],
) -> RawTask:
    logger.info("DPO task")
    model_id = await anext(models)
    logger.info(f"We picked {model_id}")

    dataset = await get_dataset(datasets, task_type=TaskType.DPOTASK, keypair=config.keypair)

    logger.info(
        f"Selected dataset: {dataset.dataset_id} "
        f"(rows: {dataset.num_rows}, bytes: {dataset.num_bytes_parquet_files})"
    )

    number_of_hours = _get_training_hours_from_num_rows(dataset.num_rows)
    assert dataset.dpo_rejected_column, "we should have a reject column"
    assert dataset.dpo_accepted_column, "we should have a accepted column"
    assert dataset.dpo_prompt_column, "we should have a prompt column"

    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=number_of_hours)

    task = DpoRawTask(
        model_id=model_id,
        ds=dataset.dataset_id,
        field_system=None,
        field_prompt=dataset.dpo_prompt_column,
        field_chosen=dataset.dpo_accepted_column,
        field_rejected=dataset.dpo_rejected_column,
        status=TaskStatus.PENDING,
        is_organic=False,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=number_of_hours,
        account_id=cst.NULL_ACCOUNT_ID,
    )
    logger.info(f"New DPO task created with dataset {dataset.dataset_id}")

    task = await add_task(task, config.psql_db)

    return task


def process_reward_functions(result: str) -> list[str]:
    """
    Process and validate the LLM-generated reward functions.
    Returns list of valid reward function definitions.
    """
    valid_reward_functions = []
    try:
        list_str = result[result.find("[") : result.rfind("]") + 1]
        func_list = literal_eval(list_str)
        if not isinstance(func_list, list):
            raise ValueError("Expected a list")
        if not all(isinstance(item, str) for item in func_list):
            raise ValueError("Expected a list of strings")

        for func_def in func_list:
            is_valid, error, _ = validate_reward_function(func_def)
            if is_valid:
                valid_reward_functions.append(func_def)
            else:
                logger.warning(f"Function validation failed: {error}")

        return valid_reward_functions
    except Exception as e:
        logger.error(f"Failed to parse LLM response as list: {e}")
        return []


async def _generate_generic_reward_functions_from_llm(keypair: Keypair, num_rewards: int) -> list[RewardFunction]:
    prompts = load_prompts()
    num_rewards_with_margin = int(num_rewards * 1.5)

    messages = [
        Message(role=Role.SYSTEM, content=prompts.reward_function_generation_sys),
        Message(role=Role.USER, content=prompts.reward_function_generation_user.format(num_rewards=num_rewards_with_margin)),
    ]

    payload = convert_to_nineteen_payload(
        messages=messages,
        model=TEXT_SYNTH_MODEL,
        temperature=TEXT_SYNTH_MODEL_TEMPERATURE,
        max_tokens=TEXT_SYNTH_MODEL_MAX_TOKENS,
    )

    result = await post_to_nineteen_chat_with_reasoning(payload, keypair, END_OF_REASONING_TAG)

    if result:
        valid_reward_functions = process_reward_functions(result)

    reward_functions = [
        RewardFunction(reward_func=valid_reward_function, is_generic=True, reward_weight=1.0)
        for valid_reward_function in valid_reward_functions[:num_rewards]
    ]
    return reward_functions


async def _get_generic_reward_functions(config: Config) -> list[RewardFunction]:
    reward_functions = []
    total_rewards = random.randint(cst.MIN_NUM_REWARD_FUNCTIONS, cst.MAX_NUM_REWARD_FUNCTIONS)

    num_generic_rewards_from_db = max(1, int(total_rewards * cst.PERCENTAGE_REWARD_FUNCTIONS_GENERIC_FROM_DB))
    num_generic_rewards_from_llm = total_rewards - num_generic_rewards_from_db

    reward_functions += await get_generic_reward_functions_from_db(config.psql_db, num_generic_rewards_from_db)

    if num_generic_rewards_from_llm > 0:
        reward_functions += await _generate_generic_reward_functions_from_llm(config.keypair, num_generic_rewards_from_llm)

    reward_functions = _randomize_reward_weights(reward_functions)

    return reward_functions


def _randomize_reward_weights(reward_functions: list[RewardFunction]) -> list[RewardFunction]:
    # Generate random weights
    random_weights = [random.uniform(0.1, 10.0) for _ in reward_functions]

    # Normalize to sum to 1
    weight_sum = sum(random_weights)
    normalized_weights = [w / weight_sum for w in random_weights]

    return [
        RewardFunction(
            reward_id=reward_function.reward_id,
            reward_func=reward_function.reward_func,
            func_hash=reward_function.func_hash,
            is_generic=reward_function.is_generic,
            reward_weight=normalized_weight,
        )
        for reward_function, normalized_weight in zip(reward_functions, normalized_weights)
    ]


@retry_with_backoff
async def create_synthetic_grpo_task(
    config: Config,
    models: AsyncGenerator[str, None],
    datasets: AsyncGenerator[Dataset, None],
) -> RawTask:
    model_id = await anext(models)

    dataset = await get_dataset(datasets, task_type=TaskType.GRPOTASK, keypair=config.keypair)

    number_of_hours = _get_training_hours_from_num_rows(dataset.num_rows)
    columns = await _get_columns_for_instruct_dataset(dataset.dataset_id, config.keypair)

    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=number_of_hours)

    reward_functions = await _get_generic_reward_functions(config)

    task = GrpoRawTask(
        model_id=model_id,
        ds=dataset.dataset_id,
        field_prompt=columns.field_instruction,
        reward_functions=reward_functions,
        status=TaskStatus.PENDING,
        is_organic=False,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=number_of_hours,
        account_id=cst.NULL_ACCOUNT_ID,
    )
    logger.info(f"New GRPO task created with dataset {dataset.dataset_id}")

    task = await add_task(task, config.psql_db)

    return task


@retry_with_backoff
async def create_synthetic_affine_grpo_task(
    config: Config,
    models: AsyncGenerator[str, None],
) -> RawTask:
    """Create a synthetic GRPO task using affine data from the content service."""
    model_id = await anext(models)

    try:
        response = await call_content_service(cst.GET_AFFINE_GRPO_DATA_ENDPOINT, config.keypair)
        logger.info(f"Retrieved affine GRPO data: {response}")

        if not isinstance(response, dict):
            raise ValueError("Expected dict response from affine GRPO data endpoint")

        s3_url = response.get("s3_url")
        if not s3_url:
            raise ValueError("No s3_url in affine GRPO data response")

        logger.info(f"Looking for affine reward functions with IDs: {cst.AFFINE_REWARD_FN_IDS}")

        affine_reward_functions = []
        for reward_id in cst.AFFINE_REWARD_FN_IDS:
            logger.debug(f"Attempting to fetch reward function with ID: {reward_id}")
            reward_function = await grpo_sql.get_reward_function_by_id(config.psql_db, UUID(reward_id))
            if reward_function:
                affine_reward_functions.append(reward_function)
            else:
                logger.warning(f"Reward function {reward_id} not found in database")

        logger.info(f"Successfully loaded {len(affine_reward_functions)} affine reward functions")

        # Normalize weights to sum to 1
        if affine_reward_functions:
            num_functions = len(affine_reward_functions)
            normalized_weight = 1.0 / num_functions
            for reward_function in affine_reward_functions:
                logger.info(f"Setting weight for {reward_function.reward_id} to {normalized_weight:.4f}")
                reward_function.reward_weight = normalized_weight

        if not affine_reward_functions:
            logger.error("No affine reward functions found in database, falling back to generic functions")
            reward_functions = await _get_generic_reward_functions(config)
        else:
            logger.info(f"Using {len(affine_reward_functions)} affine-specific reward functions")
            reward_functions = affine_reward_functions

        num_entries = response.get("num_entries", 10_000)
        number_of_hours = _get_training_hours_from_num_rows(num_entries)

        current_time = datetime.utcnow()
        end_timestamp = current_time + timedelta(hours=number_of_hours)

        task = GrpoRawTask(
            model_id=model_id,
            ds=s3_url,
            field_prompt="prompt",
            reward_functions=reward_functions,
            status=TaskStatus.PENDING,
            is_organic=False,
            created_at=current_time,
            termination_at=end_timestamp,
            hours_to_complete=number_of_hours,
            account_id=cst.NULL_ACCOUNT_ID,
            file_format=FileFormat.S3,
            extra_column="extra",
        )

        logger.info(f"New affine GRPO task created with S3 dataset: {s3_url}")

        task = await add_task(task, config.psql_db)

        return task

    except Exception as e:
        logger.error(f"Failed to create affine GRPO task: {e}")


@retry_with_backoff
async def create_synthetic_instruct_text_task(
    config: Config,
    models: AsyncGenerator[str, None],
    datasets: AsyncGenerator[Dataset, None],
) -> RawTask:
    model_id = await anext(models)

    logger.info("INSTRUCT_TASK: Starting dataset selection...")
    dataset = await get_dataset(datasets, task_type=TaskType.INSTRUCTTEXTTASK, keypair=config.keypair)
    logger.info(f"INSTRUCT_TASK: Selected dataset: {dataset.dataset_id}")

    number_of_hours = _get_training_hours_from_num_rows(dataset.num_rows)
    columns = await _get_columns_for_instruct_dataset(dataset.dataset_id, config.keypair)

    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=number_of_hours)

    task = InstructTextRawTask(
        model_id=model_id,
        ds=dataset.dataset_id,
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
    logger.info(f"INSTRUCT_TASK: Successfully created task with dataset {dataset.dataset_id}")

    task = await add_task(task, config.psql_db)
    logger.info(f"INSTRUCT_TASK: Task saved to database with ID: {task.task_id}")

    return task


async def create_synthetic_chat_task(
    config: Config,
    models: AsyncGenerator[str, None],
    datasets: AsyncGenerator[Dataset, None],
) -> RawTask:
    model_id = await anext(models)

    logger.info("CHAT_TASK: Starting dataset selection...")
    dataset = await get_dataset(datasets, task_type=TaskType.INSTRUCTTEXTTASK, keypair=config.keypair)
    logger.info(f"CHAT_TASK: Selected dataset: {dataset.dataset_id}")

    number_of_hours = _get_training_hours_from_num_rows(dataset.num_rows)
    columns = await _get_columns_for_instruct_dataset(dataset.dataset_id, config.keypair)

    input_field = columns.field_input if columns.field_input else columns.field_instruction

    chat_dataset_url = await convert_and_upload_chat_dataset(
        dataset.dataset_id, input_field=input_field, output_field=columns.field_output
    )

    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=number_of_hours)

    task = ChatRawTask(
        model_id=model_id,
        ds=chat_dataset_url,
        chat_template=cst.STANDARD_CHAT_TEMPLATE,
        chat_column=cst.STANDARD_CHAT_MESSAGES_COLUMN,
        chat_role_field=cst.STANDARD_CHAT_ROLE_FIELD,
        chat_content_field=cst.STANDARD_CHAT_CONTENT_FIELD,
        chat_user_reference=cst.STANDARD_CHAT_ROLE_USER,
        chat_assistant_reference=cst.STANDARD_CHAT_ROLE_ASSISTANT,
        status=TaskStatus.PENDING,
        is_organic=False,
        file_format=FileFormat.S3,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=number_of_hours,
        account_id=cst.NULL_ACCOUNT_ID,
    )
    logger.info(f"CHAT_TASK: Successfully created task with dataset {dataset.dataset_id}")

    task = await add_task(task, config.psql_db)
    logger.info(f"CHAT_TASK: Task saved to database with ID: {task.task_id}")

    return task


async def _add_new_task_to_network_if_not_enough(
    config: Config,
    models: AsyncGenerator[str, None],
    instruct_datasets: AsyncGenerator[Dataset, None],
    dpo_datasets: AsyncGenerator[Dataset, None],
    image_models: AsyncGenerator[ImageModelInfo, None],
    grpo_models: AsyncGenerator[str, None] | None = None,
):
    current_training_tasks = await get_tasks_with_status(TaskStatus.TRAINING, config.psql_db)
    current_preeval_tasks = await get_tasks_with_status(TaskStatus.PREEVALUATION, config.psql_db)
    current_delayed_tasks = await get_tasks_with_status(TaskStatus.DELAYED, config.psql_db, include_not_ready_tasks=True)
    current_pending_tasks = await get_tasks_with_status(TaskStatus.PENDING, config.psql_db)
    total_active_tasks = len(current_training_tasks) + len(current_preeval_tasks) + len(current_pending_tasks)

    logger.info(
        f"There are {total_active_tasks} active tasks"
        + f" ({len(current_training_tasks)} training, {len(current_preeval_tasks)} pre-evaluation)"
    )

    if len(current_delayed_tasks) == 0 and total_active_tasks < cst.MAX_CONCURRENT_SYNTHETIC_JOBS:
        logger.info(
            "Current number of training tasks is less than the maximum amount of concurrent synthetic"
            " jobs we can have. New task incoming..."
        )
        TASK_TYPES = [
            (TaskType.INSTRUCTTEXTTASK, cst.PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT),
            (TaskType.IMAGETASK, cst.PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_IMAGE),
            (TaskType.DPOTASK, cst.PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO),
            (TaskType.GRPOTASK, cst.PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_GRPO),
        ]
        selected_task_type = random.choices([t[0] for t in TASK_TYPES], weights=[t[1] for t in TASK_TYPES], k=1)[0]

        if selected_task_type == TaskType.INSTRUCTTEXTTASK:
            probability_of_chat_task = random.random()
            if probability_of_chat_task < cst.PERCENTAGE_OF_INSTRUCT_TASKS_THAT_SHOULD_BE_CHAT:
                logger.info(f"TASK_TYPE_SELECTION: Creating CHAT task (probability: {probability_of_chat_task:.3f})")
                await create_synthetic_chat_task(config, models, instruct_datasets)
            else:
                logger.info(f"TASK_TYPE_SELECTION: Creating INSTRUCT task (probability: {probability_of_chat_task:.3f})")
                await create_synthetic_instruct_text_task(config, models, instruct_datasets)
        elif selected_task_type == TaskType.CHATTASK:
            logger.info("TASK_TYPE_SELECTION: Creating CHAT task (direct selection)")
            await create_synthetic_chat_task(config, models, instruct_datasets)
        elif selected_task_type == TaskType.IMAGETASK:
            await create_synthetic_image_task(config, image_models)
        elif selected_task_type == TaskType.DPOTASK:
            await create_synthetic_dpo_task(config, models, dpo_datasets)
        elif selected_task_type == TaskType.GRPOTASK:
            grpo_models_to_use = grpo_models if grpo_models is not None else models
            if random.random() < cst.PERCENTAGE_OF_GRPO_TASKS_THAT_SHOULD_BE_AFFINE:
                logger.info("TASK_TYPE_SELECTION: Creating AFFINE GRPO task")
                await create_synthetic_affine_grpo_task(config, grpo_models_to_use)
            else:
                logger.info("TASK_TYPE_SELECTION: Creating regular GRPO task")
                await create_synthetic_grpo_task(config, grpo_models_to_use, instruct_datasets)
