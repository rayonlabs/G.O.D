import asyncio
import json
from typing import List

import yaml
from datasets import load_dataset
from fiber import Keypair

from core.models.utility_models import Message
from core.models.utility_models import Prompts
from core.models.utility_models import Role
from validator.core.constants import END_OF_REASONING_TAG
from validator.core.constants import MAX_SYNTH_DATA_POINTS
from validator.core.constants import PROMPT_PATH
from validator.core.constants import SYNTH_GEN_BATCH_SIZE
from validator.core.constants import TEXT_SYNTH_MODEL
from validator.core.constants import TEXT_SYNTH_MODEL_MAX_TOKENS
from validator.core.constants import TEXT_SYNTH_MODEL_TEMPERATURE
from validator.evaluation.utils import get_default_dataset_config
from validator.utils.llm import convert_to_nineteen_payload
from validator.utils.llm import extract_json_from_response
from validator.utils.llm import post_to_nineteen_chat_with_reasoning
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def load_prompts() -> Prompts:
    with open(PROMPT_PATH, "r") as file:
        prompts_dict = yaml.safe_load(file)
    return Prompts(**prompts_dict)


def load_and_sample_dataset(dataset_name: str, columns_to_sample: List[str]) -> List[dict]:
    try:
        config_name = get_default_dataset_config(dataset_name)
        dataset = load_dataset(dataset_name, config_name, trust_remote_code=True, streaming=True)
    except Exception as e:
        logger.exception(f"Failed to load dataset {dataset_name}: {e}")
        raise e

    logger.info(f"Loading dataset: {dataset_name}")
    train_dataset = dataset["train"]

    filtered_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in columns_to_sample])

    num_samples = MAX_SYNTH_DATA_POINTS
    logger.info(f"Taking {num_samples} samples from {dataset_name}")

    sampled_data = filtered_dataset.shuffle(seed=42, buffer_size=1000).take(num_samples)

    sampled_data_list = [sample for sample in sampled_data]
    return sampled_data_list


def create_messages_for_input_generation(
    reformulated_output: str, description: str, output_field: str, schema: dict, prompts: Prompts
) -> List[Message]:
    messages = []
    system_message = Message(role=Role.SYSTEM, content=prompts.input_field_generation_sys)
    messages.append(system_message)
    user_message = Message(
        role=Role.USER,
        content=prompts.input_field_generation_user.format(
            schema=json.dumps(schema), output_field=output_field, output=reformulated_output, description=description
        ),
    )
    messages.append(user_message)
    return messages


def create_messages_for_input_output_reformulation(row: dict, prompts: Prompts) -> List[Message]:
    messages = []
    system_message = Message(role=Role.SYSTEM, content=prompts.input_output_reformulation_sys)
    messages.append(system_message)
    user_message = Message(
        role=Role.USER,
        content=prompts.input_output_reformulation_user.format(data=json.dumps(row)),
    )
    messages.append(user_message)
    return messages


def check_the_synthetic_data(synthetic_data_point: dict, original_data_columns: List[str]) -> bool:
    return set(synthetic_data_point.keys()) == set(original_data_columns)


async def generate_paraphrased_version(row: dict, prompts: Prompts, keypair: Keypair) -> dict:
    messages = create_messages_for_input_output_reformulation(row, prompts)
    payload = convert_to_nineteen_payload(messages, TEXT_SYNTH_MODEL, TEXT_SYNTH_MODEL_TEMPERATURE, TEXT_SYNTH_MODEL_MAX_TOKENS)
    result = await post_to_nineteen_chat_with_reasoning(payload, keypair, END_OF_REASONING_TAG)
    paraphrased_data = extract_json_from_response(result) if isinstance(result, str) else result

    return paraphrased_data


async def process_row(row, prompts, keypair):
    json_synthetic_data_point = await generate_paraphrased_version(row, prompts, keypair)

    if check_the_synthetic_data(json_synthetic_data_point, row.keys()):
        return json_synthetic_data_point
    else:
        error_message = (
            f"Generated data point has incorrect schema. Expected keys: {set(row.keys())}, "
            f"got: {set(json_synthetic_data_point.keys())}"
        )
        logger.error(error_message)
        raise ValueError(error_message)


async def generate_augmented_text_dataset(sampled_data: List[dict], keypair: Keypair) -> List[dict]:
    prompts = load_prompts()
    logger.info(f"Creating an augmented dataset with {len(sampled_data)} samples...")
    synthetic_dataset = []
    json_errors = 0
    generic_errors = 0
    consecutive_errors = 0
    max_consecutive_errors = 10

    total_batches = (len(sampled_data) + SYNTH_GEN_BATCH_SIZE - 1) // SYNTH_GEN_BATCH_SIZE
    for batch_idx in range(0, len(sampled_data), SYNTH_GEN_BATCH_SIZE):
        batch = sampled_data[batch_idx : batch_idx + SYNTH_GEN_BATCH_SIZE]
        current_batch = (batch_idx // SYNTH_GEN_BATCH_SIZE) + 1
        logger.info(f"Processing batch {current_batch}/{total_batches} ({len(batch)} samples)")

        tasks = [process_row(row, prompts, keypair) for row in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        batch_results = []
        for result in results:
            if isinstance(result, Exception):
                if isinstance(result, json.JSONDecodeError):
                    json_errors += 1
                else:
                    generic_errors += 1
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Maximum consecutive errors reached when generating the augmented dataset.")
                    return None
            else:
                consecutive_errors = 0  # Reset on success
                batch_results.append(result)

        synthetic_dataset.extend(batch_results)

        if batch_results:
            logger.info(
                f"Batch {current_batch}/{total_batches} complete. "
                f"Generated {len(batch_results)}/{len(batch)} samples successfully"
            )

    logger.info(
        f"Finished processing all batches. Generated {len(synthetic_dataset)} samples total. "
        f"JSON errors: {json_errors}, Other errors: {generic_errors}"
    )

    return synthetic_dataset
