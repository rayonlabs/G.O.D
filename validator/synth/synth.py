import asyncio
import json
import random
from typing import List

import yaml
from datasets import load_dataset
from fiber import Keypair
from fiber.logging_utils import get_logger

from core.models.utility_models import Message
from core.models.utility_models import Prompts
from core.models.utility_models import Role
from validator.core.constants import MAX_SYNTH_DATA_POINTS
from validator.core.constants import OUTPUT_REFORMULATION_PROBABILITY
from validator.core.constants import PROMPT_PATH
from validator.core.constants import SYNTH_GEN_BATCH_SIZE
from validator.core.constants import SYNTH_MODEL
from validator.core.constants import SYNTH_MODEL_TEMPERATURE
from validator.evaluation.utils import get_default_dataset_config
from validator.utils.call_endpoint import post_to_nineteen_ai


logger = get_logger(__name__)


def load_prompts() -> Prompts:
    with open(PROMPT_PATH, "r") as file:
        prompts_dict = yaml.safe_load(file)
    return Prompts(**prompts_dict)


def load_and_sample_dataset(dataset_name: str, columns_to_sample: List[str]) -> List[dict]:
    try:
        config_name = get_default_dataset_config(dataset_name)
        dataset = load_dataset(dataset_name, config_name,
                               trust_remote_code=True, streaming=True)
    except Exception as e:
        logger.exception(f"Failed to load dataset {dataset_name}: {e}")
        raise e

    logger.info(f"Loading dataset: {dataset_name}")
    train_dataset = dataset["train"]

    filtered_dataset = train_dataset.remove_columns(
        [col for col in train_dataset.column_names if col not in columns_to_sample])

    num_samples = MAX_SYNTH_DATA_POINTS
    logger.info(f"Taking {num_samples} samples from {dataset_name}")

    sampled_data = filtered_dataset.shuffle(
        seed=42, buffer_size=1000).take(num_samples)

    sampled_data_list = [sample for sample in sampled_data]
    return sampled_data_list


def create_messages_for_distribution_replication(row: dict, prompts: Prompts) -> List[Message]:
    messages = []
    system_message = Message(role=Role.SYSTEM, content=prompts.in_context_learning_generation_sys)
    messages.append(system_message)
    schema = json.dumps({key: value for key, value in row.items()})
    user_message = Message(role=Role.USER, content=prompts.in_context_learning_generation_user.format(schema=schema))
    messages.append(user_message)
    return messages


def create_messages_for_output_reformulation(row: dict, output_field: str, prompts: Prompts) -> List[Message]:
    messages = []
    system_message = Message(role=Role.SYSTEM, content=prompts.output_field_reformulation_sys)
    messages.append(system_message)
    user_message = Message(role=Role.USER, content=prompts.output_field_reformulation_user.format(
        data=json.dumps(row),
        output_field=output_field
    ))
    messages.append(user_message)
    return messages


def create_messages_for_input_generation(
    reformulated_output: str,
    description: str,
    output_field: str,
    schema: dict,
    prompts: Prompts
) -> List[Message]:
    messages = []
    system_message = Message(role=Role.SYSTEM, content=prompts.input_field_generation_sys)
    messages.append(system_message)
    user_message = Message(role=Role.USER, content=prompts.input_field_generation_user.format(
        schema=json.dumps(schema),
        output_field=output_field,
        output=reformulated_output,
        description=description
    ))
    messages.append(user_message)
    return messages


def check_the_synthetic_data(synthetic_data_point: dict, original_data_columns: List[str]) -> bool:
    return set(synthetic_data_point.keys()) == set(original_data_columns)


def convert_to_nineteen_payload(
    messages: List[Message],
    model: str = SYNTH_MODEL,
    temperature: float = SYNTH_MODEL_TEMPERATURE,
    stream: bool = False
) -> dict:
    return {
        "messages": [message.model_dump() for message in messages],
        "model": model,
        "temperature": temperature,
        "stream": stream,
    }


async def generate_from_distribution(row: dict, prompts: Prompts, keypair: Keypair) -> str:
    messages = create_messages_for_distribution_replication(row, prompts)
    payload = convert_to_nineteen_payload(messages)
    synthetic_data_point = await post_to_nineteen_ai(payload, keypair)
    json_synthetic_data_point = (
        json.loads(synthetic_data_point) if isinstance(synthetic_data_point, str)
        else synthetic_data_point
    )
    return json_synthetic_data_point

async def generate_from_output(row: dict, output_field: str, prompts: Prompts, keypair: Keypair) -> str:
    # Step 1: Reformulate output and get description
    messages = create_messages_for_output_reformulation(row, output_field, prompts)
    payload = convert_to_nineteen_payload(messages)
    result = await post_to_nineteen_ai(payload, keypair)
    result_dict = json.loads(result) if isinstance(result, str) else result
    reformulated_output = result_dict["reformulated_output"]
    description = result_dict["description"]

    # Step 2: Generate inputs based on reformulated output
    schema = {k: type(v).__name__ for k, v in row.items()}
    messages = create_messages_for_input_generation(reformulated_output, description, output_field, schema, prompts)
    payload = convert_to_nineteen_payload(messages)
    result = await post_to_nineteen_ai(payload, keypair)
    generated_inputs = json.loads(result) if isinstance(result, str) else result
    generated_inputs[output_field] = reformulated_output # Double check the output is unchanged

    return generated_inputs


async def generate_synthetic_dataset(sampled_data: List[dict], column_to_reformulate: str | None, keypair: Keypair) -> List[dict]:
    prompts = load_prompts()
    logger.info("Creating synthetic dataset")  # Task id would be nice here
    synthetic_dataset = []
    json_errors = 0
    generic_errors = 0
    consecutive_errors = 0
    max_consecutive_errors = 3

    async def process_row(row, column_to_reformulate):
        # Use probability constant for deciding the method
        use_output_reformulation_method = random.random() < OUTPUT_REFORMULATION_PROBABILITY if column_to_reformulate else False

        if use_output_reformulation_method:
            json_synthetic_data_point = await generate_from_output(row, column_to_reformulate, prompts, keypair)
        else:
            json_synthetic_data_point = await generate_from_distribution(row, prompts, keypair)
        logger.info(json_synthetic_data_point)
        if check_the_synthetic_data(json_synthetic_data_point, row.keys()):
            return json_synthetic_data_point
        else:
            raise ValueError(
                f"Generated data point has incorrect schema. Expected keys: {set(row.keys())}, "
                f"got: {set(json_synthetic_data_point.keys())}"
            )

    for i in range(0, len(sampled_data), SYNTH_GEN_BATCH_SIZE):
        batch = sampled_data[i: i + SYNTH_GEN_BATCH_SIZE]
        tasks = [process_row(row, column_to_reformulate) for row in batch]
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
                    logger.error("Maximum consecutive errors reached. Stopping synth dataset process.")
                    return None
            else:
                consecutive_errors = 0  # Reset on success
                batch_results.append(result)

        synthetic_dataset.extend(batch_results)


    logger.info(
        f"Generated {len(synthetic_dataset)} synthetic data points"
    )


    return synthetic_dataset
