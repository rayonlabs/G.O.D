import asyncio
import json
from typing import List

import yaml
from datasets import load_dataset
from fiber import Keypair
from fiber.logging_utils import get_logger

from core.models.utility_models import Message
from core.models.utility_models import Prompts
from core.models.utility_models import Role
from validator.core.constants import MAX_SYNTH_DATA_POINTS
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


def create_messages_from_row(row: dict, prompts: Prompts) -> List[Message]:
    messages = []
    system_message = Message(
        role=Role.SYSTEM, content=prompts.synth_data_creation_sys)
    messages.append(system_message)
    schema = json.dumps({key: value for key, value in row.items()})
    user_message = Message(
        role=Role.USER, content=prompts.synth_data_creation_prompt.format(schema=schema))
    messages.append(user_message)
    return messages


def check_the_synthetic_data(synthetic_data_point: dict, original_data_columns: List[str]) -> bool:
    return set(synthetic_data_point.keys()) == set(original_data_columns)


async def generate_synthetic_dataset(sampled_data: List[dict], keypair: Keypair) -> List[dict]:
    prompts = load_prompts()
    logger.info("Creating synthetic dataset")  # Task id would be nice here
    synthetic_dataset = []
    json_errors = 0
    generic_errors = 0

    consecutive_errors = 0
    max_consecutive_errors = 3

    async def process_row(row):
        nonlocal consecutive_errors

        nonlocal json_errors
        nonlocal generic_errors

        messages = create_messages_from_row(row, prompts)
        payload = {
            "messages": [message.model_dump() for message in messages],
            "model": SYNTH_MODEL,
            "temperature": SYNTH_MODEL_TEMPERATURE,
            "stream": False,
        }
        try:

            synthetic_data_point = await post_to_nineteen_ai(payload, keypair)
            logger.info(synthetic_data_point)

            try:
                if isinstance(synthetic_data_point, str):
                    json_synthetic_data_point = json.loads(
                        synthetic_data_point)
                elif synthetic_data_point is not None:
                    json_synthetic_data_point = synthetic_data_point
                else:
                    consecutive_errors += 1
            except json.JSONDecodeError:

                json_errors += 1
                consecutive_errors += 1
                return None

            if check_the_synthetic_data(json_synthetic_data_point, row.keys()):
                consecutive_errors = 0  # Reset counter on success
                return json_synthetic_data_point
            else:
                consecutive_errors += 1

        except Exception as e:
            generic_errors += 1
            consecutive_errors += 1

        if consecutive_errors >= max_consecutive_errors:
            logger.error(

                f"Stopping process due to {consecutive_errors} consecutive errors with the synth production")

            raise RuntimeError(
                f"Process stopped after {consecutive_errors} consecutive errors")

        return None

    try:
        for i in range(0, len(sampled_data), SYNTH_GEN_BATCH_SIZE):
            batch = sampled_data[i: i + SYNTH_GEN_BATCH_SIZE]
            tasks = [process_row(row) for row in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, RuntimeError):

                    logger.error(
                        "Maximum consecutive errors reached. Stopping synth dataset process.")

                    return None
            valid_results = [
                r for r in results if r is not None and not isinstance(r, Exception)]
            synthetic_dataset.extend(valid_results)


        logger.info(
            f"Generated {len(synthetic_dataset)} synthetic data points"
        )


        return synthetic_dataset
    except RuntimeError:
        return None
