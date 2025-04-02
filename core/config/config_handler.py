import os
import uuid

import toml
import yaml
from fiber.logging_utils import get_logger
from transformers import AutoTokenizer

import core.constants as cst
from core.models.utility_models import DPODatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import InstructDatasetType


logger = get_logger(__name__)


def create_dataset_entry(
    dataset: str,
    dataset_type: InstructDatasetType | DPODatasetType,
    file_format: FileFormat,
) -> dict:
    dataset_entry = {"path": dataset}

    if file_format == FileFormat.JSON:
        dataset_entry = {"path": f"/workspace/input_data/{os.path.basename(dataset)}"}

    if isinstance(dataset_type, InstructDatasetType):
        instruct_type_dict = {key: value for key, value in dataset_type.model_dump().items() if value is not None}
        dataset_entry.update(_process_instruct_dataset_fields(instruct_type_dict))
    elif isinstance(dataset_type, DPODatasetType):
        dataset_entry.update(_process_dpo_dataset_fields(dataset_type))
    else:
        raise ValueError("Invalid dataset_type provided.")

    if file_format != FileFormat.HF:
        dataset_entry["ds_type"] = file_format.value
        dataset_entry["data_files"] = [os.path.basename(dataset)]

    return dataset_entry


def update_flash_attention(config: dict, model: str):
    # You might want to make this model-dependent
    config["flash_attention"] = False
    return config


def update_model_info(config: dict, model: str, job_id: str = "", expected_repo_name: str | None = None):
    logger.info("WE ARE UPDATING THE MODEL INFO")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        config["special_tokens"] = {"pad_token": tokenizer.eos_token}

    config["base_model"] = model
    config["wandb_runid"] = job_id
    config["wandb_name"] = job_id
    config["hub_model_id"] = f"{cst.HUGGINGFACE_USERNAME}/{expected_repo_name or str(uuid.uuid4())}"

    return config


def save_config(config: dict, config_path: str):
    with open(config_path, "w") as file:
        yaml.dump(config, file)


def save_config_toml(config: dict, config_path: str):
    with open(config_path, "w") as file:
        toml.dump(config, file)


def _process_dpo_dataset_fields(dataset_type: DPODatasetType) -> dict:
    # Enable below when https://github.com/axolotl-ai-cloud/axolotl/issues/1417 is fixed
    # context: https://discord.com/channels/1272221995400167588/1355226588178022452/1356982842374226125

    # dpo_type_dict = dataset_type.model_dump()
    # dpo_type_dict["type"] = "user_defined.default"
    # if not dpo_type_dict.get("prompt_format"):
    #     if dpo_type_dict.get("field_system"):
    #         dpo_type_dict["prompt_format"] = "{system} {prompt}"
    #     else:
    #         dpo_type_dict["prompt_format"] = "{prompt}"
    # return dpo_type_dict

    # Fallback to https://axolotl-ai-cloud.github.io/axolotl/docs/rlhf.html#chatml.intel
    # Column names are hardcoded in axolotl: "DPO_DEFAULT_FIELD_SYSTEM",
    # "DPO_DEFAULT_FIELD_PROMPT", "DPO_DEFAULT_FIELD_CHOSEN", "DPO_DEFAULT_FIELD_REJECTED"
    return {"type": cst.DPO_DEFAULT_DATASET_TYPE, "split": "train"}


def _process_instruct_dataset_fields(instruct_type_dict: dict) -> dict:
    if not instruct_type_dict.get("field_output"):
        return {
            "type": "completion",
            "field": instruct_type_dict.get("field_instruction"),
        }

    processed_dict = instruct_type_dict.copy()
    processed_dict.setdefault("no_input_format", "{instruction}")
    if processed_dict.get("field_input"):
        processed_dict.setdefault("format", "{instruction} {input}")
    else:
        processed_dict.setdefault("format", "{instruction}")

    return {"format": "custom", "type": processed_dict}
