import os

from datasets import get_dataset_config_names
from transformers import AutoConfig
from transformers import AutoModelForCausalLM

from validator.utils.logging import get_logger


logger = get_logger(__name__)


def model_is_a_finetune(original_repo: str, finetuned_model: AutoModelForCausalLM) -> bool:
    original_config = AutoConfig.from_pretrained(original_repo, token=os.environ.get("HUGGINGFACE_TOKEN"))
    finetuned_config = finetuned_model.config

    try:
        if hasattr(finetuned_model, "name_or_path"):
            finetuned_model_path = finetuned_model.name_or_path
        else:
            finetuned_model_path = finetuned_model.config._name_or_path

        adapter_config = os.path.join(finetuned_model_path, "adapter_config.json")
        if os.path.exists(adapter_config):
            has_lora_modules = True
            logger.info(f"Adapter config found: {adapter_config}")
        else:
            logger.info(f"Adapter config not found at {adapter_config}")
            has_lora_modules = False
        base_model_match = finetuned_config._name_or_path == original_repo
    except Exception as e:
        logger.debug(f"There is an issue with checking the finetune path {e}")
        base_model_match = True
        has_lora_modules = False

    attrs_to_compare = [
        "architectures",
        "hidden_size",
        "n_layer",
        "model_type",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    ]
    architecture_same = True
    for attr in attrs_to_compare:
        if hasattr(original_config, attr):
            if not hasattr(finetuned_config, attr):
                architecture_same = False
                break
            if getattr(original_config, attr) != getattr(finetuned_config, attr):
                architecture_same = False
                break

    logger.info(
        f"Architecture same: {architecture_same}, Base model match: {base_model_match}, Has lora modules: {has_lora_modules}"
    )
    return architecture_same and (base_model_match or has_lora_modules)


def get_default_dataset_config(dataset_name: str) -> str | None:
    try:
        logger.info(dataset_name)
        config_names = get_dataset_config_names(dataset_name)
    except Exception:
        return None
    if config_names:
        logger.info(f"Taking the first config name: {config_names[0]} for dataset: {dataset_name}")
        # logger.info(f"Dataset {dataset_name} has configs: {config_names}. Taking the first config name: {config_names[0]}")
        return config_names[0]
    else:
        return None
