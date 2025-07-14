import base64
import os
import shutil
import tempfile
import time
from io import BytesIO

from datasets import get_dataset_config_names
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoConfig
from transformers import AutoModelForCausalLM

from validator.utils.logging import get_logger
from validator.utils.retry_utils import retry_on_5xx


logger = get_logger(__name__)
hf_api = HfApi()


def model_is_a_finetune(original_repo: str, finetuned_model: AutoModelForCausalLM) -> bool:
    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            original_config = AutoConfig.from_pretrained(original_repo, token=os.environ.get("HUGGINGFACE_TOKEN"))
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise e

            error_msg = str(e).lower()
            if any(pattern in error_msg for pattern in ["connection", "timeout", "5xx", "too many requests", "couldn't connect"]):
                delay = base_delay * (2**attempt)
                logger.info(f"HuggingFace connection issue (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise e
    finetuned_config = finetuned_model.config

    try:
        architecture_classes_match = finetuned_config.architectures == original_config.architectures
    except Exception as e:
        logger.debug(f"There is an issue with checking the architecture classes {e}")
        architecture_classes_match = False

    attrs_to_compare = [
        "architectures",
        "hidden_size",
        "n_layer",
        "intermediate_size",
        "head_dim",
        "hidden_act",
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

    logger.info(f"Architecture same: {architecture_same}, Architecture classes match: {architecture_classes_match}")
    return architecture_same and architecture_classes_match


@retry_on_5xx()
def check_for_lora(model_id: str) -> bool:
    """
    Check if a Hugging Face model has LoRA adapters by looking for adapter_config.json.

    Args:
        model_id (str): The Hugging Face model ID (e.g., 'username/model-name') or path

    Returns:
        bool: True if it's a LoRA adapter, False otherwise
    """
    try:
        return "adapter_config.json" in hf_api.list_repo_files(model_id)
    except Exception as e:
        logger.error(f"Error checking for LoRA adapters: {e}")
        return False


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


def adjust_image_size(image: Image.Image) -> Image.Image:
    width, height = image.size

    if width > height:
        new_width = 1024
        new_height = int((height / width) * 1024)
    else:
        new_height = 1024
        new_width = int((width / height) * 1024)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8

    width, height = image.size
    crop_width = min(width, new_width)
    crop_height = min(height, new_height)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    image = image.crop((left, top, right, bottom))

    return image


def base64_to_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    image_stream = BytesIO(image_data)
    image = Image.open(image_stream)
    return image


def download_from_huggingface(repo_id: str, filename: str, local_dir: str) -> str:
    # Use a temp folder to ensure correct file placement
    try:
        local_filename = f"models--{repo_id.replace('/', '--')}.safetensors"
        final_path = os.path.join(local_dir, local_filename)
        if os.path.exists(final_path):
            logger.info(f"File {filename} already exists. Skipping download.")
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=temp_dir)
                shutil.move(temp_file_path, final_path)
            logger.info(f"File {filename} downloaded successfully")
        return final_path
    except Exception as e:
        logger.error(f"Error downloading file: {e}")


def list_supported_images(dataset_path: str, extensions: tuple) -> list[str]:
    return [file_name for file_name in os.listdir(dataset_path) if file_name.lower().endswith(extensions)]


def read_image_as_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    img_format = image.format if image.format else "PNG"
    image.save(buffer, format=img_format)
    return base64.b64encode(buffer.getvalue()).decode()


def read_prompt_file(text_file_path: str) -> str:
    if os.path.exists(text_file_path):
        with open(text_file_path, "r", encoding="utf-8") as text_file:
            return text_file.read()
    return None
