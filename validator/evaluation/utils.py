import base64
import os
import shutil
import tempfile
from io import BytesIO

from datasets import get_dataset_config_names
from huggingface_hub import hf_hub_download
from PIL import Image
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


def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int((max_size / width) * height)
    else:
        new_height = max_size
        new_width = int((max_size / height) * width)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def crop_center(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    img_width, img_height = image.size
    left = (img_width - target_width) // 2
    top = (img_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    return image.crop((left, top, right, bottom))


def match_image_size(image1: Image.Image, image2: Image.Image) -> tuple[Image.Image, Image.Image]:
    if image1.size != image2.size:
        image1 = resize_image(image1)
        image2 = resize_image(image2)

        width1, height1 = image1.size
        width2, height2 = image2.size

        target_width = min(width1, width2)
        target_height = min(height1, height2)

        if width1 > width2 or height1 > height2:
            image1 = crop_center(image1, target_width, target_height)
        if width2 > width1 or height2 > height1:
            image2 = crop_center(image2, target_width, target_height)

    return image1, image2


def base64_to_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    image_stream = BytesIO(image_data)
    image = Image.open(image_stream)
    return image


def download_from_huggingface(repo_id: str, filename: str, local_dir: str) -> str:
    # Use a temp folder to ensure correct file placement
    try:
        local_filename = os.path.basename(filename)
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


def read_prompt_file(text_file_path: str) -> str:
    if os.path.exists(text_file_path):
        with open(text_file_path, "r", encoding="utf-8") as text_file:
            return text_file.read()
    return None
