import argparse
import asyncio
import os
import shutil
import tempfile

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
from transformers import CLIPTokenizer

from core.models.utility_models import FileFormat
from core.models.utility_models import TaskType
from core.utils import download_s3_file
from trainer import constants as cst
import trainer.utils.training_paths as train_paths


hf_api = HfApi()


async def download_text_dataset(task_id, dataset_url, file_format, dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)

    if file_format == FileFormat.S3.value:
        input_data_path = train_paths.get_text_dataset_path(task_id)

        if not os.path.exists(input_data_path):
            local_path = await download_s3_file(dataset_url)
            shutil.copy(local_path, input_data_path)

    elif file_format == FileFormat.HF.value:
        repo_name = dataset_url.replace("/", "--")
        input_data_path = os.path.join(dataset_dir, repo_name)

        if not os.path.exists(input_data_path):
            snapshot_download(repo_id=dataset_url, repo_type="dataset", local_dir=input_data_path, local_dir_use_symlinks=False)

    return input_data_path, file_format


async def download_image_dataset(dataset_zip_url, task_id, dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    local_zip_path = train_paths.get_image_training_zip_save_path(task_id)
    print(f"Downloading dataset from: {dataset_zip_url}")
    local_path = await download_s3_file(dataset_zip_url, local_zip_path)
    print(f"Downloaded dataset to: {local_path}")
    return local_path


def is_safetensors_available(repo_id: str) -> tuple[bool, str | None]:
    files_metadata = hf_api.list_repo_tree(repo_id=repo_id, repo_type="model")
    check_size_in_gb = 6
    total_check_size = check_size_in_gb * 1024 * 1024 * 1024
    largest_file = None

    for file in files_metadata:
        if hasattr(file, "size") and file.size is not None:
            if file.path.endswith(".safetensors") and file.size > total_check_size:
                if largest_file is None or file.size > largest_file.size:
                    largest_file = file

    if largest_file:
        return True, largest_file.path
    return False, None


def download_from_huggingface(repo_id: str, filename: str, local_dir: str) -> str:
    try:
        local_dir = os.path.expanduser(local_dir)
        local_filename = f"{repo_id.replace('/', '_')}.safetensors"
        final_path = os.path.join(local_dir, local_filename)
        os.makedirs(local_dir, exist_ok=True)
        if os.path.exists(final_path):
            print(f"File {filename} already exists. Skipping download.")
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=temp_dir)
                shutil.move(temp_file_path, final_path)
            print(f"File {filename} downloaded successfully")
        return final_path
    except Exception as e:
        raise e


def download_flux_unet(repo_id: str, output_dir: str) -> str:
    files_metadata = hf_api.list_repo_tree(repo_id=repo_id, repo_type="model")
    file_path = None
    for file in files_metadata:
        if hasattr(file, "size") and file.size is not None:
            if file.path.endswith(".safetensors") and file.size > 10 * 1024 * 1024 * 1024:
                file_path = file.path
                local_path = download_from_huggingface(repo_id, file_path, output_dir)
    if not file_path:
        raise FileNotFoundError(f"No valid file found in root of repo '{repo_id}'.")

    return local_path


async def download_base_model(repo_id: str, save_root: str) -> str:
    model_name = repo_id.replace("/", "--")
    save_path = os.path.join(save_root, model_name)
    if os.path.exists(save_path):
        print(f"Model {repo_id} already exists at {save_path}. Skipping download.")
        return save_path
    else:
        has_safetensors, safetensors_path = is_safetensors_available(repo_id)
        if has_safetensors and safetensors_path:
            return download_from_huggingface(repo_id, safetensors_path, save_path)
        else:
            snapshot_download(repo_id=repo_id, repo_type="model", local_dir=save_path, local_dir_use_symlinks=False)
            return save_path


async def download_axolotl_base_model(repo_id: str, save_dir: str) -> str:
    model_dir = os.path.join(save_dir, repo_id.replace("/", "--"))
    if os.path.exists(model_dir):
        print(f"Model {repo_id} already exists at {model_dir}. Skipping download.")
        return model_dir
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=model_dir, local_dir_use_symlinks=False)
    return model_dir


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--task-type",
        required=True,
        choices=[TaskType.IMAGETASK.value, TaskType.INSTRUCTTEXTTASK.value, TaskType.DPOTASK.value, TaskType.GRPOTASK.value],
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--file-format")
    args = parser.parse_args()

    dataset_dir = cst.CACHE_DATASETS_DIR
    model_dir = cst.CACHE_MODELS_DIR
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Downloading datasets to: {dataset_dir}", flush=True)
    print(f"Downloading models to: {model_dir}", flush=True)

    if args.task_type == TaskType.IMAGETASK.value:
        dataset_zip_path = await download_image_dataset(args.dataset, args.task_id, dataset_dir)
        model_path = await download_base_model(args.model, model_dir)
        print("Downloading clip models", flush=True)
        CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cst.HUGGINGFACE_CACHE_PATH)
        CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", cache_dir=cst.HUGGINGFACE_CACHE_PATH)
        snapshot_download(
            repo_id="google/t5-v1_1-xxl",
            repo_type="model",
            cache_dir="/cache/hf_cache/",
            local_dir_use_symlinks=False,
            allow_patterns=["tokenizer_config.json", "spiece.model", "special_tokens_map.json", "config.json"],
        )
    else:
        dataset_path, _ = await download_text_dataset(args.task_id, args.dataset, args.file_format, dataset_dir)
        model_path = await download_axolotl_base_model(args.model, model_dir)

    print(f"Model path: {model_path}", flush=True)
    print(f"Dataset path: {dataset_dir}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
