#!/usr/bin/env python3
"""
Standalone script for image model training (SDXL or Flux)
"""

import os
import json
import toml
import sys
import shutil
import subprocess
import asyncio
import argparse
import uuid
from pathlib import Path

# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from core.config.config_handler import save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.utils import download_s3_file
from core.models.utility_models import ImageModelType
import core.constants as cst
from miner.utils import download_flux_unet


async def download_dataset_if_needed(dataset_zip_url, task_id):
    local_zip_path = f"{cst.DIFFUSION_DATASET_DIR}/{task_id}.zip"
    print(f"Downloading dataset from: {dataset_zip_url}")
    local_path = await download_s3_file(dataset_zip_url, local_zip_path)
    print(f"Downloaded dataset to: {local_path}")
    return local_path


def create_config(task_id, model, model_type, expected_repo_name=None, huggingface_username=None, huggingface_token=None):
    """Create the diffusion config file"""
    # Load appropriate config template
    if model_type == ImageModelType.SDXL.value:
        with open(cst.CONFIG_TEMPLATE_PATH_DIFFUSION_SDXL, "r") as file:
            config = toml.load(file)
        config["pretrained_model_name_or_path"] = model
    elif model_type == ImageModelType.FLUX.value:
        with open(cst.CONFIG_TEMPLATE_PATH_DIFFUSION_FLUX, "r") as file:
            config = toml.load(file)
        flux_unet_path = download_flux_unet(model)
        config["pretrained_model_name_or_path"] = f"{cst.CONTAINER_FLUX_PATH}/flux_unet_{model.replace('/', '_')}.safetensors"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Update config
    config["train_data_dir"] = f"/dataset/images/{task_id}/img/"
    
    # Use provided HF token or fall back to environment variable
    config["huggingface_token"] = huggingface_token or os.environ.get("HUGGINGFACE_TOKEN", "")
    
    # Use provided HF username or fall back to environment variable
    hf_username = huggingface_username or os.environ.get("HUGGINGFACE_USERNAME")
    config["huggingface_repo_id"] = f"{hf_username}/{expected_repo_name or str(uuid.uuid4())}"

    # Save config to file
    config_path = os.path.join("/dataset/configs", f"{task_id}.toml")
    save_config_toml(config, config_path)
    print(f"Created config at {config_path}")
    return config_path


def run_training(model_type, config_path):
    print(f"Starting training with config: {config_path}")
    training_command = [
        "accelerate", "launch", 
        "--dynamo_backend", "no", 
        "--dynamo_mode", "default", 
        "--mixed_precision", "bf16", 
        "--num_processes", "1", 
        "--num_machines", "1", 
        "--num_cpu_threads_per_process", "2", 
        f"/app/sd-scripts/{model_type}_train_network.py",
        "--config_file", config_path
    ]
    
    subprocess.run(training_command, check=True)


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux"], help="Model type")
    parser.add_argument("--hours-to-complete", type=int, required=True, help="Number of hours to complete the task")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--huggingface-token", help="Hugging Face token")
    parser.add_argument("--wandb-token", help="Weights & Biases token")
    parser.add_argument("--huggingface-username", help="Hugging Face username")
    args = parser.parse_args()

    # Create required directories
    os.makedirs("/dataset/configs", exist_ok=True)
    os.makedirs("/dataset/outputs", exist_ok=True)
    os.makedirs("/dataset/images", exist_ok=True)

    # Set environment variables
    if args.huggingface_token:
        os.environ["HUGGINGFACE_TOKEN"] = args.huggingface_token
    if args.wandb_token:
        os.environ["WANDB_TOKEN"] = args.wandb_token

    # Download dataset
    dataset_zip = await download_dataset_if_needed(args.dataset_zip, args.task_id)
    
    # Create config file
    config_path = create_config(
        args.task_id,
        args.model,
        args.model_type,
        args.expected_repo_name,
        args.huggingface_username,
        args.huggingface_token
    )
    
    # Prepare dataset
    print("Preparing dataset...")
    prepare_dataset(
        training_images_zip_path=dataset_zip,
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
    )
    
    # Run training
    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())