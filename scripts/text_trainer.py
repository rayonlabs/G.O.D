#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText and DPO)
"""

import os
import json
import yaml
import sys
import uuid
import shutil
import subprocess
import asyncio
import argparse
import pandas as pd
from pathlib import Path

# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from core.config.config_handler import create_dataset_entry, save_config, update_flash_attention, update_model_info
from core.utils import download_s3_file
from core.models.utility_models import FileFormat, InstructDatasetType, DPODatasetType, TaskType
from core.dpo_utils import adapt_columns_for_dpo_dataset
import core.constants as cst


async def download_dataset_if_needed(dataset_url, file_format, task_type=None, dataset_type=None):
    """Download dataset from S3 and process it if needed."""
    if file_format == FileFormat.S3.value:
        local_path = await download_s3_file(dataset_url)
        
        dataset_filename = os.path.basename(local_path)
        input_data_path = f"/workspace/input_data/{dataset_filename}"
        os.makedirs("/workspace/input_data", exist_ok=True)
        shutil.copy(local_path, input_data_path)
        
        if task_type == "DpoTask" and dataset_type:
            adapt_columns_for_dpo_dataset(input_data_path, dataset_type, apply_formatting=True)
        
        return input_data_path, FileFormat.JSON.value
    return dataset_url, file_format


def copy_dataset_if_needed(dataset_path, file_format):
    """Copy dataset to Axolotl directories for non-HF datasets."""
    if file_format != FileFormat.HF.value:
        dataset_filename = os.path.basename(dataset_path)
        
        os.makedirs("/workspace/axolotl/data", exist_ok=True)
        os.makedirs("/workspace/axolotl", exist_ok=True)
        
        data_path = f"/workspace/axolotl/data/{dataset_filename}"
        root_path = f"/workspace/axolotl/{dataset_filename}"
        
        shutil.copy(dataset_path, data_path)
        shutil.copy(dataset_path, root_path)
        
        return data_path
    return dataset_path


def create_config(task_id, model, dataset, dataset_type, file_format, expected_repo_name=None, 
                huggingface_username=None, huggingface_token=None, disable_upload=False):
    """Create the axolotl config file with appropriate settings."""
    with open("/workspace/axolotl/base.yml", "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = [create_dataset_entry(dataset, dataset_type, FileFormat(file_format))]
    config["base_model"] = model
    config["wandb_runid"] = task_id
    config["wandb_name"] = task_id
    config["dataset_prepared_path"] = "/workspace/axolotl/data_prepared"
    config["mlflow_experiment_name"] = dataset
    
    config = update_flash_attention(config, model)
    
    if isinstance(dataset_type, DPODatasetType):
        config["rl"] = "dpo"

    if not disable_upload:
        hf_username = huggingface_username or os.environ.get("HUGGINGFACE_USERNAME", "rayonlabs")
        os.environ["HUGGINGFACE_USERNAME"] = hf_username
        
        repo_name = expected_repo_name or str(uuid.uuid4())
        config["hub_model_id"] = f"{hf_username}/{repo_name}"
        
        if huggingface_token:
            os.environ["HUGGINGFACE_TOKEN"] = huggingface_token
    else:
        config.pop("hub_model_id", None)
        config.pop("hub_strategy", None)
        config.pop("hub_token", None)
    
    if file_format != FileFormat.HF.value:
        for ds in config["datasets"]:
            ds["ds_type"] = "json"
            
            if "path" in ds:
                ds["path"] = "/workspace/axolotl/data"
                
            ds["data_files"] = [os.path.basename(dataset)]

    config_path = os.path.join("/workspace/axolotl/configs", f"{task_id}.yml")
    save_config(config, config_path)
    return config_path


def make_repo_public(repo_id):
    """Make a Hugging Face repository public or create it if it doesn't exist."""
    from huggingface_hub import HfApi
    
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return False
    
    try:
        api = HfApi(token=token)
        
        try:
            api.repo_info(repo_id=repo_id)
            api.update_repo_visibility(repo_id=repo_id, private=False)
        except Exception:
            api.create_repo(repo_id=repo_id, private=False, exist_ok=True)
        
        return True
    except Exception as e:
        print(f"Error making repository public: {e}")
        return False


def run_training(config_path):
    """Run the training process using the specified config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    training_env = os.environ.copy()
    training_env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    training_env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    subprocess.run(
        ["accelerate", "launch", "-m", "axolotl.cli.train", config_path], 
        check=True,
        env=training_env
    )
    
    repo_id = config.get("hub_model_id")
    if repo_id and os.environ.get("HUGGINGFACE_TOKEN"):
        make_repo_public(repo_id)


async def main():
    """Main entry point for the text trainer script."""
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset", required=True, help="Dataset path or HF dataset name")
    parser.add_argument("--dataset-type", required=True, help="JSON string of dataset type config")
    parser.add_argument("--task-type", required=True, choices=["InstructTextTask", "DpoTask"], help="Type of task")
    parser.add_argument("--file-format", required=True, choices=["csv", "json", "hf", "s3"], help="File format")
    parser.add_argument("--hours-to-complete", type=int, required=True, help="Number of hours to complete the task")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--huggingface-token", help="Hugging Face token")
    parser.add_argument("--wandb-token", help="Weights & Biases token")
    parser.add_argument("--huggingface-username", help="Hugging Face username")
    args = parser.parse_args()
    
    for directory in [
        "/workspace/axolotl/data", 
        "/workspace/axolotl/data_prepared", 
        "/workspace/axolotl/configs", 
        "/workspace/axolotl/outputs", 
        "/workspace/input_data",
        "/workspace/axolotl"
    ]:
        os.makedirs(directory, exist_ok=True)

    if args.huggingface_token:
        subprocess.run(["huggingface-cli", "login", "--token", args.huggingface_token, "--add-to-git-credential"], check=True)

    if args.wandb_token:
        subprocess.run(["wandb", "login", args.wandb_token], check=True)

    try:
        dataset_type_dict = json.loads(args.dataset_type)
        
        if args.task_type == "DpoTask":
            dataset_type = DPODatasetType(**dataset_type_dict)
        elif args.task_type == "InstructTextTask":
            dataset_type = InstructDatasetType(**dataset_type_dict)
        else:
            sys.exit(f"Unsupported task type: {args.task_type}")
    except Exception as e:
        sys.exit(f"Error creating dataset type object: {e}")

    dataset_path, file_format = await download_dataset_if_needed(
        args.dataset, 
        args.file_format,
        args.task_type,
        dataset_type
    )
    
    dataset_path = copy_dataset_if_needed(dataset_path, file_format)
    
    config_path = create_config(
        args.task_id, 
        args.model, 
        dataset_path, 
        dataset_type, 
        file_format, 
        args.expected_repo_name,
        args.huggingface_username,
        args.huggingface_token
    )
    
    run_training(config_path)


if __name__ == "__main__":
    asyncio.run(main())