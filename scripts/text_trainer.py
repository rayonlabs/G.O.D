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


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from core.config.config_handler import create_dataset_entry, save_config, update_flash_attention
from core.models.utility_models import FileFormat, InstructTextDatasetType, DpoDatasetType, TaskType
from core.dpo_utils import adapt_columns_for_dpo_dataset
import core.constants as cst
import trainer.constants as train_cst


import os
import json

def patch_model_metadata(output_dir: str, base_model_id: str):
    try:
        adapter_config_path = os.path.join(output_dir, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                config = json.load(f)

            config["base_model_name_or_path"] = base_model_id

            with open(adapter_config_path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"Updated adapter_config.json with base_model: {base_model_id}", flush=True)
        else:
            print(" adapter_config.json not found", flush=True)

        readme_path = os.path.join(output_dir, "README.md")

        if os.path.exists(readme_path):
            with open(readme_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                if line.strip().startswith("base_model:"):
                    new_lines.append(f"base_model: {base_model_id}\n")
                else:
                    new_lines.append(line)

            with open(readme_path, "w") as f:
                f.writelines(new_lines)

            print(f"Updated README.md with base_model: {base_model_id}", flush=True)
        else:
            print("README.md not found", flush=True)

    except Exception as e:
        print(f"Error updating metadata: {e}", flush=True)
        pass


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


def create_config(task_id, model, dataset, dataset_type, file_format, output_dir, expected_repo_name=None, 
                huggingface_username=None, huggingface_token=None, disable_upload=True):
    """Create the axolotl config file with appropriate settings."""
    with open("/workspace/axolotl/base.yml", "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = [create_dataset_entry(dataset, dataset_type, FileFormat(file_format))]
    print(f"Dataset entry created: {config['datasets']}", flush=True)
    config["base_model"] = f"{train_cst.CACHE_PATH}/{task_id}/models/{model.replace('/', '--')}"

    config["dataset_prepared_path"] = "/workspace/axolotl/data_prepared"
    config["mlflow_experiment_name"] = dataset
    os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir

    config = update_flash_attention(config, model)
    
    if isinstance(dataset_type, DpoDatasetType):
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
        for key in list(config.keys()):
            if key.startswith("wandb"):
                config.pop(key)
            
    if file_format != FileFormat.HF.value:
        for ds in config["datasets"]:
            ds["ds_type"] = "json"
            
            if "path" in ds:
                ds["path"] = "/workspace/axolotl/data"
                
            ds["data_files"] = [os.path.basename(dataset)]

    config_path = os.path.join("/workspace/axolotl/configs", f"{task_id}.yml")
    save_config(config, config_path)
    return config_path


def run_training(config_path):
    print(f"Starting training with config: {config_path}", flush=True)
    """Run the training process using the specified config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    training_env = os.environ.copy()
    training_env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    training_env["HF_HUB_DISABLE_TELEMETRY"] = "1"

    training_command = [
    "accelerate", "launch", 
    "-m", "axolotl.cli.train", 
    config_path
    ]

    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)
    
    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")
    
 

async def main():
    print("---STARTING TEXT TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset", required=True, help="Dataset path or HF dataset name")
    parser.add_argument("--dataset-type", required=True, help="JSON string of dataset type config")
    parser.add_argument("--task-type", required=True, choices=["InstructTextTask", "DpoTask"], help="Type of task")
    parser.add_argument("--file-format", required=True, choices=["csv", "json", "hf", "s3"], help="File format")
    parser.add_argument("--hours-to-complete", type=int, required=True, help="Number of hours to complete the task")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
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

    try:
        dataset_type_dict = json.loads(args.dataset_type)
        
        if args.task_type == "DpoTask":
            dataset_type = DpoDatasetType(**dataset_type_dict)
        elif args.task_type == "InstructTextTask":
            dataset_type = InstructTextDatasetType(**dataset_type_dict)
        else:
            sys.exit(f"Unsupported task type: {args.task_type}")
    except Exception as e:
        sys.exit(f"Error creating dataset type object: {e}")

    base_dataset_path = f"{train_cst.CACHE_PATH}/{args.task_id}/datasets"
    dataset_path = f"{base_dataset_path}/{args.task_id}_train_data.json" if args.file_format == FileFormat.S3.value else f"{base_dataset_path}/{args.dataset.replace('/', '--')}"

    
    dataset_path = copy_dataset_if_needed(dataset_path, args.file_format)

    print(args.file_format, flush=True)

    if args.file_format == FileFormat.S3.value and args.task_type == TaskType.DPOTASK.value:
        print("Adapting columns for DPO dataset...", flush=True)
        adapt_columns_for_dpo_dataset(dataset_path, dataset_type, apply_formatting=True)

    output_dir = f"/workspace/axolotl/outputs/{args.task_id}/{args.expected_repo_name}"
    
    config_path = create_config(
        args.task_id, 
        args.model, 
        dataset_path, 
        dataset_type, 
        args.file_format,
        output_dir,
        args.expected_repo_name,
    )
    
    run_training(config_path)

    patch_model_metadata(output_dir, args.model)


if __name__ == "__main__":
    asyncio.run(main())
