#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText and DPO)
"""

import os
import json
import yaml
import sys
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
import core.constants as cst


def _dpo_format_prompt(row, format_str):
    result = format_str
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _dpo_format_chosen(row, format_str):
    result = format_str
    if "{chosen}" in format_str and cst.DPO_DEFAULT_FIELD_CHOSEN in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_CHOSEN]):
        result = result.replace("{chosen}", str(row[cst.DPO_DEFAULT_FIELD_CHOSEN]))
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _dpo_format_rejected(row, format_str):
    result = format_str
    if "{rejected}" in format_str and cst.DPO_DEFAULT_FIELD_REJECTED in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_REJECTED]):
        result = result.replace("{rejected}", str(row[cst.DPO_DEFAULT_FIELD_REJECTED]))
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _adapt_columns_for_dpo_dataset(dataset_path, dataset_type, apply_formatting=True):
    """
    Transform a DPO JSON dataset file to match axolotl's expected column names.

    Args:
        dataset_path: Path to the JSON dataset file
        dataset_type: DPODatasetType with field mappings
        apply_formatting: If True, apply formatting templates to the content
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    column_mapping = {
        dataset_type["field_prompt"]: cst.DPO_DEFAULT_FIELD_PROMPT,
        dataset_type["field_system"]: cst.DPO_DEFAULT_FIELD_SYSTEM,
        dataset_type["field_chosen"]: cst.DPO_DEFAULT_FIELD_CHOSEN,
        dataset_type["field_rejected"]: cst.DPO_DEFAULT_FIELD_REJECTED
    }
    df = df.rename(columns=column_mapping)

    if apply_formatting:
        if dataset_type.get("prompt_format") and dataset_type.get("prompt_format") != "{prompt}":
            format_str = dataset_type["prompt_format"]
            df[cst.DPO_DEFAULT_FIELD_PROMPT] = df.apply(lambda row: _dpo_format_prompt(row, format_str), axis=1)
        if dataset_type.get("chosen_format") and dataset_type.get("chosen_format") != "{chosen}":
            format_str = dataset_type["chosen_format"]
            df[cst.DPO_DEFAULT_FIELD_CHOSEN] = df.apply(lambda row: _dpo_format_chosen(row, format_str), axis=1)
        if dataset_type.get("rejected_format") and dataset_type.get("rejected_format") != "{rejected}":
            format_str = dataset_type["rejected_format"]
            df[cst.DPO_DEFAULT_FIELD_REJECTED] = df.apply(lambda row: _dpo_format_rejected(row, format_str), axis=1)

    output_data = df.to_dict(orient='records')
    with open(dataset_path, 'w') as f:
        json.dump(output_data, f, indent=2)


async def download_dataset_if_needed(dataset_url, file_format):
    if file_format == FileFormat.S3.value:
        print(f"Downloading dataset from S3: {dataset_url}")
        local_path = await download_s3_file(dataset_url)
        print(f"Downloaded dataset to: {local_path}")
        
        # Move the file to input_data directory
        dataset_filename = os.path.basename(local_path)
        input_data_path = f"/workspace/input_data/{dataset_filename}"
        shutil.copy(local_path, input_data_path)
        print(f"Copied dataset to: {input_data_path}")
        
        return input_data_path, FileFormat.JSON.value
    return dataset_url, file_format


def copy_dataset_if_needed(dataset_path, file_format):
    if file_format != FileFormat.HF.value:
        dataset_filename = os.path.basename(dataset_path)
        
        # Create destination directories if they don't exist
        os.makedirs("/workspace/axolotl/data", exist_ok=True)
        
        # Copy to all required locations
        data_path = f"/workspace/axolotl/data/{dataset_filename}"
        root_path = f"/workspace/axolotl/{dataset_filename}"
        
        print(f"Copying dataset from {dataset_path} to {data_path}")
        shutil.copy(dataset_path, data_path)
        
        print(f"Copying dataset from {dataset_path} to {root_path}")
        shutil.copy(dataset_path, root_path)
        
        return data_path
    return dataset_path


def create_config(task_id, model, dataset, dataset_type, file_format, expected_repo_name=None, huggingface_username=None):
    """Create the axolotl config file"""
    # Load base config
    with open("/workspace/axolotl/base.yml", "r") as file:
        config = yaml.safe_load(file)

    # Configure datasets
    config["datasets"] = []
    dataset_entry = create_dataset_entry(dataset, dataset_type, FileFormat(file_format))
    config["datasets"].append(dataset_entry)

    # Handle DPO tasks
    if isinstance(dataset_type, DPODatasetType):
        config["rl"] = "dpo"

    # Update config
    config = update_flash_attention(config, model)
    
    # Use provided HF username or fall back to environment variable
    hf_username = huggingface_username or os.environ.get("HUGGINGFACE_USERNAME")
    if hf_username:
        os.environ["HUGGINGFACE_USERNAME"] = hf_username
    config = update_model_info(config, model, task_id, expected_repo_name)
    
    # Set the dataset type to json for local files to avoid HF hub download
    if file_format != FileFormat.HF.value:
        for ds in config["datasets"]:
            # Make it an explicit local JSON file
            ds["ds_type"] = "json"
            
            # Ensure the path is set to a location within the container
            if "path" in ds:
                filename = os.path.basename(ds["path"])
                ds["path"] = "/workspace/axolotl/data"
                
            # Set data_files to the basename of the dataset file
            ds["data_files"] = [os.path.basename(dataset)]
            
            print(f"Setting dataset config to: {ds}")
    
    # Also set the dataset_prepared_path to avoid Hugging Face download attempts
    config["dataset_prepared_path"] = "/workspace/axolotl/data_prepared"
    
    config["mlflow_experiment_name"] = dataset

    # Save config to file
    config_path = os.path.join("/workspace/axolotl/configs", f"{task_id}.yml")
    save_config(config, config_path)
    print(f"Created config at {config_path}")
    # Debug: print the config for troubleshooting
    print("Config contents:")
    print(yaml.dump(config))
    return config_path


def run_training(config_path):
    print(f"Starting training with config: {config_path}")
    subprocess.run(["accelerate", "launch", "-m", "axolotl.cli.train", config_path], check=True)


async def main():
    # Parse command line arguments
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
    
    # Ensure all required directories exist
    os.makedirs("/workspace/axolotl/data", exist_ok=True)
    os.makedirs("/workspace/axolotl/data_prepared", exist_ok=True)
    os.makedirs("/workspace/axolotl/configs", exist_ok=True)
    os.makedirs("/workspace/axolotl/outputs", exist_ok=True)
    os.makedirs("/workspace/input_data", exist_ok=True)

    # Login to HF and WANDB if tokens provided
    if args.huggingface_token:
        print("Logging in to Hugging Face")
        subprocess.run(["huggingface-cli", "login", "--token", args.huggingface_token, "--add-to-git-credential"], check=True)

    if args.wandb_token:
        print("Logging in to W&B")
        subprocess.run(["wandb", "login", args.wandb_token], check=True)

    # Parse dataset_type from JSON string
    try:
        dataset_type_dict = json.loads(args.dataset_type)
        
        # Create the appropriate dataset type object based on task type
        if args.task_type == "DpoTask":
            dataset_type = DPODatasetType(**dataset_type_dict)
            print(f"Created DPO dataset type: {dataset_type}")
        elif args.task_type == "InstructTextTask":
            dataset_type = InstructDatasetType(**dataset_type_dict)
            print(f"Created InstructText dataset type: {dataset_type}")
        else:
            print(f"Unsupported task type: {args.task_type}")
            sys.exit(1)
            
    except json.JSONDecodeError as e:
        print(f"Error parsing dataset_type JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating dataset type object: {e}")
        sys.exit(1)

    # Download dataset if needed
    dataset_path, file_format = await download_dataset_if_needed(args.dataset, args.file_format)
    
    # Copy dataset to axolotl directories if needed
    dataset_path = copy_dataset_if_needed(dataset_path, file_format)
    
    # Handle DPO dataset adaptation if needed
    if args.task_type == "DpoTask" and file_format == FileFormat.JSON.value:
        _adapt_columns_for_dpo_dataset(dataset_path, dataset_type_dict, True)
        print(f"Adapted DPO dataset columns in {dataset_path}")
    
    # Create config file
    config_path = create_config(
        args.task_id, 
        args.model, 
        dataset_path, 
        dataset_type, 
        file_format, 
        args.expected_repo_name,
        args.huggingface_username
    )
    
    # Run training
    run_training(config_path)


if __name__ == "__main__":
    asyncio.run(main())