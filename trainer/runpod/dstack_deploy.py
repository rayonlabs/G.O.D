"""
dstack deployment utilities for RunPod
"""
import asyncio
import json
import os
import sys
import time
import uuid

from dstack.api import Task, Client, Resources, GPU

from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.utility_models import TaskType
from trainer import constants as cst
from trainer.image_manager import get_task_type
from trainer.utils.logging import logger
from trainer.tasks import log_task


def build_command(
    task: TrainerProxyRequest,
    task_type: TaskType,
    env_vars: dict[str, str],
) -> tuple[str, dict[str, str]]:
    """
    Build the command string for the training task.
    
    Args:
        task: Training task request
        task_type: Type of task
        env_vars: Dictionary to add environment variables to
        
    Returns:
        tuple: (command_string, updated_env_vars)
    """
    training_data = task.training_data
    
    if task_type == TaskType.IMAGETASK:
        # Use env var for dataset-zip if it contains special characters
        dataset_zip = training_data.dataset_zip
        if "&" in dataset_zip or "?" in dataset_zip:
            env_vars["DATASET_ZIP_URL"] = dataset_zip
            dataset_zip = "$DATASET_ZIP_URL"
        
        cmd_parts = [
            "--task-id", training_data.task_id,
            "--model", training_data.model,
            "--dataset-zip", dataset_zip,
            "--model-type", training_data.model_type,
            "--expected-repo-name", training_data.expected_repo_name,
            "--hours-to-complete", str(int(training_data.hours_to_complete)),
        ]
    else:
        # Text task - use env var for dataset URL if it contains special characters
        dataset = training_data.dataset
        if "&" in dataset or "?" in dataset:
            env_vars["DATASET_URL"] = dataset
            dataset = "$DATASET_URL"
        
        cmd_parts = [
            "--task-id", training_data.task_id,
            "--model", training_data.model,
            "--dataset", dataset,
            "--dataset-type", json.dumps(training_data.dataset_type.model_dump()),
            "--task-type", task_type.value,
            "--file-format", training_data.file_format.value,
            "--expected-repo-name", training_data.expected_repo_name,
            "--hours-to-complete", str(int(training_data.hours_to_complete)),
        ]
    
    # Build command string with proper quoting for complex arguments
    quoted_parts = []
    for arg in cmd_parts:
        if " " in arg or "&" in arg or "=" in arg or arg.startswith("$"):
            quoted_parts.append(f'"{arg}"')
        else:
            quoted_parts.append(arg)
    
    cmd_str = " ".join(quoted_parts)
    
    # Use the appropriate entrypoint based on task type
    if task_type == TaskType.IMAGETASK:
        command = f"python -m trainer.image_trainer {cmd_str}"
    else:
        command = f"python -m trainer.text_trainer {cmd_str}"
    
    return command, env_vars


async def deploy_to_runpod(
    task: TrainerProxyRequest,
    docker_image: str,
    log_labels: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """
    Deploy training task to RunPod using dstack.
    
    Args:
        task: Training task request
        docker_image: Docker image tag to use
        log_labels: Labels for logging
        
    Returns:
        tuple: (run_id, error_message)
    """
    try:
        client = Client.from_config()
        training_data = task.training_data
        task_type = get_task_type(task)
        
        # Build environment variables
        env = {
            "TRANSFORMERS_CACHE": cst.HUGGINGFACE_CACHE_PATH,
        }
        
        # Add WANDB environment if needed
        if task_type != TaskType.IMAGETASK:
            wandb_token = os.getenv("WANDB_TOKEN")
            if wandb_token:
                env["WANDB_TOKEN"] = wandb_token
                env["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "training")
        
        # Calculate GPU resources
        num_gpus = len(task.gpu_ids)
        gpu_memory = f"{num_gpus * 24}GB"  # Adjust based on your GPU requirements
        
        # Build command (this will add env vars for complex URLs)
        command, env = build_command(task, task_type, env)
        
        logger.info(f"Deploying to RunPod: {docker_image}", extra=log_labels)
        logger.info(f"Command: {command}", extra=log_labels)
        
        # Create dstack task
        dstack_task = Task(
            name=f"training-{training_data.task_id}",
            image=docker_image,
            env=env,
            commands=[command],
            resources=Resources(
                gpu=GPU(count=num_gpus, memory=gpu_memory),
                memory=f"{num_gpus * cst.MEMORY_PER_GPU_GB}GB",
            ),
            volumes=[
                {"name": "cache", "path": "/cache"},
                {"name": "checkpoints", "path": "/app/checkpoints"},
            ],
        )
        
        # Deploy
        run = client.runs.apply_configuration(
            configuration=dstack_task,
            repo=None,
        )
        
        logger.info(f"Task deployed to RunPod. Run ID: {run.id}", extra=log_labels)
        await log_task(training_data.task_id, task.hotkey, f"Deployed to RunPod. Run ID: {run.id}")
        
        return run.id, None
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to deploy to RunPod: {error_msg}", extra=log_labels)
        return None, error_msg


async def get_run_status(run_id: str) -> tuple[str | None, str | None]:
    """
    Get the current status of a dstack run.
    
    Args:
        run_id: dstack run ID
        
    Returns:
        tuple: (status, error_message)
    """
    try:
        client = Client.from_config()
        run = client.runs.get(run_id)
        
        if not run:
            return None, f"Run {run_id} not found"
        
        run.refresh()
        return str(run.status), None
        
    except Exception as e:
        return None, str(e)


async def monitor_run(
    run_id: str,
    task_id: str,
    hotkey: str,
    timeout_seconds: int,
    log_labels: dict[str, str] | None = None,
) -> tuple[bool, str | None]:
    """
    Monitor a dstack run until completion or timeout.
    
    Args:
        run_id: dstack run ID
        task_id: Training task ID
        hotkey: Hotkey
        timeout_seconds: Maximum time to wait
        log_labels: Labels for logging
        
    Returns:
        tuple: (success, error_message)
    """
    try:
        client = Client.from_config()
        run = client.runs.get(run_id)
        
        if not run:
            return False, f"Run {run_id} not found"
        
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < timeout_seconds:
            await asyncio.sleep(10)  # Check every 10 seconds
            run.refresh()
            
            status_str = str(run.status).lower()
            
            if status_str != last_status:
                logger.info(f"Run {run_id} status: {run.status}", extra=log_labels)
                await log_task(task_id, hotkey, f"RunPod status: {run.status}")
                last_status = status_str
            
            if "done" in status_str or "stopped" in status_str:
                # Get logs to check for success
                try:
                    logs = []
                    for log in run.logs():
                        logs.append(log.decode('utf-8', errors='ignore') if isinstance(log, bytes) else str(log))
                    log_text = "".join(logs)
                    
                    if "successfully" in log_text.lower() or "completed" in log_text.lower():
                        return True, None
                    else:
                        return False, "Training completed but may have failed - check logs"
                except Exception as e:
                    return False, f"Could not retrieve logs: {e}"
            
            if "failed" in status_str or "error" in status_str:
                error_msg = f"Run failed with status: {run.status}"
                try:
                    for log in run.logs():
                        log_str = log.decode('utf-8', errors='ignore') if isinstance(log, bytes) else str(log)
                        if "error" in log_str.lower():
                            error_msg = log_str[:500]  # First 500 chars
                            break
                except:
                    pass
                return False, error_msg
        
        # Timeout - try to stop the run
        try:
            logger.warning(f"Timeout reached for run {run_id}, attempting to stop...", extra=log_labels)
            await log_task(task_id, hotkey, f"Timeout reached ({timeout_seconds}s), stopping run...")
            run.stop()
        except Exception as stop_err:
            logger.warning(f"Failed to stop run {run_id}: {stop_err}", extra=log_labels)
        
        return False, f"Timeout reached ({timeout_seconds}s)"
        
    except asyncio.CancelledError:
        # Handle cancellation gracefully
        logger.info(f"Run monitoring cancelled for {run_id}", extra=log_labels)
        return False, "Monitoring cancelled"
    except Exception as e:
        return False, str(e)


async def deploy_hf_upload_task(
    task_id: str,
    hotkey: str,
    expected_repo_name: str,
    model: str,
    hf_token: str,
    hf_username: str,
    wandb_token: str | None = None,
    path_in_repo: str | None = None,
    log_labels: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """
    Deploy HF upload task to RunPod.
    
    This deploys the HF uploader container that mounts the checkpoints volume
    and uploads the trained model to HuggingFace.
    
    Args:
        task_id: Training task ID
        hotkey: Hotkey
        expected_repo_name: Expected HuggingFace repo name
        model: Model name
        hf_token: HuggingFace token
        hf_username: HuggingFace username
        wandb_token: Optional WANDB token
        path_in_repo: Optional path in repo (for image tasks)
        log_labels: Labels for logging
        
    Returns:
        tuple: (run_id, error_message)
    """
    try:
        client = Client.from_config()
        
        # Build environment variables
        env = {
            "HUGGINGFACE_TOKEN": hf_token,
            "HUGGINGFACE_USERNAME": hf_username,
            "MODEL": model,
            "TASK_ID": task_id,
            "EXPECTED_REPO_NAME": expected_repo_name,
        }
        
        if wandb_token:
            env["WANDB_TOKEN"] = wandb_token
            env["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "training")
            env["WANDB_LOGS_PATH"] = f"/app/checkpoints/wandb_logs/{task_id}_{hotkey}"
        
        if path_in_repo:
            env["HF_REPO_SUBFOLDER"] = path_in_repo
        
        # Build command - the HF uploader container expects these env vars
        # The container will look for checkpoints at /app/checkpoints/{task_id}/{expected_repo_name}
        command = f"python -m hf_uploader"
        
        logger.info(f"Deploying HF upload task to RunPod for task {task_id}", extra=log_labels)
        
        # Create dstack task for HF upload
        dstack_task = Task(
            name=f"hf-upload-{task_id}",
            image=cst.HF_UPLOAD_DOCKER_IMAGE,
            env=env,
            commands=[command],
            resources=Resources(
                memory="8GB",  # HF upload doesn't need much memory
            ),
            volumes=[
                {"name": "checkpoints", "path": "/app/checkpoints"},
            ],
        )
        
        # Deploy
        run = client.runs.apply_configuration(
            configuration=dstack_task,
            repo=None,
        )
        
        logger.info(f"HF upload task deployed. Run ID: {run.id}", extra=log_labels)
        await log_task(task_id, hotkey, f"HF upload task deployed. Run ID: {run.id}")
        
        return run.id, None
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to deploy HF upload task: {error_msg}", extra=log_labels)
        return None, error_msg

