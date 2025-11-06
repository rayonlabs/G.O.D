"""
RunPod training task orchestrator - integrates with existing training workflow
"""
import asyncio
import os
import re

from core.models.payload_models import TrainerProxyRequest
from core.models.utility_models import TaskType
from trainer import constants as cst
from trainer.image_manager import get_task_type
from trainer.runpod.docker_utils import build_and_push_image
from trainer.runpod.dstack_deploy import deploy_to_runpod, monitor_run, deploy_hf_upload_task
from trainer.runpod.log_streamer import stream_runpod_logs
from trainer.tasks import complete_task
from trainer.tasks import log_task
from trainer.tasks import update_wandb_url
from trainer.utils.logging import logger


async def start_runpod_training_task(
    task: TrainerProxyRequest,
    local_repo_path: str,
):
    """
    Start training task on RunPod using dstack.
    
    This function mirrors the logic of start_training_task but deploys to RunPod instead of local Docker.
    
    Args:
        task: Training task request
        local_repo_path: Path to local repository clone
    """
    try:
        training_data = task.training_data
        success = False
        timeout_seconds = int(training_data.hours_to_complete * 3600)
        task_type = get_task_type(task)
        training_data.hours_to_complete = int(training_data.hours_to_complete)
        
        log_labels = {
            "task_id": training_data.task_id,
            "hotkey": task.hotkey,
            "model": training_data.model,
            "task_type": task_type,
            "expected_repo": training_data.expected_repo_name,
            "backend": "runpod",
            **(
                {"dataset_type": str(training_data.dataset_type)}
                if getattr(training_data, "dataset_type", None) is not None
                else {}
            ),
        }
        
        # Determine dockerfile path
        dockerfile_path = (
            f"{local_repo_path}/{cst.DEFAULT_IMAGE_DOCKERFILE_PATH}"
            if task_type == TaskType.IMAGETASK
            else f"{local_repo_path}/{cst.DEFAULT_TEXT_DOCKERFILE_PATH}"
        )
        
        logger.info("Starting RunPod training task", extra=log_labels)
        await log_task(training_data.task_id, task.hotkey, "Starting RunPod deployment")
        
        # Get DockerHub credentials
        dockerhub_username = os.getenv("DOCKERHUB_USERNAME")
        dockerhub_password = os.getenv("DOCKERHUB_PASSWORD")
        
        if not dockerhub_username or not dockerhub_password:
            error_msg = "DOCKERHUB_USERNAME and DOCKERHUB_PASSWORD environment variables required"
            logger.error(error_msg, extra=log_labels)
            await log_task(training_data.task_id, task.hotkey, f"[ERROR] {error_msg}")
            await complete_task(training_data.task_id, task.hotkey, success=False)
            return
        
        # Build and push Docker image
        await log_task(training_data.task_id, task.hotkey, "Building Docker image...")
        tag_suffix = f"{training_data.task_id[:8]}-{task.hotkey[:8]}"
        
        docker_image, build_error = await build_and_push_image(
            dockerfile_path=dockerfile_path,
            context_path=local_repo_path,
            dockerhub_username=dockerhub_username,
            dockerhub_password=dockerhub_password,
            is_image_task=(task_type == TaskType.IMAGETASK),
            tag_suffix=tag_suffix,
            log_labels=log_labels,
        )
        
        if not docker_image:
            error_msg = f"Image build/push failed: {build_error}"
            logger.error(error_msg, extra=log_labels)
            await log_task(training_data.task_id, task.hotkey, f"[ERROR] {error_msg}")
            await complete_task(training_data.task_id, task.hotkey, success=False)
            return
        
        await log_task(training_data.task_id, task.hotkey, f"Docker image built and pushed: {docker_image}")
        
        # Deploy to RunPod
        await log_task(training_data.task_id, task.hotkey, "Deploying to RunPod...")
        run_id, deploy_error = await deploy_to_runpod(
            task=task,
            docker_image=docker_image,
            log_labels=log_labels,
        )
        
        if not run_id:
            error_msg = f"RunPod deployment failed: {deploy_error}"
            logger.error(error_msg, extra=log_labels)
            await log_task(training_data.task_id, task.hotkey, f"[ERROR] {error_msg}")
            await complete_task(training_data.task_id, task.hotkey, success=False)
            return
        
        # Store run_id in task logs for monitoring
        await log_task(training_data.task_id, task.hotkey, f"RunPod Run ID: {run_id}")
        
        # Start background log streaming
        log_stream_task = asyncio.create_task(
            stream_runpod_logs(
                run_id=run_id,
                task_id=training_data.task_id,
                hotkey=task.hotkey,
                log_labels=log_labels,
            )
        )
        
        # Monitor the run
        await log_task(training_data.task_id, task.hotkey, f"Monitoring RunPod task (timeout={timeout_seconds}s)...")
        success, monitor_error = await monitor_run(
            run_id=run_id,
            task_id=training_data.task_id,
            hotkey=task.hotkey,
            timeout_seconds=timeout_seconds,
            log_labels=log_labels,
        )
        
        # Cancel log streaming when monitoring completes
        log_stream_task.cancel()
        try:
            await log_stream_task
        except asyncio.CancelledError:
            pass
        
        if success:
            await log_task(training_data.task_id, task.hotkey, "Training completed successfully on RunPod")
            
            # Upload to HuggingFace if training succeeded
            try:
                await log_task(training_data.task_id, task.hotkey, "Starting HuggingFace upload...")
                upload_success, upload_error = await upload_runpod_checkpoints_to_hf(
                    task=task,
                    task_type=task_type,
                    log_labels=log_labels,
                )
                
                if upload_success:
                    await log_task(training_data.task_id, task.hotkey, "Repo uploaded successfully to HuggingFace")
                else:
                    error_msg = f"HF upload failed: {upload_error}"
                    await log_task(training_data.task_id, task.hotkey, f"[ERROR] {error_msg}")
                    logger.error(error_msg, extra=log_labels)
                    success = False  # Mark task as failed if upload fails
            except Exception as upload_ex:
                error_msg = f"HF upload error: {upload_ex}"
                await log_task(training_data.task_id, task.hotkey, f"[ERROR] {error_msg}")
                logger.exception(error_msg, extra=log_labels)
                success = False
        else:
            error_msg = f"Training failed: {monitor_error}"
            await log_task(training_data.task_id, task.hotkey, f"[ERROR] {error_msg}")
            logger.error(error_msg, extra=log_labels)
        
        await complete_task(training_data.task_id, task.hotkey, success=success)
        
    except Exception as e:
        error_msg = f"RunPod training task failed: {e}"
        logger.exception(error_msg, extra=log_labels)
        await log_task(training_data.task_id, task.hotkey, f"[ERROR] {error_msg}")
        await complete_task(training_data.task_id, task.hotkey, success=False)


async def upload_runpod_checkpoints_to_hf(
    task: TrainerProxyRequest,
    task_type: TaskType,
    log_labels: dict[str, str] | None = None,
) -> tuple[bool, str | None]:
    """
    Upload checkpoints from RunPod volume to HuggingFace.
    
    This deploys an HF uploader container to RunPod that mounts the checkpoints volume
    and uploads the trained model to HuggingFace.
    
    Args:
        task: Training task request
        task_type: Type of task
        log_labels: Labels for logging
        
    Returns:
        tuple: (success, error_message)
    """
    try:
        training_data = task.training_data
        
        # Get HuggingFace credentials
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        hf_username = os.getenv("HUGGINGFACE_USERNAME")
        
        if not hf_token or not hf_username:
            return False, "HUGGINGFACE_TOKEN and HUGGINGFACE_USERNAME environment variables required"
        
        # Get WANDB token if needed
        wandb_token = os.getenv("WANDB_TOKEN") if task_type != TaskType.IMAGETASK else None
        
        # Determine path in repo
        path_in_repo = cst.IMAGE_TASKS_HF_SUBFOLDER_PATH if task_type == TaskType.IMAGETASK else None
        
        # Deploy HF upload task to RunPod
        upload_run_id, deploy_error = await deploy_hf_upload_task(
            task_id=training_data.task_id,
            hotkey=task.hotkey,
            expected_repo_name=training_data.expected_repo_name,
            model=training_data.model,
            hf_token=hf_token,
            hf_username=hf_username,
            wandb_token=wandb_token,
            path_in_repo=path_in_repo,
            log_labels=log_labels,
        )
        
        if not upload_run_id:
            return False, f"Failed to deploy HF upload task: {deploy_error}"
        
        await log_task(training_data.task_id, task.hotkey, f"HF upload task deployed. Run ID: {upload_run_id}")
        
        # Monitor upload task (with 30 minute timeout)
        upload_timeout = 30 * 60  # 30 minutes
        upload_success, upload_error = await monitor_run(
            run_id=upload_run_id,
            task_id=training_data.task_id,
            hotkey=task.hotkey,
            timeout_seconds=upload_timeout,
            log_labels=log_labels,
        )
        
        if upload_success:
            # Try to extract WANDB URL from logs if available
            try:
                from dstack.api import Client
                client = Client.from_config()
                run = client.runs.get(upload_run_id)
                if run:
                    logs = []
                    for log in run.logs():
                        log_str = log.decode('utf-8', errors='ignore') if isinstance(log, bytes) else str(log)
                        logs.append(log_str)
                    log_text = "".join(logs)
                    
                    # Extract WANDB URL if present
                    wandb_match = re.search(r"https://wandb\.ai/\S+", log_text)
                    if wandb_match and wandb_token:
                        wandb_url = wandb_match.group(0)
                        await update_wandb_url(training_data.task_id, task.hotkey, wandb_url)
            except Exception as e:
                logger.warning(f"Could not extract WANDB URL: {e}", extra=log_labels)
            
            return True, None
        else:
            return False, upload_error or "Upload task failed"
            
    except Exception as e:
        return False, str(e)

