"""
FastAPI endpoints for RunPod training deployment
"""
import asyncio
import os

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request

from core.models.payload_models import TrainerProxyRequest
from trainer.endpoints import verify_orchestrator_ip
from trainer.runpod.training_task import start_runpod_training_task
from trainer.tasks import complete_task
from trainer.tasks import log_task
from trainer.tasks import start_task
from trainer.utils.misc import are_gpus_available
from trainer.utils.misc import clone_repo
from trainer import constants as cst
from trainer.utils.logging import logger
from validator.core.constants import PROXY_TRAINING_IMAGE_ENDPOINT

# Use same endpoint path as existing training endpoint
# This will be handled by the router order or can be made conditional
RUNPOD_TRAINING_ENDPOINT = PROXY_TRAINING_IMAGE_ENDPOINT


async def start_runpod_training(req: TrainerProxyRequest, request: Request):
    """
    Start training on RunPod using dstack deployment.
    
    This endpoint mirrors the existing start_training but uses RunPod backend.
    Use header 'X-Backend: runpod' or query parameter 'backend=runpod' to route here.
    """
    # Check if this request is intended for RunPod backend
    # Can be specified via header or query parameter
    backend_header = request.headers.get("X-Backend", "").lower()
    backend_query = request.query_params.get("backend", "").lower()
    
    if backend_header != "runpod" and backend_query != "runpod":
        # If not explicitly RunPod, let the main endpoint handle it
        raise HTTPException(status_code=404, detail="Use X-Backend: runpod header or ?backend=runpod query parameter")
    
    # Note: GPU availability check not needed for RunPod since we're using cloud resources
    await start_task(req)
    
    try:
        local_repo_path = await asyncio.to_thread(
            clone_repo,
            repo_url=req.github_repo,
            parent_dir=cst.TEMP_REPO_PATH,
            branch=req.github_branch,
            commit_hash=req.github_commit_hash,
        )
    except Exception as e:
        await log_task(req.training_data.task_id, req.hotkey, f"Failed to clone repo: {str(e)}")
        await complete_task(req.training_data.task_id, req.hotkey, success=False)
        return {
            "message": "Error cloning github repository",
            "task_id": req.training_data.task_id,
            "error": str(e),
            "success": False,
            "no_retry": True,
        }
    
    logger.info(
        f"Repo {req.github_repo} cloned to {local_repo_path}",
        extra={
            "task_id": req.training_data.task_id,
            "hotkey": req.hotkey,
            "model": req.training_data.model,
            "backend": "runpod",
        },
    )
    
    # Start RunPod training task
    asyncio.create_task(start_runpod_training_task(req, local_repo_path))
    
    return {"message": "Started Training!", "task_id": req.training_data.task_id}


def factory_router() -> APIRouter:
    """
    Factory function to create RunPod training router.
    Uses the same endpoint schema as the existing training endpoint.
    """
    router = APIRouter(tags=["Proxy Trainer"])
    
    router.add_api_route(
        RUNPOD_TRAINING_ENDPOINT, 
        start_runpod_training, 
        methods=["POST"], 
        dependencies=[Depends(verify_orchestrator_ip)]
    )
    
    return router

