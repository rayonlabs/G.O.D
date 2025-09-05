import asyncio
import os

from fastapi import APIRouter, Depends
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.utility_models import GPUInfo
from trainer import constants as cst
from trainer.image_manager import start_training_task
from trainer.tasks import complete_task
from trainer.tasks import get_task
from trainer.tasks import load_task_history
from trainer.tasks import log_task
from trainer.tasks import start_task
from trainer.tasks import get_recent_tasks
from trainer.utils.misc import clone_repo
from trainer.utils.misc import get_gpu_info
from validator.core.constants import GET_GPU_AVAILABILITY_ENDPOINT
from validator.core.constants import PROXY_TRAINING_IMAGE_ENDPOINT
from validator.core.constants import TASK_DETAILS_ENDPOINT
from validator.core.constants import GET_RECENT_TASKS_ENDPOINT
from trainer.utils.logging import logger

load_task_history()


async def verify_orchestrator_ip(request: Request):
    """Verify request comes from validator/orchestrator"""
    # Simple test log to see if logging works at all
    print("PRINT TEST: verify_orchestrator_ip called")
    
    try:
        # Try different log levels and formats
        logger.info("SECURITY TEST LOG - verify_orchestrator_ip called")
        logger.error("SECURITY ERROR LOG - verify_orchestrator_ip called") 
        logger.warning("SECURITY WARNING LOG - verify_orchestrator_ip called")
        print("PRINT TEST: All logger calls worked")
        
        # Try standard Python logging as backup
        import logging
        std_logger = logging.getLogger("security_test")
        std_logger.addHandler(logging.StreamHandler())
        std_logger.setLevel(logging.INFO)
        std_logger.info("STANDARD LOGGER TEST - this should appear")
        print("PRINT TEST: Standard logger test done")
        
    except Exception as e:
        print(f"PRINT TEST: logger failed: {e}")
    
    import datetime
    import os as os_mod
    timestamp = datetime.datetime.utcnow().isoformat()
    client_ip = request.client.host if request.client else "unknown"
    
    # Get forensic details
    headers = dict(request.headers)
    user_agent = headers.get("user-agent", "unknown")
    process_id = os_mod.getpid()
    
    # Get both public and private validator IPs from environment
    public_ip = os.getenv("VALIDATOR_PUBLIC_IP", "185.141.218.75")
    private_ip = os.getenv("VALIDATOR_PRIVATE_IP", "10.0.1.153")
    allowed_ips = [public_ip, private_ip]
    
    # Enhanced forensic logging
    logger.info(f"[SECURITY] [{timestamp}] IP check - Client: {client_ip}, User-Agent: {user_agent}, PID: {process_id}")
    logger.info(f"[SECURITY] [{timestamp}] Request headers: {headers}")
    
    if client_ip not in allowed_ips:
        logger.error(f"[SECURITY ALERT] [{timestamp}] BLOCKED unauthorized IP: {client_ip}, User-Agent: {user_agent}, PID: {process_id}")
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    logger.info(f"[SECURITY] [{timestamp}] ALLOWED request from validator IP: {client_ip}")
    return client_ip

async def start_training(req: TrainerProxyRequest, request: Request) -> JSONResponse:
    import datetime
    timestamp = datetime.datetime.utcnow().isoformat()
    client_ip = request.client.host if request.client else "unknown"
    
    logger.info(f"[SECURITY] [{timestamp}] start_training called - IP: {client_ip}, Hotkey: {req.hotkey}, Repo: {req.github_repo}")
    logger.info(f"[SECURITY] [{timestamp}] Full request payload - TaskID: {req.training_data.task_id}, Model: {req.training_data.model}, GPU_IDs: {req.gpu_ids}")
    
    # Special alert for malicious repo with full forensic capture
    if "haihp02/sn56-tournament-repo" in req.github_repo:
        try:
            import os as os_mod
            import traceback
            process_id = os_mod.getpid()
            user_agent = request.headers.get("user-agent", "unknown")
            stack_trace = traceback.format_stack()
            
            logger.error(f"[SECURITY ALERT] [{timestamp}] MALICIOUS REPO DETECTED!")
            logger.error(f"[SECURITY ALERT] [{timestamp}] IP: {client_ip}, User-Agent: {user_agent}, PID: {process_id}")
            logger.error(f"[SECURITY ALERT] [{timestamp}] Hotkey: {req.hotkey}, TaskID: {req.training_data.task_id}")
            logger.error(f"[SECURITY ALERT] [{timestamp}] Full payload: {req.dict()}")
            logger.error(f"[SECURITY ALERT] [{timestamp}] Call stack: {stack_trace[-3:]}")  # Last 3 stack frames
        except Exception as e:
            logger.error(f"[SECURITY ALERT] [{timestamp}] MALICIOUS REPO DETECTED! (forensics failed: {e})")
    
    await start_task(req)

    logger.info(f"[SECURITY] About to clone repo - URL: {req.github_repo}, Branch: {req.github_branch}, Commit: {req.github_commit_hash}")
    
    try:
        local_repo_path = await asyncio.to_thread(
            clone_repo,
            repo_url=req.github_repo,
            parent_dir=cst.TEMP_REPO_PATH,
            branch=req.github_branch,
            commit_hash=req.github_commit_hash,
        )
        logger.info(f"[SECURITY] Repo cloned successfully to: {local_repo_path}")
    except RuntimeError as e:
        logger.warning(f"[SECURITY] Repo clone failed - URL: {req.github_repo}, Error: {e}")
        await log_task(req.training_data.task_id, req.hotkey, f"Failed to clone repo: {e}")
        await complete_task(req.training_data.task_id, req.hotkey, success=False)
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"Repo {req.github_repo} cloned to {local_repo_path}",
                extra={"task_id": req.training_data.task_id, "hotkey": req.hotkey, "model": req.training_data.model})

    asyncio.create_task(start_training_task(req, local_repo_path))

    return {"message": "Started Training!", "task_id": req.training_data.task_id}


async def get_available_gpus() -> list[GPUInfo]:
    gpu_info = await get_gpu_info()
    return gpu_info


async def get_task_details(task_id: str, hotkey: str) -> TrainerTaskLog:
    task = get_task(task_id, hotkey)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' and hotkey '{hotkey}' not found.")
    return task


async def get_recent_tasks_list(hours:int) -> list[TrainerTaskLog]:
    tasks = get_recent_tasks(hours)
    if not tasks:
        raise HTTPException(status_code=404, detail=f"Tasks not found in the last {hours} hours.")
    return tasks


def factory_router() -> APIRouter:
    router = APIRouter(tags=["Proxy Trainer"])
    router.add_api_route(PROXY_TRAINING_IMAGE_ENDPOINT, start_training, methods=["POST"], dependencies=[Depends(verify_orchestrator_ip)])
    router.add_api_route(GET_GPU_AVAILABILITY_ENDPOINT, get_available_gpus, methods=["GET"], dependencies=[Depends(verify_orchestrator_ip)])
    router.add_api_route(GET_RECENT_TASKS_ENDPOINT, get_recent_tasks_list, methods=["GET"], dependencies=[Depends(verify_orchestrator_ip)])
    router.add_api_route(TASK_DETAILS_ENDPOINT, get_task_details, methods=["GET"], dependencies=[Depends(verify_orchestrator_ip)])
    return router
