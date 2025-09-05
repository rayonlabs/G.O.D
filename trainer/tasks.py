import json
from datetime import datetime as dt
from datetime import timedelta
from pathlib import Path
import aiofiles

from core.models.utility_models import TaskStatus
from core.models.payload_models import TrainerProxyRequest, TrainerTaskLog
from validator.utils.logging import get_logger
from trainer import constants as cst

logger = get_logger(__name__)

task_history: list[TrainerTaskLog] = []
TASK_HISTORY_FILE = Path(cst.TASKS_FILE_PATH)


async def start_task(task: TrainerProxyRequest) -> tuple[str, str]:
    timestamp = dt.utcnow().isoformat()
    task_id = task.training_data.task_id
    hotkey = task.hotkey
    
    logger.info(f"[SECURITY] [{timestamp}] start_task called - TaskID: {task_id}, Hotkey: {hotkey}, Repo: {task.github_repo}")
    
    # Alert for malicious repo in start_task too
    if "haihp02/sn56-tournament-repo" in task.github_repo:
        logger.error(f"[SECURITY ALERT] [{timestamp}] MALICIOUS REPO IN TASK CREATION! TaskID: {task_id}, Hotkey: {hotkey}")

    existing_task = get_task(task_id, hotkey)
    if existing_task:
        existing_task.logs.clear()
        existing_task.status = TaskStatus.TRAINING
        existing_task.started_at = dt.utcnow()
        existing_task.finished_at = None
        await save_task_history()
        return task_id, hotkey

    log_entry = TrainerTaskLog(
        **task.dict(),
        status=TaskStatus.TRAINING,
        started_at=dt.utcnow(),
        finished_at=None,
    )
    
    logger.info(f"[SECURITY] Adding new task to history - TaskID: {task_id}, Hotkey: {hotkey}, Repo: {task.github_repo}")
    
    task_history.append(log_entry)
    await save_task_history()
    return log_entry.training_data.task_id, log_entry.hotkey


async def complete_task(task_id: str, hotkey: str, success: bool = True):
    logger.info(f"[SECURITY] complete_task called - TaskID: {task_id}, Hotkey: {hotkey}, Success: {success}")
    task = get_task(task_id, hotkey)
    if task is None:
        logger.warning(f"[SECURITY] complete_task - Task not found: {task_id}, {hotkey}")
        return
    task.status = TaskStatus.SUCCESS if success else TaskStatus.FAILURE
    task.finished_at = dt.utcnow()
    logger.info(f"[SECURITY] Task completed - TaskID: {task_id}, Status: {task.status}")
    await save_task_history()


def get_task(task_id: str, hotkey: str) -> TrainerTaskLog | None:
    for task in task_history:
        if task.training_data.task_id == task_id and task.hotkey == hotkey:
            return task
    return None


async def log_task(task_id: str, hotkey: str, message: str):
    task = get_task(task_id, hotkey)
    if task:
        timestamped_message = f"[{dt.utcnow().isoformat()}] {message}"
        task.logs.append(timestamped_message)
        await save_task_history()


async def update_wandb_url(task_id: str, hotkey: str, wandb_url: str):
    task = get_task(task_id, hotkey)
    if task:
        task.wandb_url = wandb_url
        await save_task_history()
        logger.info(f"Updated wandb_url for task {task_id}: {wandb_url}")
    else:
        logger.warning(f"Task not found for task_id={task_id} and hotkey={hotkey}")


def get_running_tasks() -> list[TrainerTaskLog]:
    return [t for t in task_history if t.status == TaskStatus.TRAINING]


def get_recent_tasks(hours: float = 1.0) -> list[TrainerTaskLog]:
    cutoff = dt.utcnow() - timedelta(hours=hours)

    recent_tasks = [
        task for task in task_history
        if (task.started_at and task.started_at >= cutoff) or
           (task.finished_at and task.finished_at >= cutoff)
    ]

    recent_tasks.sort(
        key=lambda t: max(
            t.finished_at or dt.min,
            t.started_at or dt.min
        ),
        reverse=True
    )

    return recent_tasks


async def save_task_history():
    async with aiofiles.open(TASK_HISTORY_FILE, "w") as f:
        data = json.dumps([t.model_dump() for t in task_history], indent=2, default=str)
        await f.write(data)


def load_task_history():
    global task_history
    if TASK_HISTORY_FILE.exists():
        with open(TASK_HISTORY_FILE, "r") as f:
            data = json.load(f)
            task_history.clear()
            task_history.extend(TrainerTaskLog(**item) for item in data)
