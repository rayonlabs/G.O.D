import json
from datetime import datetime
from pathlib import Path

from core.models.utility_models import TaskStatus, ImageModelType
from core.models.payload_models import TrainerProxyRequest, TrainerTaskLog
from trainer import constants as cst

task_history: list[TrainerTaskLog] = []
TASK_HISTORY_FILE = Path(cst.TASKS_FILE_PATH)

def start_task(task: TrainerProxyRequest) -> tuple[str, str]:
    log_entry = TrainerTaskLog(
        **task.dict(),
        status=TaskStatus.TRAINING,
        started_at=datetime.utcnow(),
        finished_at=None
    )
    task_history.append(log_entry)
    save_task_history()
    return log_entry.training_data.task_id, log_entry.hotkey

def complete_task(task_id: str, hotkey: str, success: bool = True):
    task = get_task(task_id, hotkey)
    if task is None:
        return
    task.status = TaskStatus.SUCCESS if success else TaskStatus.FAILURE
    task.finished_at = datetime.utcnow()
    save_task_history()

def get_task(task_id: str, hotkey: str) -> TrainerTaskLog | None:
    for task in task_history:
        if (
            task.training_data.task_id == task_id
            and task.hotkey == hotkey
        ):
            return task
    return None

def log_task(task_id: str, hotkey: str, message: str):
    task = get_task(task_id, hotkey)
    if task:
        timestamped_message = f"[{datetime.utcnow().isoformat()}] {message}"
        task.logs.append(timestamped_message)
        save_task_history()

def get_running_tasks() -> list[TrainerTaskLog]:
    return [t for t in task_history if t.status == TaskStatus.TRAINING]

def save_task_history():
    with open(TASK_HISTORY_FILE, "w") as f:
        json.dump([t.model_dump() for t in task_history], f, indent=2, default=str)

def load_task_history():
    global task_history
    if TASK_HISTORY_FILE.exists():
        with open(TASK_HISTORY_FILE, "r") as f:
            data = json.load(f)
            task_history.clear()
            task_history.extend(TrainerTaskLog(**item) for item in data)

