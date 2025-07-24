import asyncio
from datetime import timedelta
from datetime import datetime
from pathlib import Path
import docker
import threading

from core.models.utility_models import TaskStatus
from validator.utils.logging import get_all_context_tags
from validator.utils.logging import stream_container_logs
from validator.utils.logging import get_logger
from trainer.tasks import task_history, save_task_history
from trainer import constants as cst

logger = get_logger(__name__)

def start_cleanup_loop_in_thread():
    def run():
        asyncio.run(periodically_cleanup_tasks_and_cache())
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    

async def periodically_cleanup_tasks_and_cache(poll_interval_seconds: int = 600):
    while True:
        if len(task_history) > 0:
            now = datetime.utcnow()
            for task in task_history:
                if task.status != TaskStatus.TRAINING or not task.started_at:
                    continue

                timeout = timedelta(hours=task.training_data.hours_to_complete) + timedelta(minutes=cst.STALE_TASK_GRACE_MINUTES)
                deadline = task.started_at + timeout

                if now > deadline:
                    task.status = TaskStatus.FAILURE
                    task.finished_at = now
                    task.logs.append(f"[{now.isoformat()}] Task marked as FAILED due to timeout.")
                    await save_task_history()

            client = docker.from_env()
            abs_task_path = Path(cst.TASKS_FILE_PATH).resolve()

            if abs_task_path.exists():

                logger.info("Starting cleanup container...")

                container = client.containers.run(
                    image=cst.CACHE_CLEANER_DOCKER_IMAGE,
                    volumes={
                        cst.VOLUME_NAMES[0]: {"bind": "/checkpoints", "mode": "rw"},
                        cst.VOLUME_NAMES[1]: {"bind": "/cache", "mode": "rw"},
                        str(abs_task_path): {"bind": "/app/trainer/task_history.json", "mode": "ro"},
                    },
                    remove=True,
                    detach=True
                )

                log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, get_all_context_tags()))

                logger.info("Cleanup container finished.")


            await asyncio.sleep(poll_interval_seconds)