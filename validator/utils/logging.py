import json
from contextvars import ContextVar
from logging import Formatter
from logging import Logger
from logging import LogRecord
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fiber.logging_utils import get_logger


def create_extra_log(**tags: str | None) -> dict[str, dict[str, str | None]]:
    try:
        context_task_id = current_task_id.get()
        if context_task_id is not None:
            tags["task_id"] = context_task_id
    except LookupError:
        pass
    return {"tags": {k: v for k, v in tags.items() if v is not None}}


class JSONFormatter(Formatter):
    def format(self, record: LogRecord) -> str:
        tags = record.__dict__.get("tags", {}) if hasattr(record, "tags") else {}
        clean_level = record.levelname.replace("\u001b[32m", "").replace("\u001b[1m", "").replace("\u001b[0m", "")
        log_data: dict[str, str | dict] = {
            "timestamp": self.formatTime(record),
            "level": clean_level,
            "message": record.getMessage(),
            "logger": record.name,
            "tags": tags,
        }
        return json.dumps(log_data)


def setup_json_logger(name: str) -> Logger:
    base_dir = Path(__file__).parent.parent.parent
    log_dir = base_dir / "validator" / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "validator.log"
        logger = get_logger(name)
        file_handler = RotatingFileHandler(filename=str(log_file), maxBytes=100 * 1024 * 1024, backupCount=3)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
        logger.info("Logging initialized", extra={"tags": {"setup": "complete"}})
        return logger
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        raise


# we add the current task_id to all logs that are with the task context
current_task_id = ContextVar[str | None]("current_task_id", default=None)


# works like ->
#   async with TaskContext("task-123"):
#        # Inside this block, current_task_id.get() will return "task-123"
#        logger.info("Doing something")
class TaskContext:
    def __init__(self, task_id: str | None):
        self.task_id = task_id
        self.token = None

    async def __aenter__(self):
        if self.task_id:
            self.token = current_task_id.set(self.task_id)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            current_task_id.reset(self.token)


logger = setup_json_logger(__name__)
