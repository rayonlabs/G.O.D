import json
import logging
from contextvars import ContextVar
from logging import Formatter
from logging import Logger
from logging import LogRecord
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from fiber.logging_utils import get_logger as fiber_get_logger


current_context = ContextVar[dict[str, str | dict]]("current_context", default={})


def add_context_tag(key: str, value: str | dict) -> None:
    """Add or update a tag in the current logging context"""
    try:
        context = current_context.get()
        new_context = {**context, key: value}
        current_context.set(new_context)
    except LookupError:
        current_context.set({key: value})


def remove_context_tag(key: str) -> None:
    """Remove a tag from the current logging context"""
    try:
        context = current_context.get()
        if key in context:
            new_context = context.copy()
            del new_context[key]
            current_context.set(new_context)
    except LookupError:
        pass


def clear_context() -> None:
    """
    Removes all tags from the current logging context.
    """
    current_context.set({})


def get_context_tag(key: str) -> Optional[str | dict]:
    """Get a tag value from the current logging context"""
    try:
        context = current_context.get()
        return context.get(key)
    except LookupError:
        return None


class JSONFormatter(Formatter):
    def format(self, record: LogRecord) -> str:
        if not hasattr(record, "tags"):
            try:
                context = current_context.get()
                record.tags = {k: v for k, v in context.items() if v is not None}
            except LookupError:
                record.tags = {}

        clean_level = record.levelname.replace("\u001b[32m", "").replace("\u001b[1m", "").replace("\u001b[0m", "")
        log_data: dict[str, str | dict] = {
            "timestamp": self.formatTime(record),
            "level": clean_level,
            "message": record.getMessage(),
            "logger": record.name,
            "tags": record.tags,
        }
        return json.dumps(log_data)


class LogContext:
    def __init__(self, **tags: str | dict):
        self.tags = tags
        self.token = None

    def __enter__(self):
        try:
            current = current_context.get()
            new_context = {**current, **self.tags}
        except LookupError:
            new_context = self.tags
        self.token = current_context.set(new_context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            current_context.reset(self.token)


class ContextLogger(Logger):
    def _log(
        self,
        level: int,
        msg: object,
        args: tuple,
        exc_info: Optional[Exception] = None,
        extra: Optional[dict] = None,
        stack_info: bool = False,
        stacklevel: int = 1,
    ) -> None:
        if extra is None:
            try:
                context = current_context.get()
                extra = {"tags": {k: v for k, v in context.items() if v is not None}}
            except LookupError:
                extra = {"tags": {}}

        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)


def setup_logging():
    """Initialize logging configuration for the entire application"""
    logging.setLoggerClass(ContextLogger)
    root_logger = logging.getLogger()

    fiber_logger = fiber_get_logger("root")
    root_logger.setLevel(fiber_logger.level)
    for handler in fiber_logger.handlers:
        root_logger.addHandler(handler)

    base_dir = Path(__file__).parent.parent.parent
    log_dir = base_dir / "validator" / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "validator.log"
        file_handler = RotatingFileHandler(filename=str(log_file), maxBytes=100 * 1024 * 1024, backupCount=3)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        raise
