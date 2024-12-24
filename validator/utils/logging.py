import json
import logging
from contextvars import ContextVar
from logging import Formatter
from logging import Logger
from logging import LogRecord
from logging import getLogger as fiber_get_logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from typing import Optional


current_context = ContextVar[dict[str, Any]]("current_context", default={})


def add_context_tag(key: str, value: Any) -> None:
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


def get_context_tag(key: str) -> Optional[Any]:
    """Get a tag value from the current logging context"""
    try:
        context = current_context.get()
        return context.get(key)
    except LookupError:
        return None


def create_extra_log(**tags: Any) -> dict[str, dict[str, Any]]:
    try:
        context = current_context.get()
        combined_tags = {**context, **tags}
    except LookupError:
        combined_tags = tags
    return {"tags": {k: v for k, v in combined_tags.items() if v is not None}}


class JSONFormatter(Formatter):
    def format(self, record: LogRecord) -> str:
        if not hasattr(record, "tags"):
            try:
                context = current_context.get()
                record.tags = {k: v for k, v in context.items() if v is not None}
            except LookupError:
                record.tags = {}

        clean_level = record.levelname.replace("\u001b[32m", "").replace("\u001b[1m", "").replace("\u001b[0m", "")
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": clean_level,
            "message": record.getMessage(),
            "logger": record.name,
            "tags": record.tags,
        }
        return json.dumps(log_data)


class LogContext:
    def __init__(self, **tags: Any):
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


def setup_logging():
    """Initialize logging configuration for the entire application"""
    root_logger = logging.getLogger()

    # Configure the root logger with Fiber's settings
    fiber_logger = fiber_get_logger("root")
    # Copy Fiber's handlers to root logger
    for handler in fiber_logger.handlers:
        root_logger.addHandler(handler)

    # Add the JSON file handler to root logger
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

    # Monkey patch the Logger class to include context tags
    def _log_with_context(self, *args, **kwargs):
        if "extra" not in kwargs:
            try:
                context = current_context.get()
                kwargs["extra"] = {"tags": {k: v for k, v in context.items() if v is not None}}
            except LookupError:
                kwargs["extra"] = {"tags": {}}
        return original_log(self, *args, **kwargs)

    original_log = Logger._log
    Logger._log = _log_with_context
