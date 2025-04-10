import time
from contextlib import contextmanager
from typing import Any
from typing import Dict
from typing import Optional

from opentelemetry import metrics


meter = metrics.get_meter("validator_task_metrics")


evaluation_duration = meter.create_gauge(
    name="evaluation_duration_seconds", description="Duration of evaluation in seconds", unit="s"
)

text_task_preparation_duration = meter.create_gauge(
    name="text_task_preparation_duration_seconds", description="Duration of text task preparation in seconds", unit="s"
)

text_synth_task_creation_duration = meter.create_gauge(
    name="text_synth_task_creation_duration_seconds", description="Duration of text synth task creation in seconds", unit="s"
)

text_synth_tasks_finished_total = meter.create_counter(
    name="text_synth_tasks_finished_total",
    description="Total number of synthetic text tasks created successfully",
    unit="1",
)

text_synth_tasks_started_total = meter.create_counter(
    name="text_synth_tasks_started_total",
    description="Total number of synthetic text task creations started",
    unit="1",
)

image_synth_task_creation_duration = meter.create_gauge(
    name="image_synth_task_creation_duration_seconds", description="Duration of image synth task creation in seconds", unit="s"
)

image_synth_tasks_finished_total = meter.create_counter(
    name="image_synth_tasks_finished_total",
    description="Total number of synthetic image tasks created successfully",
    unit="1",
)

image_synth_tasks_started_total = meter.create_counter(
    name="image_synth_tasks_started_total",
    description="Total number of synthetic image task creations started",
    unit="1",
)


def record_text_synth_task_finished(model_id: str, dataset_id: str, num_rows: int):
    """Records the successful creation (finish) of a synthetic text task."""
    labels = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "num_rows": num_rows,
    }
    text_synth_tasks_finished_total.add(1, labels)


def record_text_synth_task_started(model_id: str, dataset_id: str, num_rows: int):
    """Records the start of a synthetic text task creation attempt."""
    labels = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "num_rows": num_rows,
    }
    text_synth_tasks_started_total.add(1, labels)


def record_image_synth_task_finished(model_id: str, type: str, num_pairs: int):
    """Records the successful creation (finish) of a synthetic image task."""
    labels = {
        "model_id": model_id,
        "type": type,
        "num_pairs": num_pairs,
    }
    image_synth_tasks_finished_total.add(1, labels)


def record_image_synth_task_started(model_id: str, type: str, num_pairs: int):
    """Records the start of a synthetic image task creation attempt."""
    labels = {
        "model_id": model_id,
        "type": type,
        "num_pairs": num_pairs,
    }
    image_synth_tasks_started_total.add(1, labels)


@contextmanager
def measure_duration(histogram, labels: Optional[Dict[str, Any]] = None):
    """Context manager to measure and record duration using an OpenTelemetry histogram or gauge.

    Args:
        histogram: OpenTelemetry histogram/gauge metric to record duration
        labels: Optional dictionary of labels/attributes to attach to the measurement
    """
    labels = labels or {}
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        if hasattr(histogram, "record"):
            histogram.record(duration, labels)
