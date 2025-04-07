from opentelemetry import metrics

from core.models.utility_models import TaskType


meter = metrics.get_meter("validator_task_metrics")


evaluation_duration = meter.create_histogram(
    name="evaluation_duration_seconds", description="Duration of evaluation in seconds", unit="s"
)

text_task_preparation_duration = meter.create_histogram(
    name="text_task_preparation_duration_seconds", description="Duration of text task preparation in seconds", unit="s"
)

text_synth_task_creation_duration = meter.create_histogram(
    name="text_synth_task_creation_duration_seconds", description="Duration of text synth task creation in seconds", unit="s"
)


def record_text_synth_task_creation_duration(
    duration_seconds: float, model_id: str, dataset_id: str, num_rows: int, augmented_success: bool
):
    labels = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "num_rows": num_rows,
        "augmented_success": augmented_success,
    }
    text_synth_task_creation_duration.record(duration_seconds, labels)


def record_text_task_preparation_duration(task_id: str, duration_seconds: float, model_id: str, dataset_id: str):
    labels = {
        "task_id": str(task_id),
        "model_id": model_id,
        "dataset_id": dataset_id,
        "task_type": "text",
    }
    text_task_preparation_duration.record(duration_seconds, labels)


def record_evaluation_duration(
    task_id: str, duration_seconds: float, num_miners: int, model_id: str, dataset_id: str, task_type: TaskType
):
    labels = {
        "task_id": str(task_id),
        "model_id": model_id,
        "dataset_id": dataset_id,
        "task_type": task_type.value,
        "num_miners": num_miners,
    }
    evaluation_duration.record(duration_seconds, labels)
