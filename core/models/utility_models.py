import uuid
from enum import Enum
from typing import Callable
from typing import Union
from uuid import UUID

from pydantic import BaseModel
from pydantic import Field

from validator.core.models import ImageRawTask
from validator.core.models import TextRawTask
from validator.cycle.process_tasks import get_total_image_dataset_size
from validator.cycle.process_tasks import get_total_text_dataset_size
from validator.cycle.process_tasks import prepare_image_task_request
from validator.cycle.process_tasks import prepare_text_task_request
from validator.cycle.process_tasks import run_image_task_prep
from validator.cycle.process_tasks import run_text_task_prep
from validator.evaluation.docker_evaluation import run_evaluation_docker
from validator.evaluation.docker_evaluation import run_evaluation_docker_diffusion
from validator.tasks.task_prep import get_additional_synth_data


class DatasetType(str, Enum):
    INSTRUCT = "instruct"
    PRETRAIN = "pretrain"
    ALPACA = "alpaca"


class TaskType(str, Enum):
    IMAGE = "image"
    TEXT = "text"


# TODO
# being lazy here with everything as callable, can we look at the signatures and use the same for the diff
# data types
class TaskConfig(BaseModel):
    task_type: TaskType = Field(..., description="The type of task.")
    eval_container: Callable = Field(..., description="Function to evaluate the task")
    synth_data_function: Callable | None = Field(..., description="Function to evaluate the task")
    data_size_function: Callable = Field(..., description="The function used to determine the dataset size")
    task_prep_function: Callable = Field(
        ..., description="What we call in order to do the prep work - train test split and whatnot"
    )

    task_request_prepare_function: Callable = Field(..., description="Namoray will come up with a better var name for sure")


class ImageTaskConfig(TaskConfig):
    task_type: TaskType = TaskType.IMAGE
    eval_container: Callable = run_evaluation_docker
    synth_data_function: Callable | None = None
    data_size_function: Callable = get_total_image_dataset_size
    task_prep_function: Callable = run_image_task_prep
    task_request_prepare_function: Callable = prepare_image_task_request


class TextTaskConfig(TaskConfig):
    task_type: TaskType = TaskType.TEXT
    eval_container: Callable = run_evaluation_docker_diffusion
    synth_data_function: Callable | None = get_additional_synth_data
    data_size_function: Callable = get_total_text_dataset_size
    task_prep_function: Callable = run_text_task_prep
    task_request_prepare_function: Callable = prepare_text_task_request


def get_task_config(task: Union[TextRawTask, ImageRawTask]) -> TaskConfig:
    if isinstance(task, TextRawTask):
        return TextTaskConfig()
    elif isinstance(task, ImageRawTask):
        return ImageTaskConfig()
    else:
        raise ValueError(f"Unsupported task type: {type(task).__name__}")


class FileFormat(str, Enum):
    CSV = "csv"  # needs to be local file
    JSON = "json"  # needs to be local file
    HF = "hf"  # Hugging Face dataset
    S3 = "s3"


class JobStatus(str, Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    NOT_FOUND = "Not Found"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PREPARING_DATA = "preparing_data"
    IDLE = "idle"
    READY = "ready"
    SUCCESS = "success"
    LOOKING_FOR_NODES = "looking_for_nodes"
    DELAYED = "delayed"
    EVALUATING = "evaluating"
    PREEVALUATION = "preevaluation"
    TRAINING = "training"
    FAILURE = "failure"
    FAILURE_FINDING_NODES = "failure_finding_nodes"
    PREP_TASK_FAILURE = "prep_task_failure"
    NODE_TRAINING_FAILURE = "node_training_failure"


class WinningSubmission(BaseModel):
    hotkey: str
    score: float
    model_repo: str

    # Turn off protected namespace for model
    model_config = {"protected_namespaces": ()}


class MinerTaskResult(BaseModel):
    hotkey: str
    quality_score: float


# NOTE: Confusing name with the class above
class TaskMinerResult(BaseModel):
    task_id: UUID
    quality_score: float


class CustomDatasetType(BaseModel):
    system_prompt: str | None = ""
    system_format: str | None = "{system}"
    field_system: str | None = None
    field_instruction: str | None = None
    field_input: str | None = None
    field_output: str | None = None
    format: str | None = None
    no_input_format: str | None = None
    field: str | None = None


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model: str
    status: JobStatus = JobStatus.QUEUED
    error_message: str | None = None


class TextJob(Job):
    dataset: str
    dataset_type: DatasetType | CustomDatasetType
    file_format: FileFormat


class DiffusionJob(Job):
    dataset_zip: str = Field(
        ...,
        description="Link to dataset zip file",
        min_length=1,
    )
    hf_repo: str
    hf_folder: str


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class Prompts(BaseModel):
    # synthetic data generation prompts
    # in-context learning prompts
    in_context_learning_generation_sys: str
    in_context_learning_generation_user: str
    # correctness-focused prompts (step 1/2)
    output_field_reformulation_sys: str
    output_field_reformulation_user: str
    # correctness-focused prompts (step 2/2)
    input_field_generation_sys: str
    input_field_generation_user: str
