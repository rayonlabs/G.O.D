import uuid
from enum import Enum

from pydantic import BaseModel
from pydantic import Field


class DatasetType(str, Enum):
    INSTRUCT = "instruct"
    PRETRAIN = "pretrain"
    ALPACA = "alpaca"
    # there is actually loads of these supported, but laziness is key here, add when we need


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
    EVALUATING = "evaluating"
    TRAINING = "training"
    FAILURE = "failure"
    FAILURE_FINDING_NODES = "failure_finding_nodes"
    PREP_TASK_FAILURE = "prep_task_failure"
    NODE_TRAINING_FAILURE = "node_training_failure"


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
    dataset: str
    model: str
    dataset_type: DatasetType | CustomDatasetType
    file_format: FileFormat
    status: JobStatus = JobStatus.QUEUED
    error_message: str | None = None


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
