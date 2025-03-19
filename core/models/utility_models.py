import uuid
from enum import Enum
from uuid import UUID

from pydantic import BaseModel
from pydantic import Field


class DatasetType(str, Enum):
    INSTRUCT = "instruct"
    PRETRAIN = "pretrain"
    ALPACA = "alpaca"


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
    test_loss: float | None
    synth_loss: float | None
    score_reason: str | None


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
    expected_repo_name: str | None = None


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


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class Prompts(BaseModel):
    input_output_reformulation_sys: str
    input_output_reformulation_user: str


class TaskType(str, Enum):
    TEXTTASK = "TextTask"
    IMAGETASK = "ImageTask"


class ImageTextPair(BaseModel):
    image_url: str = Field(..., description="Presigned URL for the image file")
    text_url: str = Field(..., description="Presigned URL for the text file")
