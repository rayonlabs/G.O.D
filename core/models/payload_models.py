from datetime import datetime
from uuid import UUID
from uuid import uuid4

from fiber.logging_utils import get_logger
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from core import constants as cst
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import DatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import ImageTextPair
from core.models.utility_models import JobStatus
from core.models.utility_models import MinerTaskResult
from core.models.utility_models import TaskMinerResult
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.core.models import AllNodeStats


logger = get_logger(__name__)


class MinerTaskOffer(BaseModel):
    ds_size: int
    model: str
    hours_to_complete: int
    task_id: str
    task_type: TaskType


class TrainRequest(BaseModel):
    model: str = Field(..., description="Name or path of the model to be trained", min_length=1)
    task_id: str
    hours_to_complete: int
    expected_repo_name: str | None = None


class TrainRequestText(TrainRequest):
    dataset: str = Field(
        ...,
        description="Path to the dataset file or Hugging Face dataset name",
        min_length=1,
    )
    dataset_type: DatasetType | CustomDatasetType
    file_format: FileFormat


class TrainRequestImage(TrainRequest):
    dataset_zip: str = Field(
        ...,
        description="Link to dataset zip file",
        min_length=1,
    )


class TrainResponse(BaseModel):
    message: str
    task_id: UUID


class JobStatusPayload(BaseModel):
    task_id: UUID


class JobStatusResponse(BaseModel):
    task_id: UUID
    status: JobStatus


class EvaluationRequest(TrainRequest):
    original_model: str


class EvaluationRequestDiffusion(BaseModel):
    test_split_url: str
    original_model_repo: str
    models: list[str]


class DiffusionLosses(BaseModel):
    text_guided_losses: list[float]
    no_text_losses: list[float]


class EvaluationResultImage(BaseModel):
    eval_loss: DiffusionLosses | float
    is_finetune: bool | None = None


class EvaluationResultText(BaseModel):
    is_finetune: bool
    eval_loss: float
    perplexity: float


class DockerEvaluationResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    results: dict[str, EvaluationResultText | EvaluationResultImage | Exception]
    base_model_params_count: int = 0


class MinerTaskResponse(BaseModel):
    message: str
    accepted: bool


class DatasetColumnsResponse(BaseModel):
    field_instruction: str
    field_input: str | None = None
    field_output: str | None = None


class NewTaskRequest(BaseModel):
    account_id: UUID
    hours_to_complete: int = Field(..., description="The number of hours to complete the task", examples=[1])
    result_model_name: str | None = Field(None, description="The name to give to a model that is created by this task")


class NewTaskRequestText(NewTaskRequest):
    field_instruction: str = Field(..., description="The column name for the instruction", examples=["instruction"])
    field_input: str | None = Field(None, description="The column name for the input", examples=["input"])
    field_output: str | None = Field(None, description="The column name for the output", examples=["output"])
    field_system: str | None = Field(None, description="The column name for the system (prompt)", examples=["system"])

    ds_repo: str = Field(..., description="The repository for the dataset", examples=["yahma/alpaca-cleaned"])
    file_format: FileFormat = Field(
        FileFormat.HF, description="The format of the dataset", examples=[FileFormat.HF, FileFormat.S3]
    )
    model_repo: str = Field(..., description="The repository for the model", examples=["Qwen/Qwen2.5-Coder-32B-Instruct"])
    format: None = None
    no_input_format: None = None

    # Turn off protected namespace for model
    model_config = {"protected_namespaces": ()}


class NewTaskRequestImage(NewTaskRequest):
    model_repo: str = Field(..., description="The model repository to use")
    image_text_pairs: list[ImageTextPair] = Field(
        ...,
        description="List of image and text file pairs",
        min_length=cst.MIN_IMAGE_TEXT_PAIRS,
        max_length=cst.MAX_IMAGE_TEXT_PAIRS,
    )
    ds_id: str = Field(
        default_factory=lambda: str(uuid4()), description="A ds name. The actual dataset is provided via the image_text_pairs"
    )


class NewTaskWithFixedDatasetsRequest(NewTaskRequest):
    ds_repo: str | None = Field(None, description="Optional: The original repository of the dataset")
    training_data: str = Field(..., description="The prepared training dataset")
    synthetic_data: str = Field(..., description="The prepared synthetic dataset")
    test_data: str = Field(..., description="The prepared test dataset")


class NewTaskResponse(BaseModel):
    success: bool = Field(..., description="Whether the task was created successfully")
    task_id: UUID | None = Field(..., description="The ID of the task")
    created_at: datetime = Field(..., description="The creation time of the task")
    account_id: UUID | None = Field(..., description="The account ID who owns the task")


class TaskResultResponse(BaseModel):
    id: UUID
    miner_results: list[MinerTaskResult] | None


class AllOfNodeResults(BaseModel):
    success: bool
    hotkey: str
    task_results: list[TaskMinerResult] | None


class TaskDetails(BaseModel):
    id: UUID
    account_id: UUID
    status: TaskStatus
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    hours_to_complete: int
    trained_model_repository: str | None
    task_type: TaskType
    result_model_name: str | None = None


class TextTaskDetails(TaskDetails):
    task_type: TaskType = TaskType.TEXTTASK
    base_model_repository: str
    ds_repo: str

    field_system: str | None = Field(None, description="The column name for the `system (prompt)`", examples=["system"])
    field_instruction: str = Field(
        ..., description="The column name for the instruction - always needs to be provided", examples=["instruction"]
    )
    field_input: str | None = Field(None, description="The column name for the `input`", examples=["input"])
    field_output: str | None = Field(None, description="The column name for the `output`", examples=["output"])

    # NOTE: ATM can not be defined by the user, but should be able to in the future
    format: None = Field(None, description="The column name for the `format`", examples=["{instruction} {input}"])
    no_input_format: None = Field(
        None, description="If the field_input is not provided, what format should we use? ", examples=["{instruction}"]
    )
    system_format: None = Field(None, description="How to format the `system (prompt)`", examples=["{system}"])

    # Turn off protected namespace for model
    model_config = {"protected_namespaces": ()}


class ImageTaskDetails(TaskDetails):
    task_type: TaskType = TaskType.IMAGETASK
    image_text_pairs: list[ImageTextPair]
    base_model_repository: str = Field(..., description="The repository for the model")


class TaskListResponse(BaseModel):
    success: bool
    task_id: UUID
    status: TaskStatus


class LeaderboardRow(BaseModel):
    hotkey: str
    stats: AllNodeStats
