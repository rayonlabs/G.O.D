from typing import Any, Optional
from pydantic import BaseModel, Field

from core.models.utility_models import CustomDatasetType, DatasetType, FileFormat, JobStatus, TaskStatus


class TrainRequest(BaseModel):
    dataset: str = Field(
        ..., description="Path to the dataset file or Hugging Face dataset name"
    )
    model: str = Field(..., description="Name or path of the model to be trained")
    dataset_type: DatasetType | CustomDatasetType
    file_format: FileFormat

class TrainResponse(BaseModel):
    message: str
    job_id: str


class JobStatusPayload(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus


class EvaluationRequest(TrainRequest):
    original_model: str

class EvaluationResult(BaseModel):
    is_finetune: bool
    eval_loss: float
    perplexity: float

class MinerTaskRequst(BaseModel):
    hf_training_repo: str
    model: str

class TaskRequest(BaseModel):
    ds_repo: str
    system_col: str
    instruction_col: str
    input_col: str
    output_col: str
    model_repo: str
    hours_to_complete: int


class SubmitTaskSubmissionRequest(BaseModel):
    task_id: str
    node_id: str
    repo: str

class TaskResponse(BaseModel):
    success: bool
    task_id: str
    message: Optional[str] = None

class TaskStatusResponse(BaseModel):
    success: bool
    task_id: str
    status: str

class SubmissionResponse(BaseModel):
    success: bool
    message: str
    submission_id: Optional[str] = None


