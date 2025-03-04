import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field

from core.models.utility_models import FileFormat
from core.models.utility_models import ImageTextPair
from core.models.utility_models import TaskType


class TokenizerConfig(BaseModel):
    bos_token: str | None = None
    eos_token: str | None = None
    pad_token: str | None = None
    unk_token: str | None = None
    chat_template: str | None = None
    use_default_system_prompt: bool | None = None


class ModelConfig(BaseModel):
    architectures: list[str]
    model_type: str
    tokenizer_config: TokenizerConfig

    model_config = {"protected_namespaces": ()}


class DatasetData(BaseModel):
    dataset_id: str
    sparse_columns: list[str] = Field(default_factory=list)
    non_sparse_columns: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    author: str | None = None
    disabled: bool = False
    gated: bool = False
    last_modified: str | None = None
    likes: int = 0
    trending_score: int | None = None
    private: bool = False
    downloads: int = 0
    created_at: str | None = None
    description: str | None = None
    sha: str | None = None


class ModelData(BaseModel):
    model_id: str
    downloads: int | None = None
    likes: int | None = None
    private: bool | None = None
    trending_score: int | None = None
    tags: list[str] | None = None
    pipeline_tag: str | None = None
    library_name: str | None = None
    created_at: str | None = None
    config: dict
    parameter_count: int | None = None

    model_config = {"protected_namespaces": ()}


class RawTask(BaseModel):
    is_organic: bool
    task_id: UUID | None = None
    status: str
    model_id: str
    ds: str
    account_id: UUID
    times_delayed: int = 0
    hours_to_complete: int
    test_data: str | None = None
    training_data: str | None = None
    assigned_miners: list[int] | None = None
    miner_scores: list[float] | None = None
    training_repo_backup: str | None = None
    result_model_name: str | None = None

    created_at: datetime
    next_delay_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    termination_at: datetime | None = None
    completed_at: datetime | None = None
    n_eval_attempts: int = 0
    task_type: TaskType
    model_params_count: int = 0

    # Turn off protected namespace for model
    model_config = {"protected_namespaces": ()}


class TextRawTask(RawTask):
    field_system: str | None = None
    field_instruction: str
    field_input: str | None = None
    field_output: str | None = None
    format: str | None = None
    no_input_format: str | None = None
    system_format: None = None  # NOTE: Needs updating to be optional once we accept it
    synthetic_data: str | None = None
    file_format: FileFormat = FileFormat.HF
    task_type: TaskType = TaskType.TEXTTASK


class ImageRawTask(RawTask):
    image_text_pairs: list[ImageTextPair] | None = None
    task_type: TaskType = TaskType.IMAGETASK


# NOTE: As time goes on we will expand this class to be more of a 'submitted task'?
# Might wanna rename this some more
class Task(RawTask):
    trained_model_repository: str | None = None


class TextTask(TextRawTask):
    trained_model_repository: str | None = None


class ImageTask(ImageRawTask):
    trained_model_repository: str | None = None


class PeriodScore(BaseModel):
    quality_score: float
    summed_task_score: float
    average_score: float
    hotkey: str
    weight_multiplier: float
    normalised_score: float | None = 0.0


class TaskNode(BaseModel):
    task_id: str
    hotkey: str
    quality_score: float


class MiniTaskWithScoringOnly(BaseModel):
    is_organic: bool
    task_id: UUID | None = None
    model_id: str
    ds: str
    file_format: FileFormat = FileFormat.HF
    status: str
    account_id: UUID
    times_delayed: int = 0
    hours_to_complete: int
    assigned_miners: list[int] | None = None
    miner_scores: list[float] | None = None
    task_type: TaskType
    created_at: datetime
    next_delay_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    termination_at: datetime | None = None
    completed_at: datetime | None = None
    model_params_count: int | None = 0

    # Turn off protected namespace for model
    model_config = {"protected_namespaces": ()}


class TaskResults(BaseModel):
    task: MiniTaskWithScoringOnly
    node_scores: list[TaskNode]


class NodeAggregationResult(BaseModel):
    task_work_scores: list[float] = Field(default_factory=list)
    average_raw_score: float | None = Field(default=0.0)
    summed_adjusted_task_scores: float = Field(default=0.0)
    quality_score: float | None = Field(default=0.0)
    emission: float | None = Field(default=0.0)
    task_raw_scores: list[float] = Field(default_factory=list)
    hotkey: str

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True


class Submission(BaseModel):
    submission_id: UUID = Field(default_factory=uuid4)
    score: float | None = None
    task_id: UUID
    hotkey: str
    repo: str
    created_on: datetime | None = None
    updated_on: datetime | None = None


class MinerResults(BaseModel):
    hotkey: str
    test_loss: float
    synth_loss: float
    is_finetune: bool
    score: float | None = 0.0
    submission: Submission | None = None
    score_reason: str | None = None


class MinerResultsText(MinerResults):
    task_type: TaskType = TaskType.TEXTTASK


class MinerResultsImage(MinerResults):
    task_type: TaskType = TaskType.IMAGETASK


class QualityMetrics(BaseModel):
    total_score: float
    total_count: int
    total_success: int
    total_quality: int
    avg_quality_score: float
    success_rate: float
    quality_rate: float


class WorkloadMetrics(BaseModel):
    competition_hours: int = Field(ge=0)
    total_params_billions: float = Field(ge=0.0)


class ModelMetrics(BaseModel):
    modal_model: str
    unique_models: int = Field(ge=0)
    unique_datasets: int = Field(ge=0)


class NodeStats(BaseModel):
    quality_metrics: QualityMetrics
    workload_metrics: WorkloadMetrics
    model_metrics: ModelMetrics

    model_config = {"protected_namespaces": ()}


class AllNodeStats(BaseModel):
    daily: NodeStats
    three_day: NodeStats
    weekly: NodeStats
    monthly: NodeStats
    all_time: NodeStats

    @classmethod
    def get_periods_sql_mapping(cls) -> dict[str, str]:
        return {
            "daily": "24 hours",
            "three_day": "3 days",
            "weekly": "7 days",
            "monthly": "30 days",
            "all_time": "all"
        }


class DatasetUrls(BaseModel):
    test_url: str
    synthetic_url: str | None = None
    train_url: str


class DatasetFiles(BaseModel):
    prefix: str
    data: str
    temp_path: Path | None = None


class DatasetJsons(BaseModel):
    train_data: list[Any]
    test_data: list[Any]
    synthetic_data: list[Any] = Field(default_factory=list)

    def to_json_strings(self) -> dict[str, str]:
        return {
            "train_data": json.dumps(self.train_data),
            "test_data": json.dumps(self.test_data),
            "synthetic_data": json.dumps(self.synthetic_data) if self.synthetic_data else "",
        }


class Img2ImgPayload(BaseModel):
    ckpt_name: str
    lora_name: str
    steps: int
    cfg: float
    denoise: float
    comfy_template: dict
    height: int = 1024
    width: int = 1024
    is_safetensors: bool = True
    prompt: str | None = None
    base_image: str | None = None


class NetworkStats(BaseModel):
    number_of_jobs_training: int
    number_of_jobs_preevaluation: int
    number_of_jobs_evaluating: int
    number_of_jobs_success: int
    next_training_end: datetime | None
    job_can_be_made: bool = True


class HotkeyDetails(BaseModel):
    hotkey: str
    submission_id: UUID | None = None
    quality_score: float | None
    test_loss: float | None
    synth_loss: float | None
    repo: str | None
    rank: int | None
    score_reason: str | None = None
    offer_response: dict | None = None


class TextTaskWithHotkeyDetails(TextTask):
    hotkey_details: list[HotkeyDetails]


class ImageTaskWithHotkeyDetails(ImageTask):
    hotkey_details: list[HotkeyDetails]


class Dataset(BaseModel):
    dataset_id: str
    num_rows: int
    num_bytes_parquet_files: int
