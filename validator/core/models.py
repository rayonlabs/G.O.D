import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID
from uuid import uuid4

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import ImageModelType
from core.models.utility_models import ImageTextPair
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import RewardFunction
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

    model_config = ConfigDict(protected_namespaces=())


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

    model_config = ConfigDict(protected_namespaces=())


class RawTask(BaseModel):
    """
    Task data as stored in the base Task table.
    """

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
    model_config = ConfigDict(protected_namespaces=())


class DpoRawTask(RawTask):
    """
    DPO task data as stored in the database. It expand the RawTask with fields from the DpoTask table.
    """

    field_prompt: str
    field_system: str | None = None
    field_chosen: str
    field_rejected: str
    prompt_format: str | None = None
    chosen_format: str | None = None
    rejected_format: str | None = None
    synthetic_data: str | None = None
    file_format: FileFormat = FileFormat.HF
    task_type: TaskType = TaskType.DPOTASK


class GrpoRawTask(RawTask):
    """
    GRPO task data as stored in the database. It expand the RawTask with fields from the GrpoTask table.
    """

    field_prompt: str
    reward_functions: list[RewardFunction]
    file_format: FileFormat = FileFormat.HF
    task_type: TaskType = TaskType.GRPOTASK
    synthetic_data: str | None = None

    @model_validator(mode="after")
    def validate_reward_functions(self) -> "GrpoRawTask":
        for reward_function in self.reward_functions:
            if reward_function.func_hash is None:
                reward_function.func_hash = hashlib.sha256(reward_function.reward_func.encode()).hexdigest()
        return self


class InstructTextRawTask(RawTask):
    """
    Instruct Text task data as stored in the database. It expand the RawTask with fields from the instruct_text_tasks table.
    """

    field_system: str | None = None
    field_instruction: str
    field_input: str | None = None
    field_output: str | None = None
    format: str | None = None
    no_input_format: str | None = None
    system_format: None = None  # NOTE: Needs updating to be optional once we accept it
    synthetic_data: str | None = None
    file_format: FileFormat = FileFormat.HF
    task_type: TaskType = TaskType.INSTRUCTTEXTTASK


class ChatRawTask(RawTask):
    """
    Chat task data as stored in the database. It expand the RawTask with fields from the chat_tasks table.
    """

    chat_template: str | None = "chatml"
    chat_column: str | None = "conversations"
    chat_role_field: str | None = "from"
    chat_content_field: str | None = "value"
    chat_user_reference: str | None = "user"
    chat_assistant_reference: str | None = "assistant"
    synthetic_data: str | None = None
    file_format: FileFormat = FileFormat.HF
    task_type: TaskType = TaskType.CHATTASK


class ImageRawTask(RawTask):
    """
    Image task data as stored in the database. It expand the RawTask with fields from the ImageTask table.
    """

    image_text_pairs: list[ImageTextPair] | None = None
    task_type: TaskType = TaskType.IMAGETASK
    model_type: ImageModelType = ImageModelType.SDXL


# NOTE: As time goes on we will expand this class to be more of a 'submitted task'?
# Might wanna rename this some more
class Task(RawTask):
    trained_model_repository: str | None = None


class InstructTextTask(InstructTextRawTask):
    """
    Expands on the InstructTextRawTask with the trained_model_repository field.
    This field is not stored in the db directly, but is computed from the submissions table.

    """

    trained_model_repository: str | None = None


class ImageTask(ImageRawTask):
    trained_model_repository: str | None = None


class DpoTask(DpoRawTask):
    trained_model_repository: str | None = None


class GrpoTask(GrpoRawTask):
    trained_model_repository: str | None = None


class ChatTask(ChatRawTask):
    trained_model_repository: str | None = None


class PeriodScore(BaseModel):
    quality_score: float
    summed_task_score: float
    average_score: float
    std_score: float | None = 0.0
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
    model_config = ConfigDict(protected_namespaces=())


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
    model_hash: str | None = None
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
    adjusted_loss: float | None = None


class MinerResultsText(MinerResults):
    task_type: TaskType

    @field_validator("task_type")
    def validate_task_type(cls, v):
        if v not in {TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.CHATTASK}:
            raise ValueError("Must be INSTRUCTTEXTTASK, DPOTASK or GRPOTASK")
        return v


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

    model_config = ConfigDict(protected_namespaces=())


class AllNodeStats(BaseModel):
    daily: NodeStats
    three_day: NodeStats
    weekly: NodeStats
    monthly: NodeStats
    all_time: NodeStats

    @classmethod
    def get_periods_sql_mapping(cls) -> dict[str, str]:
        return {"daily": "24 hours", "three_day": "3 days", "weekly": "7 days", "monthly": "30 days", "all_time": "all"}


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
    model_type: str = "sdxl"
    is_safetensors: bool = True
    prompt: str | None = None
    base_image: str | None = None

    model_config = ConfigDict(protected_namespaces=())


class NetworkStats(BaseModel):
    number_of_jobs_training: int
    number_of_jobs_preevaluation: int
    number_of_jobs_evaluating: int
    number_of_jobs_success: int
    next_training_end: datetime | None
    job_can_be_made: bool = True


class DetailedNetworkStats(NetworkStats):
    instruct_training: int = 0
    instruct_preevaluation: int = 0
    instruct_evaluating: int = 0
    instruct_success: int = 0

    dpo_training: int = 0
    dpo_preevaluation: int = 0
    dpo_evaluating: int = 0
    dpo_success: int = 0

    grpo_training: int = 0
    grpo_preevaluation: int = 0
    grpo_evaluating: int = 0
    grpo_success: int = 0

    image_training: int = 0
    image_preevaluation: int = 0
    image_evaluating: int = 0
    image_success: int = 0


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


class InstructTextTaskWithHotkeyDetails(InstructTextTask):
    hotkey_details: list[HotkeyDetails]


class ImageTaskWithHotkeyDetails(ImageTask):
    hotkey_details: list[HotkeyDetails]


class DpoTaskWithHotkeyDetails(DpoTask):
    hotkey_details: list[HotkeyDetails]


class GrpoTaskWithHotkeyDetails(GrpoTask):
    hotkey_details: list[HotkeyDetails]


class ChatTaskWithHotkeyDetails(ChatTask):
    hotkey_details: list[HotkeyDetails]


class Dataset(BaseModel):
    dataset_id: str
    num_rows: int
    num_bytes_parquet_files: int
    dpo_available: bool = False
    dpo_prompt_column: str | None = None
    dpo_accepted_column: str | None = None
    dpo_rejected_column: str | None = None


class EvaluationArgs(BaseModel):
    dataset: str
    original_model: str
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType | ChatTemplateDatasetType
    file_format: FileFormat
    repo: str

    @field_validator("file_format", mode="before")
    def parse_file_format(cls, value):
        if isinstance(value, str):
            return FileFormat(value)
        return value

    @field_validator("dataset_type", mode="before")
    def parse_dataset_type(cls, value):
        if isinstance(value, str):
            try:
                data = json.loads(value)
                if "field_instruction" in data and "field_input" in data:
                    return InstructTextDatasetType.model_validate(data)
                elif "field_chosen" in data:
                    return DpoDatasetType.model_validate(data)
                elif "reward_functions" in data:
                    return GrpoDatasetType.model_validate(data)
            except Exception as e:
                raise ValueError(f"Failed to parse dataset type: {e}")
        return value


# Type aliases for common task type groupings
AnyTextTypeRawTask = InstructTextRawTask | DpoRawTask | GrpoRawTask | ChatRawTask
AnyTypeRawTask = AnyTextTypeRawTask | ImageRawTask
AnyTypeTask = InstructTextTask | DpoTask | ImageTask | GrpoTask | ChatTask
AnyTypeTaskWithHotkeyDetails = (
    InstructTextTaskWithHotkeyDetails | ImageTaskWithHotkeyDetails | DpoTaskWithHotkeyDetails | GrpoTaskWithHotkeyDetails | ChatTaskWithHotkeyDetails
)
