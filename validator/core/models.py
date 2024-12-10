import json
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Optional
from uuid import UUID
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field


class Task(BaseModel):
    is_organic: bool
    task_id: Optional[UUID] = None
    model_id: str
    ds_id: str
    input: Optional[str] = None
    status: str
    system: Optional[str] = None
    instruction: Optional[str] = None
    output: Optional[str] = None
    format: Optional[str] = None
    no_input_format: Optional[str] = None
    test_data: Optional[str] = None
    synthetic_data: Optional[str] = None
    training_data: Optional[str] = None
    assigned_miners: Optional[list[int]] = None
    miner_scores: Optional[list[float]] = None
    created_timestamp: Optional[datetime] = None
    delay_timestamp: Optional[datetime] = None
    delay_times: Optional[int] = 0
    updated_timestamp: Optional[datetime] = None
    started_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    completed_timestamp: Optional[datetime] = None
    hours_to_complete: int
    best_submission_repo: Optional[str] = None
    user_id: Optional[str] = None

    # Turn off protected namespace for model
    model_config = {"protected_namespaces": ()}


class PeriodScore(BaseModel):
    quality_score: float
    summed_task_score: float
    average_score: float
    hotkey: str
    normalised_score: Optional[float] = 0.0


class TaskNode(BaseModel):
    task_id: str
    hotkey: str
    quality_score: float


class TaskResults(BaseModel):
    task: Task
    node_scores: list[TaskNode]


class NodeAggregationResult(BaseModel):
    task_work_scores: list[float] = Field(default_factory=list)
    average_raw_score: Optional[float] = Field(default=0.0)
    summed_adjusted_task_scores: float = Field(default=0.0)
    quality_score: Optional[float] = Field(default=0.0)
    emission: Optional[float] = Field(default=0.0)
    task_raw_scores: list[float] = Field(default_factory=list)
    hotkey: str

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True


class Submission(BaseModel):
    submission_id: UUID = Field(default_factory=uuid4)
    score: Optional[float] = None
    task_id: UUID
    hotkey: str
    repo: str
    created_on: Optional[datetime]
    updated_on: Optional[datetime]


class MinerResults(BaseModel):
    hotkey: str
    test_loss: float
    synth_loss: float
    is_finetune: bool
    score: Optional[float] = 0.0
    submission: Optional[Submission] = None


class QualityMetrics(BaseModel):
    total_score: float
    total_count: int
    total_success: int
    total_quality: int
    avg_quality_score: float = Field(ge=0.0)
    success_rate: float = Field(ge=0.0, le=1.0)
    quality_rate: float = Field(ge=0.0, le=1.0)


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


class AllNodeStats(BaseModel):
    daily: NodeStats
    three_day: NodeStats
    weekly: NodeStats
    monthly: NodeStats
    all_time: NodeStats


class LeaderboardRow(BaseModel):
    hotkey: str
    stats: AllNodeStats


class DatasetUrls(BaseModel):
    test_url: str
    synthetic_url: Optional[str] = None
    train_url: str


class DatasetFiles(BaseModel):
    prefix: str
    data: str
    temp_path: Optional[Path] = None


class DatasetJsons(BaseModel):
    train_data: list[Any]
    test_data: list[Any]
    synthetic_data: list[Any] = Field(default_factory=list)

    def to_json_strings(self) -> dict[str, str]:
        return {
            'train_data': json.dumps(self.train_data),
            'test_data': json.dumps(self.test_data),
            'synthetic_data': json.dumps(self.synthetic_data) if self.synthetic_data else ""
        }
