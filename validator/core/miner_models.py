from pydantic import BaseModel
from pydantic import Field
from datetime import datetime
from uuid import UUID

from core.models.utility_models import TaskType


class OnChainIncentive(BaseModel):
    raw_value: int
    normalized: float
    network_share_percent: float


class CalculatedPerformanceWeight(BaseModel):
    weight_value: float
    network_share_percent: float


class TaskSourcePerformance(BaseModel):
    task_count: int = 0
    average_score: float = 0.0
    normalized_score: float = 0.0


class PeriodScore(BaseModel):
    average_score: float = 0.0
    normalized_score: float = 0.0
    weight_multiplier: float = 0.0
    weighted_contribution: float = 0.0


class TaskTypePerformance(BaseModel):
    one_day: PeriodScore = Field(default_factory=PeriodScore)
    three_day: PeriodScore = Field(default_factory=PeriodScore)
    seven_day: PeriodScore = Field(default_factory=PeriodScore)
    
    organic_performance: TaskSourcePerformance = Field(default_factory=TaskSourcePerformance)
    synthetic_performance: TaskSourcePerformance = Field(default_factory=TaskSourcePerformance)
    
    total_submissions: int = 0
    average_score: float = 0.0
    weight_contribution: float = 0.0


class TaskTypeBreakdown(BaseModel):
    instruct_text: TaskTypePerformance = Field(default_factory=TaskTypePerformance)
    dpo: TaskTypePerformance = Field(default_factory=TaskTypePerformance)
    image: TaskTypePerformance = Field(default_factory=TaskTypePerformance)
    grpo: TaskTypePerformance = Field(default_factory=TaskTypePerformance)


class PeriodTotals(BaseModel):
    one_day_total: float = 0.0
    three_day_total: float = 0.0
    seven_day_total: float = 0.0


class TaskSubmissionResult(BaseModel):
    task_id: UUID
    task_type: TaskType
    is_organic: bool
    created_at: datetime
    score: float
    rank: int
    total_participants: int
    percentile: float


class MinerPerformanceMetrics(BaseModel):
    total_tasks_participated: int = 0
    tasks_last_24h: int = 0
    tasks_last_7d: int = 0
    
    positive_score_rate: float = 0.0
    average_percentile_rank: float = 0.0
    
    task_type_distribution: dict[str, float] = Field(default_factory=dict)


class MinerDetailsResponse(BaseModel):
    hotkey: str
    node_id: int | None = None
    
    current_incentive: OnChainIncentive
    
    performance_weight: CalculatedPerformanceWeight
    
    task_type_breakdown: TaskTypeBreakdown
    
    period_totals: PeriodTotals
    
    recent_submissions: list[TaskSubmissionResult] = Field(default_factory=list)
    
    performance_metrics: MinerPerformanceMetrics