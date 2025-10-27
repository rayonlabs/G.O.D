import secrets
from datetime import datetime
from enum import Enum
from uuid import UUID

from fiber.chain.models import Node
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from core.models.payload_models import TrainingRepoResponse
from core.models.utility_models import TaskType
from core.models.utility_models import TrainingStatus
from validator.core.models import AnyTypeRawTask


class TournamentStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class RoundStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"


class RoundType(str, Enum):
    GROUP = "group"
    KNOCKOUT = "knockout"


class TournamentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"


class GpuRequirement(str, Enum):
    A100 = "A100"
    H100_1X = "1xH100"
    H100_2X = "2xH100"
    H100_4X = "4xH100"
    H100_8X = "8xH100"


def generate_tournament_id() -> str:
    hash_part = secrets.token_hex(8)
    date_part = datetime.now().strftime("%Y%m%d")
    return f"tourn_{hash_part}_{date_part}"


def generate_round_id(tournament_id: str, round_number: int) -> str:
    return f"{tournament_id}_round_{round_number:03d}"


def generate_group_id(round_id: str, group_number: int) -> str:
    return f"{round_id}_group_{group_number:03d}"


def generate_pair_id(round_id: str, pair_number: int) -> str:
    return f"{round_id}_pair_{pair_number:03d}"


class TournamentData(BaseModel):
    tournament_id: str
    tournament_type: TournamentType
    status: TournamentStatus = TournamentStatus.PENDING
    base_winner_hotkey: str | None = Field(
        default=None, description="The defending champion's real hotkey at the START of this tournament (snapshot)."
    )
    winner_hotkey: str | None = Field(
        default=None,
        description="The tournament winner's hotkey at the END of this tournament. "
        "May be EMISSION_BURN_HOTKEY if the defending champion successfully defended.",
    )
    winning_performance_difference: float | None = Field(
        default=None,
        description="Performance difference metric (0.0 to 1.0) between champion and challenger in boss round. "
        "Calculated as: (defending_champion_score - new_winner_score) / defending_champion_score. "
        "score = loss, so lower is better. Higher diff = better perf = less burn.",
    )


class TournamentRoundData(BaseModel):
    round_id: str
    tournament_id: str
    round_number: int
    round_type: RoundType
    is_final_round: bool = False
    status: RoundStatus = RoundStatus.PENDING


class TournamentGroupData(BaseModel):
    group_id: str
    round_id: str


class TournamentPairData(BaseModel):
    pair_id: str
    round_id: str
    hotkey1: str
    hotkey2: str
    winner_hotkey: str | None = None


class TournamentParticipant(BaseModel):
    tournament_id: str
    hotkey: str
    eliminated_in_round_id: str | None = None
    final_position: int | None = None
    training_repo: str | None = None
    training_commit_hash: str | None = None
    backup_repo: str | None = None


class TournamentTask(BaseModel):
    tournament_id: str
    round_id: str
    task_id: str
    group_id: str | None = None
    pair_id: str | None = None
    gpu_requirement: GpuRequirement | None = None

    @field_validator("task_id", mode="before")
    @classmethod
    def ensure_str(cls, v):
        if isinstance(v, UUID):
            return str(v)
        return v


class Group(BaseModel):
    member_ids: list[str]
    task_ids: list[str] | None = None


class GroupRound(BaseModel):
    groups: list[Group]


class KnockoutRound(BaseModel):
    # pairs of hotkeys
    pairs: list[tuple[str, str]]
    tasks: list[str] | None = None


Round = GroupRound | KnockoutRound


class TournamentRound(BaseModel):
    round_structure: Round
    tasks: list[str] = Field(default_factory=list)
    is_final_round: bool = False


class TaskTrainingAssignment(BaseModel):
    """Data for assigning a task-hotkey pair for training with repo information."""

    task_id: str
    hotkey: str
    created_at: datetime
    priority: int = Field(
        default=1, description="Training priority: 1=organic (non-tournament/non-benchmark), 2=tournament, 3=benchmark"
    )
    training_repo: str | None = None
    training_commit_hash: str | None = None


class TournamentTaskTraining(BaseModel):
    task: AnyTypeRawTask
    hotkey: str
    training_status: TrainingStatus
    n_training_attempts: int
    created_at: datetime
    updated_at: datetime
    training_repo: str | None = None
    training_commit_hash: str | None = None


class TournamentTaskScore(BaseModel):
    task_id: str
    group_id: str | None
    pair_id: str | None
    winner: str | None
    participant_scores: list[dict]


class DetailedTournamentTaskScore(TournamentTaskScore):
    task_type: TaskType | None = None


class TournamentRoundResult(BaseModel):
    round_id: str
    round_number: int
    round_type: str
    is_final_round: bool
    tasks: list[TournamentTaskScore]


class DetailedTournamentRoundResult(TournamentRoundResult):
    status: str
    participants: list[str]
    tasks: list[DetailedTournamentTaskScore]


class TournamentResults(BaseModel):
    tournament_id: str
    rounds: list[TournamentRoundResult]


class TournamentResultsWithWinners(BaseModel):
    tournament_id: str
    rounds: list[TournamentRoundResult]
    base_winner_hotkey: str | None = None
    winner_hotkey: str | None = None


class TournamentScore(BaseModel):
    hotkey: str
    score: float


class TournamentTypeResult(BaseModel):
    scores: list[TournamentScore]
    prev_winner_hotkey: str | None
    prev_winner_won_final: bool


class TaskPerformanceDifference(BaseModel):
    """Performance difference data for a single task"""

    task_id: str
    task_type: str
    boss_score: float | None
    challenger_score: float | None
    threshold_used: float  # 0.05, 0.075, or 0.10
    performance_difference: float | None  # Percentage difference (positive = challenger better)
    challenger_won: bool


class TournamentPerformanceData(BaseModel):
    """Performance data for tournament vs sync comparison"""

    tournament_task_id: str
    synthetic_task_id: str
    task_type: str
    tournament_winner_score: float
    best_synthetic_score: float
    performance_difference: float  # Percentage difference (positive = tournament better)


class TournamentDetailsResponse(BaseModel):
    tournament_id: str
    tournament_type: TournamentType
    status: TournamentStatus
    base_winner_hotkey: str | None
    winner_hotkey: str | None
    participants: list[TournamentParticipant]
    rounds: list[DetailedTournamentRoundResult]
    final_scores: list[TournamentScore]
    text_tournament_weight: float
    image_tournament_weight: float
    boss_round_performance: list[TaskPerformanceDifference] | None = None
    sync_performance: list[TournamentPerformanceData] | None = None


class TournamentAuditData(BaseModel):
    text_tournament_data: TournamentResultsWithWinners | None = None
    image_tournament_data: TournamentResultsWithWinners | None = None
    participants: list[str] = []
    text_tournament_weight: float = 0.0
    image_tournament_weight: float = 0.0
    burn_weight: float = 0.0
    weekly_participation: list["HotkeyTaskParticipation"] = []


class BossRoundTaskCompletion(BaseModel):
    total_synth_tasks: int
    completed_synth_tasks: int


class BossRoundTaskPair(BaseModel):
    tournament_task_id: str
    synthetic_task_id: str
    winner_hotkey: str
    task_type: str


class TaskScore(BaseModel):
    hotkey: str
    test_loss: float
    synth_loss: float
    quality_score: float


class RespondingNode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    node: Node
    training_repo_response: TrainingRepoResponse


class NextTournamentInfo(BaseModel):
    tournament_type: TournamentType
    next_start_date: datetime | None = None
    next_end_date: datetime | None = None
    interval_hours: int | None = None
    current_round_number: int | None = None
    tournament_status: str | None = None


class NextTournamentDates(BaseModel):
    text: NextTournamentInfo
    image: NextTournamentInfo


class ActiveTournamentParticipant(BaseModel):
    hotkey: str


class ActiveTournamentInfo(BaseModel):
    tournament_id: str
    tournament_type: TournamentType
    status: TournamentStatus
    participants: list[ActiveTournamentParticipant]
    created_at: datetime


class ActiveTournamentsResponse(BaseModel):
    text: ActiveTournamentInfo | None
    image: ActiveTournamentInfo | None


class TournamentBurnData(BaseModel):
    """Separated burn data by tournament type"""

    text_performance_diff: float | None
    image_performance_diff: float | None
    text_burn_proportion: float
    image_burn_proportion: float
    text_tournament_weight: float
    image_tournament_weight: float
    burn_weight: float


class LatestTournamentsDetailsResponse(BaseModel):
    """Response for latest tournaments with burn data"""

    text: TournamentDetailsResponse | None
    image: TournamentDetailsResponse | None
    burn_data: TournamentBurnData


class TournamentHistoryEntry(BaseModel):
    """Individual tournament entry for history response"""

    tournament_id: str
    tournament_type: TournamentType
    status: TournamentStatus
    winner_hotkey: str | None = None
    base_winner_hotkey: str | None = None
    created_at: datetime | None = None


class TournamentHistoryResponse(BaseModel):
    """Response for tournament history endpoint"""

    tournaments: list[TournamentHistoryEntry]


class BenchmarkTaskCopy(BaseModel):
    """Raw benchmark task copy data from database"""

    copy_task_id: str
    root_task_id: str
    participant_hotkey: str
    tournament_id: str | None = None
    created_at: datetime
    task_type: TaskType
    model_id: str
    dataset: str
    hours_to_complete: int
    model_params_count: int
    is_organic: bool
    task_created_at: datetime | None = None


class BenchmarkInstance(BaseModel):
    """A single benchmark instance (copy task) with its results"""

    copy_task_id: str
    participant_hotkey: str
    tournament_id: str
    created_at: datetime
    test_loss: float | None = None


class BenchmarkTimeline(BaseModel):
    """Timeline of benchmark results for a single root task"""

    root_task_id: str
    task_type: TaskType
    model_id: str
    dataset: str
    hours_to_complete: int
    model_params_count: int
    is_organic: bool
    task_created_at: datetime | None = None
    benchmarks: list[BenchmarkInstance] = Field(default_factory=list)


class BenchmarkTimelineResponse(BaseModel):
    """Response containing benchmark timelines for all tasks"""

    timelines: list[BenchmarkTimeline]


class HotkeyTournamentParticipation(BaseModel):
    """Tournament participation data for a specific hotkey"""

    hotkey: str
    participated_in_text: bool  # participated in the most recent text tournament
    participated_in_image: bool  # participated in the most recent image tournament
    text_proportion: float  # 0.0, 0.6, or 1.0 based on participation
    image_proportion: float  # 0.0, 0.4, or 1.0 based on participation


class HotkeyTaskParticipation(BaseModel):
    """Weekly task participation data for a specific hotkey"""

    hotkey: str
    text_task_proportion: float  # proportion of text tasks (0.0 to 1.0)
    image_task_proportion: float  # proportion of image tasks (0.0 to 1.0)
    total_tasks: int  # total number of tasks in the period


class NodeWeightsResult(BaseModel):
    """Result of node weight calculations"""

    node_ids: list[int]
    node_weights: list[float]

    def to_tuple(self) -> tuple[list[int], list[float]]:
        """Convert to tuple format for compatibility with existing code"""
        return self.node_ids, self.node_weights
