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
from validator.core.constants import TOURNAMENT_DPO_GPU_MULTIPLIER
from validator.core.constants import TOURNAMENT_GPU_THRESHOLD_FOR_2X_H100
from validator.core.constants import TOURNAMENT_GPU_THRESHOLD_FOR_4X_H100
from validator.core.constants import TOURNAMENT_GPU_THRESHOLD_FOR_8X_H100
from validator.core.constants import TOURNAMENT_GRPO_GPU_MULTIPLIER
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


def get_tournament_gpu_requirement(task_type: TaskType, model_params_count: int) -> GpuRequirement:
    if task_type == TaskType.IMAGETASK:
        return GpuRequirement.A100

    params_b = model_params_count / 1_000_000_000

    if task_type == TaskType.DPOTASK:
        params_b *= TOURNAMENT_DPO_GPU_MULTIPLIER
    elif task_type == TaskType.GRPOTASK:
        params_b *= TOURNAMENT_GRPO_GPU_MULTIPLIER

    if params_b <= TOURNAMENT_GPU_THRESHOLD_FOR_2X_H100:
        return GpuRequirement.H100_1X
    elif params_b <= TOURNAMENT_GPU_THRESHOLD_FOR_4X_H100:
        return GpuRequirement.H100_2X
    elif params_b <= TOURNAMENT_GPU_THRESHOLD_FOR_8X_H100:
        return GpuRequirement.H100_4X
    else:
        return GpuRequirement.H100_8X


class TournamentData(BaseModel):
    tournament_id: str
    tournament_type: TournamentType
    status: TournamentStatus = TournamentStatus.PENDING
    base_winner_hotkey: str | None = None
    winner_hotkey: str | None = None


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
    stake_required: float | None = None
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


class TournamentTaskTraining(BaseModel):
    task: AnyTypeRawTask
    hotkey: str
    training_status: TrainingStatus
    n_training_attempts: int
    created_at: datetime
    updated_at: datetime


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


class TournamentAuditData(BaseModel):
    text_tournament_data: TournamentResultsWithWinners | None = None
    image_tournament_data: TournamentResultsWithWinners | None = None
    participants: list[str] = []
    tournament_weight_multiplier: float = 0.0
    regular_weight_multiplier: float = 0.0
    burn_weight: float = 0.0


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
    boosted_stake: float
    actual_stake: float


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
    stake_requirement: float


class ActiveTournamentInfo(BaseModel):
    tournament_id: str
    tournament_type: TournamentType
    status: TournamentStatus
    participants: list[ActiveTournamentParticipant]
    created_at: datetime


class ActiveTournamentsResponse(BaseModel):
    text: ActiveTournamentInfo | None
    image: ActiveTournamentInfo | None
