import secrets
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from core.models.utility_models import TaskType
from validator.core.constants import (
    TOURNAMENT_GPU_THRESHOLD_FOR_2X_H100,
    TOURNAMENT_GPU_THRESHOLD_FOR_4X_H100,
    TOURNAMENT_GPU_THRESHOLD_FOR_8X_H100,
    TOURNAMENT_DPO_GPU_MULTIPLIER,
    TOURNAMENT_GRPO_GPU_MULTIPLIER,
)


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
    date_part = datetime.now().strftime('%Y%m%d')
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
    current_round_id: str | None = None


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


class TournamentTask(BaseModel):
    tournament_id: str
    round_id: str
    task_id: str
    group_id: str | None = None
    pair_id: str | None = None
    gpu_requirement: GpuRequirement | None = None


class Group(BaseModel):
    member_ids: list[str]


class GroupRound(BaseModel):
    groups: list[Group]


class KnockoutRound(BaseModel):
    pairs: list[tuple[str, str]]


Round = GroupRound | KnockoutRound


class TournamentRound(BaseModel):
    round_structure: Round
    tasks: list[str] = Field(default_factory=list)
    is_final_round: bool = False