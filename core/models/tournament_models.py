import secrets
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


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