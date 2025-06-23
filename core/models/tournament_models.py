import secrets
from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field

from core.models.utility_models import TaskStatus
from validator.db.sql.tasks import get_task
from validator.db.sql.submissions_and_scoring import get_task_winner, get_task_scores_for_participants, get_average_scores_for_participants
from collections import Counter


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
    task_ids: list[str]
    
    def target_num_tasks(self) -> int:
        return 3
    
    async def is_completed(self) -> bool:
        tasks = [await get_task(task_id) for task_id in self.task_ids]
        if len(tasks) < self.target_num_tasks():
            return False
        return all(task.status == TaskStatus.SUCCESS for task in tasks)
    
    async def get_winners(self, psql_db) -> list[str]:
        """ returns the hotkeys that go through to the next round """
        if not await self.is_completed():
            return []
        
        # Get winner for each task and count wins
        task_winners = []
        for task_id in self.task_ids:
            winner = await get_task_winner(UUID(task_id), psql_db)
            if winner:
                task_winners.append(winner)
        
        if not task_winners:
            return []
        
        winner_counts = Counter(task_winners)
        sorted_participants = winner_counts.most_common()
        
        # Determine winners: top 2, or all tied for first, or single winner
        if len(sorted_participants) == 1:
            return [sorted_participants[0][0]]
        
        best_count = sorted_participants[0][1]
        tied_winners = [hotkey for hotkey, count in sorted_participants if count == best_count]
        
        if len(tied_winners) > 1:
            return tied_winners 
        else:
            return [sorted_participants[0][0], sorted_participants[1][0]]  # Top 2


class GroupRound(BaseModel):
    groups: list[Group]
    
    async def is_completed(self, psql_db) -> bool:
        """Check if all groups in the round have completed their tasks"""
        return all(await group.is_completed() for group in self.groups)
    
    async def get_winners(self, psql_db) -> list[str]:
        """Get all winners from all groups that advance to the next round"""
        if not await self.is_completed(psql_db):
            return []
        
        winners = []
        for group in self.groups:
            group_winners = await group.get_winners(psql_db)
            winners.extend(group_winners)
        return winners


class KnockoutRound(BaseModel):
    pairs: list[tuple[str, str]]
    tasks: list[str]
    
    async def is_completed(self, psql_db, round_tasks: list[str]) -> bool:
        """Check if all pairs in the knockout round have completed their tasks"""
        # For knockout rounds, we need to check if all pairs have winners
        # This would typically involve checking task completion for each pair
        # For now, we'll assume this needs to be implemented based on task data
        # TODO: Implement based on actual task completion logic for pairs
        return True  # Placeholder - needs implementation based on task data
    
    async def get_winners(self, psql_db, round_tasks: list[str]) -> list[str]:
        """Get winners from all pairs that advance to the next round"""
        if not await self.is_completed(psql_db, round_tasks):
            return []
        
        winners = []
        for i, (hotkey1, hotkey2) in enumerate(self.pairs):
            # Find task for this pair (assuming tasks are in order)
            if i < len(round_tasks):
                task_id = round_tasks[i]
                
                # Get scores for both participants
                scores = await get_task_scores_for_participants(
                    UUID(task_id), [hotkey1, hotkey2], psql_db
                )
                
                hotkey1_score = scores.get(hotkey1)
                hotkey2_score = scores.get(hotkey2)
                
                # Determine winner (lower loss wins)
                if hotkey1_score is not None and hotkey2_score is not None:
                    if hotkey1_score < hotkey2_score:
                        winners.append(hotkey1)
                    elif hotkey2_score < hotkey1_score:
                        winners.append(hotkey2)
                    else:
                        # Tie - randomly choose winner
                        import random
                        winners.append(random.choice([hotkey1, hotkey2]))
                elif hotkey1_score is not None:
                    winners.append(hotkey1)
                elif hotkey2_score is not None:
                    winners.append(hotkey2)
        
        return winners


Round = GroupRound | KnockoutRound


class TournamentRound(BaseModel):
    round_structure: Round
    tasks: list[str] = Field(default_factory=list)
    is_final_round: bool = False
    
    async def is_completed(self, psql_db) -> bool:
        """Check if the round is completed based on its structure type"""
        if isinstance(self.round_structure, GroupRound):
            return await self.round_structure.is_completed(psql_db)
        elif isinstance(self.round_structure, KnockoutRound):
            return await self.round_structure.is_completed(psql_db, self.tasks)
        else:
            raise ValueError(f"Unknown round structure type: {type(self.round_structure)}")
    
    async def get_winners(self, psql_db) -> list[str]:
        """Get winners that advance to the next round based on round structure type"""
        if isinstance(self.round_structure, GroupRound):
            return await self.round_structure.get_winners(psql_db)
        elif isinstance(self.round_structure, KnockoutRound):
            return await self.round_structure.get_winners(psql_db, self.tasks)
        else:
            raise ValueError(f"Unknown round structure type: {type(self.round_structure)}")