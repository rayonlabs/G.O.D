from datetime import datetime
from datetime import timedelta
from datetime import timezone

from fiber.chain.models import Node

import validator.core.constants
import validator.db.constants as cst
from core.models.tournament_models import GroupRound
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentGroupData
from core.models.tournament_models import TournamentPairData
from core.models.tournament_models import TournamentParticipant
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentStatus
from core.models.tournament_models import TournamentTask
from core.models.tournament_models import TournamentTaskTraining
from core.models.tournament_models import TournamentType
from core.models.tournament_models import TournamentResults
from core.models.tournament_models import TournamentRoundResult
from core.models.tournament_models import TournamentTaskScore
from core.models.utility_models import GPUInfo
from core.models.utility_models import TrainerInfo
from core.models.utility_models import TrainingStatus
from validator.db.database import PSQLDB
from validator.db.sql import tasks as task_sql
from validator.db.sql.submissions_and_scoring import get_all_scores_and_losses_for_task, get_task_winners
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def create_tournament(tournament: TournamentData, psql_db: PSQLDB) -> str:
    async with await psql_db.connection() as connection:
        query = f"""
            INSERT INTO {cst.TOURNAMENTS_TABLE}
            ({cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS}, {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}, {cst.CREATED_AT}, {cst.UPDATED_AT})
            VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING {cst.TOURNAMENT_ID}
        """
        result = await connection.fetchrow(
            query,
            tournament.tournament_id,
            tournament.tournament_type,
            tournament.status,
            tournament.base_winner_hotkey,
            tournament.winner_hotkey,
        )
        logger.info(f"Created tournament: {tournament.tournament_id}")
        return result[cst.TOURNAMENT_ID]


async def insert_tournament_round(round_data: TournamentRoundData, psql_db: PSQLDB) -> str:
    async with await psql_db.connection() as connection:
        query = f"""
            INSERT INTO {cst.TOURNAMENT_ROUNDS_TABLE}
            ({cst.ROUND_ID}, {cst.TOURNAMENT_ID}, {cst.ROUND_NUMBER}, {cst.ROUND_TYPE},
             {cst.IS_FINAL_ROUND}, {cst.ROUND_STATUS}, {cst.CREATED_AT})
            VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
            RETURNING {cst.ROUND_ID}
        """
        result = await connection.fetchrow(
            query,
            round_data.round_id,
            round_data.tournament_id,
            round_data.round_number,
            round_data.round_type,
            round_data.is_final_round,
            round_data.status,
        )
        logger.info(f"Created tournament round: {round_data.round_id}")
        return result[cst.ROUND_ID]


async def insert_tournament_groups_with_members(round_id: str, round_structure: GroupRound, psql_db: PSQLDB) -> list[str]:
    group_ids = []
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            for i, group in enumerate(round_structure.groups):
                group_id = f"{round_id}_group_{i + 1:03d}"

                group_query = f"""
                    INSERT INTO {cst.TOURNAMENT_GROUPS_TABLE}
                    ({cst.GROUP_ID}, {cst.ROUND_ID}, {cst.CREATED_AT})
                    VALUES ($1, $2, CURRENT_TIMESTAMP)
                    RETURNING {cst.GROUP_ID}
                """
                await connection.execute(group_query, group_id, round_id)

                member_query = f"""
                    INSERT INTO {cst.TOURNAMENT_GROUP_MEMBERS_TABLE}
                    ({cst.GROUP_ID}, {cst.HOTKEY}, {cst.CREATED_AT})
                    VALUES ($1, $2, CURRENT_TIMESTAMP)
                """
                for hotkey in group.member_ids:
                    await connection.execute(member_query, group_id, hotkey)

                group_ids.append(group_id)

            logger.info(f"Created {len(group_ids)} groups for round {round_id}")
    return group_ids


async def insert_tournament_pairs(round_id: str, hotkey_pairs: list[tuple[str, str]], psql_db: PSQLDB) -> list[str]:
    pair_ids = []
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            query = f"""
                INSERT INTO {cst.TOURNAMENT_PAIRS_TABLE}
                ({cst.PAIR_ID}, {cst.ROUND_ID}, {cst.HOTKEY1}, {cst.HOTKEY2}, {cst.CREATED_AT})
                VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                RETURNING {cst.PAIR_ID}
            """
            for i, (hotkey1, hotkey2) in enumerate(hotkey_pairs):
                pair_id = f"{round_id}_pair_{i + 1:03d}"
                await connection.execute(query, pair_id, round_id, hotkey1, hotkey2)
                pair_ids.append(pair_id)

            logger.info(f"Created {len(pair_ids)} pairs for round {round_id}")
    return pair_ids


async def add_tournament_participants(participants: list[TournamentParticipant], psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            query = f"""
                INSERT INTO {cst.TOURNAMENT_PARTICIPANTS_TABLE} 
                ({cst.TOURNAMENT_ID}, {cst.HOTKEY}, {cst.TRAINING_REPO}, {cst.TRAINING_COMMIT_HASH}, {cst.STAKE_REQUIRED}, {cst.CREATED_AT})
                VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)
                ON CONFLICT ({cst.TOURNAMENT_ID}, {cst.HOTKEY}) DO NOTHING
            """
            for participant in participants:
                await connection.execute(
                    query,
                    participant.tournament_id,
                    participant.hotkey,
                    participant.training_repo,
                    participant.training_commit_hash,
                    participant.stake_required,
                )
            logger.info(f"Added {len(participants)} participants to tournament")


async def add_tournament_tasks(tasks: list[TournamentTask], psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            query = f"""
                INSERT INTO {cst.TOURNAMENT_TASKS_TABLE}
                ({cst.TOURNAMENT_ID}, {cst.ROUND_ID}, {cst.TASK_ID}, {cst.GROUP_ID}, {cst.PAIR_ID}, {cst.CREATED_AT})
                VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)
            """
            for task in tasks:
                await connection.execute(query, task.tournament_id, task.round_id, task.task_id, task.group_id, task.pair_id)
            logger.info(f"Added {len(tasks)} tasks to tournament")


async def get_tournament(tournament_id: str, psql_db: PSQLDB) -> TournamentData | None:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS}, {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1
        """
        result = await connection.fetchrow(query, tournament_id)
        if result:
            return TournamentData(
                tournament_id=result[cst.TOURNAMENT_ID],
                tournament_type=result[cst.TOURNAMENT_TYPE],
                status=result[cst.TOURNAMENT_STATUS],
                base_winner_hotkey=result[cst.BASE_WINNER_HOTKEY],
                winner_hotkey=result[cst.WINNER_HOTKEY],
            )
        return None


async def get_latest_completed_tournament(psql_db: PSQLDB, tournament_type: TournamentType) -> TournamentData | None:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS}, {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_TYPE} = $1 AND {cst.TOURNAMENT_STATUS} = 'completed'
            ORDER BY {cst.UPDATED_AT} DESC
            LIMIT 1
        """
        result = await connection.fetchrow(query, tournament_type.value)
        if result:
            return TournamentData(
                tournament_id=result[cst.TOURNAMENT_ID],
                tournament_type=result[cst.TOURNAMENT_TYPE],
                status=result[cst.TOURNAMENT_STATUS],
                base_winner_hotkey=result[cst.BASE_WINNER_HOTKEY],
                winner_hotkey=result[cst.WINNER_HOTKEY],
            )
        return None


async def get_tournament_round(round_id: str, psql_db: PSQLDB) -> TournamentRoundData | None:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.ROUND_ID}, {cst.TOURNAMENT_ID}, {cst.ROUND_NUMBER}, {cst.ROUND_TYPE}, 
                   {cst.IS_FINAL_ROUND}, {cst.ROUND_STATUS}
            FROM {cst.TOURNAMENT_ROUNDS_TABLE}
            WHERE {cst.ROUND_ID} = $1
        """
        result = await connection.fetchrow(query, round_id)
        if result:
            return TournamentRoundData(
                round_id=result[cst.ROUND_ID],
                tournament_id=result[cst.TOURNAMENT_ID],
                round_number=result[cst.ROUND_NUMBER],
                round_type=result[cst.ROUND_TYPE],
                is_final_round=result[cst.IS_FINAL_ROUND],
                status=result[cst.ROUND_STATUS],
            )
        return None


async def get_tournament_rounds(tournament_id: str, psql_db: PSQLDB) -> list[TournamentRoundData]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.ROUND_ID}, {cst.TOURNAMENT_ID}, {cst.ROUND_NUMBER}, {cst.ROUND_TYPE}, 
                   {cst.IS_FINAL_ROUND}, {cst.ROUND_STATUS}
            FROM {cst.TOURNAMENT_ROUNDS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1
            ORDER BY {cst.ROUND_NUMBER} ASC
        """
        results = await connection.fetch(query, tournament_id)
        return [
            TournamentRoundData(
                round_id=row[cst.ROUND_ID],
                tournament_id=row[cst.TOURNAMENT_ID],
                round_number=row[cst.ROUND_NUMBER],
                round_type=row[cst.ROUND_TYPE],
                is_final_round=row[cst.IS_FINAL_ROUND],
                status=row[cst.ROUND_STATUS],
            )
            for row in results
        ]


async def get_tournament_rounds_with_status(status: str, psql_db: PSQLDB) -> list[TournamentRoundData]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.ROUND_ID}, {cst.TOURNAMENT_ID}, {cst.ROUND_NUMBER}, {cst.ROUND_TYPE}, 
                   {cst.IS_FINAL_ROUND}, {cst.ROUND_STATUS}
            FROM {cst.TOURNAMENT_ROUNDS_TABLE}
            WHERE {cst.ROUND_STATUS} = $1
            ORDER BY {cst.TOURNAMENT_ID}, {cst.ROUND_NUMBER} ASC
        """
        results = await connection.fetch(query, status)
        return [
            TournamentRoundData(
                round_id=row[cst.ROUND_ID],
                tournament_id=row[cst.TOURNAMENT_ID],
                round_number=row[cst.ROUND_NUMBER],
                round_type=row[cst.ROUND_TYPE],
                is_final_round=row[cst.IS_FINAL_ROUND],
                status=row[cst.ROUND_STATUS],
            )
            for row in results
        ]


async def get_tournament_tasks(round_id: str, psql_db: PSQLDB) -> list[TournamentTask]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.ROUND_ID}, {cst.TASK_ID}, {cst.GROUP_ID}, {cst.PAIR_ID}
            FROM {cst.TOURNAMENT_TASKS_TABLE}
            WHERE {cst.ROUND_ID} = $1
        """
        results = await connection.fetch(query, round_id)
        return [
            TournamentTask(
                tournament_id=row[cst.TOURNAMENT_ID],
                round_id=row[cst.ROUND_ID],
                task_id=row[cst.TASK_ID],
                group_id=row[cst.GROUP_ID],
                pair_id=row[cst.PAIR_ID],
            )
            for row in results
        ]


async def get_tournament_pairs(round_id: str, psql_db: PSQLDB) -> list[TournamentPairData]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.PAIR_ID}, {cst.ROUND_ID}, {cst.HOTKEY1}, {cst.HOTKEY2}
            FROM {cst.TOURNAMENT_PAIRS_TABLE}
            WHERE {cst.ROUND_ID} = $1
        """
        results = await connection.fetch(query, round_id)
        return [
            TournamentPairData(
                pair_id=row[cst.PAIR_ID],
                round_id=row[cst.ROUND_ID],
                hotkey1=row[cst.HOTKEY1],
                hotkey2=row[cst.HOTKEY2],
            )
            for row in results
        ]


async def get_tournament_groups(round_id: str, psql_db: PSQLDB) -> list[TournamentGroupData]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.GROUP_ID}, {cst.ROUND_ID}
            FROM {cst.TOURNAMENT_GROUPS_TABLE}
            WHERE {cst.ROUND_ID} = $1
        """
        results = await connection.fetch(query, round_id)
        return [TournamentGroupData(group_id=row[cst.GROUP_ID], round_id=row[cst.ROUND_ID]) for row in results]


async def get_tournament_group_members(group_id: str, psql_db: PSQLDB) -> list[TournamentParticipant]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.GROUP_ID}, {cst.HOTKEY}
            FROM {cst.TOURNAMENT_GROUP_MEMBERS_TABLE}
            WHERE {cst.GROUP_ID} = $1
        """
        results = await connection.fetch(query, group_id)
        # TODO: join with full participant table
        return [
            TournamentParticipant(
                tournament_id="",
                hotkey=row[cst.HOTKEY],
            )
            for row in results
        ]


async def update_round_status(round_id: str, status: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENT_ROUNDS_TABLE} 
            SET {cst.ROUND_STATUS} = $2
            WHERE {cst.ROUND_ID} = $1
        """
        await connection.execute(query, round_id, status)
        logger.info(f"Updated round {round_id} status to {status}")


async def update_tournament_status(tournament_id: str, status: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENTS_TABLE} 
            SET {cst.TOURNAMENT_STATUS} = $2, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TOURNAMENT_ID} = $1
        """
        await connection.execute(query, tournament_id, status)
        logger.info(f"Updated tournament {tournament_id} status to {status}")


async def cancel_all_active_tournaments(psql_db: PSQLDB) -> int:
    """Set all active or pending tournaments to cancelled status. Returns number of tournaments cancelled."""
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENTS_TABLE} 
            SET {cst.TOURNAMENT_STATUS} = 'cancelled', {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TOURNAMENT_STATUS} IN ('pending', 'active')
        """
        result = await connection.execute(query)
        cancelled_count = result.split()[-1] if result else "0"
        logger.info(f"Cancelled {cancelled_count} active/pending tournaments")
        return int(cancelled_count)


async def update_tournament_base_winner_hotkey(tournament_id: str, base_winner_hotkey: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENTS_TABLE} 
            SET {cst.BASE_WINNER_HOTKEY} = $2, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TOURNAMENT_ID} = $1
        """
        await connection.execute(query, tournament_id, base_winner_hotkey)
        logger.info(f"Updated tournament {tournament_id} base winner hotkey to {base_winner_hotkey}")


async def update_tournament_winner_hotkey(tournament_id: str, winner_hotkey: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENTS_TABLE} 
            SET {cst.WINNER_HOTKEY} = $2, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TOURNAMENT_ID} = $1
        """
        await connection.execute(query, tournament_id, winner_hotkey)
        logger.info(f"Updated tournament {tournament_id} winner hotkey to {winner_hotkey}")


async def get_tournaments_with_status(status: TournamentStatus, psql_db: PSQLDB) -> list[TournamentData]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS}, {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_STATUS} = $1
            ORDER BY {cst.CREATED_AT} DESC
        """
        results = await connection.fetch(query, status.value)
        return [
            TournamentData(
                tournament_id=row[cst.TOURNAMENT_ID],
                tournament_type=row[cst.TOURNAMENT_TYPE],
                status=row[cst.TOURNAMENT_STATUS],
                base_winner_hotkey=row[cst.BASE_WINNER_HOTKEY],
                winner_hotkey=row[cst.WINNER_HOTKEY],
            )
            for row in results
        ]


async def is_task_in_tournament(task_id: str, psql_db: PSQLDB) -> bool:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT EXISTS(
                SELECT 1 FROM {cst.TOURNAMENT_TASKS_TABLE}
                WHERE {cst.TASK_ID} = $1
            )
        """
        result = await connection.fetchrow(query, task_id)
        return result[0]


async def are_tasks_in_tournament(task_ids: list[str], psql_db: PSQLDB) -> list[bool]:
    """Batched version that checks which tasks are in the tournament"""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TASK_ID} FROM {cst.TOURNAMENT_TASKS_TABLE}
            WHERE {cst.TASK_ID} = ANY($1)
        """
        result = await connection.fetch(query, task_ids)
        existing_task_ids = {row[cst.TASK_ID] for row in result}
        return [task_id in existing_task_ids for task_id in task_ids]


async def get_miners_for_tournament(task_id: str, psql_db: PSQLDB) -> list[Node]:
    async with await psql_db.connection() as connection:
        query = f"""
            -- Get miners for group rounds
            SELECT DISTINCT n.*
            FROM {cst.TOURNAMENT_TASKS_TABLE} tt
            JOIN {cst.TOURNAMENT_GROUPS_TABLE} tg ON tt.{cst.GROUP_ID} = tg.{cst.GROUP_ID}
            JOIN {cst.TOURNAMENT_GROUP_MEMBERS_TABLE} tgm ON tg.{cst.GROUP_ID} = tgm.{cst.GROUP_ID}
            JOIN {cst.NODES_TABLE} n ON tgm.{cst.HOTKEY} = n.{cst.HOTKEY}
            WHERE tt.{cst.TASK_ID} = $1 AND tt.{cst.GROUP_ID} IS NOT NULL

            UNION

            -- Get miners for knockout rounds
            SELECT DISTINCT n.*
            FROM {cst.TOURNAMENT_TASKS_TABLE} tt
            JOIN {cst.TOURNAMENT_PAIRS_TABLE} tp ON tt.{cst.PAIR_ID} = tp.{cst.PAIR_ID}
            JOIN {cst.NODES_TABLE} n ON tp.{cst.HOTKEY1} = n.{cst.HOTKEY}
            WHERE tt.{cst.TASK_ID} = $1 AND tt.{cst.PAIR_ID} IS NOT NULL

            UNION

            SELECT DISTINCT n.*
            FROM {cst.TOURNAMENT_TASKS_TABLE} tt
            JOIN {cst.TOURNAMENT_PAIRS_TABLE} tp ON tt.{cst.PAIR_ID} = tp.{cst.PAIR_ID}
            JOIN {cst.NODES_TABLE} n ON tp.{cst.HOTKEY2} = n.{cst.HOTKEY}
            WHERE tt.{cst.TASK_ID} = $1 AND tt.{cst.PAIR_ID} IS NOT NULL
        """
        result = await connection.fetch(query, task_id)
        return [Node(**dict(row)) for row in result]


async def update_tournament_participant_training_repo(
    tournament_id: str, hotkey: str, training_repo: str, training_commit_hash: str, psql_db: PSQLDB
):
    """Update the training repo information for a tournament participant."""
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENT_PARTICIPANTS_TABLE}
            SET {cst.TRAINING_REPO} = $1, {cst.TRAINING_COMMIT_HASH} = $2
            WHERE {cst.TOURNAMENT_ID} = $3 AND {cst.HOTKEY} = $4
        """
        await connection.execute(query, training_repo, training_commit_hash, tournament_id, hotkey)
        logger.info(f"Updated training repo for participant {hotkey} in tournament {tournament_id}")


async def get_tournament_participant(tournament_id: str, hotkey: str, psql_db: PSQLDB) -> TournamentParticipant | None:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.HOTKEY}, {cst.ELIMINATED_IN_ROUND_ID}, 
                   {cst.FINAL_POSITION}, {cst.TRAINING_REPO}, {cst.TRAINING_COMMIT_HASH}, {cst.STAKE_REQUIRED}
            FROM {cst.TOURNAMENT_PARTICIPANTS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1 AND {cst.HOTKEY} = $2
        """
        result = await connection.fetchrow(query, tournament_id, hotkey)
        if result:
            return TournamentParticipant(
                tournament_id=result[cst.TOURNAMENT_ID],
                hotkey=result[cst.HOTKEY],
                eliminated_in_round_id=result[cst.ELIMINATED_IN_ROUND_ID],
                final_position=result[cst.FINAL_POSITION],
                training_repo=result[cst.TRAINING_REPO],
                training_commit_hash=result[cst.TRAINING_COMMIT_HASH],
                stake_required=result[cst.STAKE_REQUIRED],
            )
        return None


async def get_tournament_participants(tournament_id: str, psql_db: PSQLDB) -> list[TournamentParticipant]:
    """Get all participants for a tournament with their training repo information."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.HOTKEY}, {cst.ELIMINATED_IN_ROUND_ID}, 
                   {cst.FINAL_POSITION}, {cst.TRAINING_REPO}, {cst.TRAINING_COMMIT_HASH}, {cst.STAKE_REQUIRED}
            FROM {cst.TOURNAMENT_PARTICIPANTS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1
        """
        results = await connection.fetch(query, tournament_id)
        return [
            TournamentParticipant(
                tournament_id=row[cst.TOURNAMENT_ID],
                hotkey=row[cst.HOTKEY],
                eliminated_in_round_id=row[cst.ELIMINATED_IN_ROUND_ID],
                final_position=row[cst.FINAL_POSITION],
                training_repo=row[cst.TRAINING_REPO],
                training_commit_hash=row[cst.TRAINING_COMMIT_HASH],
                stake_required=row[cst.STAKE_REQUIRED],
            )
            for row in results
        ]


async def add_trainer_gpus(trainer_ip: str, gpu_infos: list[GPUInfo], psql_db: PSQLDB):
    """Add or update GPU information for a trainer"""
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            # First, remove existing entries for this trainer
            delete_query = f"""
                DELETE FROM {cst.TRAINERS_GPUS_TABLE}
                WHERE {cst.TRAINER_IP} = $1
            """
            await connection.execute(delete_query, trainer_ip)

            # Then insert new GPU information
            insert_query = f"""
                INSERT INTO {cst.TRAINERS_GPUS_TABLE}
                ({cst.TRAINER_IP}, {cst.GPU_ID}, {cst.GPU_TYPE}, {cst.VRAM_GB}, {cst.USED_UNTIL})
                VALUES ($1, $2, $3, $4, $5)
            """

            for gpu_info in gpu_infos:
                used_until = None
                if not gpu_info.available:
                    used_until = datetime.now(timezone.utc) + timedelta(hours=48)

                await connection.execute(
                    insert_query, trainer_ip, gpu_info.gpu_id, gpu_info.gpu_type, gpu_info.vram_gb, used_until
                )

            logger.info(f"Added {len(gpu_infos)} GPUs for trainer {trainer_ip}")


async def remove_trainer(trainer_ip: str, psql_db: PSQLDB):
    """Remove a trainer and all its GPUs from the database"""
    async with await psql_db.connection() as connection:
        query = f"""
            DELETE FROM {cst.TRAINERS_GPUS_TABLE}
            WHERE {cst.TRAINER_IP} = $1
        """
        await connection.execute(query, trainer_ip)
        logger.info(f"Removed trainer {trainer_ip} from the database")


async def get_trainers(psql_db: PSQLDB) -> list[TrainerInfo]:
    """Get all trainers and their GPU information from the database"""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TRAINER_IP}, {cst.GPU_ID}, {cst.GPU_TYPE}, {cst.VRAM_GB}, {cst.USED_UNTIL}
            FROM {cst.TRAINERS_GPUS_TABLE}
            ORDER BY {cst.TRAINER_IP}, {cst.GPU_ID}
        """
        results = await connection.fetch(query)

        # Group by trainer IP
        trainers = {}
        for row in results:
            trainer_ip = row[cst.TRAINER_IP]
            if trainer_ip not in trainers:
                trainers[trainer_ip] = TrainerInfo(trainer_ip=trainer_ip, gpus=[])

            # Determine availability based on used_until
            used_until = row[cst.USED_UNTIL]
            available = used_until is None or used_until < datetime.now(timezone.utc)

            trainers[trainer_ip].gpus.append(
                GPUInfo(
                    gpu_id=row[cst.GPU_ID],
                    gpu_type=row[cst.GPU_TYPE],
                    vram_gb=row[cst.VRAM_GB],
                    available=available,
                    used_until=used_until,
                )
            )

        return list(trainers.values())


async def add_tournament_task_hotkey_pairs_for_training(task_hotkey_triples: list[tuple[str, str, datetime]], psql_db: PSQLDB):
    """
    Add task-hotkey pairs to the tournament_task_hotkey_trainings table using batch insert.
    Each task-hotkey pair defines a training task that we'll send to a trainer later.
    """
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            if not task_hotkey_triples:
                logger.info("No task-hotkey triples to insert")
                return

            query = f"""
                INSERT INTO {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
                ({cst.TASK_ID}, {cst.HOTKEY}, {cst.CREATED_AT})
                SELECT * FROM unnest($1::uuid[], $2::text[], $3::timestamptz[])
                ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}) DO NOTHING
            """

            task_ids = [triple[0] for triple in task_hotkey_triples]
            hotkeys = [triple[1] for triple in task_hotkey_triples]
            timestamps = [triple[2] for triple in task_hotkey_triples]

            await connection.execute(query, task_ids, hotkeys, timestamps)

            logger.info(f"Added {len(task_hotkey_triples)} task-hotkey training triples in batch")


async def get_tournament_training_tasks(psql_db: PSQLDB, status: TrainingStatus) -> list[TournamentTaskTraining]:
    """
    Fetch tournament tasks with specific training status in reverse chronological order.

    Args:
        psql_db: Database connection
        status: Training status to filter by

    Returns:
        List of TournamentTaskTraining objects ordered by creation time (newest first)
    """
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TASK_ID}, {cst.HOTKEY}, {cst.TRAINING_STATUS}, {cst.N_TRAINING_ATTEMPTS},
                   {cst.CREATED_AT}, {cst.UPDATED_AT}
            FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
            WHERE {cst.TRAINING_STATUS} = $1
            ORDER BY {cst.CREATED_AT} DESC
        """
        results = await connection.fetch(query, status)

        if not results:
            return []

        unique_task_ids = list({row[cst.TASK_ID] for row in results})
        tasks = await task_sql.get_tasks_by_ids(unique_task_ids, psql_db, connection)

        # Create a mapping for quick lookup
        tasks_dict = {task.task_id: task for task in tasks}

        tournament_tasks = []
        missing_tasks = []

        for row in results:
            task = tasks_dict.get(row[cst.TASK_ID])
            if task:
                tournament_tasks.append(
                    TournamentTaskTraining(
                        task=task,
                        hotkey=row[cst.HOTKEY],
                        training_status=row[cst.TRAINING_STATUS],
                        n_training_attempts=row[cst.N_TRAINING_ATTEMPTS],
                        created_at=row[cst.CREATED_AT],
                        updated_at=row[cst.UPDATED_AT],
                    )
                )

            else:
                missing_tasks.append(row[cst.TASK_ID])

        if missing_tasks:
            logger.warning(f"Tasks not found in batch load: {missing_tasks}")

        return tournament_tasks


async def update_tournament_task_training_status(task_id: str, hotkey: str, status: TrainingStatus, psql_db: PSQLDB):
    """Update the training status of a specific task-hotkey pair"""
    async with await psql_db.connection() as connection:
        increment_clause = (
            f", {cst.N_TRAINING_ATTEMPTS} = {cst.N_TRAINING_ATTEMPTS} + 1" if status == TrainingStatus.TRAINING else ""
        )

        query = f"""
            UPDATE {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
            SET {cst.TRAINING_STATUS} = $3{increment_clause}, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2
        """

        await connection.execute(query, task_id, hotkey, status)
        logger.info(f"Marked task-hotkey pair ({task_id}, {hotkey}) as {status}")


async def get_training_attempts(task_id: str, hotkey: str, psql_db: PSQLDB) -> int:
    """Get the number of training attempts for a task-hotkey pair"""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.N_TRAINING_ATTEMPTS}
            FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
            WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2
        """
        result = await connection.fetchrow(query, task_id, hotkey)
        return result[cst.N_TRAINING_ATTEMPTS] if result else 0


async def get_tournament_training_repo_and_commit(hotkey: str, psql_db: PSQLDB) -> tuple[str | None, str | None]:
    """Get the training_repo and training_commit_hash for a hotkey from tournament_participants table"""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TRAINING_REPO}, {cst.TRAINING_COMMIT_HASH}
            FROM {cst.TOURNAMENT_PARTICIPANTS_TABLE}
            WHERE {cst.HOTKEY} = $1
            ORDER BY {cst.CREATED_AT} DESC
            LIMIT 1
        """
        result = await connection.fetchrow(query, hotkey)
        if result:
            return result[cst.TRAINING_REPO], result[cst.TRAINING_COMMIT_HASH]
        return None, None


async def get_tournament_training_stats(psql_db: PSQLDB) -> dict:
    """Get statistics about tournament training status"""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT
                {cst.TRAINING_STATUS},
                COUNT(*) as count,
                AVG({cst.N_TRAINING_ATTEMPTS}) as avg_attempts,
                MAX({cst.N_TRAINING_ATTEMPTS}) as max_attempts
            FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
            GROUP BY {cst.TRAINING_STATUS}
        """
        results = await connection.fetch(query)

        stats = {"total_pairs": 0, "pending": 0, "success": 0, "failure": 0, "avg_attempts": 0, "max_attempts": 0}

        for row in results:
            status = row[cst.TRAINING_STATUS]
            count = row["count"]
            avg_attempts = row["avg_attempts"] or 0
            max_attempts = row["max_attempts"] or 0

            stats["total_pairs"] += count
            stats[status] = count
            stats["avg_attempts"] = max(stats["avg_attempts"], avg_attempts)
            stats["max_attempts"] = max(stats["max_attempts"], max_attempts)

        return stats


async def update_gpu_availability(trainer_ip: str, gpu_ids: list[int], hours_to_complete: int, psql_db: PSQLDB):
    """Update GPU availability by setting used_until based on hours_to_complete"""
    async with await psql_db.connection() as connection:
        if hours_to_complete == 0:
            # Reset GPU availability by setting used_until to NULL
            query = f"""
                UPDATE {cst.TRAINERS_GPUS_TABLE}
                SET {cst.USED_UNTIL} = NULL, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
                WHERE {cst.TRAINER_IP} = $1 AND {cst.GPU_ID} = ANY($2)
            """
        else:
            # Set GPU as used for specified hours
            used_until = f"CURRENT_TIMESTAMP + INTERVAL '{hours_to_complete} hours'"
            query = f"""
                UPDATE {cst.TRAINERS_GPUS_TABLE}
                SET {cst.USED_UNTIL} = {used_until}, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
                WHERE {cst.TRAINER_IP} = $1 AND {cst.GPU_ID} = ANY($2)
            """

        await connection.execute(query, trainer_ip, gpu_ids)
        action = "reset" if hours_to_complete == 0 else f"used for {hours_to_complete} hours"
        logger.info(f"Updated GPU availability for trainer {trainer_ip}, GPUs {gpu_ids} - {action}")


async def get_tasks_with_all_training_completed(psql_db: PSQLDB) -> list[str]:
    """Get task IDs where all training tasks (task_id, hotkey pairs) have completed (success or failure) and task is in training status"""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT DISTINCT t1.{cst.TASK_ID}
            FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE} t1
            JOIN {cst.TASKS_TABLE} {cst.TASKS_TABLE} ON t1.{cst.TASK_ID} = {cst.TASKS_TABLE}.{cst.TASK_ID}
            WHERE {cst.TASKS_TABLE}.{cst.STATUS} = 'training'
            AND {cst.TASKS_TABLE}.{cst.CREATED_AT} >= NOW() - INTERVAL '1 month'
            AND NOT EXISTS (
                SELECT 1
                FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE} t2
                WHERE t2.{cst.TASK_ID} = t1.{cst.TASK_ID}
                AND t2.{cst.TRAINING_STATUS} NOT IN ('success', 'failure')
            )
        """
        results = await connection.fetch(query)
        return [row[cst.TASK_ID] for row in results]


async def eliminate_tournament_participants(tournament_id: str, round_id: str, hotkeys: list[str], psql_db: PSQLDB):
    """Mark tournament participants as eliminated in the specified round."""
    if not hotkeys:
        return
    
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENT_PARTICIPANTS_TABLE}
            SET {cst.ELIMINATED_IN_ROUND_ID} = $2
            WHERE {cst.TOURNAMENT_ID} = $1 AND {cst.HOTKEY} = ANY($3)
            AND {cst.ELIMINATED_IN_ROUND_ID} IS NULL
        """
        await connection.execute(query, tournament_id, round_id, hotkeys)
        logger.info(f"Eliminated {len(hotkeys)} participants from tournament {tournament_id} in round {round_id}")


async def get_participants_with_insufficient_stake(tournament_id: str, psql_db: PSQLDB) -> list[str]:
    """Get list of participant hotkeys who no longer meet their stake requirement."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT tp.{cst.HOTKEY}
            FROM {cst.TOURNAMENT_PARTICIPANTS_TABLE} tp
            JOIN {cst.NODES_TABLE} n ON tp.{cst.HOTKEY} = n.{cst.HOTKEY}
            WHERE tp.{cst.TOURNAMENT_ID} = $1 
            AND tp.{cst.ELIMINATED_IN_ROUND_ID} IS NULL
            AND (n.alpha_stake < COALESCE(tp.{cst.STAKE_REQUIRED}, 0))
        """
        results = await connection.fetch(query, tournament_id)
        return [row[cst.HOTKEY] for row in results]


def calculate_boosted_stake(actual_stake: float, completed_entries: int) -> float:
    """Calculate stake with repeat participant bonus."""
    boost_percentage = min(completed_entries * validator.core.constants.TOURNAMENT_REPEAT_BOOST_PERCENTAGE, 
                          validator.core.constants.TOURNAMENT_MAX_REPEAT_BOOST_PERCENTAGE)
    return actual_stake * (1 + boost_percentage / 100)


async def count_completed_tournament_entries(hotkey: str, psql_db: PSQLDB) -> int:
    """Count how many completed tournaments a hotkey has participated in."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT COUNT(DISTINCT tp.{cst.TOURNAMENT_ID}) 
            FROM {cst.TOURNAMENT_PARTICIPANTS_TABLE} tp
            JOIN {cst.TOURNAMENTS_TABLE} t ON tp.{cst.TOURNAMENT_ID} = t.{cst.TOURNAMENT_ID}
            WHERE tp.{cst.HOTKEY} = $1 
            AND t.{cst.TOURNAMENT_STATUS} = 'completed'
        """
        result = await connection.fetchrow(query, hotkey)
        return result[0] if result else 0


async def get_active_tournament_participants(psql_db: PSQLDB) -> list[str]:
    """Get hotkeys of all active tournament participants for participation weights."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT DISTINCT tp.{cst.HOTKEY}
            FROM {cst.TOURNAMENT_PARTICIPANTS_TABLE} tp
            JOIN {cst.TOURNAMENTS_TABLE} t ON tp.{cst.TOURNAMENT_ID} = t.{cst.TOURNAMENT_ID}
            WHERE t.{cst.TOURNAMENT_STATUS} = 'active'
            AND tp.{cst.ELIMINATED_IN_ROUND_ID} IS NULL
        """
        results = await connection.fetch(query)
        hotkeys = [row[cst.HOTKEY] for row in results]
        logger.info(f"Found {len(hotkeys)} active tournament participants for participation weights")
        return hotkeys


async def get_tournament_full_results(tournament_id: str, psql_db: PSQLDB) -> TournamentResults:
    rounds = await get_tournament_rounds(tournament_id, psql_db)
    
    round_results = []
    
    for round_data in rounds:
        tasks = await get_tournament_tasks(round_data.round_id, psql_db)
        
        task_scores = []
        task_ids = [task.task_id for task in tasks]
        if task_ids:
            task_winners = await get_task_winners(task_ids, psql_db)
            
            for task in tasks:
                participant_scores = await get_all_scores_and_losses_for_task(task.task_id, psql_db)
                task_score = TournamentTaskScore(
                    task_id=str(task.task_id),
                    group_id=task.group_id,
                    pair_id=task.pair_id,
                    winner=task_winners.get(str(task.task_id)),
                    participant_scores=participant_scores
                )
                task_scores.append(task_score)
        
        round_result = TournamentRoundResult(
            round_id=round_data.round_id,
            round_number=round_data.round_number,
            round_type=round_data.round_type,
            is_final_round=round_data.is_final_round,
            tasks=task_scores
        )
        round_results.append(round_result)
    
    return TournamentResults(
        tournament_id=tournament_id,
        rounds=round_results
    )


async def get_tournament_with_created_at(tournament_id: str, psql_db: PSQLDB) -> tuple[TournamentData | None, datetime | None]:
    """Get a tournament with its created_at timestamp."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS}, 
                   {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}, {cst.CREATED_AT}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1
        """
        result = await connection.fetchrow(query, tournament_id)
        if result:
            tournament = TournamentData(
                tournament_id=result[cst.TOURNAMENT_ID],
                tournament_type=result[cst.TOURNAMENT_TYPE],
                status=result[cst.TOURNAMENT_STATUS],
                base_winner_hotkey=result[cst.BASE_WINNER_HOTKEY],
                winner_hotkey=result[cst.WINNER_HOTKEY],
            )
            created_at = result[cst.CREATED_AT]
            return tournament, created_at
        return None, None


async def get_active_tournament(psql_db: PSQLDB, tournament_type: TournamentType) -> TournamentData | None:
    """Get the active tournament for a given type (if any)."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS}, 
                   {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_TYPE} = $1 AND {cst.TOURNAMENT_STATUS} = 'active'
            ORDER BY {cst.CREATED_AT} DESC
            LIMIT 1
        """
        result = await connection.fetchrow(query, tournament_type.value)
        if result:
            return TournamentData(
                tournament_id=result[cst.TOURNAMENT_ID],
                tournament_type=result[cst.TOURNAMENT_TYPE],
                status=result[cst.TOURNAMENT_STATUS],
                base_winner_hotkey=result[cst.BASE_WINNER_HOTKEY],
                winner_hotkey=result[cst.WINNER_HOTKEY],
            )
        return None


async def get_latest_tournament_with_created_at(psql_db: PSQLDB, tournament_type: TournamentType) -> tuple[TournamentData | None, datetime | None]:
    """Get the latest tournament (active or completed) with its created_at timestamp."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS}, 
                   {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}, {cst.CREATED_AT}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_TYPE} = $1
            ORDER BY {cst.CREATED_AT} DESC
            LIMIT 1
        """
        result = await connection.fetchrow(query, tournament_type.value)
        if result:
            tournament = TournamentData(
                tournament_id=result[cst.TOURNAMENT_ID],
                tournament_type=result[cst.TOURNAMENT_TYPE],
                status=result[cst.TOURNAMENT_STATUS],
                base_winner_hotkey=result[cst.BASE_WINNER_HOTKEY],
                winner_hotkey=result[cst.WINNER_HOTKEY],
            )
            created_at = result[cst.CREATED_AT]
            return tournament, created_at
        return None, None


async def get_tournament_id_by_task_id(task_id: str, psql_db: PSQLDB) -> str | None:
    """
    Fetch the tournament_id for a given task_id from the tournament_tasks table.

    Args:
        task_id: The task ID to look up
        psql_db: The database connection

    Returns:
        The tournament_id as a string, or None if not found
    """
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}
            FROM {cst.TOURNAMENT_TASKS_TABLE}
            WHERE {cst.TASK_ID} = $1
        """
        result = await connection.fetchrow(query, task_id)
        if result:
            return result[cst.TOURNAMENT_ID]
        return None


async def is_synced_task(task_id: str, psql_db: PSQLDB) -> bool:
    """Check if a task is a synced task (should not be set to failure during round reversal)."""
    async with await psql_db.connection() as connection:
        query = """
            SELECT 1 FROM boss_round_synced_tasks 
            WHERE tournament_task_id = $1
        """
        result = await connection.fetchval(query, task_id)
        return result is not None
