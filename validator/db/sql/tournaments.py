from datetime import datetime

from fiber.chain.models import Node

import validator.db.constants as cst
from core.models.tournament_models import GroupRound
from core.models.tournament_models import KnockoutRound
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentParticipant
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentTask
from core.models.utility_models import GPUInfo
from core.models.utility_models import TrainerInfo
from validator.db.database import PSQLDB
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def create_tournament(tournament: TournamentData, psql_db: PSQLDB) -> str:
    async with await psql_db.connection() as connection:
        query = f"""
            INSERT INTO {cst.TOURNAMENTS_TABLE}
            ({cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS}, {cst.CREATED_AT}, {cst.UPDATED_AT})
            VALUES ($1, $2, $3, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING {cst.TOURNAMENT_ID}
        """
        result = await connection.fetchrow(
            query, tournament.tournament_id, tournament.tournament_type, tournament.status
        )
        logger.info(f"Created tournament: {tournament.tournament_id}")
        return result[cst.TOURNAMENT_ID]


async def create_tournament_round(round_data: TournamentRoundData, psql_db: PSQLDB) -> str:
    async with await psql_db.connection() as connection:
        query = f"""
            INSERT INTO {cst.TOURNAMENT_ROUNDS_TABLE}
            ({cst.ROUND_ID}, {cst.TOURNAMENT_ID}, {cst.ROUND_NUMBER}, {cst.ROUND_TYPE},
             {cst.IS_FINAL_ROUND}, {cst.ROUND_STATUS}, {cst.CREATED_AT})
            VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
            RETURNING {cst.ROUND_ID}
        """
        result = await connection.fetchrow(
            query, round_data.round_id, round_data.tournament_id, round_data.round_number,
            round_data.round_type, round_data.is_final_round, round_data.status
        )
        logger.info(f"Created tournament round: {round_data.round_id}")
        return result[cst.ROUND_ID]


async def add_tournament_participants(participants: list[TournamentParticipant], psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            query = f"""
                INSERT INTO {cst.TOURNAMENT_PARTICIPANTS_TABLE}
                ({cst.TOURNAMENT_ID}, {cst.HOTKEY}, {cst.CREATED_AT})
                VALUES ($1, $2, CURRENT_TIMESTAMP)
                ON CONFLICT ({cst.TOURNAMENT_ID}, {cst.HOTKEY}) DO NOTHING
            """
            for participant in participants:
                await connection.execute(query, participant.tournament_id, participant.hotkey)
            logger.info(f"Added {len(participants)} participants to tournament")


async def create_tournament_groups_with_members(
    round_id: str, round_structure: GroupRound, psql_db: PSQLDB
) -> list[str]:
    group_ids = []
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            for i, group in enumerate(round_structure.groups):
                group_id = f"{round_id}_group_{i+1:03d}"

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


async def create_tournament_pairs(
    round_id: str, round_structure: KnockoutRound, psql_db: PSQLDB
) -> list[str]:
    pair_ids = []
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            query = f"""
                INSERT INTO {cst.TOURNAMENT_PAIRS_TABLE}
                ({cst.PAIR_ID}, {cst.ROUND_ID}, {cst.HOTKEY1}, {cst.HOTKEY2}, {cst.CREATED_AT})
                VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                RETURNING {cst.PAIR_ID}
            """
            for i, (hotkey1, hotkey2) in enumerate(round_structure.pairs):
                pair_id = f"{round_id}_pair_{i+1:03d}"
                await connection.execute(query, pair_id, round_id, hotkey1, hotkey2)
                pair_ids.append(pair_id)

            logger.info(f"Created {len(pair_ids)} pairs for round {round_id}")
    return pair_ids


async def add_tournament_tasks(tasks: list[TournamentTask], psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            query = f"""
                INSERT INTO {cst.TOURNAMENT_TASKS_TABLE}
                ({cst.TOURNAMENT_ID}, {cst.ROUND_ID}, {cst.TASK_ID}, {cst.GROUP_ID}, {cst.PAIR_ID}, {cst.CREATED_AT})
                VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)
            """
            for task in tasks:
                await connection.execute(
                    query, task.tournament_id, task.round_id, task.task_id,
                    task.group_id, task.pair_id
                )
            logger.info(f"Added {len(tasks)} tasks to tournament")


async def get_tournament(tournament_id: str, psql_db: PSQLDB) -> TournamentData | None:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS}, {cst.CURRENT_ROUND_ID}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1
        """
        result = await connection.fetchrow(query, tournament_id)
        if result:
            return TournamentData(
                tournament_id=result[cst.TOURNAMENT_ID],
                tournament_type=result[cst.TOURNAMENT_TYPE],
                status=result[cst.TOURNAMENT_STATUS],
                current_round_id=result[cst.CURRENT_ROUND_ID]
            )
        return None


async def get_tournament_participants(tournament_id: str, psql_db: PSQLDB) -> list[str]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.HOTKEY} FROM {cst.TOURNAMENT_PARTICIPANTS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1
            ORDER BY {cst.CREATED_AT}
        """
        results = await connection.fetch(query, tournament_id)
        return [row[cst.HOTKEY] for row in results]


async def update_tournament_status(tournament_id: str, status: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENTS_TABLE}
            SET {cst.TOURNAMENT_STATUS} = $2, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TOURNAMENT_ID} = $1
        """
        await connection.execute(query, tournament_id, status)
        logger.info(f"Updated tournament {tournament_id} status to {status}")


async def update_round_status(round_id: str, status: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENT_ROUNDS_TABLE}
            SET {cst.ROUND_STATUS} = $2, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.ROUND_ID} = $1
        """
        await connection.execute(query, round_id, status)
        logger.info(f"Updated round {round_id} status to {status}")


async def set_pair_winner(pair_id: str, winner_hotkey: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENT_PAIRS_TABLE}
            SET {cst.WINNER_HOTKEY} = $2, {cst.COMPLETED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.PAIR_ID} = $1
        """
        await connection.execute(query, pair_id, winner_hotkey)
        logger.info(f"Set winner {winner_hotkey} for pair {pair_id}")


async def eliminate_participant(
    tournament_id: str, hotkey: str, round_id: str, psql_db: PSQLDB
):
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENT_PARTICIPANTS_TABLE}
            SET {cst.ELIMINATED_IN_ROUND_ID} = $3
            WHERE {cst.TOURNAMENT_ID} = $1 AND {cst.HOTKEY} = $2
        """
        await connection.execute(query, tournament_id, hotkey, round_id)
        logger.info(f"Eliminated participant {hotkey} in round {round_id}")


async def get_active_tournaments(psql_db: PSQLDB) -> list[TournamentData]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS}, {cst.CURRENT_ROUND_ID}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_STATUS} IN ('pending', 'active')
            ORDER BY {cst.CREATED_AT} DESC
        """
        results = await connection.fetch(query)
        return [
            TournamentData(
                tournament_id=row[cst.TOURNAMENT_ID],
                tournament_type=row[cst.TOURNAMENT_TYPE],
                status=row[cst.TOURNAMENT_STATUS],
                current_round_id=row[cst.CURRENT_ROUND_ID]
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
                    used_until = "CURRENT_TIMESTAMP + INTERVAL '48 hours'"

                await connection.execute(
                    insert_query,
                    trainer_ip,
                    gpu_info.gpu_id,
                    gpu_info.gpu_type,
                    gpu_info.vram_gb,
                    used_until
                )

            logger.info(f"Added {len(gpu_infos)} GPUs for trainer {trainer_ip}")


async def remove_trainer(trainer_ip: str, psql_db: PSQLDB):
    """Remove a trainer and all its GPUs from the database"""
    async with await psql_db.connection() as connection:
        query = f"""
            DELETE FROM {cst.TRAINERS_GPUS_TABLE}
            WHERE {cst.TRAINER_IP} = $1
        """
        result = await connection.execute(query, trainer_ip)
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
                trainers[trainer_ip] = TrainerInfo(
                    trainer_ip=trainer_ip,
                    gpus=[]
                )

            # Determine availability based on used_until
            used_until = row[cst.USED_UNTIL]
            available = used_until is None or used_until < datetime.utcnow()

            trainers[trainer_ip].gpus.append(GPUInfo(
                gpu_id=row[cst.GPU_ID],
                gpu_type=row[cst.GPU_TYPE],
                vram_gb=row[cst.VRAM_GB],
                available=available,
                used_until=used_until
            ))

        return list(trainers.values())
