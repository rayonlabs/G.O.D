from asyncpg.connection import Connection

from core.models.tournament_models import (
    TournamentData, TournamentRoundData, TournamentGroupData, TournamentPairData,
    TournamentParticipant, TournamentTask, Round, GroupRound, KnockoutRound
)
import validator.db.constants as cst
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