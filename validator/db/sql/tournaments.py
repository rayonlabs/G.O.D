from fiber.chain.models import Node

import validator.db.constants as cst
from core.models.tournament_models import GroupRound
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentGroupData
from core.models.tournament_models import TournamentPairData
from core.models.tournament_models import TournamentParticipant
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentStatus
from core.models.tournament_models import TournamentTask
from core.models.tournament_models import TournamentType
from validator.db.database import PSQLDB
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
                ({cst.TOURNAMENT_ID}, {cst.HOTKEY}, {cst.CREATED_AT})
                VALUES ($1, $2, CURRENT_TIMESTAMP)
                ON CONFLICT ({cst.TOURNAMENT_ID}, {cst.HOTKEY}) DO NOTHING
            """
            for participant in participants:
                await connection.execute(query, participant.tournament_id, participant.hotkey)
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
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS}, {cst.CURRENT_ROUND_ID}, {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1
        """
        result = await connection.fetchrow(query, tournament_id)
        if result:
            return TournamentData(
                tournament_id=result[cst.TOURNAMENT_ID],
                tournament_type=result[cst.TOURNAMENT_TYPE],
                status=result[cst.TOURNAMENT_STATUS],
                current_round_id=result[cst.CURRENT_ROUND_ID],
                base_winner_hotkey=result[cst.BASE_WINNER_HOTKEY],
                winner_hotkey=result[cst.WINNER_HOTKEY],
            )
        return None


async def get_latest_completed_tournament(psql_db: PSQLDB, tournament_type: TournamentType) -> TournamentData | None:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS}, {cst.CURRENT_ROUND_ID}, {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}
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
                current_round_id=result[cst.CURRENT_ROUND_ID],
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


async def update_tournament_current_round(tournament_id: str, round_id: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENTS_TABLE} 
            SET {cst.CURRENT_ROUND_ID} = $2, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TOURNAMENT_ID} = $1
        """
        await connection.execute(query, tournament_id, round_id)
        logger.info(f"Updated tournament {tournament_id} current round to {round_id}")


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
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS}, {cst.CURRENT_ROUND_ID}, {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}
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
                current_round_id=row[cst.CURRENT_ROUND_ID],
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
                   {cst.FINAL_POSITION}, {cst.TRAINING_REPO}, {cst.TRAINING_COMMIT_HASH}
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
            )
        return None


async def get_tournament_participants(tournament_id: str, psql_db: PSQLDB) -> list[TournamentParticipant]:
    """Get all participants for a tournament with their training repo information."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.HOTKEY}, {cst.ELIMINATED_IN_ROUND_ID}, 
                   {cst.FINAL_POSITION}, {cst.TRAINING_REPO}, {cst.TRAINING_COMMIT_HASH}
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
            )
            for row in results
        ]
