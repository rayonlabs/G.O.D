import asyncio
from core.models.tournament_models import (
    TournamentData, TournamentRoundData, TournamentParticipant, 
    TournamentType, TournamentStatus, RoundType, RoundStatus
)
from validator.db.sql.tournaments import (
    create_tournament, get_tournament, update_tournament_status,
    create_tournament_round, add_tournament_participants, get_tournament_participants
)
from validator.db.database import PSQLDB
from validator.core.config import Config
from validator.utils.logging import get_logger

logger = get_logger(__name__)


async def test_tournament_crud_operations():
    config = Config()
    psql_db = PSQLDB(config.psql_db)
    
    tournament_data = TournamentData(
        tournament_id="TEST_TOURN_001",
        tournament_type=TournamentType.TEXT,
        status=TournamentStatus.PENDING
    )
    
    await create_tournament(tournament_data, psql_db)
    
    retrieved_tournament = await get_tournament("TEST_TOURN_001", psql_db)
    assert retrieved_tournament is not None
    assert retrieved_tournament.tournament_id == "TEST_TOURN_001"
    assert retrieved_tournament.tournament_type == TournamentType.TEXT
    
    await update_tournament_status("TEST_TOURN_001", TournamentStatus.ACTIVE, psql_db)
    updated_tournament = await get_tournament("TEST_TOURN_001", psql_db)
    assert updated_tournament.status == TournamentStatus.ACTIVE
    
    logger.info("Tournament CRUD operations test passed")


async def test_participant_management():
    config = Config()
    psql_db = PSQLDB(config.psql_db)
    
    tournament_data = TournamentData(
        tournament_id="TEST_TOURN_002",
        tournament_type=TournamentType.TEXT,
        status=TournamentStatus.PENDING
    )
    await create_tournament(tournament_data, psql_db)
    
    participants = [
        TournamentParticipant(tournament_id="TEST_TOURN_002", hotkey="hotkey_1"),
        TournamentParticipant(tournament_id="TEST_TOURN_002", hotkey="hotkey_2"),
        TournamentParticipant(tournament_id="TEST_TOURN_002", hotkey="hotkey_3"),
        TournamentParticipant(tournament_id="TEST_TOURN_002", hotkey="hotkey_4")
    ]
    
    await add_tournament_participants(participants, psql_db)
    
    retrieved_hotkeys = await get_tournament_participants("TEST_TOURN_002", psql_db)
    
    assert len(retrieved_hotkeys) == 4
    for participant in participants:
        assert participant.hotkey in retrieved_hotkeys
    
    logger.info("Participant management test passed")


async def test_tournament_round_creation():
    config = Config()
    psql_db = PSQLDB(config.psql_db)
    
    tournament_data = TournamentData(
        tournament_id="TEST_TOURN_003",
        tournament_type=TournamentType.TEXT,
        status=TournamentStatus.PENDING
    )
    await create_tournament(tournament_data, psql_db)
    
    round_data = TournamentRoundData(
        round_id="TEST_ROUND_001",
        tournament_id="TEST_TOURN_003",
        round_number=1,
        round_type=RoundType.KNOCKOUT,
        is_final_round=False,
        status=RoundStatus.PENDING
    )
    
    await create_tournament_round(round_data, psql_db)
    
    logger.info("Tournament round creation test passed")


async def test_all_tournament_db_operations():
    try:
        await test_tournament_crud_operations()
        await test_participant_management()
        await test_tournament_round_creation()
        logger.info("All tournament database tests passed successfully!")
        return True
    except Exception as e:
        logger.error(f"Tournament database tests failed: {e}")
        return False


def test_tournament_database():
    return asyncio.run(test_all_tournament_db_operations())


if __name__ == "__main__":
    test_tournament_database()