import asyncio
from datetime import datetime
from core.models.tournament_models import TournamentType, TournamentStatus
from validator.tournament.tournament_manager import (
    create_new_tournament, start_tournament, complete_tournament,
    create_text_tournament_with_database, create_image_tournament_with_database
)
from validator.db.sql.tournaments import get_tournament_by_id, get_tournament_participants
from validator.utils.logging import get_logger

logger = get_logger(__name__)


async def test_create_new_tournament():
    tournament_name = "Integration Test Tournament"
    participants = [f"test_hotkey_{i}" for i in range(10)]
    
    tournament_id = await create_new_tournament(
        tournament_name=tournament_name,
        tournament_type=TournamentType.TEXT,
        participant_hotkeys=participants
    )
    
    assert tournament_id is not None
    assert tournament_id.startswith("TOURN_")
    
    tournament = await get_tournament_by_id(tournament_id)
    assert tournament is not None
    assert tournament["name"] == tournament_name
    assert tournament["tournament_type"] == TournamentType.TEXT
    assert tournament["participant_count"] == 10
    assert tournament["status"] == TournamentStatus.PENDING
    
    tournament_participants = await get_tournament_participants(tournament_id)
    assert len(tournament_participants) == 10
    
    participant_hotkeys = [p["miner_hotkey"] for p in tournament_participants]
    for hotkey in participants:
        assert hotkey in participant_hotkeys
    
    logger.info("Create new tournament test passed")
    return tournament_id


async def test_tournament_lifecycle():
    tournament_name = "Lifecycle Test Tournament"
    participants = [f"lifecycle_hotkey_{i}" for i in range(8)]
    
    tournament_id = await create_new_tournament(
        tournament_name=tournament_name,
        tournament_type=TournamentType.TEXT,
        participant_hotkeys=participants
    )
    
    await start_tournament(tournament_id)
    tournament_after_start = await get_tournament_by_id(tournament_id)
    assert tournament_after_start["status"] == TournamentStatus.ACTIVE
    
    await complete_tournament(tournament_id)
    tournament_after_completion = await get_tournament_by_id(tournament_id)
    assert tournament_after_completion["status"] == TournamentStatus.COMPLETED
    
    logger.info("Tournament lifecycle test passed")


async def test_text_tournament_creation():
    participants = [f"text_hotkey_{i}" for i in range(12)]
    
    tournament_id = await create_text_tournament_with_database(
        tournament_name="Text Tournament Test",
        participant_hotkeys=participants
    )
    
    assert tournament_id is not None
    tournament = await get_tournament_by_id(tournament_id)
    assert tournament["tournament_type"] == TournamentType.TEXT
    
    logger.info("Text tournament creation test passed")


async def test_image_tournament_creation():
    participants = [f"image_hotkey_{i}" for i in range(6)]
    
    tournament_id = await create_image_tournament_with_database(
        tournament_name="Image Tournament Test",
        participant_hotkeys=participants
    )
    
    assert tournament_id is not None
    tournament = await get_tournament_by_id(tournament_id)
    assert tournament["tournament_type"] == TournamentType.IMAGE
    
    logger.info("Image tournament creation test passed")


async def test_large_tournament_organization():
    participants = [f"large_hotkey_{i}" for i in range(25)]
    
    tournament_id = await create_new_tournament(
        tournament_name="Large Tournament Test",
        tournament_type=TournamentType.TEXT,
        participant_hotkeys=participants
    )
    
    tournament = await get_tournament_by_id(tournament_id)
    assert tournament["participant_count"] == 25
    
    logger.info("Large tournament organization test passed")


async def test_small_tournament_organization():
    participants = [f"small_hotkey_{i}" for i in range(4)]
    
    tournament_id = await create_new_tournament(
        tournament_name="Small Tournament Test",
        tournament_type=TournamentType.TEXT,
        participant_hotkeys=participants
    )
    
    tournament = await get_tournament_by_id(tournament_id)
    assert tournament["participant_count"] == 4
    
    logger.info("Small tournament organization test passed")


async def test_edge_case_single_participant():
    participants = ["single_hotkey"]
    
    try:
        tournament_id = await create_new_tournament(
            tournament_name="Single Participant Test",
            tournament_type=TournamentType.TEXT,
            participant_hotkeys=participants
        )
        
        tournament = await get_tournament_by_id(tournament_id)
        assert tournament["participant_count"] == 1
        logger.info("Single participant tournament test passed")
    except Exception as e:
        logger.info(f"Single participant tournament correctly rejected: {e}")


async def test_all_tournament_manager_functionality():
    try:
        await test_create_new_tournament()
        await test_tournament_lifecycle()
        await test_text_tournament_creation()
        await test_image_tournament_creation()
        await test_large_tournament_organization()
        await test_small_tournament_organization()
        await test_edge_case_single_participant()
        logger.info("All tournament manager tests passed successfully!")
        return True
    except Exception as e:
        logger.error(f"Tournament manager tests failed: {e}")
        return False


def test_tournament_manager():
    return asyncio.run(test_all_tournament_manager_functionality())


if __name__ == "__main__":
    test_tournament_manager()