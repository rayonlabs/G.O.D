import asyncio
from datetime import datetime
from core.models.tournament_models import TournamentType, TournamentStatus, RoundType
from validator.db.sql.tournaments import (
    create_tournament, get_tournament_by_id, update_tournament_status,
    create_tournament_round, get_rounds_by_tournament_id,
    add_tournament_participant, get_tournament_participants,
    create_tournament_group, create_tournament_pair,
    get_tournament_groups_by_round_id, get_tournament_pairs_by_round_id,
    create_tournament_task, get_tournament_tasks_by_round_id
)
from validator.utils.logging import get_logger

logger = get_logger(__name__)


async def test_tournament_crud_operations():
    tournament_id = "TEST_TOURN_001"
    
    tournament_data = {
        "tournament_id": tournament_id,
        "tournament_type": TournamentType.TEXT,
        "status": TournamentStatus.PENDING,
        "participant_count": 8,
        "created_at": datetime.now(),
        "name": "Test Tournament"
    }
    
    await create_tournament(tournament_data)
    
    retrieved_tournament = await get_tournament_by_id(tournament_id)
    assert retrieved_tournament is not None
    assert retrieved_tournament["tournament_id"] == tournament_id
    assert retrieved_tournament["tournament_type"] == TournamentType.TEXT
    
    await update_tournament_status(tournament_id, TournamentStatus.ACTIVE)
    updated_tournament = await get_tournament_by_id(tournament_id)
    assert updated_tournament["status"] == TournamentStatus.ACTIVE
    
    logger.info("Tournament CRUD operations test passed")


async def test_tournament_round_operations():
    tournament_id = "TEST_TOURN_002"
    round_id = "TEST_ROUND_001"
    
    tournament_data = {
        "tournament_id": tournament_id,
        "tournament_type": TournamentType.TEXT,
        "status": TournamentStatus.PENDING,
        "participant_count": 16,
        "created_at": datetime.now(),
        "name": "Test Tournament 2"
    }
    await create_tournament(tournament_data)
    
    round_data = {
        "round_id": round_id,
        "tournament_id": tournament_id,
        "round_number": 1,
        "round_type": RoundType.KNOCKOUT,
        "status": TournamentStatus.ACTIVE,
        "created_at": datetime.now()
    }
    await create_tournament_round(round_data)
    
    rounds = await get_rounds_by_tournament_id(tournament_id)
    assert len(rounds) == 1
    assert rounds[0]["round_id"] == round_id
    assert rounds[0]["round_type"] == RoundType.KNOCKOUT
    
    logger.info("Tournament round operations test passed")


async def test_participant_management():
    tournament_id = "TEST_TOURN_003"
    
    tournament_data = {
        "tournament_id": tournament_id,
        "tournament_type": TournamentType.TEXT,
        "status": TournamentStatus.PENDING,
        "participant_count": 4,
        "created_at": datetime.now(),
        "name": "Test Tournament 3"
    }
    await create_tournament(tournament_data)
    
    participants = ["hotkey_1", "hotkey_2", "hotkey_3", "hotkey_4"]
    
    for hotkey in participants:
        await add_tournament_participant(tournament_id, hotkey)
    
    retrieved_participants = await get_tournament_participants(tournament_id)
    participant_hotkeys = [p["miner_hotkey"] for p in retrieved_participants]
    
    assert len(retrieved_participants) == 4
    for hotkey in participants:
        assert hotkey in participant_hotkeys
    
    logger.info("Participant management test passed")


async def test_group_and_pair_creation():
    tournament_id = "TEST_TOURN_004"
    round_id = "TEST_ROUND_002"
    
    tournament_data = {
        "tournament_id": tournament_id,
        "tournament_type": TournamentType.TEXT,
        "status": TournamentStatus.PENDING,
        "participant_count": 8,
        "created_at": datetime.now(),
        "name": "Test Tournament 4"
    }
    await create_tournament(tournament_data)
    
    round_data = {
        "round_id": round_id,
        "tournament_id": tournament_id,
        "round_number": 1,
        "round_type": RoundType.GROUP,
        "status": TournamentStatus.ACTIVE,
        "created_at": datetime.now()
    }
    await create_tournament_round(round_data)
    
    group_data = {
        "group_id": "TEST_GROUP_001",
        "round_id": round_id,
        "group_name": "Group A"
    }
    await create_tournament_group(group_data)
    
    pair_data = {
        "pair_id": "TEST_PAIR_001",
        "round_id": round_id,
        "contestant1_hotkey": "hotkey_1",
        "contestant2_hotkey": "hotkey_2"
    }
    await create_tournament_pair(pair_data)
    
    groups = await get_tournament_groups_by_round_id(round_id)
    pairs = await get_tournament_pairs_by_round_id(round_id)
    
    assert len(groups) == 1
    assert groups[0]["group_id"] == "TEST_GROUP_001"
    assert len(pairs) == 1
    assert pairs[0]["pair_id"] == "TEST_PAIR_001"
    
    logger.info("Group and pair creation test passed")


async def test_task_assignment():
    tournament_id = "TEST_TOURN_005"
    round_id = "TEST_ROUND_003"
    
    tournament_data = {
        "tournament_id": tournament_id,
        "tournament_type": TournamentType.TEXT,
        "status": TournamentStatus.PENDING,
        "participant_count": 8,
        "created_at": datetime.now(),
        "name": "Test Tournament 5"
    }
    await create_tournament(tournament_data)
    
    round_data = {
        "round_id": round_id,
        "tournament_id": tournament_id,
        "round_number": 1,
        "round_type": RoundType.KNOCKOUT,
        "status": TournamentStatus.ACTIVE,
        "created_at": datetime.now()
    }
    await create_tournament_round(round_data)
    
    task_data = {
        "task_id": "test_task_001",
        "round_id": round_id,
        "group_id": None,
        "pair_id": "TEST_PAIR_001",
        "task_type": "INSTRUCT"
    }
    await create_tournament_task(task_data)
    
    tasks = await get_tournament_tasks_by_round_id(round_id)
    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "test_task_001"
    assert tasks[0]["task_type"] == "INSTRUCT"
    
    logger.info("Task assignment test passed")


async def test_all_tournament_db_operations():
    try:
        await test_tournament_crud_operations()
        await test_tournament_round_operations()
        await test_participant_management()
        await test_group_and_pair_creation()
        await test_task_assignment()
        logger.info("All tournament database tests passed successfully!")
        return True
    except Exception as e:
        logger.error(f"Tournament database tests failed: {e}")
        return False


def test_tournament_database():
    return asyncio.run(test_all_tournament_db_operations())


if __name__ == "__main__":
    test_tournament_database()