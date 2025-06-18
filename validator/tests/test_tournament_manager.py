import asyncio
from fiber.chain.models import Node
from core.models.tournament_models import TournamentType, TournamentStatus
from validator.tournament.tournament_manager import (
    create_new_tournament, start_tournament, complete_tournament,
    get_tournament_status, create_text_tournament_with_database, 
    create_image_tournament_with_database
)
from validator.db.sql.tournaments import get_tournament_participants
from validator.db.database import PSQLDB
from validator.core.config import Config
from validator.utils.logging import get_logger

logger = get_logger(__name__)


async def test_create_new_tournament():
    config = Config()
    psql_db = PSQLDB(config.psql_db)
    
    nodes = [Node(hotkey=f"test_hotkey_{i}") for i in range(10)]
    
    tournament_id = await create_new_tournament(
        tournament_type=TournamentType.TEXT,
        eligible_nodes=nodes,
        psql_db=psql_db,
        config=config
    )
    
    assert tournament_id is not None
    assert tournament_id.startswith("tourn_")
    
    tournament = await get_tournament_status(tournament_id, psql_db)
    assert tournament is not None
    assert tournament.tournament_type == TournamentType.TEXT
    assert tournament.status == TournamentStatus.PENDING
    
    participant_hotkeys = await get_tournament_participants(tournament_id, psql_db)
    assert len(participant_hotkeys) == 10
    
    for node in nodes:
        assert node.hotkey in participant_hotkeys
    
    logger.info("Create new tournament test passed")
    return tournament_id


async def test_tournament_lifecycle():
    config = Config()
    psql_db = PSQLDB(config.psql_db)
    
    nodes = [Node(hotkey=f"lifecycle_hotkey_{i}") for i in range(8)]
    
    tournament_id = await create_new_tournament(
        tournament_type=TournamentType.TEXT,
        eligible_nodes=nodes,
        psql_db=psql_db,
        config=config
    )
    
    await start_tournament(tournament_id, psql_db)
    tournament_after_start = await get_tournament_status(tournament_id, psql_db)
    assert tournament_after_start.status == TournamentStatus.ACTIVE
    
    await complete_tournament(tournament_id, psql_db)
    tournament_after_completion = await get_tournament_status(tournament_id, psql_db)
    assert tournament_after_completion.status == TournamentStatus.COMPLETED
    
    logger.info("Tournament lifecycle test passed")


async def test_text_tournament_creation():
    config = Config()
    psql_db = PSQLDB(config.psql_db)
    
    nodes = [Node(hotkey=f"text_hotkey_{i}") for i in range(12)]
    
    tournament_id = await create_text_tournament_with_database(
        eligible_nodes=nodes,
        psql_db=psql_db,
        config=config
    )
    
    assert tournament_id is not None
    tournament = await get_tournament_status(tournament_id, psql_db)
    assert tournament.tournament_type == TournamentType.TEXT
    assert tournament.status == TournamentStatus.ACTIVE
    
    logger.info("Text tournament creation test passed")


async def test_image_tournament_creation():
    config = Config()
    psql_db = PSQLDB(config.psql_db)
    
    nodes = [Node(hotkey=f"image_hotkey_{i}") for i in range(6)]
    
    tournament_id = await create_image_tournament_with_database(
        eligible_nodes=nodes,
        psql_db=psql_db,
        config=config
    )
    
    assert tournament_id is not None
    tournament = await get_tournament_status(tournament_id, psql_db)
    assert tournament.tournament_type == TournamentType.IMAGE
    assert tournament.status == TournamentStatus.ACTIVE
    
    logger.info("Image tournament creation test passed")


async def test_all_tournament_manager_functionality():
    try:
        await test_create_new_tournament()
        await test_tournament_lifecycle()
        await test_text_tournament_creation()
        await test_image_tournament_creation()
        logger.info("All tournament manager tests passed successfully!")
        return True
    except Exception as e:
        logger.error(f"Tournament manager tests failed: {e}")
        return False


def test_tournament_manager():
    return asyncio.run(test_all_tournament_manager_functionality())


if __name__ == "__main__":
    test_tournament_manager()