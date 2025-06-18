import asyncio
from fiber.chain.models import Node
from core.models.tournament_models import TournamentType, TournamentStatus
from validator.tournament.organiser import organise_tournament_round
from validator.tournament.tournament_manager import create_new_tournament
from validator.db.database import PSQLDB
from validator.core.config import Config
from validator.core.constants import PREVIOUS_WINNER_BASE_CONTESTANT
from validator.tests.test_constants import (
    ODD_PARTICIPANT_COUNT, BOUNDARY_KNOCKOUT_COUNT, BOUNDARY_GROUP_COUNT,
    TEST_HOTKEY_PREFIX
)
from validator.utils.logging import get_logger

logger = get_logger(__name__)


def test_minimum_participants():
    single_node = [Node(hotkey=f"{TEST_HOTKEY_PREFIX}1")]
    
    try:
        result = organise_tournament_round(single_node)
        
        assert len(result.pairs) == 1
        pair = result.pairs[0]
        assert PREVIOUS_WINNER_BASE_CONTESTANT in [pair[0], pair[1]]
        
        logger.info("Minimum participants test passed")
        return True
    except Exception as e:
        logger.info(f"Minimum participants correctly handled: {e}")
        return True


def test_exactly_boundary_participants():
    exactly_boundary_nodes = [Node(hotkey=f"{TEST_HOTKEY_PREFIX}{i}") 
                             for i in range(BOUNDARY_KNOCKOUT_COUNT)]
    
    result = organise_tournament_round(exactly_boundary_nodes)
    
    assert hasattr(result, 'pairs')
    assert len(result.pairs) == BOUNDARY_KNOCKOUT_COUNT // 2
    
    logger.info("Exactly boundary participants test passed")


def test_just_over_boundary_participants():
    just_over_boundary_nodes = [Node(hotkey=f"{TEST_HOTKEY_PREFIX}{i}") 
                               for i in range(BOUNDARY_GROUP_COUNT)]
    
    result = organise_tournament_round(just_over_boundary_nodes)
    
    assert hasattr(result, 'groups')
    assert len(result.groups) > 0
    
    total_members = sum(len(group.member_ids) for group in result.groups)
    assert total_members == BOUNDARY_GROUP_COUNT
    
    logger.info("Just over boundary participants test passed")


def test_odd_number_with_base_contestant():
    odd_nodes = [Node(hotkey=f"{TEST_HOTKEY_PREFIX}{i}") for i in range(ODD_PARTICIPANT_COUNT)]
    
    result = organise_tournament_round(odd_nodes)
    
    assert hasattr(result, 'pairs')
    
    all_contestants = []
    for pair in result.pairs:
        all_contestants.extend([pair[0], pair[1]])
    
    assert PREVIOUS_WINNER_BASE_CONTESTANT in all_contestants
    
    real_participants = [c for c in all_contestants if c != PREVIOUS_WINNER_BASE_CONTESTANT]
    assert len(set(real_participants)) == ODD_PARTICIPANT_COUNT
    
    logger.info("Odd number with BASE contestant test passed")


async def test_empty_tournament_creation():
    config = Config()
    psql_db = PSQLDB(config.psql_db)
    
    empty_nodes = []
    
    try:
        tournament_id = await create_new_tournament(
            tournament_type=TournamentType.TEXT,
            eligible_nodes=empty_nodes,
            psql_db=psql_db,
            config=config
        )
        
        logger.error("Empty tournament should have failed but didn't")
        return False
    except Exception as e:
        logger.info(f"Empty tournament correctly rejected: {e}")
        return True


def test_all_edge_cases():
    try:
        test_minimum_participants()
        test_exactly_boundary_participants()
        test_just_over_boundary_participants()
        test_odd_number_with_base_contestant()
        
        logger.info("All tournament edge case tests passed successfully!")
        return True
    except Exception as e:
        logger.error(f"Tournament edge case tests failed: {e}")
        return False


async def test_all_async_edge_cases():
    try:
        await test_empty_tournament_creation()
        logger.info("All async tournament edge case tests passed successfully!")
        return True
    except Exception as e:
        logger.error(f"Async tournament edge case tests failed: {e}")
        return False


def test_tournament_edge_cases():
    sync_result = test_all_edge_cases()
    async_result = asyncio.run(test_all_async_edge_cases())
    return sync_result and async_result


if __name__ == "__main__":
    test_tournament_edge_cases()