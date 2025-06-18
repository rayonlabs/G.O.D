from validator.tournament.organiser import organise_tournament_round
from core.models.tournament_models import KnockoutRound, GroupRound
from validator.core.constants import (
    MIN_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND, EXPECTED_GROUP_SIZE, 
    PREVIOUS_WINNER_BASE_CONTESTANT
)
from validator.tests.test_constants import (
    SMALL_KNOCKOUT_COUNT, LARGE_GROUP_COUNT, BOUNDARY_KNOCKOUT_COUNT, 
    BOUNDARY_GROUP_COUNT, ODD_PARTICIPANT_COUNT, TEST_HOTKEY_PREFIX
)
from validator.tests.test_helpers import create_test_node
from validator.utils.logging import get_logger

logger = get_logger(__name__)


def test_knockout_tournament_creation():
    nodes = [create_test_node(f"{TEST_HOTKEY_PREFIX}{i}") for i in range(SMALL_KNOCKOUT_COUNT)]
    
    result = organise_tournament_round(nodes)
    
    assert isinstance(result, KnockoutRound)
    assert len(result.pairs) == SMALL_KNOCKOUT_COUNT // 2
    
    all_participants = set()
    for pair in result.pairs:
        all_participants.add(pair[0])
        all_participants.add(pair[1])
    
    assert len(all_participants) == SMALL_KNOCKOUT_COUNT
    logger.info("Knockout tournament creation test passed")


def test_group_tournament_creation():
    nodes = [create_test_node(f"{TEST_HOTKEY_PREFIX}{i}") for i in range(LARGE_GROUP_COUNT)]
    
    result = organise_tournament_round(nodes)
    
    assert isinstance(result, GroupRound)
    assert len(result.groups) > 0
    
    total_members = sum(len(group.member_ids) for group in result.groups)
    assert total_members == LARGE_GROUP_COUNT
    logger.info("Group tournament creation test passed")


def test_odd_number_participants_knockout():
    nodes = [create_test_node(f"{TEST_HOTKEY_PREFIX}{i}") for i in range(ODD_PARTICIPANT_COUNT)]
    
    result = organise_tournament_round(nodes)
    
    assert isinstance(result, KnockoutRound)
    assert len(result.pairs) == 4
    
    all_contestants = []
    for pair in result.pairs:
        all_contestants.extend([pair[0], pair[1]])
    
    assert PREVIOUS_WINNER_BASE_CONTESTANT in all_contestants
    logger.info("Odd number participants knockout test passed")


def test_boundary_conditions():
    exactly_boundary_nodes = [create_test_node(f"{TEST_HOTKEY_PREFIX}{i}") for i in range(BOUNDARY_KNOCKOUT_COUNT)]
    result_boundary = organise_tournament_round(exactly_boundary_nodes)
    assert isinstance(result_boundary, KnockoutRound)
    
    just_over_boundary_nodes = [create_test_node(f"{TEST_HOTKEY_PREFIX}{i}") for i in range(BOUNDARY_GROUP_COUNT)]
    result_over = organise_tournament_round(just_over_boundary_nodes)
    assert isinstance(result_over, GroupRound)
    
    logger.info("Boundary conditions test passed")


def test_group_size_distribution():
    nodes = [create_test_node(f"{TEST_HOTKEY_PREFIX}{i}") for i in range(25)]
    
    result = organise_tournament_round(nodes)
    
    assert isinstance(result, GroupRound)
    group_sizes = [len(group.member_ids) for group in result.groups]
    
    for size in group_sizes:
        assert size >= 6
        assert size <= 10
    
    logger.info("Group size distribution test passed")


def test_participant_uniqueness():
    nodes = [create_test_node(f"{TEST_HOTKEY_PREFIX}{i}") for i in range(12)]
    
    result = organise_tournament_round(nodes)
    
    assert isinstance(result, KnockoutRound)
    all_participants = set()
    for pair in result.pairs:
        all_participants.add(pair[0])
        all_participants.add(pair[1])
    
    assert len(all_participants) == 12
    logger.info("Participant uniqueness test passed")


def test_all_organiser_functionality():
    try:
        test_knockout_tournament_creation()
        test_group_tournament_creation()
        test_odd_number_participants_knockout()
        test_boundary_conditions()
        test_group_size_distribution()
        test_participant_uniqueness()
        logger.info("All tournament organiser tests passed successfully!")
        return True
    except Exception as e:
        logger.error(f"Tournament organiser tests failed: {e}")
        return False


if __name__ == "__main__":
    test_all_organiser_functionality()