import asyncio
from datetime import datetime
from fiber.chain.models import Node
from core.models.tournament_models import (
    TournamentData, TournamentRoundData, TournamentParticipant,
    TournamentType, TournamentStatus, RoundType, RoundStatus
)
from validator.tournament.organiser import organise_tournament_round
from validator.tournament.tournament_manager import create_new_tournament
from validator.core.constants import (
    MIN_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND, 
    PREVIOUS_WINNER_BASE_CONTESTANT,
    TEXT_TASKS_PER_GROUP, IMAGE_TASKS_PER_GROUP
)
from validator.tests.test_constants import TEST_HOTKEY_PREFIX
from validator.utils.logging import get_logger

logger = get_logger(__name__)


def create_test_node(hotkey: str) -> Node:
    return Node(
        hotkey=hotkey,
        coldkey="test_coldkey",
        node_id=0,
        incentive=0.0,
        netuid=181,
        alpha_stake=0.0,
        tao_stake=0.0,
        stake=0.0,
        trust=0.0,
        vtrust=0.0,
        last_updated=0.0,
        ip="0.0.0.0",
        ip_type=4,
        port=8080,
        protocol=4
    )


def test_tournament_data_models():
    """Test that all tournament data models work correctly"""
    tournament = TournamentData(
        tournament_id="test_tourn_001",
        tournament_type=TournamentType.TEXT,
        status=TournamentStatus.PENDING
    )
    
    assert tournament.tournament_id == "test_tourn_001"
    assert tournament.tournament_type == TournamentType.TEXT
    assert tournament.status == TournamentStatus.PENDING
    
    round_data = TournamentRoundData(
        round_id="test_round_001",
        tournament_id="test_tourn_001",
        round_number=1,
        round_type=RoundType.KNOCKOUT,
        is_final_round=False,
        status=RoundStatus.PENDING
    )
    
    assert round_data.round_type == RoundType.KNOCKOUT
    assert round_data.is_final_round is False
    
    participant = TournamentParticipant(
        tournament_id="test_tourn_001",
        hotkey="test_hotkey_1"
    )
    
    assert participant.hotkey == "test_hotkey_1"
    assert participant.eliminated_in_round_id is None
    
    logger.info("Tournament data models test passed")


def test_knockout_tournament_logic():
    """Test knockout tournament bracket generation"""
    # Test with 8 participants (should create 4 pairs)
    nodes = [create_test_node(f"{TEST_HOTKEY_PREFIX}{i}") for i in range(8)]
    result = organise_tournament_round(nodes)
    
    assert hasattr(result, 'pairs')
    assert len(result.pairs) == 4
    
    # Verify all participants are included
    all_participants = set()
    for pair in result.pairs:
        all_participants.add(pair[0])
        all_participants.add(pair[1])
    
    assert len(all_participants) == 8
    
    # Test with odd number (should add BASE contestant)
    odd_nodes = [create_test_node(f"{TEST_HOTKEY_PREFIX}{i}") for i in range(7)]
    odd_result = organise_tournament_round(odd_nodes)
    
    assert hasattr(odd_result, 'pairs')
    assert len(odd_result.pairs) == 4
    
    # Should include BASE contestant
    all_odd_contestants = []
    for pair in odd_result.pairs:
        all_odd_contestants.extend([pair[0], pair[1]])
    
    assert PREVIOUS_WINNER_BASE_CONTESTANT in all_odd_contestants
    
    logger.info("Knockout tournament logic test passed")


def test_group_tournament_logic():
    """Test group tournament bracket generation"""
    # Test with 20 participants (should create groups)
    nodes = [create_test_node(f"{TEST_HOTKEY_PREFIX}{i}") for i in range(20)]
    result = organise_tournament_round(nodes)
    
    assert hasattr(result, 'groups')
    assert len(result.groups) > 0
    
    # Verify all participants are included
    total_members = sum(len(group.member_ids) for group in result.groups)
    assert total_members == 20
    
    # Check group sizes are reasonable
    group_sizes = [len(group.member_ids) for group in result.groups]
    for size in group_sizes:
        assert size >= 6
        assert size <= 10
    
    logger.info("Group tournament logic test passed")


def test_boundary_conditions():
    """Test the boundary between knockout and group tournaments"""
    # Exactly at boundary should be knockout
    boundary_nodes = [create_test_node(f"{TEST_HOTKEY_PREFIX}{i}") 
                     for i in range(MIN_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND)]
    boundary_result = organise_tournament_round(boundary_nodes)
    assert hasattr(boundary_result, 'pairs')
    
    # Just over boundary should be group
    over_boundary_nodes = [create_test_node(f"{TEST_HOTKEY_PREFIX}{i}") 
                          for i in range(MIN_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND + 1)]
    over_boundary_result = organise_tournament_round(over_boundary_nodes)
    assert hasattr(over_boundary_result, 'groups')
    
    logger.info("Boundary conditions test passed")


def test_tournament_constants_validation():
    """Test that tournament constants are sensible"""
    assert MIN_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND > 0
    assert TEXT_TASKS_PER_GROUP > 0
    assert IMAGE_TASKS_PER_GROUP > 0
    assert PREVIOUS_WINNER_BASE_CONTESTANT == "BASE"
    
    # Test enum values
    assert TournamentType.TEXT == "text"
    assert TournamentType.IMAGE == "image"
    assert TournamentStatus.PENDING == "pending"
    assert RoundType.KNOCKOUT == "knockout"
    assert RoundType.GROUP == "group"
    
    logger.info("Tournament constants validation test passed")


def test_participant_distribution():
    """Test that participants are distributed fairly"""
    nodes = [create_test_node(f"{TEST_HOTKEY_PREFIX}{i}") for i in range(24)]
    result = organise_tournament_round(nodes)
    
    assert hasattr(result, 'groups')
    group_sizes = [len(group.member_ids) for group in result.groups]
    
    # No group should be significantly larger than others
    min_size = min(group_sizes)
    max_size = max(group_sizes)
    assert max_size - min_size <= 1  # Difference should be at most 1
    
    logger.info("Participant distribution test passed")


def test_single_participant_edge_case():
    """Test what happens with just 1 participant"""
    single_node = [create_test_node(f"{TEST_HOTKEY_PREFIX}0")]
    result = organise_tournament_round(single_node)
    
    # Should create a knockout round with BASE contestant
    assert hasattr(result, 'pairs')
    assert len(result.pairs) == 1
    
    pair = result.pairs[0]
    assert PREVIOUS_WINNER_BASE_CONTESTANT in [pair[0], pair[1]]
    
    logger.info("Single participant edge case test passed")


def test_all_tournament_integration():
    try:
        test_tournament_data_models()
        test_knockout_tournament_logic()
        test_group_tournament_logic() 
        test_boundary_conditions()
        test_tournament_constants_validation()
        test_participant_distribution()
        test_single_participant_edge_case()
        
        logger.info("All tournament integration tests passed successfully!")
        return True
    except Exception as e:
        logger.error(f"Tournament integration tests failed: {e}")
        return False


if __name__ == "__main__":
    test_all_tournament_integration()