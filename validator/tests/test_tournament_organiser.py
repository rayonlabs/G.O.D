from validator.tournament.organiser import organize_tournament
from core.models.tournament_models import TournamentType, RoundType
from validator.core.constants import MIN_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND, EXPECTED_GROUP_SIZE
from validator.utils.logging import get_logger

logger = get_logger(__name__)


def test_knockout_tournament_creation():
    participants = [f"hotkey_{i}" for i in range(8)]
    
    result = organize_tournament("TOURN_001", "ROUND_001", participants, TournamentType.TEXT)
    
    assert result["round_type"] == RoundType.KNOCKOUT
    assert len(result["pairs"]) == 4
    assert len(result["groups"]) == 0
    
    all_participants = set()
    for pair in result["pairs"]:
        all_participants.add(pair["contestant1_hotkey"])
        all_participants.add(pair["contestant2_hotkey"])
    
    assert len(all_participants) == 8
    logger.info("Knockout tournament creation test passed")


def test_group_tournament_creation():
    participants = [f"hotkey_{i}" for i in range(20)]
    
    result = organize_tournament("TOURN_002", "ROUND_002", participants, TournamentType.TEXT)
    
    assert result["round_type"] == RoundType.GROUP
    assert len(result["groups"]) > 0
    assert len(result["pairs"]) == 0
    
    total_members = sum(len(group["members"]) for group in result["groups"])
    assert total_members == 20
    logger.info("Group tournament creation test passed")


def test_odd_number_participants_knockout():
    participants = [f"hotkey_{i}" for i in range(7)]
    
    result = organize_tournament("TOURN_003", "ROUND_003", participants, TournamentType.TEXT)
    
    assert result["round_type"] == RoundType.KNOCKOUT
    assert len(result["pairs"]) == 4
    
    all_contestants = []
    for pair in result["pairs"]:
        all_contestants.extend([pair["contestant1_hotkey"], pair["contestant2_hotkey"]])
    
    assert "BASE" in all_contestants
    logger.info("Odd number participants knockout test passed")


def test_boundary_conditions():
    exactly_16_participants = [f"hotkey_{i}" for i in range(16)]
    result_16 = organize_tournament("TOURN_004", "ROUND_004", exactly_16_participants, TournamentType.TEXT)
    assert result_16["round_type"] == RoundType.KNOCKOUT
    
    exactly_17_participants = [f"hotkey_{i}" for i in range(17)]
    result_17 = organize_tournament("TOURN_005", "ROUND_005", exactly_17_participants, TournamentType.TEXT)
    assert result_17["round_type"] == RoundType.GROUP
    
    logger.info("Boundary conditions test passed")


def test_group_size_distribution():
    participants = [f"hotkey_{i}" for i in range(25)]
    
    result = organize_tournament("TOURN_006", "ROUND_006", participants, TournamentType.TEXT)
    
    group_sizes = [len(group["members"]) for group in result["groups"]]
    
    for size in group_sizes:
        assert size >= 6
        assert size <= 10
    
    logger.info("Group size distribution test passed")


def test_participant_uniqueness():
    participants = [f"hotkey_{i}" for i in range(12)]
    
    result = organize_tournament("TOURN_007", "ROUND_007", participants, TournamentType.TEXT)
    
    all_participants = set()
    for pair in result["pairs"]:
        all_participants.add(pair["contestant1_hotkey"])
        all_participants.add(pair["contestant2_hotkey"])
    
    assert len(all_participants) == len(participants) + (1 if len(participants) % 2 == 1 else 0)
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