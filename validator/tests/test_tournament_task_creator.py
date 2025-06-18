import asyncio
from core.models.tournament_models import GroupRound, KnockoutRound, Group, TournamentType
from validator.tournament.task_creator import create_text_tournament_round, create_image_tournament_round
from validator.core.config import Config
from validator.core.constants import TEXT_TASKS_PER_GROUP, IMAGE_TASKS_PER_GROUP
from validator.tests.test_constants import (
    SMALL_KNOCKOUT_COUNT, LARGE_GROUP_COUNT, TEST_HOTKEY_PREFIX
)
from validator.utils.logging import get_logger

logger = get_logger(__name__)


async def test_text_tournament_task_creation_knockout():
    config = Config()
    
    pairs = [(f"{TEST_HOTKEY_PREFIX}{i}", f"{TEST_HOTKEY_PREFIX}{i+1}") 
             for i in range(0, SMALL_KNOCKOUT_COUNT, 2)]
    knockout_round = KnockoutRound(pairs=pairs)
    
    tournament_round = await create_text_tournament_round(knockout_round, config, is_final=False)
    
    assert len(tournament_round.tasks) > 0
    assert len(tournament_round.tasks) <= len(pairs)
    
    logger.info("Text tournament knockout task creation test passed")


async def test_text_tournament_task_creation_group():
    config = Config()
    
    groups = [Group(member_ids=[f"{TEST_HOTKEY_PREFIX}{i}" for i in range(j*8, (j+1)*8)]) 
              for j in range(3)]
    group_round = GroupRound(groups=groups)
    
    tournament_round = await create_text_tournament_round(group_round, config, is_final=False)
    
    expected_tasks = len(groups) * TEXT_TASKS_PER_GROUP
    assert len(tournament_round.tasks) == expected_tasks
    
    logger.info("Text tournament group task creation test passed")


async def test_image_tournament_task_creation_group():
    config = Config()
    
    groups = [Group(member_ids=[f"{TEST_HOTKEY_PREFIX}{i}" for i in range(j*8, (j+1)*8)]) 
              for j in range(2)]
    group_round = GroupRound(groups=groups)
    
    tournament_round = await create_image_tournament_round(group_round, config)
    
    expected_tasks = len(groups) * IMAGE_TASKS_PER_GROUP
    assert len(tournament_round.tasks) == expected_tasks
    
    logger.info("Image tournament group task creation test passed")


async def test_final_round_handling():
    config = Config()
    
    pairs = [(f"{TEST_HOTKEY_PREFIX}1", f"{TEST_HOTKEY_PREFIX}2")]
    knockout_round = KnockoutRound(pairs=pairs)
    
    tournament_round = await create_text_tournament_round(knockout_round, config, is_final=True)
    
    assert len(tournament_round.tasks) > 0
    assert tournament_round.is_final_round is True
    
    logger.info("Final round handling test passed")


async def test_all_task_creator_functionality():
    try:
        await test_text_tournament_task_creation_knockout()
        await test_text_tournament_task_creation_group()
        await test_image_tournament_task_creation_group()
        await test_final_round_handling()
        logger.info("All tournament task creator tests passed successfully!")
        return True
    except Exception as e:
        logger.error(f"Tournament task creator tests failed: {e}")
        return False


def test_tournament_task_creator():
    return asyncio.run(test_all_task_creator_functionality())


if __name__ == "__main__":
    test_tournament_task_creator()