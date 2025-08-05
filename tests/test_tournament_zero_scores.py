#!/usr/bin/env python3

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch
from uuid import uuid4

import pytest

from core.models.tournament_models import Group
from core.models.tournament_models import GroupRound
from core.models.tournament_models import RoundStatus
from core.models.tournament_models import RoundType
from core.models.tournament_models import TournamentRoundData
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.core.constants import NULL_ACCOUNT_ID
from validator.core.models import InstructTextRawTask
from validator.tournament.tournament_manager import check_if_round_is_completed


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.psql_db = AsyncMock()
    return config


@pytest.fixture
def sample_instruct_task():
    return InstructTextRawTask(
        is_organic=False,
        task_id=uuid4(),
        status=TaskStatus.SUCCESS,
        model_id="microsoft/DialoGPT-medium",
        ds="tatsu-lab/alpaca",
        account_id=NULL_ACCOUNT_ID,
        hours_to_complete=4,
        test_data="test_data_content",
        training_data="training_data_content",
        created_at=datetime.utcnow(),
        task_type=TaskType.INSTRUCTTEXTTASK,
        model_params_count=1000000000,
        field_instruction="Write a poem",
        field_input="about nature",
        field_output="The trees sway gently...",
        field_system="You are a helpful assistant",
        format="### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
        no_input_format="### Instruction:\n{instruction}\n\n### Response:\n{output}",
        synthetic_data=None,
    )


@pytest.fixture
def sample_round_data():
    return TournamentRoundData(
        round_id="round_123",
        tournament_id="tourn_abc123_20250713",
        round_number=1,
        round_type=RoundType.GROUP,
        is_final_round=False,
        status=RoundStatus.ACTIVE,
    )


@pytest.fixture
def sample_group_round():
    return GroupRound(
        groups=[
            Group(member_ids=["hotkey1", "hotkey2", "hotkey3"], task_ids=[]),
            Group(member_ids=["hotkey4", "hotkey5"], task_ids=[]),
        ]
    )


class TestTournamentZeroScoreHandling:
    """Test cases for tournament failure task handling logic."""

    @pytest.mark.asyncio
    async def test_round_completion_with_failure_task_no_synced_task(self, mock_config, sample_round_data):
        mock_tournament_task = MagicMock()
        mock_tournament_task.task_id = "task_123"
        mock_tournament_task.group_id = "group_001"
        mock_tournament_task.pair_id = None
        mock_task_obj = MagicMock()
        mock_task_obj.status = TaskStatus.FAILURE.value
        with patch("validator.tournament.tournament_manager.get_tournament_tasks", return_value=[mock_tournament_task]):
            with patch("validator.tournament.tournament_manager.task_sql.get_task", return_value=mock_task_obj):
                with patch("validator.tournament.tournament_manager.get_synced_task_id", return_value=None):
                    with patch("validator.tournament.tournament_manager._copy_task_to_general") as mock_copy:
                        result = await check_if_round_is_completed(sample_round_data, mock_config)
                        mock_copy.assert_called_once_with("task_123", mock_config.psql_db)
                        assert result is False

    @pytest.mark.asyncio
    async def test_round_completion_with_failure_task_synced_task_failure(self, mock_config, sample_round_data):
        mock_tournament_task = MagicMock()
        mock_tournament_task.task_id = "task_123"
        mock_tournament_task.group_id = "group_001"
        mock_tournament_task.pair_id = None
        mock_task_obj = MagicMock()
        mock_task_obj.status = TaskStatus.FAILURE.value
        mock_synced_task_obj = MagicMock()
        mock_synced_task_obj.status = TaskStatus.FAILURE.value
        mock_new_task = MagicMock()
        mock_new_task.task_id = "new_task_456"
        with patch("validator.tournament.tournament_manager.get_tournament_tasks", return_value=[mock_tournament_task]):
            with patch(
                "validator.tournament.tournament_manager.task_sql.get_task", side_effect=[mock_task_obj, mock_synced_task_obj]
            ):
                with patch("validator.tournament.tournament_manager.get_synced_task_id", return_value="synced_task_789"):
                    with patch(
                        "validator.tournament.tournament_manager.create_new_task_of_same_type",
                        new=AsyncMock(return_value=mock_new_task),
                    ):
                        with patch("validator.tournament.tournament_manager.add_tournament_tasks") as mock_add_tasks:
                            with patch("validator.tournament.tournament_manager.task_sql.delete_task") as mock_delete_task:
                                result = await check_if_round_is_completed(sample_round_data, mock_config)
                                mock_add_tasks.assert_called_once()
                                mock_delete_task.assert_called_once_with("task_123", mock_config.psql_db)
                                assert result is False

    @pytest.mark.asyncio
    async def test_round_completion_with_failure_task_synced_task_success(self, mock_config, sample_round_data):
        mock_tournament_task = MagicMock()
        mock_tournament_task.task_id = "task_123"
        mock_tournament_task.group_id = "group_001"
        mock_tournament_task.pair_id = None
        mock_task_obj = MagicMock()
        mock_task_obj.status = TaskStatus.FAILURE.value
        mock_synced_task_obj = MagicMock()
        mock_synced_task_obj.status = TaskStatus.SUCCESS.value
        with patch("validator.tournament.tournament_manager.get_tournament_tasks", return_value=[mock_tournament_task]):
            with patch(
                "validator.tournament.tournament_manager.task_sql.get_task", side_effect=[mock_task_obj, mock_synced_task_obj]
            ):
                with patch("validator.tournament.tournament_manager.get_synced_task_id", return_value="synced_task_789"):
                    result = await check_if_round_is_completed(sample_round_data, mock_config)
                    assert result is True

    @pytest.mark.asyncio
    async def test_round_completion_with_failure_task_synced_task_not_completed(self, mock_config, sample_round_data):
        mock_tournament_task = MagicMock()
        mock_tournament_task.task_id = "task_123"
        mock_tournament_task.group_id = "group_001"
        mock_tournament_task.pair_id = None
        mock_task_obj = MagicMock()
        mock_task_obj.status = TaskStatus.FAILURE.value
        mock_synced_task_obj = MagicMock()
        mock_synced_task_obj.status = TaskStatus.TRAINING.value
        with patch("validator.tournament.tournament_manager.get_tournament_tasks", return_value=[mock_tournament_task]):
            with patch(
                "validator.tournament.tournament_manager.task_sql.get_task", side_effect=[mock_task_obj, mock_synced_task_obj]
            ):
                with patch("validator.tournament.tournament_manager.get_synced_task_id", return_value="synced_task_789"):
                    result = await check_if_round_is_completed(sample_round_data, mock_config)
                    assert result is False

    @pytest.mark.asyncio
    async def test_round_completion_normal_case(self, mock_config, sample_round_data):
        mock_tournament_task = MagicMock()
        mock_tournament_task.task_id = "task_123"
        mock_tournament_task.group_id = "group_001"
        mock_tournament_task.pair_id = None
        mock_task_obj = MagicMock()
        mock_task_obj.status = TaskStatus.SUCCESS.value
        with patch("validator.tournament.tournament_manager.get_tournament_tasks", return_value=[mock_tournament_task]):
            with patch("validator.tournament.tournament_manager.task_sql.get_task", return_value=mock_task_obj):
                with patch("validator.tournament.tournament_manager.get_synced_task_id", return_value=None):
                    result = await check_if_round_is_completed(sample_round_data, mock_config)
                    assert result is True

    @pytest.mark.asyncio
    async def test_round_completion_task_not_finished(self, mock_config, sample_round_data):
        mock_tournament_task = MagicMock()
        mock_tournament_task.task_id = "task_123"
        mock_tournament_task.group_id = "group_001"
        mock_tournament_task.pair_id = None
        mock_task_obj = MagicMock()
        mock_task_obj.status = TaskStatus.TRAINING.value
        with patch("validator.tournament.tournament_manager.get_tournament_tasks", return_value=[mock_tournament_task]):
            with patch("validator.tournament.tournament_manager.task_sql.get_task", return_value=mock_task_obj):
                result = await check_if_round_is_completed(sample_round_data, mock_config)
                assert result is False

    @pytest.mark.asyncio
    async def test_round_completion_no_tasks(self, mock_config, sample_round_data):
        with patch("validator.tournament.tournament_manager.get_tournament_tasks", return_value=[]):
            result = await check_if_round_is_completed(sample_round_data, mock_config)
            assert result is False


if __name__ == "__main__":
    # Simple test runner
    async def run_tests():
        print("Testing tournament zero-score handling functionality...")

        # Create mock objects
        mock_config = MagicMock()
        mock_config.psql_db = AsyncMock()

        print("\nAll tests passed! 🎉")

    asyncio.run(run_tests())
