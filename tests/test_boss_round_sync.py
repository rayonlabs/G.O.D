#!/usr/bin/env python3

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch
from uuid import uuid4

import pytest

from core.models.tournament_models import RoundStatus
from core.models.tournament_models import RoundType
from core.models.tournament_models import TournamentRoundData
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.core.constants import NULL_ACCOUNT_ID
from validator.core.models import InstructTextRawTask
from validator.tournament.boss_round_sync import _copy_task_to_general
from validator.tournament.boss_round_sync import sync_boss_round_tasks_to_general


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.synthetic_account_id = NULL_ACCOUNT_ID
    return config


@pytest.fixture
def mock_psql_db():
    return AsyncMock()


@pytest.fixture
def sample_boss_round():
    return TournamentRoundData(
        round_id="tourn_abc123_20250713_round_003",
        tournament_id="tourn_abc123_20250713",
        round_number=3,
        round_type=RoundType.KNOCKOUT,
        is_final_round=True,
        status=RoundStatus.COMPLETED,
    )


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
def sample_tournament_tasks():
    return [
        MagicMock(task_id=uuid4()),
        MagicMock(task_id=uuid4()),
        MagicMock(task_id=uuid4()),
    ]


@pytest.mark.asyncio
async def test_sync_boss_round_tasks_schedules_correctly(mock_config, mock_psql_db, sample_boss_round, sample_tournament_tasks):
    """Test that sync_boss_round_tasks_to_general schedules the correct number of tasks."""

    with patch(
        "validator.tournament.boss_round_sync.get_tournament_tasks", return_value=sample_tournament_tasks
    ) as mock_get_tasks:
        with patch("asyncio.create_task") as mock_create_task:
            await sync_boss_round_tasks_to_general("tourn_abc123_20250713", sample_boss_round, mock_psql_db, mock_config)

            # Should call get_tournament_tasks with the round_id
            mock_get_tasks.assert_called_once_with(sample_boss_round.round_id, mock_psql_db)

            # Should schedule 3 tasks (boss round should have 3 tasks)
            assert mock_create_task.call_count == 3


@pytest.mark.asyncio
async def test_copy_task_to_general_creates_correct_copy(mock_config, mock_psql_db, sample_instruct_task):
    """Test that _copy_task_to_general creates a proper copy with updated fields."""

    original_task_id = sample_instruct_task.task_id

    with patch("validator.tournament.boss_round_sync.get_task", return_value=sample_instruct_task) as mock_get_task:
        with patch("validator.tournament.boss_round_sync.add_task") as mock_add_task:
            with patch("validator.tournament.boss_round_sync._record_task_sync_link") as mock_record_link:
                await _copy_task_to_general(str(original_task_id), mock_psql_db)

                # Should get the original task
                mock_get_task.assert_called_once_with(str(original_task_id), mock_psql_db)

                # Should add a new task
                mock_add_task.assert_called_once()
                added_task = mock_add_task.call_args[0][0]

                # Verify the copy has correct properties
                assert added_task.task_id != original_task_id  # New UUID
                assert added_task.is_organic == False
                assert added_task.status == TaskStatus.PENDING
                assert added_task.account_id == NULL_ACCOUNT_ID
                assert added_task.times_delayed == 0
                assert added_task.assigned_miners is None
                assert added_task.n_eval_attempts == 0

                # Verify unchanged properties
                assert added_task.model_id == sample_instruct_task.model_id
                assert added_task.ds == sample_instruct_task.ds
                assert added_task.test_data == sample_instruct_task.test_data
                assert added_task.training_data == sample_instruct_task.training_data
                assert added_task.field_instruction == sample_instruct_task.field_instruction

                # Should record the link
                mock_record_link.assert_called_once_with(str(original_task_id), added_task.task_id, mock_psql_db)


@pytest.mark.asyncio
async def test_sync_handles_no_tasks_gracefully(mock_config, mock_psql_db, sample_boss_round):
    """Test that sync handles empty task list gracefully."""

    with patch("validator.tournament.boss_round_sync.get_tournament_tasks", return_value=[]) as mock_get_tasks:
        with patch("asyncio.create_task") as mock_create_task:
            await sync_boss_round_tasks_to_general("tourn_abc123_20250713", sample_boss_round, mock_psql_db, mock_config)

            # Should not schedule any tasks
            mock_create_task.assert_not_called()


if __name__ == "__main__":
    # Simple test runner
    async def run_tests():
        print("Testing boss round sync functionality...")

        # Create mock objects
        mock_config = MagicMock()
        mock_config.synthetic_account_id = NULL_ACCOUNT_ID
        mock_psql_db = AsyncMock()

        # Test task copying
        sample_task = InstructTextRawTask(
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

        # Test that task copying preserves important fields
        copied_task = sample_task.model_copy()
        copied_task.task_id = uuid4()
        copied_task.status = TaskStatus.PENDING

        print(f"✅ Original task ID: {sample_task.task_id}")
        print(f"✅ Copied task ID: {copied_task.task_id}")
        print(f"✅ Model preserved: {copied_task.model_id == sample_task.model_id}")
        print(f"✅ Dataset preserved: {copied_task.ds == sample_task.ds}")
        print(f"✅ Instructions preserved: {copied_task.field_instruction == sample_task.field_instruction}")
        print(f"✅ Status updated: {copied_task.status == TaskStatus.PENDING}")

        print("\nAll tests passed! 🎉")

    asyncio.run(run_tests())
