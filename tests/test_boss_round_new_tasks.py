#!/usr/bin/env python3
"""Test file for new boss round synthetic task creation functionality."""

from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch
from uuid import uuid4

import pytest

from core.models.tournament_models import TournamentTask
from core.models.utility_models import RewardFunction
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.core.constants import NULL_ACCOUNT_ID
from validator.core.models import DpoRawTask
from validator.core.models import GrpoRawTask
from validator.core.models import ImageRawTask
from validator.core.models import InstructTextRawTask
from validator.tournament import constants as t_cst
from validator.tournament.task_creator import _create_new_image_boss_round_tasks
from validator.tournament.task_creator import _create_new_text_boss_round_tasks
from validator.tournament.task_creator import _create_single_new_text_task
from validator.tournament.task_creator import create_image_tournament_tasks
from validator.tournament.task_creator import create_text_tournament_tasks


@pytest.fixture
def mock_config():
    """Create a mock Config object."""
    config = MagicMock()
    config.keypair = MagicMock()
    config.psql_db = AsyncMock()
    return config


@pytest.fixture
def mock_instruct_task():
    """Create a mock InstructTextTask."""
    return InstructTextRawTask(
        is_organic=False,
        task_id=uuid4(),
        status=TaskStatus.LOOKING_FOR_NODES,
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
    )


@pytest.fixture
def mock_dpo_task():
    """Create a mock DpoTask."""
    return DpoRawTask(
        is_organic=False,
        task_id=uuid4(),
        status=TaskStatus.LOOKING_FOR_NODES,
        model_id="microsoft/DialoGPT-medium",
        ds="tatsu-lab/alpaca",
        account_id=NULL_ACCOUNT_ID,
        hours_to_complete=4,
        test_data="test_data_content",
        training_data="training_data_content",
        created_at=datetime.utcnow(),
        task_type=TaskType.DPOTASK,
        model_params_count=1000000000,
        field_prompt="prompt",
        field_chosen="chosen",
        field_rejected="rejected",
    )


@pytest.fixture
def mock_grpo_task():
    """Create a mock GrpoTask."""
    return GrpoRawTask(
        is_organic=False,
        task_id=uuid4(),
        status=TaskStatus.LOOKING_FOR_NODES,
        model_id="microsoft/DialoGPT-medium",
        ds="tatsu-lab/alpaca",
        account_id=NULL_ACCOUNT_ID,
        hours_to_complete=4,
        test_data="test_data_content",
        training_data="training_data_content",
        created_at=datetime.utcnow(),
        task_type=TaskType.GRPOTASK,
        model_params_count=1000000000,
        field_prompt="prompt",
        reward_functions=[
            RewardFunction(
                reward_func="def reward_func(completions, **kwargs):\n    return [1.0] * len(completions)",
                reward_weight=1.0,
            )
        ],
    )


@pytest.fixture
def mock_image_task():
    """Create a mock ImageTask."""
    return ImageRawTask(
        is_organic=False,
        task_id=uuid4(),
        status=TaskStatus.LOOKING_FOR_NODES,
        model_id="runwayml/stable-diffusion-v1-5",
        ds="test_dataset",
        account_id=NULL_ACCOUNT_ID,
        hours_to_complete=4,
        test_data="test_data_content",
        training_data="training_data_content",
        created_at=datetime.utcnow(),
        task_type=TaskType.IMAGETASK,
        model_params_count=1000000000,
    )


class TestNewTextBossRoundTasks:
    """Test new synthetic text task creation for boss rounds."""

    @pytest.mark.asyncio
    async def test_create_new_text_boss_round_tasks_creates_correct_count(self, mock_config, mock_instruct_task, mock_dpo_task, mock_grpo_task):
        """Test that _create_new_text_boss_round_tasks creates 6 tasks (2 of each type)."""
        tournament_id = str(uuid4())
        round_id = str(uuid4())

        # Mock the dependencies
        with patch("validator.tournament.task_creator.get_tournament_tasks", return_value=[]):
            with patch("validator.tournament.task_creator._get_text_models", return_value=["model1", "model2"]):
                with patch("validator.tournament.task_creator._get_instruct_text_datasets", return_value=["dataset1"]):
                    with patch("validator.tournament.task_creator._get_dpo_datasets", return_value=["dataset1"]):
                        with patch("validator.tournament.task_creator.create_synthetic_instruct_text_task", return_value=mock_instruct_task):
                            with patch("validator.tournament.task_creator.create_synthetic_dpo_task", return_value=mock_dpo_task):
                                with patch("validator.tournament.task_creator.create_synthetic_grpo_task", return_value=mock_grpo_task):
                                    with patch("validator.tournament.task_creator.add_tournament_tasks") as mock_add_tournament:
                                        with patch("validator.tournament.task_creator.get_tournament_gpu_requirement", return_value="A100"):
                                            tasks = await _create_new_text_boss_round_tasks(tournament_id, round_id, mock_config)

        # Should create 6 tasks total (2 of each type)
        assert len(tasks) == t_cst.FINAL_ROUND_TEXT_TASKS

        # Count task types
        instruct_count = sum(1 for t in tasks if t.task_type == TaskType.INSTRUCTTEXTTASK)
        dpo_count = sum(1 for t in tasks if t.task_type == TaskType.DPOTASK)
        grpo_count = sum(1 for t in tasks if t.task_type == TaskType.GRPOTASK)

        assert instruct_count == 2, f"Expected 2 InstructTextTask, got {instruct_count}"
        assert dpo_count == 2, f"Expected 2 DpoTask, got {dpo_count}"
        assert grpo_count == 2, f"Expected 2 GrpoTask, got {grpo_count}"

        # Verify tournament tasks were added
        assert mock_add_tournament.call_count == 6
        for call in mock_add_tournament.call_args_list:
            tournament_task = call[0][0][0]
            assert isinstance(tournament_task, TournamentTask)
            assert tournament_task.tournament_id == tournament_id
            assert tournament_task.round_id == round_id
            assert tournament_task.pair_id == f"{round_id}_pair_001"
            assert tournament_task.group_id is None

    @pytest.mark.asyncio
    async def test_create_new_text_boss_round_tasks_with_existing_tasks(self, mock_config, mock_instruct_task, mock_dpo_task, mock_grpo_task):
        """Test that _create_new_text_boss_round_tasks handles existing tasks correctly."""
        tournament_id = str(uuid4())
        round_id = str(uuid4())
        pair_id = f"{round_id}_pair_001"

        # Create existing tournament task
        existing_tournament_task = TournamentTask(
            tournament_id=tournament_id,
            round_id=round_id,
            task_id=mock_instruct_task.task_id,
            group_id=None,
            pair_id=pair_id,
        )

        with patch("validator.tournament.task_creator.get_tournament_tasks", return_value=[existing_tournament_task]):
            with patch("validator.tournament.task_creator.task_sql.get_task", return_value=mock_instruct_task):
                with patch("validator.tournament.task_creator._get_text_models", return_value=["model1"]):
                    with patch("validator.tournament.task_creator._get_instruct_text_datasets", return_value=["dataset1"]):
                        with patch("validator.tournament.task_creator._get_dpo_datasets", return_value=["dataset1"]):
                            with patch("validator.tournament.task_creator.create_synthetic_instruct_text_task", return_value=mock_instruct_task):
                                with patch("validator.tournament.task_creator.create_synthetic_dpo_task", return_value=mock_dpo_task):
                                    with patch("validator.tournament.task_creator.create_synthetic_grpo_task", return_value=mock_grpo_task):
                                        with patch("validator.tournament.task_creator.add_tournament_tasks") as mock_add_tournament:
                                            with patch("validator.tournament.task_creator.get_tournament_gpu_requirement", return_value="A100"):
                                                tasks = await _create_new_text_boss_round_tasks(tournament_id, round_id, mock_config)

        # Should have 1 existing + 5 new tasks = 6 total
        assert len(tasks) == t_cst.FINAL_ROUND_TEXT_TASKS
        # Should only add 5 new tournament tasks (1 already exists)
        assert mock_add_tournament.call_count == 5

    @pytest.mark.asyncio
    async def test_create_new_text_boss_round_tasks_skips_when_complete(self, mock_config, mock_instruct_task, mock_dpo_task, mock_grpo_task):
        """Test that _create_new_text_boss_round_tasks skips creation when 6 tasks already exist."""
        tournament_id = str(uuid4())
        round_id = str(uuid4())
        pair_id = f"{round_id}_pair_001"

        # Create 6 existing tournament tasks
        existing_tasks = []
        for i in range(6):
            task = MagicMock()
            task.task_id = uuid4()
            task.task_type = [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK][i % 3]
            existing_tasks.append(TournamentTask(
                tournament_id=tournament_id,
                round_id=round_id,
                task_id=task.task_id,
                group_id=None,
                pair_id=pair_id,
            ))

        with patch("validator.tournament.task_creator.get_tournament_tasks", return_value=existing_tasks):
            with patch("validator.tournament.task_creator.task_sql.get_task") as mock_get_task:
                # Mock get_task to return appropriate task types
                def get_task_side_effect(task_id, db):
                    task = MagicMock()
                    task.task_id = task_id
                    idx = [t.task_id for t in existing_tasks].index(task_id)
                    task.task_type = [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK][idx % 3]
                    return task
                mock_get_task.side_effect = get_task_side_effect

                with patch("validator.tournament.task_creator.add_tournament_tasks") as mock_add_tournament:
                    tasks = await _create_new_text_boss_round_tasks(tournament_id, round_id, mock_config)

        # Should return existing tasks without creating new ones
        assert len(tasks) == 6
        mock_add_tournament.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_single_new_text_task_instruct(self, mock_config, mock_instruct_task):
        """Test _create_single_new_text_task for InstructTextTask."""
        tournament_id = str(uuid4())
        round_id = str(uuid4())
        pair_id = f"{round_id}_pair_001"

        with patch("validator.tournament.task_creator.create_synthetic_instruct_text_task", return_value=mock_instruct_task):
            with patch("validator.tournament.task_creator.add_tournament_tasks") as mock_add_tournament:
                with patch("validator.tournament.task_creator.get_tournament_gpu_requirement", return_value="A100"):
                    task = await _create_single_new_text_task(
                        TaskType.INSTRUCTTEXTTASK, tournament_id, round_id, pair_id, mock_config,
                        ["model1"], ["dataset1"], ["dataset1"]
                    )

        assert task is not None
        assert task.task_type == TaskType.INSTRUCTTEXTTASK
        mock_add_tournament.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_single_new_text_task_dpo(self, mock_config, mock_dpo_task):
        """Test _create_single_new_text_task for DpoTask."""
        tournament_id = str(uuid4())
        round_id = str(uuid4())
        pair_id = f"{round_id}_pair_001"

        with patch("validator.tournament.task_creator.create_synthetic_dpo_task", return_value=mock_dpo_task):
            with patch("validator.tournament.task_creator.add_tournament_tasks") as mock_add_tournament:
                with patch("validator.tournament.task_creator.get_tournament_gpu_requirement", return_value="A100"):
                    task = await _create_single_new_text_task(
                        TaskType.DPOTASK, tournament_id, round_id, pair_id, mock_config,
                        ["model1"], ["dataset1"], ["dataset1"]
                    )

        assert task is not None
        assert task.task_type == TaskType.DPOTASK
        mock_add_tournament.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_single_new_text_task_grpo(self, mock_config, mock_grpo_task):
        """Test _create_single_new_text_task for GrpoTask."""
        tournament_id = str(uuid4())
        round_id = str(uuid4())
        pair_id = f"{round_id}_pair_001"

        with patch("validator.tournament.task_creator.create_synthetic_grpo_task", return_value=mock_grpo_task):
            with patch("validator.tournament.task_creator.add_tournament_tasks") as mock_add_tournament:
                with patch("validator.tournament.task_creator.get_tournament_gpu_requirement", return_value="A100"):
                    task = await _create_single_new_text_task(
                        TaskType.GRPOTASK, tournament_id, round_id, pair_id, mock_config,
                        ["model1"], ["dataset1"], ["dataset1"]
                    )

        assert task is not None
        assert task.task_type == TaskType.GRPOTASK
        mock_add_tournament.assert_called_once()


class TestNewImageBossRoundTasks:
    """Test new synthetic image task creation for boss rounds."""

    @pytest.mark.asyncio
    async def test_create_new_image_boss_round_tasks_creates_correct_count(self, mock_config, mock_image_task):
        """Test that _create_new_image_boss_round_tasks creates 6 image tasks."""
        tournament_id = str(uuid4())
        round_id = str(uuid4())

        with patch("validator.tournament.task_creator.get_tournament_tasks", return_value=[]):
            with patch("validator.tournament.task_creator._get_image_models", return_value=["model1", "model2"]):
                with patch("validator.tournament.task_creator._create_single_image_task_with_retry", return_value=mock_image_task):
                    with patch("validator.tournament.task_creator.add_tournament_tasks") as mock_add_tournament:
                        with patch("validator.tournament.task_creator.get_tournament_gpu_requirement", return_value="A100"):
                            tasks = await _create_new_image_boss_round_tasks(tournament_id, round_id, mock_config)

        # Should create 6 image tasks
        assert len(tasks) == t_cst.FINAL_ROUND_IMAGE_TASKS
        assert all(task.task_type == TaskType.IMAGETASK for task in tasks)

        # Verify tournament tasks were added
        assert mock_add_tournament.call_count == 6
        for call in mock_add_tournament.call_args_list:
            tournament_task = call[0][0][0]
            assert isinstance(tournament_task, TournamentTask)
            assert tournament_task.tournament_id == tournament_id
            assert tournament_task.round_id == round_id
            assert tournament_task.pair_id == f"{round_id}_pair_001"
            assert tournament_task.group_id is None

    @pytest.mark.asyncio
    async def test_create_new_image_boss_round_tasks_with_existing_tasks(self, mock_config, mock_image_task):
        """Test that _create_new_image_boss_round_tasks handles existing tasks correctly."""
        tournament_id = str(uuid4())
        round_id = str(uuid4())
        pair_id = f"{round_id}_pair_001"

        # Create 2 existing tournament tasks
        existing_tournament_task = TournamentTask(
            tournament_id=tournament_id,
            round_id=round_id,
            task_id=mock_image_task.task_id,
            group_id=None,
            pair_id=pair_id,
        )

        with patch("validator.tournament.task_creator.get_tournament_tasks", return_value=[existing_tournament_task]):
            with patch("validator.tournament.task_creator.task_sql.get_task", return_value=mock_image_task):
                with patch("validator.tournament.task_creator._get_image_models", return_value=["model1"]):
                    with patch("validator.tournament.task_creator._create_single_image_task_with_retry", return_value=mock_image_task):
                        with patch("validator.tournament.task_creator.add_tournament_tasks") as mock_add_tournament:
                            with patch("validator.tournament.task_creator.get_tournament_gpu_requirement", return_value="A100"):
                                tasks = await _create_new_image_boss_round_tasks(tournament_id, round_id, mock_config)

        # Should have 1 existing + 5 new tasks = 6 total
        assert len(tasks) == t_cst.FINAL_ROUND_IMAGE_TASKS
        # Should only add 5 new tournament tasks (1 already exists)
        assert mock_add_tournament.call_count == 5

    @pytest.mark.asyncio
    async def test_create_new_image_boss_round_tasks_skips_when_complete(self, mock_config, mock_image_task):
        """Test that _create_new_image_boss_round_tasks skips creation when 6 tasks already exist."""
        tournament_id = str(uuid4())
        round_id = str(uuid4())
        pair_id = f"{round_id}_pair_001"

        # Create 6 existing tournament tasks
        existing_tasks = [
            TournamentTask(
                tournament_id=tournament_id,
                round_id=round_id,
                task_id=uuid4(),
                group_id=None,
                pair_id=pair_id,
            ) for _ in range(6)
        ]

        with patch("validator.tournament.task_creator.get_tournament_tasks", return_value=existing_tasks):
            with patch("validator.tournament.task_creator.task_sql.get_task", return_value=mock_image_task):
                with patch("validator.tournament.task_creator.add_tournament_tasks") as mock_add_tournament:
                    tasks = await _create_new_image_boss_round_tasks(tournament_id, round_id, mock_config)

        # Should return existing tasks without creating new ones
        assert len(tasks) == 6
        mock_add_tournament.assert_not_called()


class TestBossRoundTaskCreationIntegration:
    """Integration tests for boss round task creation through main functions."""

    @pytest.mark.asyncio
    async def test_create_text_tournament_tasks_final_round(self, mock_config, mock_instruct_task, mock_dpo_task, mock_grpo_task):
        """Test create_text_tournament_tasks with is_final_round=True."""
        tournament_id = str(uuid4())
        round_id = str(uuid4())
        round_data = MagicMock()  # Not a GroupRound, so it will go to final round path

        with patch("validator.tournament.task_creator._create_new_text_boss_round_tasks") as mock_create_boss:
            mock_create_boss.return_value = [mock_instruct_task, mock_dpo_task, mock_grpo_task] * 2
            task_ids = await create_text_tournament_tasks(
                round_data, tournament_id, round_id, mock_config, is_final_round=True
            )

        mock_create_boss.assert_called_once_with(tournament_id, round_id, mock_config)
        assert len(task_ids) == 6
        assert all(isinstance(task_id, str) for task_id in task_ids)

    @pytest.mark.asyncio
    async def test_create_image_tournament_tasks_final_round(self, mock_config, mock_image_task):
        """Test create_image_tournament_tasks with is_final_round=True."""
        tournament_id = str(uuid4())
        round_id = str(uuid4())
        round_data = MagicMock()  # Not a GroupRound, so it will go to final round path

        with patch("validator.tournament.task_creator._create_new_image_boss_round_tasks") as mock_create_boss:
            mock_create_boss.return_value = [mock_image_task] * 6
            task_ids = await create_image_tournament_tasks(
                round_data, tournament_id, round_id, mock_config, is_final_round=True
            )

        mock_create_boss.assert_called_once_with(tournament_id, round_id, mock_config)
        assert len(task_ids) == 6
        assert all(isinstance(task_id, str) for task_id in task_ids)

