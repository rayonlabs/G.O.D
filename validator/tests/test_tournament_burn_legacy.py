from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

import validator.core.constants as cts
from core.models.tournament_models import BossRoundTaskCompletion
from core.models.tournament_models import BossRoundTaskPair
from core.models.tournament_models import TaskScore
from core.models.utility_models import TaskType
from validator.core.weight_setting import calculate_burn_proportion
from validator.core.weight_setting import calculate_performance_difference
from validator.core.weight_setting import check_boss_round_synthetic_tasks_complete


class TestTournamentBurn:
    @pytest.fixture
    def mock_psql_db(self):
        return AsyncMock()

    def test_calculate_burn_proportion_zero_performance(self):
        """Test burn proportion calculation with zero performance difference"""
        result = calculate_burn_proportion(0.0)
        assert result == 0.0

    def test_calculate_burn_proportion_negative_performance(self):
        """Test burn proportion calculation with negative performance difference"""
        result = calculate_burn_proportion(-0.1)
        assert result == 0.0

    def test_calculate_burn_proportion_normal_performance(self):
        """Test burn proportion calculation with normal performance difference"""
        result = calculate_burn_proportion(0.1)  # 10% performance difference
        expected = 0.1 * cts.BURN_REDUCTION_RATE  # 10% * 5.0 = 0.5 (50% burn reduction)
        assert result == expected

    def test_calculate_burn_proportion_max_capped(self):
        """Test burn proportion calculation hits maximum cap"""
        result = calculate_burn_proportion(0.5)  # 50% performance difference
        assert result == cts.MAX_BURN_REDUCTION  # Should be capped at 0.9

    @pytest.mark.asyncio
    async def test_check_boss_round_synthetic_tasks_complete_true(self, mock_psql_db):
        """Test boss round synthetic tasks completion check - completed"""
        mock_completion = BossRoundTaskCompletion(total_synth_tasks=5, completed_synth_tasks=5)
        with patch("validator.core.weight_setting.get_boss_round_synthetic_task_completion", return_value=mock_completion):
            result = await check_boss_round_synthetic_tasks_complete("test_tournament", mock_psql_db)
            assert result is True

    @pytest.mark.asyncio
    async def test_check_boss_round_synthetic_tasks_complete_false(self, mock_psql_db):
        """Test boss round synthetic tasks completion check - incomplete"""
        mock_completion = BossRoundTaskCompletion(total_synth_tasks=5, completed_synth_tasks=3)
        with patch("validator.core.weight_setting.get_boss_round_synthetic_task_completion", return_value=mock_completion):
            result = await check_boss_round_synthetic_tasks_complete("test_tournament", mock_psql_db)
            assert result is False

    @pytest.mark.asyncio
    async def test_check_boss_round_synthetic_tasks_complete_none(self, mock_psql_db):
        """Test boss round synthetic tasks completion check - no tasks"""
        mock_completion = BossRoundTaskCompletion(total_synth_tasks=0, completed_synth_tasks=0)
        with patch("validator.core.weight_setting.get_boss_round_synthetic_task_completion", return_value=mock_completion):
            result = await check_boss_round_synthetic_tasks_complete("test_tournament", mock_psql_db)
            assert result is False

    @pytest.mark.asyncio
    async def test_calculate_performance_difference_no_tasks(self, mock_psql_db):
        """Test performance difference calculation with no task pairs"""
        with patch("validator.core.weight_setting.get_boss_round_winner_task_pairs", return_value=[]):
            result = await calculate_performance_difference("test_tournament", mock_psql_db)
            assert result == 0.0

    @pytest.mark.asyncio
    async def test_calculate_performance_difference_grpo_task(self, mock_psql_db):
        """Test performance difference calculation for GRPO task (higher is better)"""
        task_pair = BossRoundTaskPair(
            tournament_task_id="tourn_task_1",
            synthetic_task_id="synth_task_1",
            winner_hotkey="winner_hotkey",
            task_type=TaskType.GRPOTASK.value,
        )

        tournament_scores = [TaskScore(hotkey="winner_hotkey", test_loss=0.8, synth_loss=0.7, quality_score=0.9)]
        synthetic_scores = [TaskScore(hotkey="winner_hotkey", test_loss=0.6, synth_loss=0.5, quality_score=0.7)]

        with patch("validator.core.weight_setting.get_boss_round_winner_task_pairs", return_value=[task_pair]):
            with patch(
                "validator.core.weight_setting.get_task_scores_as_models", side_effect=[tournament_scores, synthetic_scores]
            ):
                result = await calculate_performance_difference("test_tournament", mock_psql_db)

                # For GRPO: tournament_score=0.8, synthetic_score=0.6
                # Performance diff = (0.8 - 0.6) / 0.6 = 0.333...
                expected = (0.8 - 0.6) / 0.6
                assert abs(result - expected) < 0.001

    @pytest.mark.asyncio
    async def test_calculate_performance_difference_non_grpo_task(self, mock_psql_db):
        """Test performance difference calculation for non-GRPO task (lower is better)"""
        task_pair = BossRoundTaskPair(
            tournament_task_id="tourn_task_1",
            synthetic_task_id="synth_task_1",
            winner_hotkey="winner_hotkey",
            task_type=TaskType.DPOTASK.value,
        )

        tournament_scores = [TaskScore(hotkey="winner_hotkey", test_loss=0.4, synth_loss=0.3, quality_score=0.9)]
        synthetic_scores = [TaskScore(hotkey="winner_hotkey", test_loss=0.6, synth_loss=0.5, quality_score=0.7)]

        with patch("validator.core.weight_setting.get_boss_round_winner_task_pairs", return_value=[task_pair]):
            with patch(
                "validator.core.weight_setting.get_task_scores_as_models", side_effect=[tournament_scores, synthetic_scores]
            ):
                result = await calculate_performance_difference("test_tournament", mock_psql_db)

                # For DPO: tournament_score=0.4, synthetic_score=0.6
                # Performance diff = (0.6 - 0.4) / 0.4 = 0.5
                expected = (0.6 - 0.4) / 0.4
                assert abs(result - expected) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
