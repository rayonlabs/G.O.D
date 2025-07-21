import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.models.tournament_models import BossRoundTaskCompletion, BossRoundTaskPair, TaskScore, TournamentType
from core.models.utility_models import TaskType
from validator.core.weight_setting import (
    calculate_performance_difference,
    calculate_burn_proportion,
    calculate_weight_redistribution,
    get_active_tournament_burn_data,
    check_boss_round_synthetic_tasks_complete
)
import validator.core.constants as cts


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
    
    def test_calculate_weight_redistribution_no_burn(self):
        """Test weight redistribution with no performance difference"""
        tournament_weight, regular_weight, burn_weight = calculate_weight_redistribution(0.0)
        
        assert tournament_weight == cts.BASE_TOURNAMENT_WEIGHT  # 0.5
        assert regular_weight == cts.BASE_REGULAR_WEIGHT  # 0.25
        assert burn_weight == (1 - cts.BASE_REGULAR_WEIGHT - cts.BASE_TOURNAMENT_WEIGHT)  # 0.25
        assert abs(tournament_weight + regular_weight + burn_weight - 1.0) < 0.0001
    
    def test_calculate_weight_redistribution_with_burn(self):
        """Test weight redistribution with performance difference causing burn"""
        performance_diff = 0.1  # 10% performance difference
        tournament_weight, regular_weight, burn_weight = calculate_weight_redistribution(performance_diff)
        
        # Expected: 10% * 5.0 = 50% burn reduction
        # Tournament burn = 0.5 * 0.5 = 0.25
        # Tournament weight = 0.5 - 0.25 = 0.25
        # Regular weight = 0.25 + (0.25 / 2) = 0.375
        # Burn weight = 0.25 + (0.25 / 2) = 0.375
        
        assert tournament_weight == 0.25
        assert regular_weight == 0.375
        assert burn_weight == 0.375
        assert abs(tournament_weight + regular_weight + burn_weight - 1.0) < 0.0001
    
    def test_calculate_weight_redistribution_max_burn(self):
        """Test weight redistribution with maximum burn scenario"""
        performance_diff = 0.2  # 20% performance difference (will be capped)
        tournament_weight, regular_weight, burn_weight = calculate_weight_redistribution(performance_diff)
        
        # Expected: 20% * 5.0 = 100%, but capped at 90%
        # Tournament burn = 0.5 * 0.9 = 0.45
        # Tournament weight = 0.5 - 0.45 = 0.05
        # Regular weight = 0.25 + (0.45 / 2) = 0.475
        # Burn weight = 0.25 + (0.45 / 2) = 0.475
        
        assert abs(tournament_weight - 0.05) < 0.0001
        assert regular_weight == 0.475
        assert burn_weight == 0.475
        assert abs(tournament_weight + regular_weight + burn_weight - 1.0) < 0.0001
    
    @pytest.mark.asyncio
    async def test_check_boss_round_synthetic_tasks_complete_true(self, mock_psql_db):
        """Test boss round synthetic tasks completion check - completed"""
        from validator.db.sql.tournament_performance import get_boss_round_synthetic_task_completion
        
        mock_completion = BossRoundTaskCompletion(total_synth_tasks=5, completed_synth_tasks=5)
        
        with patch('validator.core.weight_setting.get_boss_round_synthetic_task_completion', return_value=mock_completion):
            result = await check_boss_round_synthetic_tasks_complete("test_tournament", mock_psql_db)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_check_boss_round_synthetic_tasks_complete_false(self, mock_psql_db):
        """Test boss round synthetic tasks completion check - incomplete"""
        from validator.db.sql.tournament_performance import get_boss_round_synthetic_task_completion
        
        mock_completion = BossRoundTaskCompletion(total_synth_tasks=5, completed_synth_tasks=3)
        
        with patch('validator.core.weight_setting.get_boss_round_synthetic_task_completion', return_value=mock_completion):
            result = await check_boss_round_synthetic_tasks_complete("test_tournament", mock_psql_db)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_check_boss_round_synthetic_tasks_complete_none(self, mock_psql_db):
        """Test boss round synthetic tasks completion check - no tasks"""
        from validator.db.sql.tournament_performance import get_boss_round_synthetic_task_completion
        
        mock_completion = BossRoundTaskCompletion(total_synth_tasks=0, completed_synth_tasks=0)
        
        with patch('validator.core.weight_setting.get_boss_round_synthetic_task_completion', return_value=mock_completion):
            result = await check_boss_round_synthetic_tasks_complete("test_tournament", mock_psql_db)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_calculate_performance_difference_no_tasks(self, mock_psql_db):
        """Test performance difference calculation with no task pairs"""
        with patch('validator.core.weight_setting.get_boss_round_winner_task_pairs', return_value=[]):
            result = await calculate_performance_difference("test_tournament", mock_psql_db)
            assert result == 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_performance_difference_grpo_task(self, mock_psql_db):
        """Test performance difference calculation for GRPO task (higher is better)"""
        task_pair = BossRoundTaskPair(
            tournament_task_id="tourn_task_1",
            synthetic_task_id="synth_task_1", 
            winner_hotkey="winner_hotkey",
            task_type=TaskType.GRPOTASK.value
        )
        
        tournament_scores = [
            TaskScore(hotkey="winner_hotkey", test_loss=0.8, synth_loss=0.7, quality_score=0.9)
        ]
        synthetic_scores = [
            TaskScore(hotkey="winner_hotkey", test_loss=0.6, synth_loss=0.5, quality_score=0.7)
        ]
        
        with patch('validator.core.weight_setting.get_boss_round_winner_task_pairs', return_value=[task_pair]):
            with patch('validator.core.weight_setting.get_task_scores_as_models', side_effect=[tournament_scores, synthetic_scores]):
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
            task_type=TaskType.DPOTASK.value
        )
        
        tournament_scores = [
            TaskScore(hotkey="winner_hotkey", test_loss=0.4, synth_loss=0.3, quality_score=0.9)
        ]
        synthetic_scores = [
            TaskScore(hotkey="winner_hotkey", test_loss=0.6, synth_loss=0.5, quality_score=0.7)
        ]
        
        with patch('validator.core.weight_setting.get_boss_round_winner_task_pairs', return_value=[task_pair]):
            with patch('validator.core.weight_setting.get_task_scores_as_models', side_effect=[tournament_scores, synthetic_scores]):
                result = await calculate_performance_difference("test_tournament", mock_psql_db)
                
                # For DPO: tournament_score=0.4, synthetic_score=0.6
                # Performance diff = (0.6 - 0.4) / 0.4 = 0.5
                expected = (0.6 - 0.4) / 0.4
                assert abs(result - expected) < 0.001
    
    @pytest.mark.asyncio
    async def test_get_active_tournament_burn_data_no_tournaments(self, mock_psql_db):
        """Test burn data calculation with no tournament data"""
        with patch('validator.core.weight_setting.get_latest_completed_tournament', return_value=None):
            tournament_weight, regular_weight, burn_weight = await get_active_tournament_burn_data(mock_psql_db)
            
            # Should burn entire tournament allocation
            assert tournament_weight == 0.0
            assert regular_weight == 0.5  # 0.25 + (0.5 / 2)
            assert burn_weight == 0.5  # 0.25 + (0.5 / 2)
    
    @pytest.mark.asyncio
    async def test_get_active_tournament_burn_data_with_completed_tournaments(self, mock_psql_db):
        """Test burn data calculation with completed tournaments"""
        mock_tournament = MagicMock()
        mock_tournament.tournament_id = "test_tournament"
        
        with patch('validator.core.weight_setting.get_latest_completed_tournament', return_value=mock_tournament):
            with patch('validator.core.weight_setting.check_boss_round_synthetic_tasks_complete', return_value=True):
                with patch('validator.core.weight_setting.calculate_performance_difference', return_value=0.1):
                    tournament_weight, regular_weight, burn_weight = await get_active_tournament_burn_data(mock_psql_db)
                    
                    # Should use performance difference of 0.1 for both text and image
                    # Weighted average = (0.1 * 0.6 + 0.1 * 0.4) / 1.0 = 0.1
                    # This gives 50% burn reduction -> 25% tournament burn
                    assert tournament_weight == 0.25
                    assert regular_weight == 0.375
                    assert burn_weight == 0.375
    
    @pytest.mark.asyncio
    async def test_get_active_tournament_burn_data_fallback_to_previous(self, mock_psql_db):
        """Test burn data calculation falling back to previous tournament"""
        mock_latest_tournament = MagicMock()
        mock_latest_tournament.tournament_id = "latest_tournament"
        
        with patch('validator.core.weight_setting.get_latest_completed_tournament', return_value=mock_latest_tournament):
            with patch('validator.core.weight_setting.check_boss_round_synthetic_tasks_complete', side_effect=[False, True, False, True]):
                with patch('validator.core.weight_setting.get_previous_completed_tournament', return_value="previous_tournament"):
                    with patch('validator.core.weight_setting.calculate_performance_difference', return_value=0.05):
                        tournament_weight, regular_weight, burn_weight = await get_active_tournament_burn_data(mock_psql_db)
                        
                        # Should use previous tournament data with 0.05 performance difference
                        # 5% * 5.0 = 25% burn reduction -> 12.5% tournament burn
                        expected_tournament_burn = 0.5 * 0.25
                        expected_tournament_weight = 0.5 - expected_tournament_burn
                        expected_regular_weight = 0.25 + (expected_tournament_burn / 2)
                        expected_burn_weight = 0.25 + (expected_tournament_burn / 2)
                        
                        assert abs(tournament_weight - expected_tournament_weight) < 0.001
                        assert abs(regular_weight - expected_regular_weight) < 0.001
                        assert abs(burn_weight - expected_burn_weight) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])