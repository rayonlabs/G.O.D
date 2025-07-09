import pytest
from unittest.mock import AsyncMock, MagicMock
from core.models.utility_models import TaskType
from core.models.tournament_models import TournamentType, TournamentData, TournamentTaskScore, TournamentRoundResult, TournamentResults
from validator.evaluation.tournament_scoring import (
    calculate_final_round_winner,
    calculate_tournament_type_scores,
    get_tournament_scores,
    tournament_scores_to_weights,
    get_tournament_weights
)


class TestCalculateFinalRoundWinner:
    def test_insufficient_participants(self):
        task = TournamentTaskScore(
            task_id="task1",
            group_id=None,
            pair_id="pair1",
            winner="hotkey1",
            participant_scores=[{"hotkey": "hotkey1", "test_loss": 0.5, "synth_loss": 0.6}]
        )
        result = calculate_final_round_winner(task, "prev_winner", TaskType.INSTRUCTTEXTTASK)
        assert result is None
    
    def test_missing_prev_winner(self):
        task = TournamentTaskScore(
            task_id="task1",
            group_id=None,
            pair_id="pair1",
            winner="hotkey1",
            participant_scores=[
                {"hotkey": "hotkey1", "test_loss": 0.5, "synth_loss": 0.6},
                {"hotkey": "hotkey2", "test_loss": 0.7, "synth_loss": 0.8}
            ]
        )
        result = calculate_final_round_winner(task, "prev_winner", TaskType.INSTRUCTTEXTTASK)
        assert result is None
    
    def test_instruct_task_contender_wins_by_5_percent(self):
        task = TournamentTaskScore(
            task_id="task1",
            group_id=None,
            pair_id="pair1",
            winner="contender",
            participant_scores=[
                {"hotkey": "prev_winner", "test_loss": 1.0, "synth_loss": 0.9},
                {"hotkey": "contender", "test_loss": 0.8, "synth_loss": 0.7}
            ]
        )
        # Contender's max loss (0.8) * 1.05 = 0.84 < prev_winner's max loss (1.0)
        result = calculate_final_round_winner(task, "prev_winner", TaskType.INSTRUCTTEXTTASK)
        assert result == "contender"
    
    def test_instruct_task_prev_winner_wins_margin_too_small(self):
        task = TournamentTaskScore(
            task_id="task1",
            group_id=None,
            pair_id="pair1",
            winner="prev_winner",
            participant_scores=[
                {"hotkey": "prev_winner", "test_loss": 0.5, "synth_loss": 0.6},
                {"hotkey": "contender", "test_loss": 0.8, "synth_loss": 0.7}
            ]
        )
        # Contender's max loss (0.8) * 1.05 = 0.84 > prev_winner's max loss (0.6)
        result = calculate_final_round_winner(task, "prev_winner", TaskType.INSTRUCTTEXTTASK)
        assert result == "prev_winner"
    
    def test_grpo_task_contender_wins_by_5_percent(self):
        task = TournamentTaskScore(
            task_id="task1",
            group_id=None,
            pair_id="pair1",
            winner="contender",
            participant_scores=[
                {"hotkey": "prev_winner", "test_loss": 0.8, "synth_loss": 0.7},
                {"hotkey": "contender", "test_loss": 1.0, "synth_loss": 0.9}
            ]
        )
        # For GRPO, higher is better: contender's max (1.0) > prev_winner's max (0.8) * 1.05 = 0.84
        result = calculate_final_round_winner(task, "prev_winner", TaskType.GRPOTASK)
        assert result == "contender"
    
    def test_grpo_task_prev_winner_wins_margin_too_small(self):
        task = TournamentTaskScore(
            task_id="task1",
            group_id=None,
            pair_id="pair1",
            winner="prev_winner",
            participant_scores=[
                {"hotkey": "prev_winner", "test_loss": 1.0, "synth_loss": 0.9},
                {"hotkey": "contender", "test_loss": 0.8, "synth_loss": 0.7}
            ]
        )
        # For GRPO, higher is better: contender's max (0.8) <= prev_winner's max (1.0) * 1.05 = 1.05
        result = calculate_final_round_winner(task, "prev_winner", TaskType.GRPOTASK)
        assert result == "prev_winner"


class TestTournamentScoresToWeights:
    def test_empty_scores(self):
        result = tournament_scores_to_weights({})
        assert result == {}
    
    def test_all_zero_scores_excluded(self):
        scores = {"hotkey1": 0, "hotkey2": 0, "hotkey3": 0}
        result = tournament_scores_to_weights(scores)
        assert result == {}
    
    def test_zero_scores_excluded_from_ranking(self):
        scores = {"hotkey1": 10.0, "hotkey2": 0.0, "hotkey3": 5.0}
        result = tournament_scores_to_weights(scores)
        # hotkey2 should be excluded, only hotkey1 and hotkey3 ranked
        assert "hotkey2" not in result
        assert len(result) == 2
        assert result["hotkey1"] > result["hotkey3"]
    
    def test_tied_participants_get_same_weight(self):
        scores = {"hotkey1": 10.0, "hotkey2": 10.0, "hotkey3": 5.0}
        result = tournament_scores_to_weights(scores)
        # Both tied participants should get same weight
        assert result["hotkey1"] == result["hotkey2"]
        assert result["hotkey1"] > result["hotkey3"]


class TestCalculateTournamentTypeScores:
    @pytest.mark.asyncio
    async def test_no_tournament_found(self):
        mock_db = MagicMock()
        
        with pytest.mock.patch('validator.evaluation.tournament_scoring.get_latest_completed_tournament', return_value=None):
            result = await calculate_tournament_type_scores(TournamentType.TEXT, mock_db)
            assert result == {}
    
    @pytest.mark.asyncio
    async def test_excludes_prev_winner_from_regular_scoring(self):
        mock_db = MagicMock()
        
        mock_tournament = TournamentData(
            tournament_id="tourn_123",
            tournament_type=TournamentType.TEXT,
            status="completed",
            base_winner_hotkey="prev_winner",
            winner_hotkey="winner"
        )
        
        mock_results = TournamentResults(
            tournament_id="tourn_123",
            rounds=[
                TournamentRoundResult(
                    round_id="round_1",
                    round_number=1,
                    round_type="GROUP",
                    is_final_round=False,
                    tasks=[
                        TournamentTaskScore(
                            task_id="task1",
                            group_id="group1",
                            pair_id=None,
                            winner="prev_winner",  # Should be excluded
                            participant_scores=[]
                        ),
                        TournamentTaskScore(
                            task_id="task2",
                            group_id="group1",
                            pair_id=None,
                            winner="hotkey2",  # Should be included
                            participant_scores=[]
                        )
                    ]
                )
            ]
        )
        
        with pytest.mock.patch('validator.evaluation.tournament_scoring.get_latest_completed_tournament', return_value=mock_tournament), \
             pytest.mock.patch('validator.evaluation.tournament_scoring.get_tournament_full_results', return_value=mock_results):
            
            result = await calculate_tournament_type_scores(TournamentType.TEXT, mock_db)
            
            # Only hotkey2 should be included, prev_winner excluded
            assert "prev_winner" not in result
            assert result == {"hotkey2": 0.6}  # round 1 * text weight 0.6
    
    @pytest.mark.asyncio
    async def test_scores_weighted_by_round_number(self):
        mock_db = MagicMock()
        
        mock_tournament = TournamentData(
            tournament_id="tourn_123",
            tournament_type=TournamentType.TEXT,
            status="completed",
            base_winner_hotkey="prev_winner",
            winner_hotkey="winner"
        )
        
        mock_results = TournamentResults(
            tournament_id="tourn_123",
            rounds=[
                TournamentRoundResult(
                    round_id="round_1",
                    round_number=1,
                    round_type="GROUP",
                    is_final_round=False,
                    tasks=[
                        TournamentTaskScore(
                            task_id="task1",
                            group_id="group1",
                            pair_id=None,
                            winner="hotkey1",
                            participant_scores=[]
                        )
                    ]
                ),
                TournamentRoundResult(
                    round_id="round_2",
                    round_number=2,
                    round_type="KNOCKOUT",
                    is_final_round=False,
                    tasks=[
                        TournamentTaskScore(
                            task_id="task2",
                            group_id=None,
                            pair_id="pair1",
                            winner="hotkey1",
                            participant_scores=[]
                        )
                    ]
                )
            ]
        )
        
        with pytest.mock.patch('validator.evaluation.tournament_scoring.get_latest_completed_tournament', return_value=mock_tournament), \
             pytest.mock.patch('validator.evaluation.tournament_scoring.get_tournament_full_results', return_value=mock_results):
            
            result = await calculate_tournament_type_scores(TournamentType.TEXT, mock_db)
            
            # hotkey1 should have (1 + 2) * 0.6 = 1.8 (round 1 + round 2, text weight = 0.6)
            assert result == {"hotkey1": 1.8}


class TestGetTournamentScores:
    @pytest.mark.asyncio
    async def test_combines_text_and_image_scores(self):
        mock_db = MagicMock()
        
        async def mock_calculate_scores(tournament_type, db):
            if tournament_type == TournamentType.TEXT:
                return {"hotkey1": 1.8, "hotkey2": 0.6}
            elif tournament_type == TournamentType.IMAGE:
                return {"hotkey1": 0.8, "hotkey3": 0.4}
            return {}
        
        with pytest.mock.patch('validator.evaluation.tournament_scoring.calculate_tournament_type_scores', side_effect=mock_calculate_scores):
            result = await get_tournament_scores(mock_db)
            
            # hotkey1 should have 1.8 + 0.8 = 2.6 (text + image)
            # hotkey2 should have 0.6 (text only)
            # hotkey3 should have 0.4 (image only)
            assert result == {"hotkey1": 2.6, "hotkey2": 0.6, "hotkey3": 0.4}


class TestGetTournamentWeights:
    @pytest.mark.asyncio
    async def test_end_to_end_conversion(self):
        mock_db = MagicMock()
        
        mock_scores = {"hotkey1": 10.0, "hotkey2": 5.0, "hotkey3": 0.0}
        
        with pytest.mock.patch('validator.evaluation.tournament_scoring.get_tournament_scores', return_value=mock_scores):
            result = await get_tournament_weights(mock_db)
            
            # Should exclude zero scores and convert to weights
            assert "hotkey3" not in result  # Zero score should be excluded
            assert len(result) == 2
            assert result["hotkey1"] > result["hotkey2"]  # Higher score gets higher weight


if __name__ == "__main__":
    pytest.main([__file__, "-v"])