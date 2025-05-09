import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from core.models.utility_models import TaskType
from validator.core.models import MinerResultsText
from validator.evaluation.scoring import (
    calculate_test_synth_ratio_penalty,
    calculate_weighted_loss,
    calculate_miner_ranking_and_scores
)

def test_calculate_test_synth_ratio_penalty():
    # Test when test_loss >= synth_loss (no penalty)
    assert calculate_test_synth_ratio_penalty(10.0, 5.0) == 1.0
    assert calculate_test_synth_ratio_penalty(5.0, 5.0) == 1.0
    
    # Test when test_loss < synth_loss (apply ratio penalty)
    assert abs(calculate_test_synth_ratio_penalty(2.0, 10.0) - 0.2) < 1e-10
    assert abs(calculate_test_synth_ratio_penalty(0.01, 0.1) - 0.1) < 1e-10
    
    # Test edge cases
    assert calculate_test_synth_ratio_penalty(0.0, 1.0) < 1.0  # Should handle zeros gracefully
    assert calculate_test_synth_ratio_penalty(1e-7, 1.0) < 1.0  # Should handle very small values
    
    # Values very close to each other should not get penalty
    assert calculate_test_synth_ratio_penalty(1.0000001, 1.0) == 1.0

def test_calculate_weighted_loss():
    # For non-DPO tasks
    non_dpo_loss = calculate_weighted_loss(10.0, 5.0, is_dpo=False)
    # Should use weighted average
    from validator.core import constants as cts
    expected = cts.TEST_SCORE_WEIGHTING * 10.0 + (1 - cts.TEST_SCORE_WEIGHTING) * 5.0
    assert non_dpo_loss == pytest.approx(expected)
    
    # For DPO tasks with test_loss >= synth_loss (no penalty)
    dpo_no_penalty = calculate_weighted_loss(10.0, 5.0, is_dpo=True)
    assert dpo_no_penalty == 10.0
    
    # For DPO tasks with test_loss < synth_loss (apply ratio penalty)
    dpo_with_penalty = calculate_weighted_loss(2.0, 10.0, is_dpo=True)
    assert dpo_with_penalty == 0.4  # 2.0 * (2.0/10.0)

@patch('validator.evaluation.scoring.logger')
def test_miner_ranking_with_dpo_penalty(mock_logger):
    # Create mock DPO task results
    miner_results = [
        MinerResultsText(
            hotkey="miner1",
            test_loss=0.5,      # Better test loss but much worse than synth
            synth_loss=2.0,
            is_finetune=True,
            task_type=TaskType.DPOTASK
        ),
        MinerResultsText(
            hotkey="miner2",
            test_loss=0.6,      # Slightly worse test loss but closer to synth
            synth_loss=0.7,
            is_finetune=True,
            task_type=TaskType.DPOTASK
        ),
        MinerResultsText(
            hotkey="miner3",
            test_loss=0.4,      # Best test loss but warning sign: very low vs synth
            synth_loss=2.5,
            is_finetune=True,
            task_type=TaskType.DPOTASK
        ),
        MinerResultsText(
            hotkey="miner4",
            test_loss=1.0,      # Worst test loss but synth is very close
            synth_loss=1.1,
            is_finetune=True,
            task_type=TaskType.DPOTASK
        )
    ]
    
    # Without our DPO penalty, ranking should be: miner3, miner1, miner2, miner4
    # With our penalty, ranking should be: miner2, miner4, miner1, miner3
    
    # Mock the _is_synth_loss_valid_for_group to return True
    with patch('validator.evaluation.scoring._is_synth_loss_valid_for_group', return_value=True):
        scored_results = calculate_miner_ranking_and_scores(miner_results)
    
    # Find winner and extract scores
    winner = None
    scores = {}
    for result in scored_results:
        scores[result.hotkey] = result.score
        if result.score > 0 and 'Ranked 1st' in (result.score_reason or ''):
            winner = result.hotkey
    
    # The winner should be miner2 with the penalty applied
    assert winner == "miner2", f"Expected miner2 to win but got {winner}"
    
    # Verify miner3 (which would win without penalty) is not the winner
    assert scores["miner3"] < scores["miner2"], "miner3 should be penalized due to test_loss << synth_loss"
    
    # Check for expected log message about DPO tasks
    mock_logger.info.assert_any_call("Processing DPO task with ratio-based penalty")

if __name__ == "__main__":
    test_calculate_test_synth_ratio_penalty()
    test_calculate_weighted_loss()
    test_miner_ranking_with_dpo_penalty()
    print("All tests passed!")