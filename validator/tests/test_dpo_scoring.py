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
    assert dpo_no_penalty == pytest.approx(10.0)
    
    # For DPO tasks with test_loss < synth_loss (apply ratio penalty)
    dpo_with_penalty = calculate_weighted_loss(2.0, 10.0, is_dpo=True)
    assert abs(dpo_with_penalty - 0.4) < 1e-10  # 2.0 * (2.0/10.0)

@patch('validator.evaluation.scoring.logger')
def test_miner_ranking_with_dpo_penalty(mock_logger):
    # Let's print debug output to understand what's happening
    print("\n===== DPO Penalty Test =====")
    
    # Create mock DPO task results with EXTREMELY different values to isolate the issue
    # We'll make the differences so large that there's no way they could be sorted incorrectly
    miner_results = [
        MinerResultsText(
            hotkey="miner1",
            test_loss=10.0,     # Very high test loss
            synth_loss=11.0,    # Close to test
            is_finetune=True,
            task_type=TaskType.DPOTASK
        ),
        MinerResultsText(
            hotkey="miner2",
            test_loss=1.0,      # Good test loss and same as synth (no penalty)
            synth_loss=1.0,     # Equal to test (NO penalty)
            is_finetune=True,
            task_type=TaskType.DPOTASK
        ),
        MinerResultsText(
            hotkey="miner3",
            test_loss=0.1,      # Extremely good test loss but vs huge synth (severe penalty)
            synth_loss=10.0,    # 100x higher than test (should become 0.01 after penalty)
            is_finetune=True, 
            task_type=TaskType.DPOTASK
        ),
        MinerResultsText(
            hotkey="miner4",
            test_loss=5.0,      # Poor test loss
            synth_loss=5.5,     # Close to test (small penalty)
            is_finetune=True,
            task_type=TaskType.DPOTASK
        )
    ]
    
    # Expected adjusted scores (after penalty):
    # miner1: 0.5 * (0.5/1.5) = 0.5 * 0.33 = 0.167
    # miner2: 0.6 * (0.6/0.65) = 0.6 * 0.92 = 0.552
    # miner3: 0.2 * (0.2/2.0) = 0.2 * 0.1 = 0.02
    # miner4: 1.0 * (1.0/1.1) = 1.0 * 0.91 = 0.91
    
    # Calculate expected scores with penalty
    print("Expected scores after penalty:")
    for miner in miner_results:
        ratio = min(1.0, miner.test_loss / miner.synth_loss) if miner.test_loss < miner.synth_loss else 1.0
        adjusted = miner.test_loss * ratio
        print(f"{miner.hotkey}: test={miner.test_loss:.3f}, synth={miner.synth_loss:.3f}, " +
              f"ratio={ratio:.3f}, adjusted={adjusted:.3f}")
        
    # Ranking with penalty should be: miner2, miner4, miner1, miner3
    
    # Mock just the validation function - we want to see the real behavior
    with patch('validator.evaluation.scoring._is_synth_loss_valid_for_group', return_value=True):
        # Directly patch the ranking sort to see the values being used
        original_sort = list.sort
        ranked_pairs = []
        
        def capture_sort(self, *args, **kwargs):
            nonlocal ranked_pairs
            if len(self) > 0 and isinstance(self[0], tuple) and len(self[0]) == 2:
                # This looks like our ranked_results list
                if hasattr(self[0][0], 'hotkey'):
                    ranked_pairs = [(r[0].hotkey, r[1]) for r in self]
                    print("\nValues being sorted for ranking:")
                    for miner, val in ranked_pairs:
                        print(f"{miner}: {val:.6f}")
            return original_sort(self, *args, **kwargs)
        
        with patch('builtins.list.sort', capture_sort):
            scored_results = calculate_miner_ranking_and_scores(miner_results)
    
    # Removed weighted_loss_values reference that was causing an error
    
    # Find winner and extract scores
    winner = None
    scores = {}
    print("\nActual scores from function:")
    for result in scored_results:
        scores[result.hotkey] = result.score
        print(f"{result.hotkey}: score={result.score}, reason={result.score_reason}")
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