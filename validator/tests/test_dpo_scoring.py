import pytest
import numpy as np
import math
from unittest.mock import patch, MagicMock
from core.models.utility_models import TaskType
from validator.core.models import MinerResultsText
from validator.evaluation.scoring import (
    calculate_weighted_loss,
    calculate_miner_ranking_and_scores
)

def test_weighted_loss_for_different_scenarios():
    # We're testing the calculate_weighted_loss function directly
    # rather than calculate_test_synth_ratio_penalty which we removed
    
    # Standard non-DPO case (weighted average)
    from validator.core import constants as cts
    non_dpo_case = calculate_weighted_loss(3.0, 1.0, is_dpo=False)
    expected = 3.0 * cts.TEST_SCORE_WEIGHTING + 1.0 * (1 - cts.TEST_SCORE_WEIGHTING)
    assert abs(non_dpo_case - expected) < 1e-10
    
    # DPO case 1: test_loss > synth_loss (use test_loss)
    dpo_case1 = calculate_weighted_loss(5.0, 3.0, is_dpo=True)
    assert dpo_case1 == 5.0
    
    # DPO case 2: test_loss == synth_loss (use either)
    dpo_case2 = calculate_weighted_loss(2.0, 2.0, is_dpo=True)
    assert dpo_case2 == 2.0
    
    # DPO case 3: test_loss < synth_loss (use synth_loss)
    dpo_case3 = calculate_weighted_loss(1.0, 4.0, is_dpo=True)
    assert dpo_case3 == 4.0
    
    # Edge cases
    assert calculate_weighted_loss(0.0, 5.0, is_dpo=True) == 5.0
    assert calculate_weighted_loss(5.0, 0.0, is_dpo=True) == 5.0

# We replaced this with test_weighted_loss_for_different_scenarios

def test_edge_case_miner_scenarios():
    test_cases = [
        # Test miner with NaN test_loss (should be excluded from ranking)
        {"test_loss": float('nan'), "synth_loss": 1.0, "expected_score": 0.0, "expected_reason": "Invalid test loss"},
        
        # Test miner with non-finetuned submission (should be excluded)
        {"test_loss": 1.0, "synth_loss": 1.0, "is_finetune": False, "expected_score": 0.0, 
         "expected_reason": "Non-finetuned submission"},
        
        # Test miner with synth_loss=1000 (outside top-4)
        {"test_loss": 1.0, "synth_loss": 1000.0, "expected_score": 0.0, 
         "expected_reason": "Outside of top-4 test doesn't get scored."},
    ]
    
    for i, case in enumerate(test_cases):
        miner_results = [
            MinerResultsText(
                hotkey=f"test{i}",
                test_loss=case.get("test_loss", 1.0),
                synth_loss=case.get("synth_loss", 1.0),
                is_finetune=case.get("is_finetune", True),
                task_type=TaskType.DPOTASK
            )
        ]
        
        with patch('validator.evaluation.scoring.logger'):
            with patch('validator.evaluation.scoring._is_synth_loss_valid_for_group', return_value=True):
                results = calculate_miner_ranking_and_scores(miner_results)
                
        assert results[0].score == case["expected_score"]
        if "expected_reason" in case:
            assert case["expected_reason"] in (results[0].score_reason or "")

@patch('validator.evaluation.scoring.logger')
def test_multiple_miner_dpo_scenarios(mock_logger):
    # Test several realistic miner scoring scenarios
    scenarios = [
        # Scenario 1: Standard case with clear winner
        {
            "miners": [
                {"hotkey": "miner1", "test_loss": 2.0, "synth_loss": 2.2},  # Penalized to 2.2
                {"hotkey": "miner2", "test_loss": 1.5, "synth_loss": 1.5},  # No penalty, stays 1.5
                {"hotkey": "miner3", "test_loss": 3.0, "synth_loss": 2.8},  # No penalty, stays 3.0
                {"hotkey": "miner4", "test_loss": 0.5, "synth_loss": 5.0},  # Penalized to 5.0
            ],
            "expected_winner": "miner2",
            "expected_loser": "miner4"  # Due to suspicious test/synth ratio
        },
        
        # Scenario 2: Close competition between miners with similar adjusted scores
        {
            "miners": [
                {"hotkey": "miner1", "test_loss": 1.01, "synth_loss": 1.02},  # Penalized to 1.02
                {"hotkey": "miner2", "test_loss": 1.0, "synth_loss": 1.0},    # No penalty, stays 1.0 (winner)
                {"hotkey": "miner3", "test_loss": 0.9, "synth_loss": 1.8},    # Penalized to 1.8
                {"hotkey": "miner4", "test_loss": 0.8, "synth_loss": 4.0},    # Penalized to 4.0
            ],
            "expected_winner": "miner2",
            "expected_ranks": ["miner2", "miner1", "miner3", "miner4"]  # Expected ordering
        },
        
        # Scenario 3: All miners have penalty applied except one
        {
            "miners": [
                {"hotkey": "miner1", "test_loss": 2.0, "synth_loss": 4.0},   # Penalized to 4.0
                {"hotkey": "miner2", "test_loss": 1.0, "synth_loss": 5.0},   # Penalized to 5.0 
                {"hotkey": "miner3", "test_loss": 0.5, "synth_loss": 5.0},   # Penalized to 5.0
                {"hotkey": "miner4", "test_loss": 3.0, "synth_loss": 2.0},   # No penalty, stays at 3.0
            ],
            "expected_winner": "miner4",
            "expected_ranks": ["miner4", "miner1", "miner2", "miner3"]
        }
    ]
    
    for scenario_idx, scenario in enumerate(scenarios):
        miners = [
            MinerResultsText(
                hotkey=m["hotkey"],
                test_loss=m["test_loss"],
                synth_loss=m["synth_loss"],
                is_finetune=m.get("is_finetune", True),
                task_type=TaskType.DPOTASK
            ) for m in scenario["miners"]
        ]
        
        with patch('validator.evaluation.scoring._is_synth_loss_valid_for_group', return_value=True):
            results = calculate_miner_ranking_and_scores(miners)
        
        # Map results to score by hotkey
        scores = {r.hotkey: r.score for r in results}
        
        # Find the winner
        winner = None
        for r in results:
            if r.score > 0 and 'Ranked 1st' in (r.score_reason or ''):
                winner = r.hotkey
                break
        
        # Verify the expected winner
        assert winner == scenario["expected_winner"], \
            f"Scenario {scenario_idx+1}: Expected {scenario['expected_winner']} to win, got {winner}"
        
        # Verify expected loser if specified
        if "expected_loser" in scenario:
            assert scores[scenario["expected_loser"]] == 0, \
                f"Scenario {scenario_idx+1}: Expected {scenario['expected_loser']} to score 0"
        
        # Verify ranking order if specified
        if "expected_ranks" in scenario:
            # Calculate adjusted losses for verification
            adjusted_losses = {}
            for m in scenario["miners"]:
                if m["test_loss"] < m["synth_loss"]:
                    penalty = m["test_loss"] / m["synth_loss"]
                    adj_loss = m["test_loss"] * penalty
                else:
                    adj_loss = m["test_loss"]
                adjusted_losses[m["hotkey"]] = adj_loss
            
            # Sort by adjusted loss (lower is better)
            expected_order = sorted(adjusted_losses.items(), key=lambda x: x[1])
            expected_hotkeys = [item[0] for item in expected_order]
            
            assert expected_hotkeys == scenario["expected_ranks"], \
                f"Scenario {scenario_idx+1}: Expected rank order {scenario['expected_ranks']}, " \
                f"calculated {expected_hotkeys}"

@patch('validator.evaluation.scoring.logger')
def test_extreme_dpo_case(mock_logger):
    miner_results = [
        MinerResultsText(
            hotkey="miner1",
            test_loss=10.0,
            synth_loss=11.0,  # max(10.0, 11.0) = 11.0
            is_finetune=True,
            task_type=TaskType.DPOTASK
        ),
        MinerResultsText(
            hotkey="miner2",
            test_loss=1.0,
            synth_loss=1.0,   # max(1.0, 1.0) = 1.0
            is_finetune=True,
            task_type=TaskType.DPOTASK
        ),
        MinerResultsText(
            hotkey="miner3",
            test_loss=0.1,
            synth_loss=10.0,  # max(0.1, 10.0) = 10.0
            is_finetune=True, 
            task_type=TaskType.DPOTASK
        ),
        MinerResultsText(
            hotkey="miner4",
            test_loss=5.0,
            synth_loss=5.5,   # max(5.0, 5.5) = 5.5
            is_finetune=True,
            task_type=TaskType.DPOTASK
        )
    ]
    
    with patch('validator.evaluation.scoring._is_synth_loss_valid_for_group', return_value=True):
        scored_results = calculate_miner_ranking_and_scores(miner_results)
    
    winner = None
    scores = {}
    for result in scored_results:
        scores[result.hotkey] = result.score
        if result.score > 0 and 'Ranked 1st' in (result.score_reason or ''):
            winner = result.hotkey
    
    # Miner2 should win with max(test, synth) approach
    assert winner == "miner2", f"Expected miner2 to win with lowest adjusted loss, got {winner}"
    
    # The winner should have higher score than others
    for miner_hotkey in ["miner1", "miner3", "miner4"]:
        assert scores["miner2"] > scores.get(miner_hotkey, 0)
    
    mock_logger.info.assert_any_call(f"DPO using test_loss: test=1.000000 >= synth=1.000000")

if __name__ == "__main__":
    test_weighted_loss_for_different_scenarios()
    test_edge_case_miner_scenarios()
    test_multiple_miner_dpo_scenarios()
    test_extreme_dpo_case()
    print("All tests passed!")