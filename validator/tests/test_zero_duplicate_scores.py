import math  # For math.isnan

import numpy as np

import validator.core.constants as cts
from core.models.utility_models import TaskType  # Needed for MinerResultsText
from validator.core.models import MinerResultsText
from validator.evaluation.scoring import zero_duplicate_scores


# --- Mocking logger and LogContext for test simplicity ---
class MockLogger:
    def info(self, msg):
        # print(f"INFO: {msg}")
        pass

    def warning(self, msg):
        # print(f"WARNING: {msg}")
        pass

    def error(self, msg):
        # print(f"ERROR: {msg}")
        pass


logger_instance = MockLogger()


class LogContext:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Patching the logger and LogContext in the imported scoring module
import validator.evaluation.scoring as scoring_module


scoring_module.logger = logger_instance
scoring_module.LogContext = LogContext

# --- Test Cases ---


def test_no_duplicates():
    print("\n--- Test Case 1: No duplicates ---")
    miners_data = [
        {
            "hotkey": "M1",
            "test_loss": 0.1,
            "synth_loss": 0.1,
            "is_finetune": True,
            "score": 1.0,
            "score_reason": "Initial",
            "task_type": TaskType.INSTRUCTTEXTTASK,
        },
        {
            "hotkey": "M2",
            "test_loss": 0.2,
            "synth_loss": 0.2,
            "is_finetune": True,
            "score": 0.9,
            "score_reason": "Valid",
            "task_type": TaskType.INSTRUCTTEXTTASK,
        },
    ]
    miners = [MinerResultsText(**data) for data in miners_data]
    keep_submission = {"M1": True, "M2": True}

    results = zero_duplicate_scores(miners, keep_submission)
    for r in results:
        print(r)

    m1_res = next(r for r in results if r.hotkey == "M1")
    m2_res = next(r for r in results if r.hotkey == "M2")

    assert m1_res.score == 1.0
    assert m1_res.score_reason == "Initial"
    assert m1_res.test_loss == 0.1
    assert m1_res.synth_loss == 0.1
    assert m1_res.is_finetune is True

    assert m2_res.score == 0.9
    assert m2_res.score_reason == "Valid"
    assert m2_res.test_loss == 0.2
    assert m2_res.synth_loss == 0.2
    assert m2_res.is_finetune is True
    print("Test Case 1 Passed!")


def test_one_duplicate_others_valid():
    print("\n--- Test Case 2: One duplicate, other valid submissions exist ---")
    miners_data = [
        {
            "hotkey": "M1",
            "test_loss": 0.1,
            "synth_loss": 0.1,
            "is_finetune": True,
            "score": 1.0,
            "task_type": TaskType.INSTRUCTTEXTTASK,
        },  # Kept, contributes to remaining_valid_count
        {
            "hotkey": "M2",
            "test_loss": 0.2,
            "synth_loss": 0.2,
            "is_finetune": True,
            "score": 0.9,
            "task_type": TaskType.INSTRUCTTEXTTASK,
        },  # Duplicate
        {
            "hotkey": "M3",
            "test_loss": 0.3,
            "synth_loss": 0.3,
            "is_finetune": True,
            "score": 0.8,
            "task_type": TaskType.INSTRUCTTEXTTASK,
        },  # Kept, contributes to remaining_valid_count
    ]
    miners = [MinerResultsText(**data) for data in miners_data]
    keep_submission = {"M1": True, "M2": False, "M3": True}
    # remaining_valid_count = 2 (M1, M3)

    results = zero_duplicate_scores(miners, keep_submission)
    for r in results:
        print(r)

    m1_res = next(r for r in results if r.hotkey == "M1")
    m2_res = next(r for r in results if r.hotkey == "M2")
    m3_res = next(r for r in results if r.hotkey == "M3")

    # M1 should be unchanged
    assert m1_res.score == 1.0
    assert m1_res.test_loss == 0.1
    assert m1_res.is_finetune is True

    # M2 is the duplicate
    assert m2_res.score == cts.SCORE_PENALTY
    assert m2_res.score_reason == "Duplicated submission"
    assert math.isnan(m2_res.test_loss)
    assert math.isnan(m2_res.synth_loss)
    assert m2_res.is_finetune is False

    # M3 should be unchanged
    assert m3_res.score == 0.8
    assert m3_res.test_loss == 0.3
    assert m3_res.is_finetune is True
    print("Test Case 2 Passed!")


def test_one_duplicate_no_other_valid_kept_is_not_finetune():
    print("\n--- Test Case 3: One duplicate, no other *valid* submissions (kept one is not finetune) ---")
    miners_data = [
        {
            "hotkey": "M1",
            "test_loss": 0.1,
            "synth_loss": 0.1,
            "is_finetune": False,
            "score": 0.5,
            "task_type": TaskType.INSTRUCTTEXTTASK,
        },  # Kept, but not finetune
        {
            "hotkey": "M2",
            "test_loss": 0.2,
            "synth_loss": 0.2,
            "is_finetune": True,
            "score": 0.9,
            "task_type": TaskType.INSTRUCTTEXTTASK,
        },  # Duplicate
    ]
    miners = [MinerResultsText(**data) for data in miners_data]
    keep_submission = {"M1": True, "M2": False}
    # remaining_valid_count = 0 (M1 is not finetune, M2 is not kept)

    results = zero_duplicate_scores(miners, keep_submission)
    for r in results:
        print(r)

    m1_res = next(r for r in results if r.hotkey == "M1")
    m2_res = next(r for r in results if r.hotkey == "M2")

    # M1 should be unchanged
    assert m1_res.score == 0.5
    assert m1_res.test_loss == 0.1
    assert m1_res.is_finetune is False

    # M2 is the duplicate, no other valid submissions remain
    assert m2_res.score == 0.0
    assert m2_res.score_reason == "Duplicated submission"
    assert math.isnan(m2_res.test_loss)
    assert math.isnan(m2_res.synth_loss)
    assert m2_res.is_finetune is False
    print("Test Case 3 Passed!")


def test_one_duplicate_no_other_valid_all_duplicates():
    print("\n--- Test Case 4: One duplicate, no other valid submissions (all are duplicates or not kept effectively) ---")
    miners_data = [
        {
            "hotkey": "M1",
            "test_loss": 0.1,
            "synth_loss": 0.1,
            "is_finetune": True,
            "score": 1.0,
            "task_type": TaskType.INSTRUCTTEXTTASK,
        },  # Duplicate
        {
            "hotkey": "M2",
            "test_loss": 0.2,
            "synth_loss": 0.2,
            "is_finetune": True,
            "score": 0.9,
            "task_type": TaskType.INSTRUCTTEXTTASK,
        },  # Duplicate
    ]
    miners = [MinerResultsText(**data) for data in miners_data]
    keep_submission = {"M1": False, "M2": False}
    # remaining_valid_count = 0

    results = zero_duplicate_scores(miners, keep_submission)
    for r in results:
        print(r)

    m1_res = next(r for r in results if r.hotkey == "M1")
    m2_res = next(r for r in results if r.hotkey == "M2")

    assert m1_res.score == 0.0
    assert m1_res.score_reason == "Duplicated submission"
    assert math.isnan(m1_res.test_loss)
    assert math.isnan(m1_res.synth_loss)
    assert m1_res.is_finetune is False

    assert m2_res.score == 0.0
    assert m2_res.score_reason == "Duplicated submission"
    assert math.isnan(m2_res.test_loss)
    assert math.isnan(m2_res.synth_loss)
    assert m2_res.is_finetune is False
    print("Test Case 4 Passed!")


def test_duplicate_with_pre_existing_reason():
    print("\n--- Test Case 5: Duplicate miner already has a score_reason ---")
    miners_data = [
        {
            "hotkey": "M1",
            "test_loss": 0.1,
            "synth_loss": 0.1,
            "is_finetune": True,
            "score": 1.0,
            "task_type": TaskType.INSTRUCTTEXTTASK,
        },  # Kept
        {
            "hotkey": "M2",
            "test_loss": 0.2,
            "synth_loss": 0.2,
            "is_finetune": True,
            "score": 0.9,
            "score_reason": "Pre-existing reason",
            "task_type": TaskType.INSTRUCTTEXTTASK,
        },  # Duplicate
    ]
    miners = [MinerResultsText(**data) for data in miners_data]
    keep_submission = {"M1": True, "M2": False}
    # remaining_valid_count = 1 (M1)

    results = zero_duplicate_scores(miners, keep_submission)
    for r in results:
        print(r)

    m2_res = next(r for r in results if r.hotkey == "M2")

    assert m2_res.score == cts.SCORE_PENALTY
    assert m2_res.score_reason == "Pre-existing reason"  # Should NOT change
    assert math.isnan(m2_res.test_loss)
    assert math.isnan(m2_res.synth_loss)
    assert m2_res.is_finetune is False
    print("Test Case 5 Passed!")


def test_empty_task_results():
    print("\n--- Test Case 6: Empty task_results ---")
    miners_data = []
    miners = [MinerResultsText(**data) for data in miners_data]
    keep_submission = {}

    results = zero_duplicate_scores(miners, keep_submission)
    print(results)

    assert len(results) == 0
    print("Test Case 6 Passed!")


def test_duplicate_with_initial_nan_loss_and_valid_kept():
    print("\n--- Test Case 7: Duplicate with initial NaN test_loss, valid submission kept ---")
    miners_data = [
        {
            "hotkey": "M1",
            "test_loss": 0.1,
            "synth_loss": 0.1,
            "is_finetune": True,
            "score": 1.0,
            "task_type": TaskType.INSTRUCTTEXTTASK,
        },  # Kept
        {
            "hotkey": "M2",
            "test_loss": np.nan,
            "synth_loss": 0.2,
            "is_finetune": True,
            "score": 0.0,
            "score_reason": "Invalid loss initially",
            "task_type": TaskType.INSTRUCTTEXTTASK,
        },  # Duplicate
    ]
    miners = [MinerResultsText(**data) for data in miners_data]
    keep_submission = {"M1": True, "M2": False}
    # remaining_valid_count = 1 (M1)
    # M2 would not contribute to remaining_valid_count even if keep_submission["M2"] was True, due to NaN test_loss.

    results = zero_duplicate_scores(miners, keep_submission)
    for r in results:
        print(r)

    m2_res = next(r for r in results if r.hotkey == "M2")

    assert m2_res.score == cts.SCORE_PENALTY
    assert m2_res.score_reason == "Invalid loss initially"  # Original reason preserved
    assert math.isnan(m2_res.test_loss)  # Stays NaN
    assert math.isnan(m2_res.synth_loss)  # Set to NaN
    assert m2_res.is_finetune is False
    print("Test Case 7 Passed!")


def test_duplicate_score_reason_becomes_default_if_none():
    print("\n--- Test Case 8: Duplicate miner with None score_reason ---")
    miners_data = [
        {
            "hotkey": "M1",
            "test_loss": 0.1,
            "synth_loss": 0.1,
            "is_finetune": True,
            "score": 1.0,
            "task_type": TaskType.INSTRUCTTEXTTASK,
        },  # Kept
        {
            "hotkey": "M2",
            "test_loss": 0.2,
            "synth_loss": 0.2,
            "is_finetune": True,
            "score": 0.9,
            "score_reason": None,
            "task_type": TaskType.INSTRUCTTEXTTASK,
        },  # Duplicate
    ]
    miners = [MinerResultsText(**data) for data in miners_data]
    keep_submission = {"M1": True, "M2": False}
    # remaining_valid_count = 1 (M1)

    results = zero_duplicate_scores(miners, keep_submission)
    for r in results:
        print(r)

    m2_res = next(r for r in results if r.hotkey == "M2")

    assert m2_res.score == cts.SCORE_PENALTY
    assert m2_res.score_reason == "Duplicated submission"  # Should change to default
    assert math.isnan(m2_res.test_loss)
    assert math.isnan(m2_res.synth_loss)
    assert m2_res.is_finetune is False
    print("Test Case 8 Passed!")


if __name__ == "__main__":
    test_no_duplicates()
    test_one_duplicate_others_valid()
    test_one_duplicate_no_other_valid_kept_is_not_finetune()
    test_one_duplicate_no_other_valid_all_duplicates()
    test_duplicate_with_pre_existing_reason()
    test_empty_task_results()
    test_duplicate_with_initial_nan_loss_and_valid_kept()
    test_duplicate_score_reason_becomes_default_if_none()
    print("\nAll specified tests for zero_duplicate_scores completed.")
