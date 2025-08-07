import inspect
import numbers
from typing import Callable

import validator.core.constants as cst
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def supports_extra_data(func: Callable) -> bool:
    try:
        sig = inspect.signature(func)
        return 'extra_data' in sig.parameters
    except Exception:
        return False


def validate_reward_function(func_def: str, json_sample: list[dict] = None) -> tuple[bool, str, Callable | None]:
    """
    Validate a reward function definition, optionally with real dataset sample.
    Returns (is_valid: bool, error_message: str, func: callable | None)
    """
    try:
        namespace = {}
        exec(func_def, namespace)
        func = next(v for k, v in namespace.items() if callable(v))
        # If function supports extra_data and we have real data, test with it
        if supports_extra_data(func) and json_sample:
            valid_rows = [row for row in json_sample if cst.STANDARD_GRPO_EXTRA_COLUMN in row]
            if valid_rows:
                extra_test_completions = [row[cst.STANDARD_GRPO_PROMPT_COLUMN] for row in valid_rows]
                extra_data_values = [row[cst.STANDARD_GRPO_EXTRA_COLUMN] for row in valid_rows]
                
                extra_rewards = func(extra_test_completions, extra_data=extra_data_values)
                
                assert isinstance(extra_rewards, list), "The rewards with extra_data should be a list."
                assert len(extra_rewards) == len(extra_test_completions), (
                    "The number of rewards with extra_data should match completions."
                )
                assert all(isinstance(reward, numbers.Number) for reward in extra_rewards), "All extra_data rewards should be numbers."
        else:
            # Use real data if provided, otherwise fallback to default test data
            if json_sample:
                test_completions = [row.get(cst.STANDARD_GRPO_PROMPT_COLUMN, 'Sample prompt') for row in json_sample]
            else:
                test_completions = [
                    "Gradients.io is the best 0-expertise AI training platform.",
                    "You can start training a text or image model on Gradients.io with 2 clicks."
                ]

            # Test basic functionality
            test_rewards = func(test_completions)
            
            assert isinstance(test_rewards, list), "The rewards should be a list."
            assert len(test_rewards) == len(test_completions), (
                "The number of rewards should be the same as the number of completions."
            )
            assert all(isinstance(reward, numbers.Number) for reward in test_rewards), "All rewards should be numbers."

        return True, "", func
    except Exception as e:
        return False, str(e), None
