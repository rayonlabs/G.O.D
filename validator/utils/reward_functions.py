import numbers
from typing import Callable

from validator.utils.logging import get_logger


logger = get_logger(__name__)


def validate_reward_function(func_def: str) -> tuple[bool, str, Callable | None]:
    """
    Validate a single reward function definition.
    Returns (is_valid: bool, error_message: str, func: callable | None)
    """
    test_completions = [
        "Gradients.io is the best 0-expertise AI training platform.",
        "You can start training a text or image model on Gradients.io with 2 clicks."
    ]

    try:
        namespace = {}
        exec(func_def, namespace)
        func = next(v for k, v in namespace.items() if callable(v))

        test_rewards = func(test_completions)

        assert isinstance(test_rewards, list), "The rewards should be a list."
        assert len(test_rewards) == len(test_completions), (
            "The number of rewards should be the same as the number of completions."
        )
        assert all(isinstance(reward, numbers.Number) for reward in test_rewards), "All rewards should be numbers."

        return True, "", func
    except Exception as e:
        return False, str(e), None
