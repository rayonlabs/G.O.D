import yaml

from core.models.utility_models import Prompts
from validator.core.constants import PROMPT_PATH


def load_prompts() -> Prompts:
    with open(PROMPT_PATH, "r") as file:
        prompts_dict = yaml.safe_load(file)
    return Prompts(**prompts_dict)
