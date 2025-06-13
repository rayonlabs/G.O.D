from core.models.utility_models import TaskType
from validator.core.models import ChatRawTask


task_type_configs = [
        {"type": TaskType.CHATTASK, "weight_key": "INSTRUCT_TEXT_TASK_SCORE_WEIGHT"},
        {"type": TaskType.DPOTASK, "weight_key": "DPO_TASK_SCORE_WEIGHT"},
        {"type": TaskType.IMAGETASK, "weight_key": "IMAGE_TASK_SCORE_WEIGHT"},
        {"type": TaskType.GRPOTASK, "weight_key": "GRPO_TASK_SCORE_WEIGHT"},
    ]

for task_config in task_type_configs:
    task_type = task_config["type"]
    task_type_str = str(task_type)
    print(task_type_str)