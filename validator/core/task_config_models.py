from typing import Callable
from typing import Union

from pydantic import BaseModel
from pydantic import Field

from core.models.utility_models import TaskType
from validator.core import constants as cst
from validator.core.models import ImageRawTask
from validator.core.models import TextRawTask
from validator.cycle.util_functions import get_total_image_dataset_size
from validator.cycle.util_functions import get_total_text_dataset_size
from validator.cycle.util_functions import prepare_image_task_request
from validator.cycle.util_functions import prepare_text_task_request
from validator.cycle.util_functions import run_image_task_prep
from validator.cycle.util_functions import run_text_task_prep
from validator.evaluation.docker_evaluation import run_evaluation_docker_image
from validator.evaluation.docker_evaluation import run_evaluation_docker_text
from validator.tasks.task_prep import get_additional_synth_data


# TODO
# being lazy here with everything as callable, can we look at the signatures and use the same for the diff
# data types
class TaskConfig(BaseModel):
    task_type: TaskType = Field(..., description="The type of task.")
    eval_container: Callable = Field(..., description="Container to evaluate the task")
    synth_data_function: Callable | None = Field(..., description="Function to generate synthetic data")
    data_size_function: Callable = Field(..., description="The function used to determine the dataset size")
    task_prep_function: Callable = Field(
        ..., description="What we call in order to do the prep work - train test split and whatnot"
    )

    task_request_prepare_function: Callable = Field(..., description="Namoray will come up with a better var name for sure")
    start_training_endpoint: str = Field(..., description="The endpoint to start training")


class ImageTaskConfig(TaskConfig):
    task_type: TaskType = TaskType.IMAGETASK
    eval_container: Callable = run_evaluation_docker_image
    synth_data_function: Callable | None = None
    data_size_function: Callable = get_total_image_dataset_size
    task_prep_function: Callable = run_image_task_prep
    task_request_prepare_function: Callable = prepare_image_task_request
    start_training_endpoint: str = cst.START_TRAINING_IMAGE_ENDPOINT


class TextTaskConfig(TaskConfig):
    task_type: TaskType = TaskType.TEXTTASK
    eval_container: Callable = run_evaluation_docker_text
    synth_data_function: Callable | None = get_additional_synth_data
    data_size_function: Callable = get_total_text_dataset_size
    task_prep_function: Callable = run_text_task_prep
    task_request_prepare_function: Callable = prepare_text_task_request
    start_training_endpoint: str = cst.START_TRAINING_ENDPOINT


def get_task_config(task: Union[TextRawTask, ImageRawTask]) -> TaskConfig:
    if isinstance(task, TextRawTask):
        return TextTaskConfig()
    elif isinstance(task, ImageRawTask):
        return ImageTaskConfig()
    else:
        raise ValueError(f"Unsupported task type: {type(task).__name__}")
