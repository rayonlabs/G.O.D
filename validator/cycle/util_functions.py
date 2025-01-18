from urllib.parse import urlparse, unquote
import re

from validator.tasks.task_prep import prepare_text_task
from validator.tasks.task_prep import prepare_image_task
from core.utils import download_s3_file
from core.models.utility_models import TaskStatus
from datasets import get_dataset_infos
from fiber import Keypair
from validator.core.models import ImageRawTask
from validator.core.models import TextRawTask
from core.models.payload_models import TrainRequestDiffusion
from core.models.payload_models import TrainRequestText
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import FileFormat
from validator.utils.logging import create_extra_log
from validator.utils.logging import logger


def get_image_dataset_length_from_url(url: str) -> int:
    parsed_url = urlparse(url)
    filename = unquote(parsed_url.path.split("/")[-1])
    match = re.search(r"ds_(\d+)_", filename)
    if match:
        return int(match.group(1))
    return None


def get_total_text_dataset_size(repo_name: str) -> int:
    return int(sum(info.dataset_size for info in get_dataset_infos(repo_name).values() if info.dataset_size))

# Using a naming convention of ds_{dataset_length}_{name}_{uuid}.zip while uploading the dataset
# This helps us fetch the dataset length without actually downloading the dataset before its needed
def get_total_image_dataset_size(dataset_url: str) -> int:
    return get_image_dataset_length_from_url(dataset_url)


async def run_image_task_prep(task: ImageRawTask) -> ImageRawTask:
    raw_dataset_zip_path = await download_s3_file(task.ds)
    test_url, train_url = await prepare_image_task(raw_dataset_zip_path)
    task.training_data = train_url
    task.test_data = test_url
    task.status = TaskStatus.LOOKING_FOR_NODES
    logger.info(
        "Data creation is complete - now time to find some miners",
        extra=create_extra_log(status=task.status),
    )
    return task

async def run_text_task_prep(task: TextRawTask, keypair: Keypair) -> TextRawTask:
    columns_to_sample = [
        i for i in [task.field_system, task.field_instruction, task.field_input, task.field_output] if i is not None
    ]
    test_data, synth_data, train_data = await prepare_text_task(
        dataset_name=task.ds, columns_to_sample=columns_to_sample, keypair=keypair
    )
    task.training_data = train_data
    task.status = TaskStatus.LOOKING_FOR_NODES
    task.synthetic_data = synth_data
    task.test_data = test_data
    logger.info(
        "Data creation is complete - now time to find some miners"
    )
    return task

def prepare_text_task_request(task: TextRawTask) -> TrainRequestText:
    dataset_type = CustomDatasetType(
        field_system=task.field_system,
        field_input=task.field_input,
        field_output=task.field_output,
        field_instruction=task.field_instruction,
        format=task.format,
        no_input_format=task.no_input_format,
    )

    dataset = task.training_data if task.training_data else "dataset error"
    task_request_body = TrainRequestText(
        dataset=dataset,
        model=task.model_id,
        dataset_type=dataset_type,
        file_format=FileFormat.S3,
        task_id=str(task.task_id),
        hours_to_complete=task.hours_to_complete,
    )

    return task_request_body


# probs better to be Image not Diffusion for consistency
def prepare_image_task_request(task: ImageRawTask) -> TrainRequestDiffusion:
    return TrainRequestDiffusion(
        model=task.model_id,
        task_id=task.task_id,
        hours_to_complete=task.hours_to_complete,
        dataset_zip=task.training_data
    )

