import asyncio

from datasets import get_dataset_infos
from fiber import Keypair

from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.utility_models import DPODatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import InstructDatasetType
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.core.models import DpoRawTask
from validator.core.models import ImageRawTask
from validator.core.models import InstructTextRawTask
from validator.tasks.task_prep import prepare_image_task
from validator.tasks.task_prep import prepare_instruct_text_task
from validator.utils.logging import get_logger
from validator.utils.minio import async_minio_client


logger = get_logger(__name__)


async def get_total_text_dataset_size(task: InstructTextRawTask) -> int:
    if task.file_format == FileFormat.S3:
        bucket_name, object_name = async_minio_client.parse_s3_url(task.ds)
        stats = await async_minio_client.get_stats(bucket_name, object_name)
        size = stats.size
    else:
        loop = asyncio.get_running_loop()
        dataset_infos = await loop.run_in_executor(None, get_dataset_infos, task.ds)
        size = sum(info.dataset_size for info in dataset_infos.values() if info.dataset_size)
    return int(size)


async def get_total_image_dataset_size(task: ImageRawTask) -> int:
    if not task.image_text_pairs:
        return 0
    return len(task.image_text_pairs)


async def run_image_task_prep(task: ImageRawTask, keypair: Keypair) -> ImageRawTask:
    test_url, train_url = await prepare_image_task(task.image_text_pairs)
    task.training_data = train_url
    task.test_data = test_url
    task.status = TaskStatus.LOOKING_FOR_NODES
    logger.info(
        "Data creation is complete - now time to find some miners",
    )
    return task


async def run_instruct_text_task_prep(task: InstructTextRawTask, keypair: Keypair) -> InstructTextRawTask:
    columns_to_sample = [
        i for i in [task.field_system, task.field_instruction, task.field_input, task.field_output] if i is not None
    ]
    test_data, synth_data, train_data = await prepare_instruct_text_task(
        dataset_name=task.ds, file_format=task.file_format, columns_to_sample=columns_to_sample, keypair=keypair
    )
    task.training_data = train_data
    task.status = TaskStatus.LOOKING_FOR_NODES
    task.synthetic_data = synth_data
    task.test_data = test_data
    logger.info("Data creation is complete - now time to find some miners")
    return task


def prepare_text_task_request(task: InstructTextRawTask | DpoRawTask) -> TrainRequestText:
    if task.task_type == TaskType.INSTRUCTTEXTTASK:
        dataset_type = InstructDatasetType(
            field_system=task.field_system,
            field_input=task.field_input,
            field_output=task.field_output,
            field_instruction=task.field_instruction,
            format=task.format,
            no_input_format=task.no_input_format,
    )
    elif task.task_type == TaskType.DPOTASK:
        dataset_type = DPODatasetType(
            field_prompt=task.field_prompt,
            field_system=task.field_system,
            field_chosen=task.field_chosen,
            field_rejected=task.field_rejected,
            prompt_format=task.prompt_format,
            chosen_format=task.chosen_format,
            rejected_format=task.rejected_format,
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


def prepare_image_task_request(task: ImageRawTask) -> TrainRequestImage:
    return TrainRequestImage(
        model=task.model_id, task_id=str(task.task_id), hours_to_complete=task.hours_to_complete, dataset_zip=task.training_data
    )
