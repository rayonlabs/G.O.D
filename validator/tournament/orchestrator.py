import asyncio

import httpx
from dotenv import load_dotenv
from trainer.endpoints import GET_GPU_AVAILABILITY_ENDPOINT
from trainer.endpoints import PROXY_TRAINING_IMAGE_ENDPOINT
from trainer.endpoints import TASK_DETAILS_ENDPOINT

import validator.tournament.constants as cst
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.tournament_models import GpuRequirement
from core.models.tournament_models import get_tournament_gpu_requirement
from core.models.utility_models import GPUInfo
from core.models.utility_models import GPUType
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from core.models.utility_models import TournamentTaskTraining
from core.models.utility_models import TrainingStatus
from validator.core.config import Config
from validator.core.config import load_config
from validator.core.models import AnyTypeRawTask
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournaments as tournament_sql
from validator.evaluation.scoring import _get_dataset_type
from validator.utils.logging import get_logger
from validator.utils.util import try_db_connections


load_dotenv(".vali.env", override=True)
logger = get_logger(__name__)


async def fetch_trainer_gpus(trainer_ip: str) -> list[GPUInfo]:
    """
    Fetch GPU availability information from a trainer.

    Args:
        trainer_ip: IP address of the trainer to contact

    Returns:
        List of GPUInfo objects from the trainer
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        url = f"http://{trainer_ip}{GET_GPU_AVAILABILITY_ENDPOINT}"
        logger.info(f"Fetching GPU availability from trainer at {url}")

        response = await client.get(url)
        response.raise_for_status()

        gpu_data = response.json()
        gpu_infos = [GPUInfo.model_validate(gpu_info) for gpu_info in gpu_data]

        logger.info(f"Retrieved {len(gpu_infos)} GPUs from trainer {trainer_ip}")
        return gpu_infos


async def request_training(trainer_ip: str, training_request: TrainerProxyRequest) -> bool:
    """
    Request training from a trainer.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        url = f"http://{trainer_ip}{PROXY_TRAINING_IMAGE_ENDPOINT}"
        logger.info(f"Requesting training from trainer at {url}")

        response = await client.post(url, json=training_request.model_dump())
        response.raise_for_status()

        return response.json()["message"] == "Started Training!"


async def get_training_task_details(trainer_ip: str, task_id: str, hotkey: str) -> TrainerTaskLog:
    """
    Get the details of a training task from a trainer.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        url = f"http://{trainer_ip}{TASK_DETAILS_ENDPOINT}"
        logger.info(f"Getting task details from trainer at {url}")

        response = await client.get(url, params={"task_id": task_id, "hotkey": hotkey})
        response.raise_for_status()

        return TrainerTaskLog.model_validate(response.json())


async def fetch_tournament_tasks_ready_to_train(config: Config):
    """
    Main function to run all tournament orchestrator cycles.
    """
    while True:
        try:
            logger.info("Fetching tournament tasks ready to train")
            await _fetch_tournament_tasks_ready_to_train(config)
        except Exception as e:
            logger.error(f"Error in tournament orchestrator cycles: {str(e)}")
        finally:
            await asyncio.sleep(15 * 60)  # 15 minutes in seconds


async def _fetch_tournament_tasks_ready_to_train(config: Config):
    """
    Fetch tasks that are looking for nodes and are part of tournaments,
    then move them to training status and record the hotkey assignments.
    """
    tasks = await task_sql.get_tasks_with_status(TaskStatus.LOOKING_FOR_NODES, config.psql_db, tournament_filter="only")
    logger.info(f"Found {len(tasks)} tournament tasks looking for nodes")

    if not tasks:
        logger.info("No tournament tasks found, skipping cycle")
        return

    task_hotkey_triples = []
    tasks_to_update = []

    for task in tasks:
        nodes = await task_sql.get_nodes_assigned_to_task(task.task_id, config.psql_db)
        hotkeys = [node.hotkey for node in nodes]

        if hotkeys:
            for hotkey in hotkeys:
                task_hotkey_triples.append((task.task_id, hotkey, task.created_at))
            tasks_to_update.append(task)

    if task_hotkey_triples:
        await tournament_sql.add_tournament_task_hotkey_trainings(task_hotkey_triples, config.psql_db)

    for task in tasks_to_update:
        task.status = TaskStatus.TRAINING
        await task_sql.update_task(task, config.psql_db)
        logger.info(f"Updated task {task.task_id} to training status with {len(task_hotkey_triples)} hotkeys")

    logger.info(f"Successfully processed {len(tasks_to_update)} tasks in fetch_tournament_tasks_ready_to_train cycle")


async def process_pending_tournament_tasks(config: Config):
    """
    Main function to run the pending tournament tasks processing cycle.
    """
    while True:
        try:
            logger.info("Processing pending tournament tasks")
            pending_training_tasks = await tournament_sql.get_tournament_training_tasks(
                config.psql_db,
                TrainingStatus.PENDING,
                )
            logger.info(f"Fetched {len(pending_training_tasks)} pending tournament tasks")

            if not pending_training_tasks:
                logger.info("No pending tasks found, waiting 15 minutes to avoid tight loop")
                await asyncio.sleep(15 * 60)  # 15 minutes
                continue

            await schedule_tasks_for_training(pending_training_tasks, config)
        except Exception as e:
            logger.error(f"Error in process_pending_tournament_tasks cycle: {str(e)}")
            await asyncio.sleep(15 * 60)  # 15 minutes


async def schedule_tasks_for_training(pending_training_tasks: list[TournamentTaskTraining], config: Config):
    """
    Process tasks from the list and schedule them for training.
    Only pop tasks when we're 100% sure GPUs are available.
    """
    while pending_training_tasks:
        oldest_task_training = pending_training_tasks[-1]
        task = oldest_task_training.task

        # Check max attempts
        if oldest_task_training.n_training_attempts >= cst.MAX_TRAINING_ATTEMPTS:
            logger.warning(f"Task {task.task_id} with hotkey {oldest_task_training.hotkey} has exceeded max attempts ({oldest_task_training.n_training_attempts}), marking as failed")
            await tournament_sql.update_tournament_task_training_status(
                task.task_id, oldest_task_training.hotkey, TrainingStatus.FAILURE, config.psql_db
            )
            pending_training_tasks.pop()
            continue

        # Determine required GPUs for this task
        required_gpus = get_tournament_gpu_requirement(task.task_type, task.model_params_count)
        logger.info(f"Task {task.task_id} requires {required_gpus.value}")
        suitable_gpus_result = await _check_suitable_gpus(config, required_gpus)

        if not suitable_gpus_result:
            logger.info(f"No suitable GPUs found for requirement {required_gpus.value}, waiting 30 minutes before retry")
            await asyncio.sleep(30 * 60)
            continue

        trainer_ip, gpu_ids = suitable_gpus_result

        try:
            training_task = pending_training_tasks[-1]
            training_request = await _create_training_request(training_task.task, training_task.hotkey, gpu_ids, config)
            training_success = await request_training(trainer_ip, training_request)

            if training_success:
                await tournament_sql.update_tournament_task_training_status(
                    training_task.task.task_id, training_task.hotkey, TrainingStatus.TRAINING, config.psql_db
                )
                await tournament_sql.update_gpu_availability(
                    trainer_ip, gpu_ids, training_task.task.hours_to_complete, config.psql_db
                    )
                pending_training_tasks.pop()
                logger.info(
                    f"Successfully scheduled task {training_task.task.task_id} with hotkey {training_task.hotkey} for training "
                    f"on trainer {trainer_ip} with GPUs {gpu_ids} for {training_task.task.hours_to_complete} hours"
                    )
            else:
                logger.error(f"Failed to start training for task {training_task.task.task_id} on trainer {trainer_ip}")
                await asyncio.sleep(15 * 60)
                continue
        except Exception as e:
            logger.error(f"Exception while scheduling training: {str(e)}")
            await asyncio.sleep(15 * 60)
            continue

    logger.info(f"Completed scheduling cycle, {len(pending_training_tasks)} tasks remaining")


async def _check_suitable_gpus(config: Config, required_gpus: GpuRequirement) -> tuple[str, list[int]] | None:
    """
    Check if there are any suitable GPUs across all trainers for the given GPU requirement.

    Returns:
        tuple[str, list[int]] | None: (trainer_ip, gpu_ids) if suitable GPUs found, None otherwise
    """
    try:
        trainers = await tournament_sql.get_trainers(config.psql_db)

        for trainer in trainers:
            gpu_ids = _trainer_has_sufficient_gpus(trainer.gpus, required_gpus)
            if gpu_ids:
                logger.info(f"Found suitable GPUs on trainer {trainer.trainer_ip} for requirement {required_gpus.value}")
                return trainer.trainer_ip, gpu_ids

        logger.info(f"No suitable GPUs found across all trainers for requirement {required_gpus.value}")

    except Exception as e:
        logger.error(f"Error checking suitable GPUs: {str(e)}")


async def _create_training_request(task: AnyTypeRawTask, hotkey: str, available_gpu_ids: list[int], config: Config) -> TrainerProxyRequest:
    """
    Create a TrainerProxyRequest based on the task type.

    Args:
        task: The task to create a training request for
        hotkey: The hotkey of the miner
        available_gpu_ids: List of available GPU IDs
        config: Configuration object for database access

    Returns:
        TrainerProxyRequest: The training request
    """
    expected_repo_name = await task_sql.get_expected_repo_name(task.task_id, hotkey, config.psql_db)
    training_repo, training_commit_hash = await tournament_sql.get_tournament_training_repo_and_commit(hotkey, config.psql_db)

    if task.task_type == TaskType.IMAGETASK:
        training_data = TrainRequestImage(
            model=task.model_id,
            task_id=str(task.task_id),
            hours_to_complete=task.hours_to_complete,
            expected_repo_name=expected_repo_name,
            dataset_zip=task.training_data,
            model_type=task.model_type,
        )
    else:
        dataset_type = _get_dataset_type(task)
        training_data = TrainRequestText(
            model=task.model_id,
            task_id=str(task.task_id),
            hours_to_complete=task.hours_to_complete,
            expected_repo_name=expected_repo_name,
            dataset=task.training_data,
            dataset_type=dataset_type,
            file_format=task.file_format,
        )

    return TrainerProxyRequest(
        training_data=training_data,
        github_repo=training_repo,
        gpu_ids=available_gpu_ids,
        hotkey=hotkey,
        github_commit_hash=training_commit_hash,
    )


def _trainer_has_sufficient_gpus(trainer_gpus: list[GPUInfo], requirement: GpuRequirement) -> list[int]:
    """
    Check if a trainer has sufficient GPUs to meet the requirement.

    Args:
        trainer_gpus: List of GPUs on the trainer
        requirement: Required GPU specification

    Returns:
        list[int]: List of GPU IDs needed for the requirement, empty list if insufficient
    """
    available_h100s = [gpu for gpu in trainer_gpus if gpu.available and gpu.gpu_type == GPUType.H100]
    available_a100s = [gpu for gpu in trainer_gpus if gpu.available and gpu.gpu_type == GPUType.A100]

    if requirement == GpuRequirement.A100:
        return [available_a100s[0].gpu_id] if len(available_a100s) >= 1 else []
    elif requirement == GpuRequirement.H100_1X:
        return [available_h100s[0].gpu_id] if len(available_h100s) >= 1 else []
    elif requirement == GpuRequirement.H100_2X:
        return [gpu.gpu_id for gpu in available_h100s[:2]] if len(available_h100s) >= 2 else []
    elif requirement == GpuRequirement.H100_4X:
        return [gpu.gpu_id for gpu in available_h100s[:4]] if len(available_h100s) >= 4 else []
    elif requirement == GpuRequirement.H100_8X:
        return [gpu.gpu_id for gpu in available_h100s[:8]] if len(available_h100s) >= 8 else []

    return []


async def run_tournament_orchestrator_cycles():
    config = load_config()
    await try_db_connections(config)

    await asyncio.gather(
        fetch_tournament_tasks_ready_to_train(config),
        process_pending_tournament_tasks(config),
    )


if __name__ == "__main__":
    asyncio.run(run_tournament_orchestrator_cycles())
