import asyncio

import httpx
from dotenv import load_dotenv
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential

import validator.tournament.constants as cst
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.tournament_models import GpuRequirement
from core.models.tournament_models import TournamentTaskTraining
from core.models.tournament_models import get_tournament_gpu_requirement
from core.models.utility_models import FileFormat
from core.models.utility_models import GPUInfo
from core.models.utility_models import GPUType
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from core.models.utility_models import TrainingStatus
from validator.core.config import Config
from validator.core.config import load_config
from validator.core.constants import GET_GPU_AVAILABILITY_ENDPOINT
from validator.core.constants import PROXY_TRAINING_IMAGE_ENDPOINT
from validator.core.constants import TASK_DETAILS_ENDPOINT
from validator.core.models import AnyTypeRawTask
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournaments as tournament_sql
from validator.db.sql.tournaments import get_tournament_id_by_task_id
from validator.evaluation.scoring import _get_dataset_type
from validator.utils.logging import LogContext
from validator.utils.logging import get_logger
from validator.utils.util import try_db_connections


logger = get_logger(__name__)


simple_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=10),
    reraise=True,
)


@simple_retry
async def fetch_trainer_gpus(trainer_ip: str) -> list[GPUInfo]:
    """
    Fetch GPU availability information from a trainer.

    Args:
        trainer_ip: IP address of the trainer to contact

    Returns:
        List of GPUInfo objects from the trainer
    """
    async with httpx.AsyncClient(timeout=cst.TRAINER_HTTP_TIMEOUT) as client:
        # Default to port 8001 if no port is specified
        if ":" not in trainer_ip:
            trainer_ip_with_port = f"{trainer_ip}:8001"
        else:
            trainer_ip_with_port = trainer_ip

        url = f"http://{trainer_ip_with_port}{GET_GPU_AVAILABILITY_ENDPOINT}"
        logger.info(f"Fetching GPU availability from trainer at {url}")

        response = await client.get(url)
        response.raise_for_status()

        gpu_data = response.json()
        gpu_infos = [GPUInfo.model_validate(gpu_info) for gpu_info in gpu_data]

        logger.info(f"Retrieved {len(gpu_infos)} GPUs from trainer {trainer_ip}")
        return gpu_infos


@simple_retry
async def start_training_task(trainer_ip: str, training_request: TrainerProxyRequest) -> bool:
    """
    Ask trainer to start training.


    Args:
        trainer_ip: IP address of the trainer
        training_request: The training request to send


    Returns:
        bool: True if training started successfully, False otherwise
    """
    try:
        # Validate the request by converting to dict and back
        validated_request = TrainerProxyRequest.model_validate(training_request.model_dump())
        logger.info("Schema validation passed for training request")
    except Exception as e:
        logger.error(f"Schema validation failed for training request: {str(e)}")
        logger.error(f"Request payload: {training_request.model_dump()}")
        return False

    async with httpx.AsyncClient(timeout=cst.TRAINER_HTTP_TIMEOUT) as client:
        # Default to port 8001 if no port is specified
        if ":" not in trainer_ip:
            trainer_ip_with_port = f"{trainer_ip}:8001"
        else:
            trainer_ip_with_port = trainer_ip

        url = f"http://{trainer_ip_with_port}{PROXY_TRAINING_IMAGE_ENDPOINT}"
        
        # Log key information about the training request
        training_data = validated_request.training_data
        logger.info(f"Starting training task:")
        logger.info(f"  - Trainer: {trainer_ip_with_port}")
        logger.info(f"  - Task ID: {training_data.task_id}")
        logger.info(f"  - Model: {training_data.model}")
        logger.info(f"  - Hours to complete: {training_data.hours_to_complete}")
        logger.info(f"  - GitHub repo: {validated_request.github_repo}")
        logger.info(f"  - Hotkey: {validated_request.hotkey[:8]}...")
        logger.info(f"  - GPU IDs: {validated_request.gpu_ids}")
        
        logger.debug(f"Full request payload: {validated_request.model_dump()}")

        try:
            response = await client.post(url, json=validated_request.model_dump())
            response.raise_for_status()
            
            response_data = response.json()
            success = response_data.get("message") == cst.EXPECTED_TRAINING_START_MESSAGE
            
            if success:
                logger.info(f"✓ Training started successfully on trainer {trainer_ip} for task {training_data.task_id}")
            else:
                logger.error(f"✗ Training failed to start on trainer {trainer_ip}. Response: {response_data}")
                
            return success
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error when starting training on {trainer_ip}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when starting training on {trainer_ip}: {str(e)}")
            raise


@simple_retry
async def get_training_task_details(trainer_ip: str, task_id: str, hotkey: str) -> TrainerTaskLog:
    """
    Get the details of a training task from a trainer.

    Args:
        trainer_ip: IP address of the trainer
        task_id: The task ID to get details for
        hotkey: The hotkey of the miner

    Returns:
        TrainerTaskLog: The task log from the trainer
    """
    async with httpx.AsyncClient(timeout=cst.TRAINER_HTTP_TIMEOUT) as client:
        # Default to port 8001 if no port is specified
        if ":" not in trainer_ip:
            trainer_ip_with_port = f"{trainer_ip}:8001"
        else:
            trainer_ip_with_port = trainer_ip

        url = f"http://{trainer_ip_with_port}{TASK_DETAILS_ENDPOINT.format(task_id=task_id)}"
        logger.debug(f"Getting task details from trainer at {url} for task {task_id}")

        response = await client.get(url, params={"hotkey": hotkey})
        response.raise_for_status()

        return TrainerTaskLog.model_validate(response.json())


async def fetch_tournament_tasks_ready_to_train(config: Config):
    """
    Fill the `tournament_task_hotkey_trainings` table with task-hotkey pairs that haven't been trained yet.
    """
    while True:
        try:
            logger.info("Fetching tournament tasks ready to train")
            await _fetch_tournament_tasks_ready_to_train(config)
        except Exception as e:
            logger.error(f"Error in tournament orchestrator cycles: {str(e)}", exc_info=True)
        finally:
            await asyncio.sleep(cst.FETCH_TASKS_CYCLE_INTERVAL)


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

    # Log details about each task found
    for task in tasks:
        logger.info(f"Task {task.task_id}: created_at={task.created_at}, model_id={task.model_id}, task_type={task.task_type}")

    task_hotkey_triples = []
    tasks_to_update = []
    tasks_without_nodes = []

    for task in tasks:
        nodes = await task_sql.get_nodes_assigned_to_task(task.task_id, config.psql_db)
        hotkeys = [node.hotkey for node in nodes]
        
        logger.info(f"Task {task.task_id}: Found {len(nodes)} assigned nodes")
        if nodes:
            for node in nodes:
                logger.debug(f"  - Node: hotkey={node.hotkey[:8]}..., node_id={node.node_id}")

        if hotkeys:
            for hotkey in hotkeys:
                task_hotkey_triples.append((task.task_id, hotkey, task.created_at))
            tasks_to_update.append(task)
        else:
            tasks_without_nodes.append(task)
            logger.warning(f"Task {task.task_id} has no nodes assigned - will remain in LOOKING_FOR_NODES status")

    if tasks_without_nodes:
        logger.warning(f"Found {len(tasks_without_nodes)} tasks without assigned nodes: {[str(t.task_id) for t in tasks_without_nodes]}")

    if task_hotkey_triples:
        logger.info(f"Adding {len(task_hotkey_triples)} task-hotkey pairs to tournament_task_hotkey_trainings table")
        await tournament_sql.add_tournament_task_hotkey_pairs_for_training(task_hotkey_triples, config.psql_db)

    for task in tasks_to_update:
        hotkey_count = len([t for t in task_hotkey_triples if t[0] == task.task_id])
        logger.info(f"Updating task {task.task_id} from LOOKING_FOR_NODES to TRAINING status (with {hotkey_count} hotkeys)")
        task.status = TaskStatus.TRAINING
        await task_sql.update_task(task, config.psql_db)

    logger.info(f"Successfully processed {len(tasks_to_update)} tasks in fetch_tournament_tasks_ready_to_train cycle")


async def process_pending_tournament_tasks(config: Config):
    while True:
        try:
            logger.info("Processing pending tournament tasks")
            pending_training_tasks = await tournament_sql.get_tournament_training_tasks(
                config.psql_db,
                TrainingStatus.PENDING,
            )

            logger.info(f"Fetched {len(pending_training_tasks)} pending tournament tasks")
            
            # Log details about pending tasks
            if pending_training_tasks:
                for task_training in pending_training_tasks[:5]:  # Log first 5 for brevity
                    logger.info(f"Pending task: task_id={task_training.task.task_id}, hotkey={task_training.hotkey[:8]}..., "
                              f"n_training_attempts={task_training.n_training_attempts}, "
                              f"model={task_training.task.model_id}, task_type={task_training.task.task_type}")

            if not pending_training_tasks:
                logger.info("No pending tasks found, waiting to avoid tight loop")
                await asyncio.sleep(cst.PROCESS_PENDING_TASKS_CYCLE_INTERVAL)
                continue

            await schedule_tasks_for_training(pending_training_tasks, config)
        except Exception as e:
            logger.error(f"Error in process_pending_tournament_tasks cycle: {str(e)}", exc_info=True)
            await asyncio.sleep(cst.PROCESS_PENDING_TASKS_CYCLE_INTERVAL)


async def schedule_tasks_for_training(pending_training_tasks: list[TournamentTaskTraining], config: Config):
    """
    Process tasks from the list and schedule them for training.
    Only pop tasks when we're 100% sure GPUs are available.
    """
    # Track failed attempts for this scheduling session
    failed_attempts = {}
    MAX_SCHEDULING_ATTEMPTS = 3
    
    logger.info(f"Starting schedule_tasks_for_training with {len(pending_training_tasks)} pending tasks")

    while pending_training_tasks:
        oldest_task_training = pending_training_tasks[-1]
        tournament_id = await get_tournament_id_by_task_id(oldest_task_training.task.task_id, config.psql_db)
        with LogContext(
            task_id=oldest_task_training.task.task_id, hotkey=oldest_task_training.hotkey, tournament_id=tournament_id
        ):
            task = oldest_task_training.task
            task_key = f"{task.task_id}_{oldest_task_training.hotkey}"
            
            logger.info(f"Processing task {task.task_id} with hotkey {oldest_task_training.hotkey[:8]}... "
                       f"(attempt {oldest_task_training.n_training_attempts}/{cst.MAX_TRAINING_ATTEMPTS})")

            # Check max attempts
            if oldest_task_training.n_training_attempts >= cst.MAX_TRAINING_ATTEMPTS:
                logger.warning(
                    f"Task {task.task_id} with hotkey {oldest_task_training.hotkey} has exceeded max attempts ({oldest_task_training.n_training_attempts}), marking as failed"
                )

                await tournament_sql.update_tournament_task_training_status(
                    task.task_id, oldest_task_training.hotkey, TrainingStatus.FAILURE, config.psql_db
                )
                pending_training_tasks.pop()
                continue

            # Determine required GPUs for this task
            required_gpus = get_tournament_gpu_requirement(task.task_type, task.model_params_count)
            logger.info(f"Task {task.task_id} (model: {task.model_id}, params: {task.model_params_count}) requires {required_gpus.value}")
            
            logger.info(f"Checking for suitable GPUs for requirement {required_gpus.value}...")
            suitable_gpus_result = await _check_suitable_gpus(config, required_gpus)

            if not suitable_gpus_result:
                logger.warning(f"No suitable GPUs found for requirement {required_gpus.value}, waiting {cst.GPU_AVAILABILITY_CHECK_RETRY_INTERVAL}s before retry")
                logger.info(f"Keeping task {task.task_id} in pending queue, will retry later")
                await asyncio.sleep(cst.GPU_AVAILABILITY_CHECK_RETRY_INTERVAL)
                continue

            trainer_ip, gpu_ids = suitable_gpus_result
            logger.info(f"Found suitable GPUs on trainer {trainer_ip}: GPU IDs {gpu_ids}")

        try:
            training_task = pending_training_tasks[-1]
            tournament_id = await get_tournament_id_by_task_id(training_task.task.task_id, config.psql_db)
            with LogContext(task_id=str(training_task.task.task_id), hotkey=training_task.hotkey, tournament_id=tournament_id):
                training_request = await _create_training_request(training_task.task, training_task.hotkey, gpu_ids, config)
                training_success = await start_training_task(trainer_ip, training_request)

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
                    # Track failed attempts for this scheduling session
                    failed_attempts[task_key] = failed_attempts.get(task_key, 0) + 1

                    if failed_attempts[task_key] >= MAX_SCHEDULING_ATTEMPTS:
                        logger.warning(
                            f"Task {training_task.task.task_id} with hotkey {training_task.hotkey} has exceeded max scheduling attempts ({failed_attempts[task_key]}), marking as FAILURE"
                        )
                        logger.info(
                            f"Current n_training_attempts: {oldest_task_training.n_training_attempts} for task {training_task.task.task_id} with hotkey {training_task.hotkey}"
                        )
                        await tournament_sql.update_tournament_task_training_status(
                            training_task.task.task_id, training_task.hotkey, TrainingStatus.FAILURE, config.psql_db
                        )
                        pending_training_tasks.pop()
                    else:
                        logger.info(
                            f"Task {training_task.task.task_id} with hotkey {training_task.hotkey} failed, scheduling attempt {failed_attempts[task_key]}/{MAX_SCHEDULING_ATTEMPTS}"
                        )
                        await asyncio.sleep(cst.TRAINING_START_RETRY_INTERVAL)
                    continue
        except Exception as e:
            logger.error(f"Exception while scheduling training: {str(e)}")
            # Track failed attempts for this scheduling session
            failed_attempts[task_key] = failed_attempts.get(task_key, 0) + 1

            if failed_attempts[task_key] >= MAX_SCHEDULING_ATTEMPTS:
                logger.warning(
                    f"Task {training_task.task.task_id} with hotkey {training_task.hotkey} has exceeded max scheduling attempts ({failed_attempts[task_key]}) due to exception, marking as FAILURE"
                )
                logger.info(
                    f"Current n_training_attempts: {oldest_task_training.n_training_attempts} for task {training_task.task.task_id} with hotkey {training_task.hotkey}"
                )
                await tournament_sql.update_tournament_task_training_status(
                    training_task.task.task_id, training_task.hotkey, TrainingStatus.FAILURE, config.psql_db
                )
                pending_training_tasks.pop()
            else:
                logger.info(
                    f"Task {training_task.task.task_id} with hotkey {training_task.hotkey} failed due to exception, scheduling attempt {failed_attempts[task_key]}/{MAX_SCHEDULING_ATTEMPTS}"
                )
                await asyncio.sleep(cst.TRAINING_START_RETRY_INTERVAL)
            continue

    logger.info(f"Completed scheduling cycle, {len(pending_training_tasks)} tasks remaining")


async def _check_suitable_gpus(config: Config, required_gpus: GpuRequirement) -> tuple[str, list[int]] | None:
    """
    Check if there are any suitable GPUs across all trainers for the given GPU requirement.

    Args:
        config: Configuration object for database access
        required_gpus: Required GPU specification

    Returns:
        tuple[str, list[int]] | None: (trainer_ip, gpu_ids) if suitable GPUs found, None otherwise
    """
    try:
        trainers = await tournament_sql.get_trainers(config.psql_db)
        logger.info(f"Checking {len(trainers)} trainers for suitable GPUs (requirement: {required_gpus.value})")

        total_gpus = sum(len(trainer.gpus) for trainer in trainers)
        logger.info(f"Total GPUs across all trainers: {total_gpus}")

        for trainer in trainers:
            logger.info(f"Checking trainer {trainer.trainer_ip} with {len(trainer.gpus)} GPUs")
            
            # Count available GPUs by type
            available_h100s = 0
            available_a100s = 0
            total_h100s = 0
            total_a100s = 0
            
            for gpu in trainer.gpus:
                if gpu.gpu_type == GPUType.H100:
                    total_h100s += 1
                    if gpu.available:
                        available_h100s += 1
                elif gpu.gpu_type == GPUType.A100:
                    total_a100s += 1
                    if gpu.available:
                        available_a100s += 1
                        
                logger.debug(f"  GPU {gpu.gpu_id} ({gpu.gpu_type}): available={gpu.available}, used_until={gpu.used_until}")
            
            logger.info(f"  Trainer {trainer.trainer_ip} summary: H100s: {available_h100s}/{total_h100s} available, "
                       f"A100s: {available_a100s}/{total_a100s} available")

            gpu_ids = _trainer_has_sufficient_gpus(trainer.gpus, required_gpus)
            if gpu_ids:
                logger.info(f"✓ Found suitable GPUs on trainer {trainer.trainer_ip} for requirement {required_gpus.value}: GPU IDs {gpu_ids}")
                return trainer.trainer_ip, gpu_ids
            else:
                logger.info(f"✗ Trainer {trainer.trainer_ip} does not have sufficient GPUs for requirement {required_gpus.value}")

        logger.warning(f"No suitable GPUs found across all {len(trainers)} trainers for requirement {required_gpus.value}")
        return None

    except Exception as e:
        logger.error(f"Error checking suitable GPUs: {str(e)}", exc_info=True)
        return None


async def _create_training_request(
    task: AnyTypeRawTask, hotkey: str, available_gpu_ids: list[int], config: Config
) -> TrainerProxyRequest:
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

    logger.info(f"Creating training request for hotkey {hotkey}, task {task.task_id}")
    logger.info(f"Expected repo name: {expected_repo_name}")
    logger.info(f"Training repo from DB: {training_repo}")
    logger.info(f"Training commit hash from DB: {training_commit_hash}")

    # Validate that training repo exists for this hotkey
    if training_repo is None:
        logger.error(f"No training repository found for hotkey {hotkey} in tournament_participants table")
        logger.error(
            "This hotkey may not be registered as a tournament participant or the training repo was not properly set during tournament registration"
        )
        raise ValueError(
            f"No training repository found for hotkey {hotkey}. This hotkey may not be registered as a tournament participant or the training repo was not properly set during tournament registration."
        )

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
            file_format=FileFormat.S3,  # always an S3 since we task prep
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
    
    logger.debug(f"_trainer_has_sufficient_gpus: Found {len(available_h100s)} available H100s, {len(available_a100s)} available A100s")
    logger.debug(f"Checking against requirement: {requirement.value}")

    if requirement == GpuRequirement.A100:
        if len(available_a100s) >= 1:
            gpu_ids = [available_a100s[0].gpu_id]
            logger.debug(f"Requirement A100 met: returning GPU IDs {gpu_ids}")
            return gpu_ids
        else:
            logger.debug(f"Requirement A100 NOT met: need 1, have {len(available_a100s)}")
            return []
    elif requirement == GpuRequirement.H100_1X:
        if len(available_h100s) >= 1:
            gpu_ids = [available_h100s[0].gpu_id]
            logger.debug(f"Requirement H100_1X met: returning GPU IDs {gpu_ids}")
            return gpu_ids
        else:
            logger.debug(f"Requirement H100_1X NOT met: need 1, have {len(available_h100s)}")
            return []
    elif requirement == GpuRequirement.H100_2X:
        if len(available_h100s) >= 2:
            gpu_ids = [gpu.gpu_id for gpu in available_h100s[:2]]
            logger.debug(f"Requirement H100_2X met: returning GPU IDs {gpu_ids}")
            return gpu_ids
        else:
            logger.debug(f"Requirement H100_2X NOT met: need 2, have {len(available_h100s)}")
            return []
    elif requirement == GpuRequirement.H100_4X:
        if len(available_h100s) >= 4:
            gpu_ids = [gpu.gpu_id for gpu in available_h100s[:4]]
            logger.debug(f"Requirement H100_4X met: returning GPU IDs {gpu_ids}")
            return gpu_ids
        else:
            logger.debug(f"Requirement H100_4X NOT met: need 4, have {len(available_h100s)}")
            return []
    elif requirement == GpuRequirement.H100_8X:
        if len(available_h100s) >= 8:
            gpu_ids = [gpu.gpu_id for gpu in available_h100s[:8]]
            logger.debug(f"Requirement H100_8X met: returning GPU IDs {gpu_ids}")
            return gpu_ids
        else:
            logger.debug(f"Requirement H100_8X NOT met: need 8, have {len(available_h100s)}")
            return []

    logger.warning(f"Unknown GPU requirement: {requirement}")
    return []


async def monitor_training_tasks(config: Config):
    """
    Monitor training tasks and update GPU availability based on completion status.
    """
    while True:
        try:
            logger.info("Monitoring training tasks")
            await _monitor_training_tasks(config)
        except Exception as e:
            logger.error(f"Error in monitor_training_tasks cycle: {str(e)}", exc_info=True)
        finally:
            await asyncio.sleep(cst.MONITOR_TRAINING_TASKS_CYCLE_INTERVAL)


async def _monitor_training_tasks(config: Config):
    """
    Monitor training tasks and update GPU availability based on completion status.
    """
    # Get all tasks currently in training status
    training_tasks = await tournament_sql.get_tournament_training_tasks(config.psql_db, TrainingStatus.TRAINING)
    logger.info(f"Found {len(training_tasks)} tasks currently in training")

    if not training_tasks:
        logger.info("No tasks in training, skipping monitoring cycle")
        return

    # Track if any tasks completed to determine if we need to update GPU availability
    any_completed = False

    # Get all trainers to check task status
    trainers = await tournament_sql.get_trainers(config.psql_db)

    # Check each training task
    for training_task in training_tasks:
        tournament_id = await get_tournament_id_by_task_id(training_task.task.task_id, config.psql_db)
        if tournament_id is None:
            logger.warning(f"Task {training_task.task.task_id} not found in tournament_tasks table - no tournament_id available")
        with LogContext(task_id=str(training_task.task.task_id), hotkey=training_task.hotkey, tournament_id=tournament_id):
            try:
                # Query all trainers for this task
                logger.info(
                    f"Checking task {training_task.task.task_id} with hotkey {training_task.hotkey} "
                    f"on trainers {[trainer.trainer_ip for trainer in trainers]}"
                )
                responses = []
                for trainer in trainers:
                    try:
                        task_log = await get_training_task_details(
                            trainer.trainer_ip, str(training_task.task.task_id), training_task.hotkey
                        )
                        if task_log:
                            responses.append((trainer.trainer_ip, task_log))
                    except httpx.HTTPStatusError as e:
                        status_code = e.response.status_code
                        if 500 <= status_code < 600:
                            logger.error(f"Server error ({status_code}) from trainer {trainer.trainer_ip}: {str(e)}")
                        continue
                    except Exception as e:
                        logger.info(f"Could not get task details from trainer {trainer.trainer_ip}: {str(e)}")
                        continue

                if not responses:
                    logger.warning(
                        f"Could not find trainer for task {training_task.task.task_id} with hotkey {training_task.hotkey}"
                    )
                    # Move task back to PENDING since trainer may have restarted or lost the task
                    await tournament_sql.update_tournament_task_training_status(
                        training_task.task.task_id, training_task.hotkey, TrainingStatus.PENDING, config.psql_db
                    )
                    logger.info(
                        f"Moved task {training_task.task.task_id} with hotkey {training_task.hotkey} back to PENDING status"
                    )
                    continue

                # Gather all statuses
                statuses = [task_log.status for _, task_log in responses]
                # Priority: SUCCESS > TRAINING > FAILURE
                if TaskStatus.SUCCESS in statuses:
                    any_completed = True
                    logger.info(
                        f"Task {training_task.task.task_id} with hotkey {training_task.hotkey} completed with status SUCCESS "
                        f"(at least one trainer)"
                    )
                    await tournament_sql.update_tournament_task_training_status(
                        training_task.task.task_id, training_task.hotkey, TrainingStatus.SUCCESS, config.psql_db
                    )
                elif all(s == TaskStatus.FAILURE for s in statuses):
                    any_completed = True
                    logger.info(f"Task {training_task.task.task_id} with hotkey {training_task.hotkey} failed on all trainers")
                    await tournament_sql.update_tournament_task_training_status(
                        training_task.task.task_id, training_task.hotkey, TrainingStatus.PENDING, config.psql_db
                    )
                else:
                    logger.info(
                        f"Task {training_task.task.task_id} with hotkey {training_task.hotkey} is still training on at least "
                        f"one trainer"
                    )

            except Exception as e:
                logger.error(f"Error checking task {training_task.task.task_id} with hotkey {training_task.hotkey}: {str(e)}")
                continue

    # If any tasks completed, update all trainers' GPU availability
    if any_completed:
        logger.info("Found completed tasks, updating GPU availability across all trainers")
        await _update_all_trainers_gpu_availability(config)

    logger.info(f"Completed monitoring cycle, processed {len(training_tasks)} tasks")


async def _update_all_trainers_gpu_availability(config: Config):
    """
    Update GPU availability for all trainers by fetching current status and syncing with database.
    """
    try:
        # Get all trainers from database
        trainers = await tournament_sql.get_trainers(config.psql_db)

        for trainer in trainers:
            try:
                # Fetch current GPU availability from trainer
                current_gpus = await fetch_trainer_gpus(trainer.trainer_ip)

                # Find GPUs that are free according to trainer but marked as used in DB
                gpus_to_reset = []
                for current_gpu in current_gpus:
                    if current_gpu.available:
                        # Check if this GPU is marked as used in our database
                        for db_gpu in trainer.gpus:
                            if db_gpu.gpu_id == current_gpu.gpu_id and not db_gpu.available:
                                gpus_to_reset.append(current_gpu.gpu_id)
                                break

                # Reset GPU availability in database if needed
                if gpus_to_reset:
                    await tournament_sql.update_gpu_availability(trainer.trainer_ip, gpus_to_reset, 0, config.psql_db)
                    logger.info(f"Reset {len(gpus_to_reset)} GPUs for trainer {trainer.trainer_ip}: {gpus_to_reset}")

            except Exception as e:
                logger.error(f"Error updating GPU availability for trainer {trainer.trainer_ip}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in _update_all_trainers_gpu_availability: {str(e)}")


async def move_completed_tasks_to_preevaluation(config: Config):
    """
    Find tasks where all training tasks (task_id, hotkey) pairs have completed
    and move those tasks to preevaluation status.
    """
    while True:
        try:
            logger.info("Moving completed tournament tasks to preevaluation")
            await _move_completed_tasks_to_preevaluation(config)
        except Exception as e:
            logger.error(f"Error in move_completed_tasks_to_preevaluation cycle: {str(e)}", exc_info=True)
        finally:
            await asyncio.sleep(cst.MOVE_COMPLETED_TASKS_CYCLE_INTERVAL)


async def _move_completed_tasks_to_preevaluation(config: Config):
    """
    Find tasks where all training tasks (task_id, hotkey) pairs have completed
    and move those tasks to preevaluation status.
    """
    # Get task IDs where all training tasks have completed (only from last month)
    completed_task_ids = await tournament_sql.get_tasks_with_all_training_completed(config.psql_db)
    logger.info(f"Found {len(completed_task_ids)} tasks with all training completed")

    if not completed_task_ids:
        logger.info("No tasks with all training completed, skipping cycle")
        return

    # Get the actual task objects for these IDs
    tasks_to_move = []
    for task_id in completed_task_ids:
        task = await task_sql.get_task(task_id, config.psql_db)
        if task:
            tasks_to_move.append(task)

    logger.info(f"Moving {len(tasks_to_move)} tasks to preevaluation status")

    # Move tasks to preevaluation
    for task in tasks_to_move:
        tournament_id = await get_tournament_id_by_task_id(task.task_id, config.psql_db)
        with LogContext(task_id=str(task.task_id), tournament_id=tournament_id):
            try:
                task.status = TaskStatus.PREEVALUATION
                await task_sql.update_task(task, config.psql_db)
                logger.info(f"Moved task {task.task_id} from training to preevaluation status")
            except Exception as e:
                logger.error(f"Error moving task {task.task_id} to preevaluation: {str(e)}")

    logger.info(f"Successfully moved {len(tasks_to_move)} tasks to preevaluation status")


async def reset_all_gpu_availability(config: Config):
    """
    Manually reset GPU availability for all trainers by setting used_until to NULL.
    This can be used to fix stuck GPU availability states.
    """
    try:
        logger.info("Manually resetting GPU availability for all trainers")
        trainers = await tournament_sql.get_trainers(config.psql_db)

        for trainer in trainers:
            gpu_ids = [gpu.gpu_id for gpu in trainer.gpus]
            if gpu_ids:
                await tournament_sql.update_gpu_availability(trainer.trainer_ip, gpu_ids, 0, config.psql_db)
                logger.info(f"Reset {len(gpu_ids)} GPUs for trainer {trainer.trainer_ip}")

        logger.info("Successfully reset GPU availability for all trainers")
    except Exception as e:
        logger.error(f"Error resetting GPU availability: {str(e)}")


async def update_all_trainers_gpu_availability_cycle(config: Config):
    """
    Periodically update GPU availability for all trainers.
    """
    while True:
        try:
            logger.info("Periodically updating all trainers' GPU availability")
            await _update_all_trainers_gpu_availability(config)
        except Exception as e:
            logger.error(f"Error in periodic GPU availability update: {str(e)}", exc_info=True)
        finally:
            await asyncio.sleep(cst.PERIODIC_GPU_AVAILABILITY_UPDATE_INTERVAL)


async def run_tournament_orchestrator_cycles():
    config = load_config()
    await try_db_connections(config)

    logger.info("Starting tournament orchestrator cycles")
    await asyncio.gather(
        fetch_tournament_tasks_ready_to_train(config),
        process_pending_tournament_tasks(config),
        monitor_training_tasks(config),
        move_completed_tasks_to_preevaluation(config),
        update_all_trainers_gpu_availability_cycle(config),
    )


if __name__ == "__main__":
    load_dotenv(".vali.env", override=True)
    asyncio.run(run_tournament_orchestrator_cycles())
