import asyncio
import datetime
import random

from datasets import get_dataset_infos
from fiber import Keypair
from fiber.chain.models import Node
from fiber.logging_utils import get_logger

import validator.core.constants as cst
import validator.db.sql.nodes as nodes_sql
import validator.db.sql.tasks as tasks_sql
from core.models.payload_models import MinerTaskRequest
from core.models.payload_models import MinerTaskResponse
from core.models.payload_models import TrainRequest
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.models import Task
from validator.core.refresh_nodes import get_and_store_nodes
from validator.evaluation.scoring import evaluate_and_score
from validator.evaluation.weight_setting import set_weights_periodically
from validator.tasks.scheduler import synthetic_task_loop
from validator.tasks.task_prep import prepare_task
from validator.utils.call_endpoint import process_non_stream_fiber


logger = get_logger(__name__)


def _get_total_dataset_size(repo_name: str) -> int:
    return sum(info.dataset_size for info in get_dataset_infos(repo_name).values() if info.dataset_size)


async def _run_task_prep(task: Task, keypair: Keypair) -> Task:
    logger.info(f"The task coming into task prep is {task}")
    columns_to_sample = [i for i in [task.system, task.instruction, task.input, task.output] if i is not None]
    test_data, synth_data, train_data = await prepare_task(
        dataset_name=task.ds_id, columns_to_sample=columns_to_sample, keypair=keypair
    )
    task.training_data = train_data
    task.status = TaskStatus.LOOKING_FOR_NODES
    task.synthetic_data = synth_data
    task.test_data = test_data
    logger.info("Data creation is complete - now time to find some miners")
    return task


# TODO: Improve by batching these up
async def _make_offer(node: Node, request: MinerTaskRequest, config: Config) -> MinerTaskResponse:
    response = await process_non_stream_fiber(cst.TASK_OFFER_ENDPOINT, config, node, request.model_dump(), timeout=3)
    logger.info(f"The response from make offer was {response}")
    if response is None:
        response = {}
    return MinerTaskResponse(
        message=response.get("message", "No message given"),
        accepted=response.get("accepted", False),
    )


async def _select_miner_pool_and_add_to_task(task: Task, nodes: list[Node], config: Config) -> Task:
    if len(nodes) < cst.MINIMUM_MINER_POOL:
        logger.warning(f"Not enough nodes available. Need at least {cst.MINIMUM_MINER_POOL}, but only have {len(nodes)}.")
        task = attempt_delay_task(task)
        return task

    selected_miners: list[str] = []
    ds_size = _get_total_dataset_size(task.ds_id)
    task_request = MinerTaskRequest(
        ds_size=ds_size,
        model=task.model_id,
        hours_to_complete=task.hours_to_complete,
        task_id=str(task.task_id),
        parameter_count=task.parameter_count,
    )
    logger.info(f"We are offering the following task to the miners: {task_request.model_dump()}")
    miners_already_assigned = await tasks_sql.get_miners_for_task(task.task_id, config.psql_db)

    already_assigned_hotkeys = [miner.hotkey for miner in miners_already_assigned]
    logger.info(f"There are {len(already_assigned_hotkeys)} miners already assigned to this task")

    # Filter out nodes that are already assigned to this task - this will occur if we had to restart a task due to all miners
    # failing
    available_nodes = [node for node in nodes if node.hotkey not in already_assigned_hotkeys]
    if not available_nodes:
        logger.error("No nodes available to assign to the task! Why not?!")
        task = attempt_delay_task(task)
        await tasks_sql.update_task(task, config.psql_db)
        return task

    num_of_miners_to_try_for = random.randint(cst.MIN_IDEAL_NUM_MINERS_IN_POOL, cst.MAX_IDEAL_NUM_MINERS_IN_POOL)
    random.shuffle(available_nodes)

    # TODO: Improve by selecting high score miners first, then lower score miners, etc
    i = 0
    while len(selected_miners) < num_of_miners_to_try_for and available_nodes:
        node = available_nodes.pop()
        # try:
        # TODO: Batch the boi
        if i > 0 and i % 5 == 0:
            logger.info(f"We have made {i} offers so far for task {task.task_id}")
        offer_response = await _make_offer(node, task_request, config)

        if offer_response.accepted is True:
            selected_miners.append(node.hotkey)
            await tasks_sql.assign_node_to_task(str(task.task_id), node, config.psql_db)
            logger.info(f"The miner {node.node_id} has officially been assigned the task {task.task_id}!!")

    if len(selected_miners) < cst.MINIMUM_MINER_POOL:
        logger.warning(
            f"Not enough miners accepted the task. We only have {len(selected_miners)} but we "
            f"need at least {cst.MINIMUM_MINER_POOL}"
        )
        task = attempt_delay_task(task)
        return task
    else:
        task.assigned_miners = selected_miners
        logger.info(f"We have {len(selected_miners)} miners assigned to the task - which is enough to get going 🚀")
        task.status = TaskStatus.READY
        logger.info("Task status should be READY")
        return task


async def _let_miners_know_to_start_training(task: Task, nodes: list[Node], config: Config):
    dataset_type = CustomDatasetType(
        field_system=task.system,
        field_input=task.input,
        field_output=task.output,
        field_instruction=task.instruction,
        format=task.format,
        no_input_format=task.no_input_format,
    )

    dataset = task.training_data if task.training_data else "dataset error"
    task_request_body = TrainRequest(
        dataset=dataset,
        model=task.model_id,
        dataset_type=dataset_type,
        file_format=FileFormat.S3,
        task_id=str(task.task_id),
        hours_to_complete=task.hours_to_complete,
        parameter_count=task.parameter_count,
    )

    logger.info(f"We are telling miners to start training, there are {len(nodes)}")

    for node in nodes:
        response = await process_non_stream_fiber(cst.START_TRAINING_ENDPOINT, config, node, task_request_body.model_dump())
        logger.info(f"The response we got from {node.node_id} was {response}")


async def assign_miners(task: Task, config: Config):
    try:
        nodes = await nodes_sql.get_all_nodes(config.psql_db)
        task = await _select_miner_pool_and_add_to_task(task, nodes, config)
        logger.info(f"After assigning miners here is the current task info {task}")
        await tasks_sql.update_task(task, config.psql_db)

    except Exception as e:
        logger.error(f"Error assigning miners to task {task.task_id}: {e}", exc_info=True)
        task = attempt_delay_task(task)
        await tasks_sql.update_task(task, config.psql_db)


def attempt_delay_task(task: Task):
    assert (
        task.created_timestamp is not None and task.delay_timestamp is not None and task.delay_times is not None
    ), "We wanted to check delay vs created timestamps but they are missing"

    if task.delay_times >= cst.MAX_DELAY_TIMES or not task.is_organic:
        if task.is_organic:
            logger.info(f"We have already delayed {task.delay_times}")
        else:
            logger.info("This is a synth task - no need to add a delay when the network is busy")

        task.status = TaskStatus.FAILURE_FINDING_NODES
    else:
        logger.info(f"Adding in a delay of {cst.TASK_TIME_DELAY} minutes for now since no miners accepted the task")

        task.delay_timestamp = task.delay_timestamp + datetime.timedelta(minutes=cst.TASK_TIME_DELAY)
        task.status = TaskStatus.DELAYED
        task.delay_times += 1
    return task


async def _find_miners_for_task(config: Config):
    pending_tasks = await tasks_sql.get_tasks_with_status(status=TaskStatus.LOOKING_FOR_NODES, psql_db=config.psql_db)
    await asyncio.gather(*[assign_miners(task, config) for task in pending_tasks[: cst.MAX_CONCURRENT_MINER_ASSIGNMENTS]])


async def prep_task(task: Task, config: Config):
    logger.info("Task is being prepped ready for miners")
    try:
        task.status = TaskStatus.PREPARING_DATA
        await tasks_sql.update_task(task, config.psql_db)
        task = await _run_task_prep(task, config.keypair)
        await tasks_sql.update_task(task, config.psql_db)
        logger.info(f"After prep task we have {task}")
    except Exception:
        task.status = TaskStatus.PREP_TASK_FAILURE
        await tasks_sql.update_task(task, config.psql_db)


async def _process_selected_tasks(config: Config):
    miner_selected_tasks = await tasks_sql.get_tasks_with_status(status=TaskStatus.PENDING, psql_db=config.psql_db)
    await asyncio.gather(*[prep_task(task, config) for task in miner_selected_tasks[: cst.MAX_CONCURRENT_TASK_PREPS]])


async def _start_training_task(task: Task, config: Config) -> None:
    task.started_timestamp = datetime.datetime.now()
    task.end_timestamp = task.started_timestamp + datetime.timedelta(hours=task.hours_to_complete)
    assigned_miners = await tasks_sql.get_nodes_assigned_to_task(str(task.task_id), config.psql_db)
    logger.info(f"Here are the miners that have been assigned {assigned_miners}")
    await _let_miners_know_to_start_training(task, assigned_miners, config)
    task.status = TaskStatus.TRAINING
    await tasks_sql.update_task(task, config.psql_db)
    logger.info("SUCCESS IN STARTING TRAINING")


async def _process_ready_to_train_tasks(config: Config):
    ready_to_train_tasks = await tasks_sql.get_tasks_with_status(status=TaskStatus.READY, psql_db=config.psql_db)
    if len(ready_to_train_tasks) > 0:
        logger.info(f"There are {len(ready_to_train_tasks)} ready to train")
        await asyncio.gather(
            *[_start_training_task(task, config) for task in ready_to_train_tasks[: cst.MAX_CONCURRENT_TRAININGS]]
        )
    else:
        logger.info("No pending tasks - waiting for 30 seconds")
        await asyncio.sleep(30)


async def _evaluate_task(task: Task, config: Config):
    try:
        task.status = TaskStatus.EVALUATING
        await tasks_sql.update_task(task, config.psql_db)
        task = await evaluate_and_score(task, config)
        await tasks_sql.update_task(task, config.psql_db)
    except Exception as e:
        logger.error(f"Error evaluating task {task.task_id}: {e}", exc_info=True)
        task.status = TaskStatus.FAILURE
        await tasks_sql.update_task(task, config.psql_db)


async def process_completed_tasks(config: Config) -> None:
    while True:
        completed_tasks = await tasks_sql.get_tasks_ready_to_evaluate(config.psql_db)
        if len(completed_tasks) > 0:
            logger.info(f"There are {len(completed_tasks)} awaiting evaluation")
            for task in completed_tasks:
                await _evaluate_task(task, config)
        if len(completed_tasks) == 0:
            logger.info("There are no tasks to evaluate - waiting 30 seconds")
            await asyncio.sleep(30)


async def _move_back_to_looking_for_nodes(task: Task, config: Config):
    task.status = TaskStatus.LOOKING_FOR_NODES
    await tasks_sql.update_task(task, config.psql_db)


async def _handle_delayed_tasks(config: Config):
    finished_delay_tasks = await tasks_sql.get_tasks_with_status(TaskStatus.DELAYED, psql_db=config.psql_db)
    logger.info(f"We have {len(finished_delay_tasks)} that we're ready to offer to miners again")
    await asyncio.gather(*[_move_back_to_looking_for_nodes(task, config) for task in finished_delay_tasks])


async def process_pending_tasks(config: Config) -> None:
    while True:
        try:
            await _process_selected_tasks(config)
            await _handle_delayed_tasks(config)
            await _find_miners_for_task(config)
            await _process_ready_to_train_tasks(config)
        except Exception as e:
            logger.info(f"There was a problem in processing: {e}")
            await asyncio.sleep(30)


async def validator_cycle(config: Config) -> None:
    await asyncio.gather(
        process_completed_tasks(config),
        process_pending_tasks(config),
    )


async def node_refresh_cycle(config: Config) -> None:
    while True:
        try:
            logger.info("Attempting to refresh_nodes")
            # 1 minute timeout
            await asyncio.wait_for(get_and_store_nodes(config), timeout=60)
            await asyncio.sleep(900)
        except asyncio.TimeoutError:
            logger.error("Node refresh timed out after 5 minutes... :( Please look into this!!")
            await asyncio.sleep(60)


async def run_validator_cycles(config: Config) -> None:
    await asyncio.gather(
        node_refresh_cycle(config),
        synthetic_task_loop(config),
        set_weights_periodically(config),
        _run_main_validator_loop(config),
    )


async def _run_main_validator_loop(config: Config) -> None:
    while True:
        await validator_cycle(config)
        # Add small sleep to prevent tight loop
        await asyncio.sleep(30)


def init_validator_cycles(config: Config) -> Task:
    return asyncio.create_task(run_validator_cycles(config))
