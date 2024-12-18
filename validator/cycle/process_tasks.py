import asyncio
import datetime
import random

from datasets import get_dataset_infos
from fiber import Keypair
from fiber.chain.models import Node


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
from validator.core.models import RawTask
from validator.evaluation.scoring import evaluate_and_score
from validator.tasks.task_prep import prepare_task
from validator.utils.call_endpoint import process_non_stream_fiber
from validator.utils.logging import TaskContext, create_extra_log
from validator.utils.logging import logger


def _get_total_dataset_size(repo_name: str) -> int:
    return sum(info.dataset_size for info in get_dataset_infos(repo_name).values() if info.dataset_size)


async def _run_task_prep(task: RawTask, keypair: Keypair) -> RawTask:
    columns_to_sample = [
        i for i in [task.field_system, task.field_instruction, task.field_input, task.field_output] if i is not None
    ]
    test_data, synth_data, train_data = await prepare_task(
        dataset_name=task.ds_id, columns_to_sample=columns_to_sample, keypair=keypair
    )
    task.training_data = train_data
    task.status = TaskStatus.LOOKING_FOR_NODES
    task.synthetic_data = synth_data
    task.test_data = test_data
    logger.info(
        "Data creation is complete - now time to find some miners",
        extra=create_extra_log(status=task.status),
    )
    return task


# TODO: Improve by batching these up
async def _make_offer(node: Node, request: MinerTaskRequest, config: Config) -> MinerTaskResponse:
    response = await process_non_stream_fiber(cst.TASK_OFFER_ENDPOINT, config, node, request.model_dump(), timeout=3)
    logger.info(
        f"The response from make offer for node {node.node_id} was {response}",
        extra=create_extra_log(node_hotkey=node.hotkey, status=TaskStatus.LOOKING_FOR_NODES.value),
    )
    if response is None:
        response = {}
    return MinerTaskResponse(
        message=response.get("message", "No message given"),
        accepted=response.get("accepted", False),
    )


async def _select_miner_pool_and_add_to_task(task: RawTask, nodes: list[Node], config: Config) -> RawTask:
    if len(nodes) < cst.MINIMUM_MINER_POOL:
        logger.warning(
            f"Not enough nodes available. Need at least {cst.MINIMUM_MINER_POOL}, but only have {len(nodes)}.",
            extras=create_extra_log(status=task.status),
        )
        task = _attempt_delay_task(task)
        return task

    selected_miners: list[str] = []
    ds_size = _get_total_dataset_size(task.ds_id)
    task_request = MinerTaskRequest(
        ds_size=ds_size,
        model=task.model_id,
        hours_to_complete=task.hours_to_complete,
        task_id=str(task.task_id),
    )
    logger.info(
        f"We are offering the following task to the miners: {task_request.model_dump()}",
        extras=create_extra_log(task.status),
    )
    miners_already_assigned = await tasks_sql.get_miners_for_task(task.task_id, config.psql_db)

    already_assigned_hotkeys = [miner.hotkey for miner in miners_already_assigned]
    logger.info(
        f"There are {len(already_assigned_hotkeys)} miners already assigned to this task",
        extras=create_extra_log(task.status),
    )

    # Filter out nodes that are already assigned to this task - this will occur if we had to restart a task due to all miners
    # failing
    available_nodes = [node for node in nodes if node.hotkey not in already_assigned_hotkeys]
    if not available_nodes:
        logger.error(
            "No nodes available to assign to the task! Why not?!",
            extras=create_extra_log(status=task.status),
        )
        task = _attempt_delay_task(task)
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
            logger.info(
                f"We have made {i} offers so far for task {task.task_id}",
                extras=create_extra_log(status=task.status),
            )
        offer_response = await _make_offer(node, task_request, config)

        if offer_response.accepted is True:
            selected_miners.append(node.hotkey)
            await tasks_sql.assign_node_to_task(str(task.task_id), node, config.psql_db)
            logger.info(
                f"The miner {node.node_id} has officially been assigned the task {task.task_id}!!",
                extras=create_extra_log(node_hotkey=node.hotkey, status=task.status),
            )

    if len(selected_miners) < cst.MINIMUM_MINER_POOL:
        logger.warning(
            f"Not enough miners accepted the task. We only have {len(selected_miners)} but we "
            f"need at least {cst.MINIMUM_MINER_POOL}",
            extras=create_extra_log(status=task.status),
        )
        task = _attempt_delay_task(task)
        return task
    else:
        task.assigned_miners = selected_miners
        logger.info(
            f"We have {len(selected_miners)} miners assigned to the task - which is enough to get going 🚀",
            extras=create_extra_log(status=task.status),
        )
        task.status = TaskStatus.READY
        logger.info("Task status should be READY", extras=create_extra_log(status=task.status))
        return task


async def _let_miners_know_to_start_training(task: RawTask, nodes: list[Node], config: Config):
    dataset_type = CustomDatasetType(
        field_system=task.field_system,
        field_input=task.field_input,
        field_output=task.field_output,
        field_instruction=task.field_instruction,
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
    )

    logger.info(
        f"We are telling miners to start training, there are {len(nodes)}",
        extras=create_extra_log(status=task.status),
    )

    for node in nodes:
        response = await process_non_stream_fiber(cst.START_TRAINING_ENDPOINT, config, node, task_request_body.model_dump())
        logger.info(
            f"The response we got from {node.node_id} was {response}",
            extras=create_extra_log(node_hokey=node.hotkey, status=task.status),
        )


async def _find_and_select_miners_for_task(task: RawTask, config: Config):
    async with TaskContext(str(task.task_id)):
        try:
            nodes = await nodes_sql.get_all_nodes(config.psql_db)
            task = await _select_miner_pool_and_add_to_task(task, nodes, config)
            logger.info(
                f"After assigning miners here is the current task info {task}",
                extras=create_extra_log(status=task.status),
            )
            await tasks_sql.update_task(task, config.psql_db)

        except Exception as e:
            logger.error(
                f"Error assigning miners to task {task.task_id}: {e}",
                exc_info=True,
                extras=create_extra_log(status=task.status),
            )
            task = _attempt_delay_task(task)
            await tasks_sql.update_task(task, config.psql_db)


def _attempt_delay_task(task: RawTask):
    assert (
        task.created_at is not None and task.next_delay_at is not None and task.times_delayed is not None
    ), "We wanted to check delay vs created timestamps but they are missing"

    if task.times_delayed >= cst.MAX_DELAY_TIMES or not task.is_organic:
        if task.is_organic:
            logger.info(
                f"We have already delayed {task.times_delayed}",
                extras=create_extra_log(status=task.status),
            )
        else:
            logger.info(
                "This is a synth task - no need to add a delay when the network is busy",
                extras=create_extra_log(status=task.status),
            )

        task.status = TaskStatus.FAILURE_FINDING_NODES
    else:
        logger.info(
            f"Adding in a delay of {cst.TASK_TIME_DELAY} minutes for now since no miners accepted the task",
            extras=create_extra_log(status=task.status),
        )

        task.next_delay_at = task.next_delay_at + datetime.timedelta(minutes=cst.TASK_TIME_DELAY)
        task.status = TaskStatus.DELAYED
        task.times_delayed += 1
    return task


async def _find_miners_for_task(config: Config):
    pending_tasks = await tasks_sql.get_tasks_with_status(status=TaskStatus.LOOKING_FOR_NODES, psql_db=config.psql_db)
    await asyncio.gather(
        *[_find_and_select_miners_for_task(task, config) for task in pending_tasks[: cst.MAX_CONCURRENT_MINER_ASSIGNMENTS]]
    )


async def _prep_task(task: RawTask, config: Config):
    async with TaskContext(str(task.task_id)):
        try:
            task.status = TaskStatus.PREPARING_DATA
            await tasks_sql.update_task(task, config.psql_db)
            task = await _run_task_prep(task, config.keypair)
            logger.info(f"THE TASK HAS BEEN PREPPED {task}", extra=create_extra_log(status=task.status))
            await tasks_sql.update_task(task, config.psql_db)
        except Exception:
            task.status = TaskStatus.PREP_TASK_FAILURE
            await tasks_sql.update_task(task, config.psql_db)


async def _processing_pending_tasks(config: Config):
    logger.debug("Processing pending tasks")

    pending_tasks = await tasks_sql.get_tasks_with_status(status=TaskStatus.PENDING, psql_db=config.psql_db)
    logger.info(f"Found {len(pending_tasks)} pending tasks! Will prep them all now...")
    await asyncio.gather(*[_prep_task(task, config) for task in pending_tasks[: cst.MAX_CONCURRENT_TASK_PREPS]])


async def _start_training_task(task: RawTask, config: Config) -> None:
    async with TaskContext(str(task.task_id)):
        task.started_at = datetime.datetime.now()
        task.termination_at = task.started_at + datetime.timedelta(hours=task.hours_to_complete)
        assigned_miners = await tasks_sql.get_nodes_assigned_to_task(str(task.task_id), config.psql_db)
        logger.info(
            f"Here are the miners that have been assigned {assigned_miners}",
            extras=create_extra_log(status=task.status),
        )
        await _let_miners_know_to_start_training(task, assigned_miners, config)
        task.status = TaskStatus.TRAINING
        await tasks_sql.update_task(task, config.psql_db)
        logger.info("SUCCESS IN STARTING TRAINING", extras=create_extra_log(status=task.status))


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


async def _evaluate_task(task: RawTask, config: Config):
    try:
        task.status = TaskStatus.EVALUATING
        await tasks_sql.update_task(task, config.psql_db)
        task = await evaluate_and_score(task, config)
        await tasks_sql.update_task(task, config.psql_db)
    except Exception as e:
        logger.error(
            f"Error evaluating task {task.task_id}: {e}",
            exc_info=True,
            extras=create_extra_log(status=task.status),
        )
        task.status = TaskStatus.FAILURE
        await tasks_sql.update_task(task, config.psql_db)


async def _move_back_to_looking_for_nodes(task: RawTask, config: Config):
    logger.info(
        "Moving back from delay to looking for nodes",
        extras=create_extra_log(task_id=str(task.task_id), status=task.status),
    )
    task.status = TaskStatus.LOOKING_FOR_NODES
    await tasks_sql.update_task(task, config.psql_db)


async def _handle_delayed_tasks(config: Config):
    finished_delay_tasks = await tasks_sql.get_tasks_with_status(TaskStatus.DELAYED, psql_db=config.psql_db)
    logger.info(f"We have {len(finished_delay_tasks)} that we're ready to offer to miners again")
    await asyncio.gather(*[_move_back_to_looking_for_nodes(task, config) for task in finished_delay_tasks])


async def process_pending_tasks(config: Config) -> None:
    while True:
        try:
            await _processing_pending_tasks(config)
            await _handle_delayed_tasks(config)
            await _find_miners_for_task(config)
            await _process_ready_to_train_tasks(config)
        except Exception as e:
            logger.info(f"There was a problem in processing: {e}")
            await asyncio.sleep(30)


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
