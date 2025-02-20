import asyncio
import datetime
import random
import uuid

from fiber.chain.models import Node

import validator.core.constants as cst
import validator.db.sql.nodes as nodes_sql
import validator.db.sql.tasks as tasks_sql
from core.models.payload_models import MinerTaskOffer
from core.models.payload_models import MinerTaskResponse
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.core.config import Config
from validator.core.models import ImageRawTask
from validator.core.models import TextRawTask
from validator.core.task_config_models import get_task_config
from validator.evaluation.scoring import evaluate_and_score
from validator.utils.cache_clear import clean_all_hf_datasets_cache
from validator.utils.call_endpoint import process_non_stream_fiber
from validator.utils.logging import LogContext
from validator.utils.logging import add_context_tag
from validator.utils.logging import get_logger


logger = get_logger(__name__)

def _weighted_random_shuffle(nodes: list[Node]):
    top_node_chance = 3  # This is the chance that the top node is picked compared to the bottom node
    nodes.sort(key=lambda x: x.incentive if x.incentive is not None else 0, reverse=True)
    weights = [top_node_chance - i * (top_node_chance - 1) / len(nodes) for i in range(1, len(nodes) + 1)]

    shuffled_nodes = []
    for _ in range(len(nodes)):
        index = random.choices(range(len(nodes)), weights=weights, k=1)[0]
        shuffled_nodes.append(nodes[index])
        nodes.pop(index)
        weights.pop(index)

    return shuffled_nodes


# TODO: Improve by batching these up
async def _make_offer(node: Node, request: MinerTaskOffer, config: Config) -> MinerTaskResponse:
    endpoint = cst.TASK_OFFER_IMAGE_ENDPOINT if request.task_type == TaskType.IMAGETASK else cst.TASK_OFFER_ENDPOINT
    response = await process_non_stream_fiber(endpoint, config, node, request.model_dump(), timeout=3)
    logger.info(f"The response from make {request.task_type} offer for node {node.node_id} was {response}")
    if response is None:
        response = {}
    return MinerTaskResponse(
        message=response.get("message", "No message given"),
        accepted=response.get("accepted", False),
    )


async def _select_miner_pool_and_add_to_task(
    task: TextRawTask | ImageRawTask, nodes: list[Node], config: Config
) -> TextRawTask | ImageRawTask:
    if len(nodes) < cst.MINIMUM_MINER_POOL:
        logger.warning(f"Not enough nodes available. Need at least {cst.MINIMUM_MINER_POOL}, but only have {len(nodes)}.")
        task = _attempt_delay_task(task)
        return task

    selected_miners: list[str] = []
    ds_size = get_task_config(task).data_size_function(task)
    task_request = MinerTaskOffer(
        ds_size=ds_size,
        model=task.model_id,
        hours_to_complete=task.hours_to_complete,
        task_id=str(task.task_id),
        task_type=task.task_type,
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
        task = _attempt_delay_task(task)
        await tasks_sql.update_task(task, config.psql_db)
        return task

    num_of_miners_to_try_for = random.randint(cst.MIN_IDEAL_NUM_MINERS_IN_POOL, cst.MAX_IDEAL_NUM_MINERS_IN_POOL)
    nodes_to_try_for = _weighted_random_shuffle(available_nodes)

    # TODO: Improve by selecting high score miners first, then lower score miners, etc
    i = 0
    while len(selected_miners) < num_of_miners_to_try_for and nodes_to_try_for:
        node = nodes_to_try_for.pop()
        with LogContext(node_id=node.node_id, miner_hotkey=node.hotkey):
            # try:
            # TODO: Batch the boi
            if i > 0 and i % 5 == 0:
                logger.info(f"We have made {i} offers so far for task {task.task_id}")

            offer_response = await _make_offer(node, task_request, config)

            # Store offer response
            await tasks_sql.store_offer_response(task.task_id, node.hotkey, offer_response.model_dump_json(), config.psql_db)

            if offer_response.accepted is True:
                selected_miners.append(node.hotkey)
                await tasks_sql.assign_node_to_task(str(task.task_id), node, config.psql_db)
                logger.info(f"The miner {node.node_id} has officially been assigned the task {task.task_id}!!")

    if len(selected_miners) < cst.MINIMUM_MINER_POOL:
        logger.warning(
            f"Not enough miners accepted the task. We only have {len(selected_miners)} but we "
            f"need at least {cst.MINIMUM_MINER_POOL}"
        )
        task = _attempt_delay_task(task)
        return task
    else:
        task.assigned_miners = selected_miners
        logger.info(f"We have {len(selected_miners)} miners assigned to the task - which is enough to get going 🚀")
        task.status = TaskStatus.READY
        add_context_tag("status", task.status.value)
        return task


async def _let_miners_know_to_start_training(task: ImageRawTask | TextRawTask, nodes: list[Node], config: Config):
    task_request_body = get_task_config(task).task_request_prepare_function(task)
    miner_endpoint = get_task_config(task).start_training_endpoint

    logger.info(f"We are telling miners to start training, there are {len(nodes)}")

    for node in nodes:
        with LogContext(node_id=node.node_id, miner_hotkey=node.hotkey):
            expected_repo_name = str(uuid.uuid4())
            await tasks_sql.set_expected_repo_name(str(task.task_id), node, config.psql_db, expected_repo_name)
            task_request_body.expected_repo_name = expected_repo_name

            response = await process_non_stream_fiber(miner_endpoint, config, node, task_request_body.model_dump())
            logger.info(f"The response we got from {node.node_id} was {response}")


async def _find_and_select_miners_for_task(task: TextRawTask | ImageRawTask, config: Config):
    with LogContext(task_id=str(task.task_id)):
        try:
            nodes = await nodes_sql.get_all_nodes(config.psql_db)
            task = await _select_miner_pool_and_add_to_task(task, nodes, config)
            logger.info(f"After assigning miners here is the current task info {task}")
            await tasks_sql.update_task(task, config.psql_db)

        except Exception as e:
            logger.error(f"Error assigning miners to task {task.task_id}: {e}", exc_info=True)
            task = _attempt_delay_task(task)
            await tasks_sql.update_task(task, config.psql_db)


def _attempt_delay_task(task: TextRawTask | ImageRawTask):
    assert task.created_at is not None and task.next_delay_at is not None and task.times_delayed is not None, (
        "We wanted to check delay vs created timestamps but they are missing"
    )

    if task.times_delayed >= cst.MAX_DELAY_TIMES or not task.is_organic:
        if task.is_organic:
            logger.info(f"We have already delayed {task.times_delayed}")
        else:
            logger.info("This is a synth task - no need to add a delay when the network is busy")

        task.status = TaskStatus.FAILURE_FINDING_NODES
        add_context_tag("status", task.status.value)
    else:
        logger.info(f"Adding in a delay of {cst.TASK_TIME_DELAY} minutes for now since no miners accepted the task")
        task.next_delay_at = task.next_delay_at + datetime.timedelta(minutes=cst.TASK_TIME_DELAY)
        task.status = TaskStatus.DELAYED
        add_context_tag("status", task.status.value)
        task.times_delayed += 1
    return task


async def _find_miners_for_task(config: Config):
    pending_tasks = await tasks_sql.get_tasks_with_status(status=TaskStatus.LOOKING_FOR_NODES, psql_db=config.psql_db)
    await asyncio.gather(
        *[_find_and_select_miners_for_task(task, config) for task in pending_tasks[: cst.MAX_CONCURRENT_MINER_ASSIGNMENTS]]
    )


async def _prep_task(task: TextRawTask | ImageRawTask, config: Config):
    with LogContext(task_id=str(task.task_id)):
        try:
            task.status = TaskStatus.PREPARING_DATA
            add_context_tag("status", task.status.value)
            await tasks_sql.update_task(task, config.psql_db)
            task = await get_task_config(task).task_prep_function(task, config.keypair)
            logger.info(f"THE TASK HAS BEEN PREPPED {task}")
            await tasks_sql.update_task(task, config.psql_db)
        except Exception as e:
            logger.error(f"Error during task prep: {e}", exc_info=True)
            task.status = TaskStatus.PREP_TASK_FAILURE
            add_context_tag("status", task.status.value)
            await tasks_sql.update_task(task, config.psql_db)


async def _processing_pending_tasks(config: Config):
    logger.debug("Processing pending tasks")

    pending_tasks = await tasks_sql.get_tasks_with_status(status=TaskStatus.PENDING, psql_db=config.psql_db)
    logger.info(f"Found {len(pending_tasks)} pending tasks! Will prep them all now...")
    await asyncio.gather(*[_prep_task(task, config) for task in pending_tasks[: cst.MAX_CONCURRENT_TASK_PREPS]])
    clean_all_hf_datasets_cache()


async def _start_training_task(task: TextRawTask | ImageRawTask, config: Config) -> None:
    with LogContext(task_id=str(task.task_id)):
        task.started_at = datetime.datetime.now(datetime.timezone.utc)
        task.termination_at = task.started_at + datetime.timedelta(hours=task.hours_to_complete)
        assigned_miners = await tasks_sql.get_nodes_assigned_to_task(str(task.task_id), config.psql_db)
        logger.info(f"Here are the miners that have been assigned {assigned_miners}")
        await _let_miners_know_to_start_training(task, assigned_miners, config)
        task.status = TaskStatus.TRAINING
        add_context_tag("status", task.status.value)
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


async def _evaluate_task(task: TextRawTask | ImageRawTask, gpu_ids: list[int], config: Config):
    gpu_ids_str = "," + ",".join(str(gpu_id) for gpu_id in gpu_ids) + ","
    with LogContext(task_id=str(task.task_id), gpu_ids=gpu_ids_str):
        try:
            task.status = TaskStatus.EVALUATING
            add_context_tag("status", task.status.value)
            await tasks_sql.update_task(task, config.psql_db)
            task = await evaluate_and_score(task, gpu_ids, config)
            await tasks_sql.update_task(task, config.psql_db)
        except Exception as e:
            logger.error(f"Error evaluating task {task.task_id}: {e}", exc_info=True)
            task.status = TaskStatus.FAILURE
            add_context_tag("status", task.status.value)
            await tasks_sql.update_task(task, config.psql_db)


async def _move_back_to_looking_for_nodes(task: TextRawTask | ImageRawTask, config: Config):
    logger.info("Moving back from delay to looking for nodes")
    task.status = TaskStatus.LOOKING_FOR_NODES
    add_context_tag("status", task.status.value)
    await tasks_sql.update_task(task, config.psql_db)


async def _handle_delayed_tasks(config: Config):
    finished_delay_tasks = await tasks_sql.get_tasks_with_status(TaskStatus.DELAYED, psql_db=config.psql_db)
    logger.info(f"We have {len(finished_delay_tasks)} that we're ready to offer to miners again")
    await asyncio.gather(*[_move_back_to_looking_for_nodes(task, config) for task in finished_delay_tasks])


async def _move_to_preevaluation_status(task, config):
    task.status = TaskStatus.PREEVALUATION
    add_context_tag("status", task.status.value)
    logger.info(f"Changing status to {task.status}")
    await tasks_sql.update_task(task, config.psql_db)


async def _move_any_evaluating_tasks_to_pending_evaluation(config: Config):
    stopped_mid_evaluation = await tasks_sql.get_tasks_with_status(TaskStatus.EVALUATING, psql_db=config.psql_db)
    logger.info(f"WE  ARE MOVING {len(stopped_mid_evaluation)} TASKS TO PREEVALUATION")
    await asyncio.gather(*[_move_to_preevaluation_status(task, config) for task in stopped_mid_evaluation])


async def _move_back_to_pending_status(task, config):
    task.status = TaskStatus.PENDING
    add_context_tag("status", task.status.value)
    await tasks_sql.update_task(task, config.psql_db)


async def _move_any_prep_data_to_pending(config):
    stopped_in_prep = await tasks_sql.get_tasks_with_status(TaskStatus.PREPARING_DATA, psql_db=config.psql_db)
    await asyncio.gather(*[_move_back_to_pending_status(task, config) for task in stopped_in_prep])


async def _move_to_preevaluation(tasks: list[TextRawTask | ImageRawTask], config: Config):
    await asyncio.gather(*[_move_to_preevaluation_status(task, config) for task in tasks])


async def process_pending_tasks(config: Config) -> None:
    await _move_any_prep_data_to_pending(config)
    while True:
        try:
            await _processing_pending_tasks(config)
            await _handle_delayed_tasks(config)
            await _find_miners_for_task(config)
            await _process_ready_to_train_tasks(config)
        except Exception as e:
            logger.info(f"There was a problem in processing: {e}")
            await asyncio.sleep(30)


async def move_tasks_to_preevaluation_loop(config: Config):
    await _move_any_evaluating_tasks_to_pending_evaluation(config)
    while True:
        completed_tasks = await tasks_sql.get_tasks_ready_to_evaluate(config.psql_db)
        if completed_tasks:
            await _move_to_preevaluation(completed_tasks, config)
        else:
            logger.info("No tasks to move to preevaluation - waiting 60 seconds")
        await asyncio.sleep(60)


async def evaluate_tasks_loop(config: Config):
    task_queue = asyncio.Queue()
    gpu_queue = asyncio.Queue()
    processing_task_ids = set()

    for gpu_id in cst.GPU_IDS:
        await gpu_queue.put(gpu_id)

    async def evaluation_worker():
        while True:
            try:
                task = await asyncio.wait_for(task_queue.get(), timeout=1)
                gpu_id = await gpu_queue.get()

                try:
                    await _evaluate_task(task, [gpu_id], config)
                finally:
                    await gpu_queue.put(gpu_id)
                    processing_task_ids.remove(task.task_id)
                    task_queue.task_done()
            except asyncio.TimeoutError:
                await asyncio.sleep(5)
                continue
            except Exception as e:
                logger.error(f"Error in evaluation worker: {str(e)}")
                continue

    for _ in cst.GPU_IDS:
        asyncio.create_task(evaluation_worker())

    while True:
        if len(processing_task_ids) < 2 * len(cst.GPU_IDS):
            tasks_to_evaluate = await tasks_sql.get_tasks_with_status(TaskStatus.PREEVALUATION, psql_db=config.psql_db)
            if tasks_to_evaluate:
                logger.info(f"Found {len(tasks_to_evaluate)} new tasks awaiting evaluation, adding to queue")
                for task in tasks_to_evaluate:
                    # Only add to queue if not already added, some tasks in the queue might still have TaskStatus.PREEVALUATION
                    if task.task_id not in processing_task_ids:
                        processing_task_ids.add(task.task_id)
                        await task_queue.put(task)
            else:
                logger.info("No new tasks awaiting evaluation - waiting 30 seconds")
        else:
            logger.info("Evaluation queue is full - waiting for 30 seconds")
        await asyncio.sleep(30)


async def process_completed_tasks(config: Config) -> None:
    await asyncio.gather(move_tasks_to_preevaluation_loop(config), evaluate_tasks_loop(config))
