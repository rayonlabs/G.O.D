import random

from core.models.tournament_models import GroupRound
from core.models.tournament_models import KnockoutRound
from core.models.tournament_models import Round
from core.models.tournament_models import TournamentTask
from core.models.tournament_models import get_tournament_gpu_requirement
from core.models.utility_models import TaskType
from validator.core.config import Config
from validator.core.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO
from validator.core.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_GRPO
from validator.core.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT
from validator.core.constants import PERCENTAGE_OF_INSTRUCT_TASKS_THAT_SHOULD_BE_CHAT
from validator.core.models import RawTask
from validator.db.sql import tasks as task_sql
from validator.db.sql.tournaments import add_tournament_tasks
from validator.db.sql.tournaments import get_tournament_tasks
from validator.tasks.synthetic_scheduler import _get_dpo_datasets
from validator.tasks.synthetic_scheduler import _get_image_models
from validator.tasks.synthetic_scheduler import _get_instruct_text_datasets
from validator.tasks.synthetic_scheduler import _get_text_models
from validator.tasks.synthetic_scheduler import create_synthetic_dpo_task
from validator.tasks.synthetic_scheduler import create_synthetic_grpo_task
from validator.tasks.synthetic_scheduler import create_synthetic_image_task
from validator.tasks.synthetic_scheduler import create_synthetic_instruct_text_task
from validator.tasks.synthetic_scheduler import create_synthetic_chat_task
from validator.tournament import constants as t_cst
from validator.utils.logging import get_logger
from validator.db.sql.historical_tasks import get_random_historical_task_by_type
from validator.tournament.boss_round_sync import copy_historical_task_into_boss_round_tournament
import validator.core.constants as cst
from core.models.utility_models import TaskType


logger = get_logger(__name__)


async def create_text_tournament_tasks(
    round_data: Round,
    tournament_id: str,
    round_id: str,
    config: Config,
    is_final_round: bool = False,
) -> list[str]:
    if isinstance(round_data, GroupRound):
        num_groups = len(round_data.groups)
        logger.info(f"Creating text tournament for {num_groups} groups (1 task per group)")
        tasks = await _create_group_text_tasks(round_data, tournament_id, round_id, config, is_final_round)
    elif is_final_round:
        logger.info("Creating final text tournament using historical tasks (1 instruct + 1 DPO + 1 GRPO)")
        tasks = await _create_historical_text_boss_round_tasks(tournament_id, round_id, config)
    else:
        num_pairs = len(round_data.pairs)
        logger.info(f"Creating text tournament for {num_pairs} knockout pairs (probability-based)")
        tasks = await _create_probability_based_text_tasks(round_data, tournament_id, round_id, config)

    return [str(task.task_id) for task in tasks]


async def create_image_tournament_tasks(
    round_data: Round, tournament_id: str, round_id: str, config: Config, is_final_round: bool = False
) -> list[str]:
    image_models = _get_image_models(config.keypair)
    tasks = []

    if isinstance(round_data, GroupRound):
        tasks = await _create_group_image_tasks(round_data, tournament_id, round_id, config, image_models)
    elif is_final_round:
        tasks = await _create_historical_image_boss_round_tasks(tournament_id, round_id, config)
    else:
        tasks = await _create_knockout_image_tasks(round_data, tournament_id, round_id, config, image_models)

    return [str(task.task_id) for task in tasks]


async def _create_group_image_tasks(
    round_data: GroupRound, tournament_id: str, round_id: str, config: Config, image_models: list
) -> list[RawTask]:
    num_groups = len(round_data.groups)
    logger.info(f"Creating image tournament for {num_groups} groups ({t_cst.IMAGE_TASKS_PER_GROUP} per group)")
    tasks = []

    for i, group in enumerate(round_data.groups):
        group_tasks = await _create_single_group_image_tasks(group, i, tournament_id, round_id, config, image_models)
        tasks.extend(group_tasks)

    return tasks


async def _create_single_group_image_tasks(
    group, group_index: int, tournament_id: str, round_id: str, config: Config, image_models: list
) -> list[RawTask]:
    group_id = f"{round_id}_group_{group_index + 1:03d}"
    logger.info(f"  Group {group_index + 1} ({len(group.member_ids)} members):")

    existing_tasks = await get_tournament_tasks(round_id, config.psql_db)
    existing_group_tasks = [task for task in existing_tasks if task.group_id == group_id]
    existing_count = len(existing_group_tasks)

    assert t_cst.IMAGE_TASKS_PER_GROUP == 1, "Only 1 image task per group is supported"
    if existing_count >= t_cst.IMAGE_TASKS_PER_GROUP:
        logger.info(f"    Group {group_index + 1} already has {existing_count} task(s), skipping task creation")
        return await _get_existing_tasks(existing_group_tasks, config)

    logger.info(f"    Group {group_index + 1} has {existing_count}/{t_cst.IMAGE_TASKS_PER_GROUP} task, creating 1 more")

    task = await _create_single_image_task_with_retry(config, image_models, 0, group_index)
    tournament_task = TournamentTask(
        tournament_id=tournament_id,
        round_id=round_id,
        task_id=task.task_id,
        group_id=group_id,
        pair_id=None,
    )
    await add_tournament_tasks([tournament_task], config.psql_db)
    gpu_req = get_tournament_gpu_requirement(task.task_type, task.model_params_count)
    logger.info(f"    Image: {task.task_id} - Model: {task.model_id} - GPU: {gpu_req}")

    return [task]


async def _create_final_image_tasks(tournament_id: str, round_id: str, config: Config, image_models: list) -> list[RawTask]:
    logger.info(f"Creating final image tournament ({t_cst.FINAL_ROUND_IMAGE_TASKS} image tasks)")
    pair_id = f"{round_id}_pair_001"

    existing_tasks = await get_tournament_tasks(round_id, config.psql_db)
    existing_pair_tasks = [task for task in existing_tasks if task.pair_id == pair_id]
    existing_count = len(existing_pair_tasks)

    if existing_count >= t_cst.FINAL_ROUND_IMAGE_TASKS:
        logger.info(f"    Final round already has {existing_count} tasks, skipping task creation")
        return await _get_existing_tasks(existing_pair_tasks, config)

    logger.info(
        f"    Final round has {existing_count}/{t_cst.FINAL_ROUND_IMAGE_TASKS} tasks, creating {t_cst.FINAL_ROUND_IMAGE_TASKS - existing_count} more"
    )

    tasks = []
    for i in range(existing_count, t_cst.FINAL_ROUND_IMAGE_TASKS):
        task = await _create_single_image_task_with_retry(config, image_models, i, None, is_final=True)
        tournament_task = TournamentTask(
            tournament_id=tournament_id,
            round_id=round_id,
            task_id=task.task_id,
            group_id=None,
            pair_id=pair_id,
        )
        await add_tournament_tasks([tournament_task], config.psql_db)
        gpu_req = get_tournament_gpu_requirement(task.task_type, task.model_params_count)
        logger.info(f"    Image {i + 1}: {task.task_id} - Model: {task.model_id} - GPU: {gpu_req}")
        tasks.append(task)

    return tasks


async def _create_knockout_image_tasks(
    round_data: KnockoutRound, tournament_id: str, round_id: str, config: Config, image_models: list
) -> list[RawTask]:
    num_pairs = len(round_data.pairs)
    logger.info(f"Creating image tournament for {num_pairs} knockout pairs ({t_cst.KNOCKOUT_PAIR_TASKS} per pair)")
    tasks = []

    for i, pair in enumerate(round_data.pairs):
        pair_tasks = await _create_single_knockout_image_task(pair, i, tournament_id, round_id, config, image_models)
        tasks.extend(pair_tasks)

    return tasks


async def _create_single_knockout_image_task(
    pair, pair_index: int, tournament_id: str, round_id: str, config: Config, image_models: list
) -> list[RawTask]:
    pair_id = f"{round_id}_pair_{pair_index + 1:03d}"
    logger.info(f"  Pair {pair_index + 1} ({pair[0]} vs {pair[1]}):")

    existing_tasks = await get_tournament_tasks(round_id, config.psql_db)
    existing_pair_tasks = [task for task in existing_tasks if task.pair_id == pair_id]

    if existing_pair_tasks:
        if len(existing_pair_tasks) > t_cst.KNOCKOUT_PAIR_TASKS:
            logger.warning(
                f"   Pair {pair_index + 1} has {len(existing_pair_tasks)} tasks when it should only have {t_cst.KNOCKOUT_PAIR_TASKS}!"
            )
        logger.info(f"    Pair {pair_index + 1} already has {len(existing_pair_tasks)} task(s), skipping task creation")
        return await _get_existing_tasks(existing_pair_tasks, config)

    logger.info(f"    Pair {pair_index + 1} has no tasks, creating {t_cst.KNOCKOUT_PAIR_TASKS}")
    task = await _create_single_image_task_with_retry(config, image_models, 0, pair_index)
    tournament_task = TournamentTask(
        tournament_id=tournament_id,
        round_id=round_id,
        task_id=task.task_id,
        group_id=None,
        pair_id=pair_id,
    )
    await add_tournament_tasks([tournament_task], config.psql_db)
    gpu_req = get_tournament_gpu_requirement(task.task_type, task.model_params_count)
    logger.info(f"    Image: {task.task_id} - Model: {task.model_id} - GPU: {gpu_req}")
    return [task]


async def _create_single_image_task_with_retry(
    config: Config, image_models: list, task_num: int, group_index: int = None, is_final: bool = False
) -> RawTask:
    while True:
        try:
            task = await create_synthetic_image_task(config, image_models)
            break
        except Exception as e:
            context = f"final image task {task_num + 1}" if is_final else f"image task {task_num + 1} for group {group_index + 1}"
            logger.warning(f"Failed to create {context}: {e}. Retrying...")
    return task


async def _get_existing_tasks(existing_tournament_tasks: list, config: Config) -> list[RawTask]:
    tasks = []
    for task in existing_tournament_tasks:
        task_obj = await task_sql.get_task(task.task_id, config.psql_db)
        if task_obj:
            tasks.append(task_obj)
    return tasks


async def _create_group_text_tasks(
    round_data: GroupRound, tournament_id: str, round_id: str, config: Config, is_final_round: bool
) -> list[RawTask]:
    models = _get_text_models(config.keypair, smallest_size_b=0.1, largest_size_b=4.0)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    tasks = []
    for i, group in enumerate(round_data.groups):
        logger.info(f"  Group {i + 1} ({len(group.member_ids)} members): creating 1 instruct task")
        group_tasks = await _create_single_group_text_tasks(
            group, i, tournament_id, round_id, config, models, instruct_datasets, dpo_datasets
        )
        tasks.extend(group_tasks)

    return tasks


async def _create_single_group_text_tasks(
    group,
    group_index: int,
    tournament_id: str,
    round_id: str,
    config: Config,
    models: list,
    instruct_datasets: list,
    dpo_datasets: list,
) -> list[RawTask]:
    group_id = f"{round_id}_group_{group_index + 1:03d}"

    existing_tasks = await get_tournament_tasks(round_id, config.psql_db)
    existing_group_tasks = [task for task in existing_tasks if task.group_id == group_id]
    existing_count = len(existing_group_tasks)

    if existing_count >= t_cst.TEXT_TASKS_PER_GROUP:
        logger.info(f"    Group {group_index + 1} already has {existing_count} task(s), skipping task creation")
        return await _get_existing_tasks(existing_group_tasks, config)

    logger.info(f"    Group {group_index + 1} has {existing_count}/{t_cst.TEXT_TASKS_PER_GROUP} task, creating 1 more")
    assert t_cst.TEXT_TASKS_PER_GROUP == 1, "Only 1 text task per group is supported"
    task = await create_synthetic_instruct_text_task(config, models, instruct_datasets)
    
    task.hours_to_complete = 2
    await task_sql.update_task(task, config.psql_db)
    
    tournament_task = TournamentTask(
        tournament_id=tournament_id,
        round_id=round_id,
        task_id=task.task_id,
        group_id=group_id,
        pair_id=None,
    )
    await add_tournament_tasks([tournament_task], config.psql_db)
    gpu_req = get_tournament_gpu_requirement(task.task_type, task.model_params_count)
    logger.info(f"    Instruct: {task.task_id} - Model: {task.model_id} - Dataset: {task.ds} - GPU: {gpu_req} - Duration: 2 hours")

    return [task]


def _get_missing_task_types(existing_group_tasks: list) -> list[TaskType]:
    existing_types = [task.task_type for task in existing_group_tasks]
    task_types_to_create = []

    if TaskType.INSTRUCTTEXTTASK not in existing_types:
        task_types_to_create.append(TaskType.INSTRUCTTEXTTASK)
    if TaskType.CHATTASK not in existing_types:
        task_types_to_create.append(TaskType.CHATTASK)
    if TaskType.DPOTASK not in existing_types:
        task_types_to_create.append(TaskType.DPOTASK)
    if TaskType.GRPOTASK not in existing_types:
        task_types_to_create.append(TaskType.GRPOTASK)

    return task_types_to_create


async def _create_text_task_by_type(
    task_type: TaskType, config: Config, models: list, instruct_datasets: list, dpo_datasets: list
) -> RawTask:
    if task_type == TaskType.INSTRUCTTEXTTASK:
        return await create_synthetic_instruct_text_task(config, models, instruct_datasets)
    elif task_type == TaskType.CHATTASK:
        return await create_synthetic_chat_task(config, models, instruct_datasets)
    elif task_type == TaskType.DPOTASK:
        return await create_synthetic_dpo_task(config, models, dpo_datasets)
    elif task_type == TaskType.GRPOTASK:
        return await create_synthetic_grpo_task(config, models, instruct_datasets)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


async def _create_one_of_each_text_task(tournament_id: str, round_id: str, config: Config, use_big_model: bool) -> list[RawTask]:
    # TODO: get this from db instead of hardcoding
    pair_id = f"{round_id}_pair_{1:03d}"
    small_models = _get_text_models(config.keypair)
    big_models = _get_text_models(
        config.keypair, smallest_size_b=t_cst.BIG_MODEL_MIN_SIZE_B, largest_size_b=t_cst.BIG_MODEL_MAX_SIZE_B
    )
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    existing_tasks = await get_tournament_tasks(round_id, config.psql_db)
    existing_pair_tasks = [task for task in existing_tasks if task.pair_id == pair_id]
    existing_count = len(existing_pair_tasks)

    if existing_count >= t_cst.FINAL_ROUND_TEXT_TASKS:
        if len(existing_pair_tasks) > t_cst.FINAL_ROUND_TEXT_TASKS:
            logger.warning(
                f"   Pair {pair_id} has {len(existing_pair_tasks)} tasks when it should only have {t_cst.FINAL_ROUND_TEXT_TASKS}!"
            )
        logger.info(f"    Final round already has {existing_count} tasks, skipping task creation")
        return await _get_existing_tasks(existing_pair_tasks, config)

    logger.info(
        f"    Final round has {existing_count}/{t_cst.FINAL_ROUND_TEXT_TASKS} tasks, creating {t_cst.FINAL_ROUND_TEXT_TASKS - existing_count} more"
    )

    task_types_to_create = _get_missing_task_types(existing_pair_tasks)
    tasks = []

    for task_type in task_types_to_create:
        task = await _create_final_text_task_by_type(
            task_type, config, small_models, big_models, instruct_datasets, dpo_datasets, use_big_model
        )
        tournament_task = TournamentTask(
            tournament_id=tournament_id,
            round_id=round_id,
            task_id=task.task_id,
            group_id=None,
            pair_id=pair_id,
        )
        await add_tournament_tasks([tournament_task], config.psql_db)
        gpu_req = get_tournament_gpu_requirement(task.task_type, task.model_params_count)
        model_size = "BIG" if use_big_model and task_type == TaskType.INSTRUCTTEXTTASK else ""
        logger.info(
            f"  {task_type.value} {model_size}: {task.task_id} - Model: {task.model_id} - Dataset: {task.ds} - GPU: {gpu_req}"
        )
        tasks.append(task)

    return tasks


async def _create_final_text_task_by_type(
    task_type: TaskType,
    config: Config,
    small_models: list,
    big_models: list,
    instruct_datasets: list,
    dpo_datasets: list,
    use_big_model: bool,
) -> RawTask:
    if task_type == TaskType.INSTRUCTTEXTTASK:
        models_to_use = big_models if use_big_model else small_models
        return await create_synthetic_instruct_text_task(config, models_to_use, instruct_datasets)
    elif task_type == TaskType.CHATTASK:
        return await create_synthetic_chat_task(config, models_to_use, instruct_datasets)
    elif task_type == TaskType.DPOTASK:
        return await create_synthetic_dpo_task(config, small_models, dpo_datasets)
    elif task_type == TaskType.GRPOTASK:
        return await create_synthetic_grpo_task(config, small_models, instruct_datasets)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


async def _create_probability_based_text_tasks(
    round_data: KnockoutRound, tournament_id: str, round_id: str, config: Config
) -> list[RawTask]:
    num_tasks = len(round_data.pairs)
    models = _get_text_models(config.keypair)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    text_total = (
        PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT
        + PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO
        + PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_GRPO
    )
    instruct_prob = PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT / text_total
    dpo_prob = PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO / text_total

    tasks = []
    for i in range(num_tasks):
        pair = round_data.pairs[i]
        logger.info(f"  Pair {i + 1} ({pair[0]} vs {pair[1]}):")
        pair_id = f"{round_id}_pair_{i + 1:03d}"

        existing_tasks = await get_tournament_tasks(round_id, config.psql_db)
        existing_pair_tasks = [task for task in existing_tasks if task.pair_id == pair_id]

        if existing_pair_tasks:
            if len(existing_pair_tasks) > t_cst.KNOCKOUT_PAIR_TASKS:
                logger.warning(
                    f"   Pair {i + 1} has {len(existing_pair_tasks)} tasks when it should only have {t_cst.KNOCKOUT_PAIR_TASKS}!"
                )
            logger.info(f"    Pair {i + 1} already has {len(existing_pair_tasks)} task(s), skipping task creation")
            pair_task_objects = await _get_existing_tasks(existing_pair_tasks, config)
            tasks.extend(pair_task_objects)
            continue

        logger.info(f"    Pair {i + 1} has no tasks, creating {t_cst.KNOCKOUT_PAIR_TASKS}")
        task = await _create_single_probability_task(config, models, instruct_datasets, dpo_datasets, instruct_prob, dpo_prob)

        tournament_task = TournamentTask(
            tournament_id=tournament_id,
            round_id=round_id,
            task_id=task.task_id,
            group_id=None,
            pair_id=pair_id,
        )
        await add_tournament_tasks([tournament_task], config.psql_db)
        gpu_req = get_tournament_gpu_requirement(task.task_type, task.model_params_count)
        logger.info(f"    {task.task_type.value}: {task.task_id} - Model: {task.model_id} - Dataset: {task.ds} - GPU: {gpu_req}")
        tasks.append(task)
    return tasks


async def _create_single_probability_task(
    config: Config, models: list, instruct_datasets: list, dpo_datasets: list, instruct_prob: float, dpo_prob: float
) -> RawTask:
    rand_val = random.random()
    if rand_val < instruct_prob:
        chat_prob = random.random()
        if chat_prob < PERCENTAGE_OF_INSTRUCT_TASKS_THAT_SHOULD_BE_CHAT:
            return await create_synthetic_chat_task(config, models, instruct_datasets)
        else:
            return await create_synthetic_instruct_text_task(config, models, instruct_datasets)
    elif rand_val < (instruct_prob + dpo_prob):
        return await create_synthetic_dpo_task(config, models, dpo_datasets)
    else:
        return await create_synthetic_grpo_task(config, models, instruct_datasets)


async def create_new_task_of_same_type(task: RawTask, config: Config) -> RawTask:
    if task.task_type == TaskType.IMAGETASK:
        return await create_synthetic_image_task(config, _get_image_models(config.keypair))

    model_params_b = int(task.model_params_count / t_cst.MODEL_PARAMS_TO_BILLIONS)

    # Handle case where model params is 0 or very small
    if model_params_b < t_cst.DEFAULT_MODEL_MIN_SIZE_B:
        logger.warning(
            f"Original task has very small model params ({task.model_params_count}), using default range {t_cst.DEFAULT_MODEL_MIN_SIZE_B}-{t_cst.DEFAULT_MODEL_MAX_SIZE_B}B"
        )
        models = _get_text_models(
            config.keypair, smallest_size_b=t_cst.DEFAULT_MODEL_MIN_SIZE_B, largest_size_b=t_cst.DEFAULT_MODEL_MAX_SIZE_B
        )
    else:
        models = _get_text_models(
            config.keypair,
            smallest_size_b=model_params_b * t_cst.MODEL_SIZE_RANGE_MULTIPLIER_MIN,
            largest_size_b=model_params_b * t_cst.MODEL_SIZE_RANGE_MULTIPLIER_MAX,
        )
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    if task.task_type == TaskType.INSTRUCTTEXTTASK:
        return await create_synthetic_instruct_text_task(config, models, instruct_datasets)
    elif task.task_type == TaskType.CHATTASK:
        return await create_synthetic_chat_task(config, models, instruct_datasets)
    elif task.task_type == TaskType.DPOTASK:
        return await create_synthetic_dpo_task(config, models, dpo_datasets)
    elif task.task_type == TaskType.GRPOTASK:
        return await create_synthetic_grpo_task(config, models, instruct_datasets)


async def _create_historical_text_boss_round_tasks(
    tournament_id: str,
    round_id: str,
    config: Config
) -> list[RawTask]:
    """Create boss round text tasks using random historical tasks."""
    pair_id = f"{round_id}_pair_001"
    
    existing_tasks = await get_tournament_tasks(round_id, config.psql_db)
    existing_pair_tasks = [task for task in existing_tasks if task.pair_id == pair_id]
    existing_count = len(existing_pair_tasks)
    
    if existing_count >= t_cst.FINAL_ROUND_TEXT_TASKS:
        logger.info(f"Final round already has {existing_count} tasks, skipping task creation")
        return await _get_existing_tasks(existing_pair_tasks, config)
    
    logger.info(f"Creating boss round text tasks using historical tasks")
    
    # Check which specific task types we already have
    existing_task_types = set()
    tasks = []
    
    for task in existing_pair_tasks:
        task_obj = await task_sql.get_task(task.task_id, config.psql_db)
        if task_obj:
            existing_task_types.add(task_obj.task_type.value if hasattr(task_obj.task_type, 'value') else task_obj.task_type)
            tasks.append(task_obj)
    
    # Create only the missing task types
    if TaskType.INSTRUCTTEXTTASK.value not in existing_task_types:
        task = await _create_single_historical_text_task(
            TaskType.INSTRUCTTEXTTASK,
            tournament_id,
            round_id,
            pair_id,
            config
        )
        if task:
            tasks.append(task)

    if TaskType.CHATTASK.value not in existing_task_types:
        task = await _create_single_historical_text_task(
            TaskType.CHATTASK,
            tournament_id,
            round_id,
            pair_id,
            config
        )
        if task:
            tasks.append(task)
    
    if TaskType.DPOTASK.value not in existing_task_types:
        task = await _create_single_historical_text_task(
            TaskType.DPOTASK,
            tournament_id,
            round_id,
            pair_id,
            config
        )
        if task:
            tasks.append(task)
    
    if TaskType.GRPOTASK.value not in existing_task_types:
        task = await _create_single_historical_text_task(
            TaskType.GRPOTASK,
            tournament_id,
            round_id,
            pair_id,
            config
        )
        if task:
            tasks.append(task)
    
    return tasks


async def _create_single_historical_text_task(
    task_type: TaskType,
    tournament_id: str,
    round_id: str,
    pair_id: str,
    config: Config
) -> RawTask | None:
    """Create a single historical text task of a specific type."""
    historical_task_id = await get_random_historical_task_by_type(
        task_type=task_type.value,
        start_date=cst.BOSS_ROUND_HISTORICAL_START_DATE,
        end_date=cst.BOSS_ROUND_HISTORICAL_END_DATE,
        min_successful_scores=cst.MIN_SUCCESSFUL_SCORES_FOR_HISTORICAL_TASK,
        psql_db=config.psql_db
    )
    
    if historical_task_id:
        new_task = await copy_historical_task_into_boss_round_tournament(
            historical_task_id,
            tournament_id,
            round_id,
            pair_id,
            config.psql_db
        )
        
        if new_task:
            logger.info(f"Created boss round {task_type.value} task from historical task {historical_task_id}")
            return new_task
    else:
        logger.error(f"No historical {task_type.value} tasks found for boss round. Cannot proceed without historical data.")
        return None


async def _create_historical_image_boss_round_tasks(
    tournament_id: str,
    round_id: str,
    config: Config
) -> list[RawTask]:
    """Create boss round image tasks using random historical tasks."""
    pair_id = f"{round_id}_pair_001"
    
    existing_tasks = await get_tournament_tasks(round_id, config.psql_db)
    existing_pair_tasks = [task for task in existing_tasks if task.pair_id == pair_id]
    existing_count = len(existing_pair_tasks)
    
    if existing_count >= t_cst.FINAL_ROUND_IMAGE_TASKS:
        logger.info(f"Final round already has {existing_count} tasks, skipping task creation")
        return await _get_existing_tasks(existing_pair_tasks, config)
    
    logger.info(f"Creating boss round image tasks using historical tasks")
    
    tasks = []
    num_needed = t_cst.FINAL_ROUND_IMAGE_TASKS - existing_count
    
    # Get unique historical image tasks
    selected_ids = []
    for i in range(num_needed):
        historical_task_id = await get_random_historical_task_by_type(
            task_type=TaskType.IMAGETASK.value,
            start_date=cst.BOSS_ROUND_HISTORICAL_START_DATE,
            end_date=cst.BOSS_ROUND_HISTORICAL_END_DATE,
            min_successful_scores=cst.MIN_SUCCESSFUL_SCORES_FOR_HISTORICAL_TASK,
            psql_db=config.psql_db,
            exclude_task_ids=selected_ids
        )
        
        if historical_task_id:
            selected_ids.append(historical_task_id)
            new_task = await copy_historical_task_into_boss_round_tournament(
                historical_task_id,
                tournament_id,
                round_id,
                pair_id,
                config.psql_db
            )
            
            if new_task:
                tasks.append(new_task)
                logger.info(f"Created boss round image task {i+1} from historical task {historical_task_id}")
        else:
            logger.error(f"Only found {len(selected_ids)} historical image tasks, needed {num_needed}. Cannot proceed without enough historical data.")
            break
    
    # Add existing tasks to the return list
    for existing_task in existing_pair_tasks:
        existing_raw_task = await task_sql.get_task(existing_task.task_id, config.psql_db)
        if existing_raw_task:
            tasks.append(existing_raw_task)
    
    return tasks
