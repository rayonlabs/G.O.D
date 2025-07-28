import random

from core.models.tournament_models import GroupRound
from core.models.tournament_models import KnockoutRound
from core.models.tournament_models import Round
from core.models.tournament_models import TournamentTask
from core.models.tournament_models import get_tournament_gpu_requirement
from core.models.utility_models import TaskType
from validator.core.config import Config
from validator.core.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO
from validator.tournament.constants import IMAGE_TASKS_PER_GROUP
from validator.core.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_GRPO
from validator.core.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT
from validator.core.models import RawTask
from validator.db.sql.tournaments import add_tournament_tasks
from validator.tasks.synthetic_scheduler import _get_dpo_datasets
from validator.tasks.synthetic_scheduler import _get_image_models
from validator.tasks.synthetic_scheduler import _get_instruct_text_datasets
from validator.tasks.synthetic_scheduler import _get_text_models
from validator.tasks.synthetic_scheduler import create_synthetic_dpo_task
from validator.tasks.synthetic_scheduler import create_synthetic_grpo_task
from validator.tasks.synthetic_scheduler import create_synthetic_image_task
from validator.tasks.synthetic_scheduler import create_synthetic_instruct_text_task
from validator.utils.logging import get_logger


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
        logger.info(f"Creating text tournament for {num_groups} groups (1 instruct + 1 DPO + 1 GRPO per group)")
        tasks = await _create_group_text_tasks(round_data, tournament_id, round_id, config, is_final_round)
    elif is_final_round:
        logger.info("Creating final text tournament (1 instruct + 1 DPO + 1 GRPO with 1 big model)")
        tasks = await _create_one_of_each_text_task(tournament_id, round_id, config, use_big_model=True)
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
        num_groups = len(round_data.groups)
        logger.info(f"Creating image tournament for {num_groups} groups ({IMAGE_TASKS_PER_GROUP} per group)")

        for i, group in enumerate(round_data.groups):
            group_id = f"{round_id}_group_{i + 1:03d}"
            logger.info(f"  Group {i + 1} ({len(group.member_ids)} members):")
            
            for task_num in range(IMAGE_TASKS_PER_GROUP):
                while True:
                    try:
                        task = await create_synthetic_image_task(config, image_models)
                        tournament_task = TournamentTask(
                            tournament_id=tournament_id,
                            round_id=round_id,
                            task_id=task.task_id,
                            group_id=group_id,
                            pair_id=None,
                            gpu_requirement=get_tournament_gpu_requirement(task.task_type, task.model_params_count),
                        )
                        await add_tournament_tasks([tournament_task], config.psql_db)
                        break
                    except Exception as e:
                        logger.warning(f"Failed to create image task {task_num + 1} for group {i + 1}: {e}. Retrying...")
                gpu_req = get_tournament_gpu_requirement(task.task_type, task.model_params_count)
                logger.info(f"    Image {task_num + 1}: {task.task_id} - Model: {task.model_id} - GPU: {gpu_req}")
                tasks.append(task)
    elif is_final_round:
        logger.info("Creating final image tournament (3 image tasks)")
        pair_id = f"{round_id}_pair_001"
        
        for i in range(3):
            while True:
                try:
                    task = await create_synthetic_image_task(config, image_models)
                    tournament_task = TournamentTask(
                        tournament_id=tournament_id,
                        round_id=round_id,
                        task_id=task.task_id,
                        group_id=None,
                        pair_id=pair_id,
                    )
                    await add_tournament_tasks([tournament_task], config.psql_db)
                    break
                except Exception as e:
                    logger.warning(f"Failed to create final image task {i + 1}: {e}. Retrying...")
            gpu_req = get_tournament_gpu_requirement(task.task_type, task.model_params_count)
            logger.info(f"    Image {i + 1}: {task.task_id} - Model: {task.model_id} - GPU: {gpu_req}")
            tasks.append(task)
    else:
        num_pairs = len(round_data.pairs)
        logger.info(f"Creating image tournament for {num_pairs} knockout pairs (1 per pair)")

        for i, pair in enumerate(round_data.pairs):
            pair_id = f"{round_id}_pair_{i + 1:03d}"
            logger.info(f"  Pair {i + 1} ({pair[0]} vs {pair[1]}):")
            while True:
                try:
                    task = await create_synthetic_image_task(config, image_models)
                    tournament_task = TournamentTask(
                        tournament_id=tournament_id,
                        round_id=round_id,
                        task_id=task.task_id,
                        group_id=None,
                        pair_id=pair_id,
                    )
                    await add_tournament_tasks([tournament_task], config.psql_db)
                    break
                except Exception as e:
                    logger.warning(f"Failed to create image task for pair {i + 1}: {e}. Retrying...")
            gpu_req = get_tournament_gpu_requirement(task.task_type, task.model_params_count)
            logger.info(f"    Image: {task.task_id} - Model: {task.model_id} - GPU: {gpu_req}")
            tasks.append(task)

    return [str(task.task_id) for task in tasks]


async def _create_group_text_tasks(
    round_data: GroupRound, tournament_id: str, round_id: str, config: Config, is_final_round: bool
) -> list[RawTask]:
    models = _get_text_models(config.keypair)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    tasks = []
    for i, group in enumerate(round_data.groups):
        logger.info(f"  Group {i + 1} ({len(group.member_ids)} members): creating 1 instruct + 1 DPO + 1 GRPO task")
        group_id = f"{round_id}_group_{i + 1:03d}"
        instruct_task = await create_synthetic_instruct_text_task(config, models, instruct_datasets)
        instruct_tournament_task = TournamentTask(
            tournament_id=tournament_id,
            round_id=round_id,
            task_id=instruct_task.task_id,
            group_id=group_id,
            pair_id=None,
        )
        await add_tournament_tasks([instruct_tournament_task], config.psql_db)
        gpu_req = get_tournament_gpu_requirement(instruct_task.task_type, instruct_task.model_params_count)
        logger.info(
            f"    Instruct: {instruct_task.task_id} - Model: {instruct_task.model_id} - Dataset: {instruct_task.ds} - GPU: {gpu_req}"
        )
        tasks.append(instruct_task)

        dpo_task = await create_synthetic_dpo_task(config, models, dpo_datasets)
        dpo_tournament_task = TournamentTask(
            tournament_id=tournament_id,
            round_id=round_id,
            task_id=dpo_task.task_id,
            group_id=group_id,
            pair_id=None,
        )
        await add_tournament_tasks([dpo_tournament_task], config.psql_db)
        gpu_req = get_tournament_gpu_requirement(dpo_task.task_type, dpo_task.model_params_count)
        logger.info(f"    DPO: {dpo_task.task_id} - Model: {dpo_task.model_id} - Dataset: {dpo_task.ds} - GPU: {gpu_req}")
        tasks.append(dpo_task)

        grpo_task = await create_synthetic_grpo_task(config, models, instruct_datasets)
        grpo_tournament_task = TournamentTask(
            tournament_id=tournament_id,
            round_id=round_id,
            task_id=grpo_task.task_id,
            group_id=group_id,
            pair_id=None,
        )
        await add_tournament_tasks([grpo_tournament_task], config.psql_db)
        gpu_req = get_tournament_gpu_requirement(grpo_task.task_type, grpo_task.model_params_count)
        logger.info(f"    GRPO: {grpo_task.task_id} - Model: {grpo_task.model_id} - Dataset: {grpo_task.ds} - GPU: {gpu_req}")
        tasks.append(grpo_task)
    return tasks


async def _create_one_of_each_text_task(tournament_id: str, round_id: str, config: Config, use_big_model: bool) -> list[RawTask]:
    pair_id = f"{round_id}_pair_{1:03d}"
    small_models = _get_text_models(config.keypair)
    big_models = _get_text_models(config.keypair, smallest_size_b=12.0, largest_size_b=71.0)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    tasks = []

    instruct_task = await create_synthetic_instruct_text_task(
        config, big_models if use_big_model else small_models, instruct_datasets
    )
    instruct_tournament_task = TournamentTask(
        tournament_id=tournament_id,
        round_id=round_id,
        task_id=instruct_task.task_id,
        group_id=None,
        pair_id=pair_id,
    )
    await add_tournament_tasks([instruct_tournament_task], config.psql_db)
    gpu_req = get_tournament_gpu_requirement(instruct_task.task_type, instruct_task.model_params_count)
    logger.info(
        f"  Instruct (BIG): {instruct_task.task_id} - Model: {instruct_task.model_id} - Dataset: {instruct_task.ds} - GPU: {gpu_req}"
    )
    tasks.append(instruct_task)

    dpo_task = await create_synthetic_dpo_task(config, small_models, dpo_datasets)
    dpo_tournament_task = TournamentTask(
        tournament_id=tournament_id,
        round_id=round_id,
        task_id=dpo_task.task_id,
        group_id=None,
        pair_id=pair_id,
    )
    await add_tournament_tasks([dpo_tournament_task], config.psql_db)
    gpu_req = get_tournament_gpu_requirement(dpo_task.task_type, dpo_task.model_params_count)
    logger.info(f"  DPO: {dpo_task.task_id} - Model: {dpo_task.model_id} - Dataset: {dpo_task.ds} - GPU: {gpu_req}")
    tasks.append(dpo_task)

    grpo_task = await create_synthetic_grpo_task(config, small_models, instruct_datasets)
    grpo_tournament_task = TournamentTask(
        tournament_id=tournament_id,
        round_id=round_id,
        task_id=grpo_task.task_id,
        group_id=None,
        pair_id=pair_id,
    )
    await add_tournament_tasks([grpo_tournament_task], config.psql_db)
    gpu_req = get_tournament_gpu_requirement(grpo_task.task_type, grpo_task.model_params_count)
    logger.info(f"  GRPO: {grpo_task.task_id} - Model: {grpo_task.model_id} - Dataset: {grpo_task.ds} - GPU: {gpu_req}")
    tasks.append(grpo_task)

    return tasks


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
        rand_val = random.random()
        if rand_val < instruct_prob:
            task = await create_synthetic_instruct_text_task(config, models, instruct_datasets)
            task_type = TaskType.INSTRUCTTEXTTASK
        elif rand_val < (instruct_prob + dpo_prob):
            task = await create_synthetic_dpo_task(config, models, dpo_datasets)
            task_type = TaskType.DPOTASK
        else:
            task = await create_synthetic_grpo_task(config, models, instruct_datasets)
            task_type = TaskType.GRPOTASK

        tournament_task = TournamentTask(
            tournament_id=tournament_id,
            round_id=round_id,
            task_id=task.task_id,
            group_id=None,
            pair_id=pair_id,
        )
        await add_tournament_tasks([tournament_task], config.psql_db)
        gpu_req = get_tournament_gpu_requirement(task.task_type, task.model_params_count)
        logger.info(f"    {task_type}: {task.task_id} - Model: {task.model_id} - Dataset: {task.ds} - GPU: {gpu_req}")
        tasks.append(task)
    return tasks


async def create_new_task_of_same_type(task: RawTask, config: Config) -> RawTask:
    if task.task_type == TaskType.IMAGETASK:
        return await create_synthetic_image_task(config, _get_image_models(config.keypair))

    model_params_b = int(task.model_params_count / 1e9)

    models = _get_text_models(config.keypair, smallest_size_b=model_params_b * 0.8, largest_size_b=model_params_b * 1.2)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    if task.task_type == TaskType.INSTRUCTTEXTTASK:
        return await create_synthetic_instruct_text_task(config, models, instruct_datasets)
    elif task.task_type == TaskType.DPOTASK:
        return await create_synthetic_dpo_task(config, models, dpo_datasets)
    elif task.task_type == TaskType.GRPOTASK:
        return await create_synthetic_grpo_task(config, models, instruct_datasets)
