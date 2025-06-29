import random

from core.models.tournament_models import GroupRound
from core.models.tournament_models import KnockoutRound
from core.models.tournament_models import Round
from core.models.tournament_models import TournamentRound
from validator.core.config import Config
from validator.core.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO
from validator.core.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_GRPO
from validator.core.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT
from validator.core.models import RawTask
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


async def create_text_tournament_round(
    round_data: Round,
    config: Config,
    is_final_round: bool = False,
) -> TournamentRound:
    if isinstance(round_data, GroupRound):
        num_groups = len(round_data.groups)
        logger.info(f"Creating text tournament for {num_groups} groups (1 instruct + 1 DPO + 1 GRPO per group)")
        tasks = await _create_group_text_tasks(round_data, config, is_final_round)
    elif is_final_round:
        logger.info("Creating final text tournament (1 instruct + 1 DPO + 1 GRPO with 1 big model)")
        tasks = await _create_one_of_each_text_task(config, use_big_model=True)
    else:
        num_pairs = len(round_data.pairs)
        logger.info(f"Creating text tournament for {num_pairs} knockout pairs (probability-based)")
        tasks = await _create_probability_based_text_tasks(round_data, config)

    return TournamentRound(round_structure=round_data, tasks=[str(task.task_id) for task in tasks], is_final_round=is_final_round)


async def create_image_tournament_round(round_data: Round, config: Config) -> TournamentRound:
    image_models = _get_image_models(config.keypair)
    tasks = []

    if isinstance(round_data, GroupRound):
        num_groups = len(round_data.groups)
        logger.info(f"Creating image tournament for {num_groups} groups (1 per group)")

        for i, group in enumerate(round_data.groups):
            logger.info(f"  Group {i + 1} ({len(group.member_ids)} members):")
            task = await create_synthetic_image_task(config, image_models)
            logger.info(f"    Image: {task.task_id} - Model: {task.model_id}")
            tasks.append(task)
    else:
        num_pairs = len(round_data.pairs)
        logger.info(f"Creating image tournament for {num_pairs} knockout pairs (1 per pair)")

        for i, pair in enumerate(round_data.pairs):
            logger.info(f"  Pair {i + 1} ({pair[0]} vs {pair[1]}):")
            task = await create_synthetic_image_task(config, image_models)
            logger.info(f"    Image: {task.task_id} - Model: {task.model_id}")
            tasks.append(task)

    return TournamentRound(round_structure=round_data, tasks=[str(task.task_id) for task in tasks], is_final_round=False)


async def _create_group_text_tasks(round_data: GroupRound, config: Config, is_final_round: bool) -> list[RawTask]:
    models = _get_text_models(config.keypair)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    tasks = []
    for i, group in enumerate(round_data.groups):
        logger.info(f"  Group {i + 1} ({len(group.member_ids)} members): creating 1 instruct + 1 DPO + 1 GRPO task")

        instruct_task = await create_synthetic_instruct_text_task(config, models, instruct_datasets)
        logger.info(f"    Instruct: {instruct_task.task_id} - Model: {instruct_task.model_id} - Dataset: {instruct_task.ds}")
        tasks.append(instruct_task)

        dpo_task = await create_synthetic_dpo_task(config, models, dpo_datasets)
        logger.info(f"    DPO: {dpo_task.task_id} - Model: {dpo_task.model_id} - Dataset: {dpo_task.ds}")
        tasks.append(dpo_task)

        grpo_task = await create_synthetic_grpo_task(config, models, instruct_datasets)
        logger.info(f"    GRPO: {grpo_task.task_id} - Model: {grpo_task.model_id} - Dataset: {grpo_task.ds}")
        tasks.append(grpo_task)
    return tasks


async def _create_one_of_each_text_task(config: Config, use_big_model: bool) -> list[RawTask]:
    small_models = _get_text_models(config.keypair)
    big_models = _get_text_models(config.keypair, smallest_size_b=12.0, largest_size_b=71.0)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    tasks = []

    instruct_task = await create_synthetic_instruct_text_task(
        config, big_models if use_big_model else small_models, instruct_datasets
    )
    logger.info(f"  Instruct (BIG): {instruct_task.task_id} - Model: {instruct_task.model_id} - Dataset: {instruct_task.ds}")
    tasks.append(instruct_task)

    dpo_task = await create_synthetic_dpo_task(config, small_models, dpo_datasets)
    logger.info(f"  DPO: {dpo_task.task_id} - Model: {dpo_task.model_id} - Dataset: {dpo_task.ds}")
    tasks.append(dpo_task)

    grpo_task = await create_synthetic_grpo_task(config, small_models, instruct_datasets)
    logger.info(f"  GRPO: {grpo_task.task_id} - Model: {grpo_task.model_id} - Dataset: {grpo_task.ds}")
    tasks.append(grpo_task)

    return tasks


async def _create_probability_based_text_tasks(round_data: KnockoutRound, config: Config) -> list[RawTask]:
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

        rand_val = random.random()
        if rand_val < instruct_prob:
            task = await create_synthetic_instruct_text_task(config, models, instruct_datasets)
            task_type = "Instruct"
        elif rand_val < (instruct_prob + dpo_prob):
            task = await create_synthetic_dpo_task(config, models, dpo_datasets)
            task_type = "DPO"
        else:
            task = await create_synthetic_grpo_task(config, models, instruct_datasets)
            task_type = "GRPO"

        logger.info(f"    {task_type}: {task.task_id} - Model: {task.model_id} - Dataset: {task.ds}")
        tasks.append(task)
    return tasks
