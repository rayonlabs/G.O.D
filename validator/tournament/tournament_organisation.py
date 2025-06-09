import math
import random
from typing import AsyncGenerator

from core.models.payload_models import ImageModelInfo
from core.models.utility_models import Round
from core.models.utility_models import KnockoutRound
from core.models.utility_models import GroupRound
from core.models.utility_models import Group
from core.models.utility_models import TaskType
from core.models.utility_models import TournamentRound
from validator.core.config import Config
from validator.core.constants import EXPECTED_GROUP_SIZE
from validator.core.constants import MIN_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND
from validator.core.constants import PREVIOUS_WINNER_BASE_CONTESTANT
from validator.core.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT
from validator.core.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO
from validator.core.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_GRPO
from validator.core.constants import PROBABILITY_OF_A_BIG_TEXT_MODEL
from validator.core.models import Dataset, RawTask
from validator.tasks.synthetic_scheduler import (
    create_synthetic_instruct_text_task,
    create_synthetic_dpo_task,
    create_synthetic_grpo_task,
    create_synthetic_image_task,
    _get_text_models,
    _get_image_models,
    _get_instruct_text_datasets,
    _get_dpo_datasets,
)


def organise_tournament_round(ids: list[str]) -> Round:
    ids_copy = ids.copy()
    random.shuffle(ids_copy)

    if len(ids_copy) <= MIN_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND:
        if len(ids_copy) % 2 != 0:
            ids_copy.append(PREVIOUS_WINNER_BASE_CONTESTANT)

        random.shuffle(ids_copy)
        pairs = []
        for i in range(0, len(ids_copy), 2):
            pairs.append((ids_copy[i], ids_copy[i+1]))

        random.shuffle(pairs)
        return KnockoutRound(pairs=pairs)
    else:
        num_groups = math.ceil(len(ids_copy) / EXPECTED_GROUP_SIZE)
        if len(ids_copy) / num_groups < 6:
            num_groups = max(1, math.ceil(len(ids_copy) / EXPECTED_GROUP_SIZE - 1))

        groups = [[] for _ in range(num_groups)]
        base_size = len(ids_copy) // num_groups
        remainder = len(ids_copy) % num_groups
        group_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_groups)]

        random.shuffle(ids_copy)
        idx = 0
        for g in range(num_groups):
            group_members = ids_copy[idx:idx + group_sizes[g]]
            groups[g] = Group(member_ids=group_members)
            idx += group_sizes[g]

        random.shuffle(groups)
        return GroupRound(groups=groups)


def summarise_result(result: Round, ids_count: int):
    print(f"\n--- Tournament with {ids_count} contestants ---")

    if isinstance(result, GroupRound):
        print(f"Group Round: {len(result.groups)} groups")
        group_sizes = [len(group.member_ids) for group in result.groups]
        print(f"Group sizes: {group_sizes}")
        print(f"Sample group: {result.groups[0].member_ids}")
    else:
        print(f"Knockout Round: {len(result.pairs)} pairs")
        has_bye = any(id == PREVIOUS_WINNER_BASE_CONTESTANT for pair in result.pairs for id in pair)

        if has_bye:
            bye_count = sum(1 for pair in result.pairs if PREVIOUS_WINNER_BASE_CONTESTANT in pair)
            print(f"Contains {bye_count} previous winner entries")

        if len(result.pairs) <= MIN_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND:
            for pair in result.pairs:
                print(f"{pair[0]} vs {pair[1]}")


async def create_text_tournament_round(
    round_data: Round,
    config: Config,
    is_final_round: bool = False,
) -> TournamentRound:
    if isinstance(round_data, GroupRound):
        num_groups = len(round_data.groups)
        print(f"Creating text tournament for {num_groups} groups (1 instruct + 1 DPO + 1 GRPO per group)")
        tasks = await _create_group_text_tasks(round_data, config, is_final_round)
    elif is_final_round:
        print("Creating final text tournament (1 instruct + 1 DPO + 1 GRPO with 1 big model)")
        tasks = await _create_one_of_each_text_task(config, use_big_model=True)
    else:
        num_pairs = len(round_data.pairs)
        print(f"Creating text tournament for {num_pairs} knockout pairs (probability-based)")
        tasks = await _create_probability_based_text_tasks(round_data, config)
    
    return TournamentRound(
        round_structure=round_data,
        tasks=[str(task.task_id) for task in tasks],
        is_final_round=is_final_round
    )


async def create_image_tournament_round(round_data: Round, config: Config) -> TournamentRound:
    image_models = _get_image_models(config.keypair)
    tasks = []
    
    if isinstance(round_data, GroupRound):
        num_groups = len(round_data.groups)
        print(f"Creating image tournament for {num_groups} groups (1 per group)")
        
        for i, group in enumerate(round_data.groups):
            print(f"  Group {i+1} ({len(group.member_ids)} members):")
            task = await create_synthetic_image_task(config, image_models)
            print(f"    Image: {task.task_id} - Model: {task.model_id}")
            tasks.append(task)
    else:
        num_pairs = len(round_data.pairs)
        print(f"Creating image tournament for {num_pairs} knockout pairs (1 per pair)")
        
        for i, pair in enumerate(round_data.pairs):
            print(f"  Pair {i+1} ({pair[0]} vs {pair[1]}):")
            task = await create_synthetic_image_task(config, image_models)
            print(f"    Image: {task.task_id} - Model: {task.model_id}")
            tasks.append(task)
    
    return TournamentRound(
        round_structure=round_data,
        tasks=[str(task.task_id) for task in tasks],
        is_final_round=False
    )


async def _create_group_text_tasks(round_data: GroupRound, config: Config, is_final_round: bool) -> list[RawTask]:
    models = _get_text_models(config.keypair)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)
    
    tasks = []
    for i, group in enumerate(round_data.groups):
        print(f"  Group {i+1} ({len(group.member_ids)} members): creating 1 instruct + 1 DPO + 1 GRPO task")
        
        instruct_task = await create_synthetic_instruct_text_task(config, models, instruct_datasets)
        print(f"    Instruct: {instruct_task.task_id} - Model: {instruct_task.model_id} - Dataset: {instruct_task.ds}")
        tasks.append(instruct_task)
        
        dpo_task = await create_synthetic_dpo_task(config, models, dpo_datasets)
        print(f"    DPO: {dpo_task.task_id} - Model: {dpo_task.model_id} - Dataset: {dpo_task.ds}")
        tasks.append(dpo_task)
        
        grpo_task = await create_synthetic_grpo_task(config, models, instruct_datasets)
        print(f"    GRPO: {grpo_task.task_id} - Model: {grpo_task.model_id} - Dataset: {grpo_task.ds}")
        tasks.append(grpo_task)
    return tasks


async def _create_one_of_each_text_task(config: Config, use_big_model: bool) -> list[RawTask]:
    small_models = _get_text_models(config.keypair)
    big_models = _get_text_models(config.keypair, smallest_size_b=12.0, largest_size_b=71.0)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)
    
    tasks = []
    
    instruct_task = await create_synthetic_instruct_text_task(config, big_models, instruct_datasets)
    print(f"  Instruct (BIG): {instruct_task.task_id} - Model: {instruct_task.model_id} - Dataset: {instruct_task.ds}")
    tasks.append(instruct_task)
    
    dpo_task = await create_synthetic_dpo_task(config, small_models, dpo_datasets)
    print(f"  DPO: {dpo_task.task_id} - Model: {dpo_task.model_id} - Dataset: {dpo_task.ds}")
    tasks.append(dpo_task)
    
    grpo_task = await create_synthetic_grpo_task(config, small_models, instruct_datasets)
    print(f"  GRPO: {grpo_task.task_id} - Model: {grpo_task.model_id} - Dataset: {grpo_task.ds}")
    tasks.append(grpo_task)
    
    return tasks


async def _create_probability_based_text_tasks(round_data: KnockoutRound, config: Config) -> list[RawTask]:
    num_tasks = len(round_data.pairs)
    models = _get_text_models(config.keypair)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)
    
    text_total = PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT + PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO + PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_GRPO
    instruct_prob = PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT / text_total
    dpo_prob = PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO / text_total
    
    tasks = []
    for i in range(num_tasks):
        pair = round_data.pairs[i]
        print(f"  Pair {i+1} ({pair[0]} vs {pair[1]}):")
        
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
        
        print(f"    {task_type}: {task.task_id} - Model: {task.model_id} - Dataset: {task.ds}")
        tasks.append(task)
    return tasks


if __name__ == "__main__":
    test_sizes = [250, 144, 37, 10, 5, 2, 1]

    for size in test_sizes:
        contestant_ids = [f"player_{i}" for i in range(1, size + 1)]
        result = organise_tournament_round(contestant_ids)
        summarise_result(result, size)
        
        print("\n--- Task Creation Preview ---")
        if isinstance(result, GroupRound):
            num_groups = len(result.groups)
            print(f"Text Tournament: {num_groups} groups × 3 tasks = {num_groups * 3} total text tasks")
            print(f"Image Tournament: {num_groups} groups × 1 task = {num_groups} total image tasks")
        else:
            num_pairs = len(result.pairs)
            is_final = size <= 2
            if is_final:
                print(f"Text Tournament (FINAL): 3 tasks (1 instruct big model + 1 DPO + 1 GRPO)")
            else:
                print(f"Text Tournament: {num_pairs} pairs × 1 task = {num_pairs} total text tasks (probability-based)")
            print(f"Image Tournament: {num_pairs} pairs × 1 task = {num_pairs} total image tasks")
        print()

