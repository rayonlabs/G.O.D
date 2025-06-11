import math
import random

from fiber.chain.models import Node
from core.models.tournament_models import Round, KnockoutRound, GroupRound, Group
from validator.core.constants import EXPECTED_GROUP_SIZE
from validator.core.constants import MIN_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND
from validator.core.constants import PREVIOUS_WINNER_BASE_CONTESTANT


def organise_tournament_round(nodes: list[Node]) -> Round:
    nodes_copy = nodes.copy()
    random.shuffle(nodes_copy)

    if len(nodes_copy) <= MIN_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND:
        hotkeys = [node.hotkey for node in nodes_copy]
        if len(hotkeys) % 2 != 0:
            hotkeys.append(PREVIOUS_WINNER_BASE_CONTESTANT)

        random.shuffle(hotkeys)
        pairs = []
        for i in range(0, len(hotkeys), 2):
            pairs.append((hotkeys[i], hotkeys[i+1]))

        random.shuffle(pairs)
        return KnockoutRound(pairs=pairs)
    else:
        num_groups = math.ceil(len(nodes_copy) / EXPECTED_GROUP_SIZE)
        if len(nodes_copy) / num_groups < 6:
            num_groups = max(1, math.ceil(len(nodes_copy) / EXPECTED_GROUP_SIZE - 1))

        groups = [[] for _ in range(num_groups)]
        base_size = len(nodes_copy) // num_groups
        remainder = len(nodes_copy) % num_groups
        group_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_groups)]

        random.shuffle(nodes_copy)
        idx = 0
        for g in range(num_groups):
            group_nodes = nodes_copy[idx:idx + group_sizes[g]]
            group_hotkeys = [node.hotkey for node in group_nodes]
            groups[g] = Group(member_ids=group_hotkeys)
            idx += group_sizes[g]

        random.shuffle(groups)
        return GroupRound(groups=groups)


def summarise_result(result: Round, node_count: int):
    print(f"\n--- Tournament with {node_count} contestants ---")

    if isinstance(result, GroupRound):
        print(f"Group Round: {len(result.groups)} groups")
        group_sizes = [len(group.member_ids) for group in result.groups]
        print(f"Group sizes: {group_sizes}")
        print(f"Sample group: {result.groups[0].member_ids}")
    else:
        print(f"Knockout Round: {len(result.pairs)} pairs")
        has_bye = any(hotkey == PREVIOUS_WINNER_BASE_CONTESTANT for pair in result.pairs for hotkey in pair)

        if has_bye:
            bye_count = sum(1 for pair in result.pairs if PREVIOUS_WINNER_BASE_CONTESTANT in pair)
            print(f"Contains {bye_count} previous winner entries")

        if len(result.pairs) <= MIN_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND:
            for pair in result.pairs:
                print(f"{pair[0]} vs {pair[1]}")