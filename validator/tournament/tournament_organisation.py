import math
import random

from core.models.utility_models import Round
from core.models.utility_models import KnockoutRound
from core.models.utility_models import GroupRound
from core.models.utility_models import Group
from validator.core.constants import EXPECTED_GROUP_SIZE
from validator.core.constants import MIN_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND
from validator.core.constants import PREVIOUS_WINNER_BASE_CONTESTANT


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


if __name__ == "__main__":
    test_sizes = [250, 144, 37, 10, 5, 2, 1]

    for size in test_sizes:
        contestant_ids = [f"player_{i}" for i in range(1, size + 1)]
        result = organise_tournament_round(contestant_ids)
        summarise_result(result, size)

