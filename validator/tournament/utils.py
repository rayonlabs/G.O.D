#!/usr/bin/env python3


from collections import Counter

import numpy as np

from core.models.tournament_models import RoundType
from core.models.tournament_models import TournamentParticipant
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentTask
from core.models.tournament_models import TournamentType
from core.models.utility_models import TaskType
from validator.core.config import Config
from validator.core.constants import DEFAULT_PARTICIPANT_COMMIT
from validator.core.constants import DEFAULT_PARTICIPANT_REPO
from validator.core.constants import EMISSION_BURN_HOTKEY
from validator.core.models import MinerResultsImage
from validator.core.models import MinerResultsText
from validator.db import constants as db_cst
from validator.db.database import PSQLDB
from validator.db.sql import tasks as task_sql
from validator.db.sql.submissions_and_scoring import get_all_scores_and_losses_for_task
from validator.db.sql.submissions_and_scoring import get_task_winner
from validator.db.sql.submissions_and_scoring import get_task_winners
from validator.db.sql.tasks import get_task
from validator.db.sql.tournaments import add_tournament_tasks
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament_group_members
from validator.db.sql.tournaments import get_tournament_groups
from validator.db.sql.tournaments import get_tournament_participant
from validator.db.sql.tournaments import get_tournament_tasks
from validator.evaluation.scoring import calculate_miner_ranking_and_scores
from validator.tournament.task_creator import create_new_task_of_same_type
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def replace_tournament_task(
    original_task_id: str, tournament_id: str, round_id: str, group_id: str | None, pair_id: str | None, config: Config
) -> str:
    original_task_obj = await task_sql.get_task(original_task_id, config.psql_db)
    if not original_task_obj:
        logger.error(f"Could not find original task {original_task_id}")
        raise ValueError(f"Original task {original_task_id} not found")

    new_task = await create_new_task_of_same_type(original_task_obj, config)

    new_tournament_task = TournamentTask(
        tournament_id=tournament_id,
        round_id=round_id,
        task_id=new_task.task_id,
        group_id=group_id,
        pair_id=pair_id,
    )
    await add_tournament_tasks([new_tournament_task], config.psql_db)
    logger.info(f"Created replacement task {new_task.task_id} for round {round_id}")

    original_assigned_nodes = await task_sql.get_nodes_assigned_to_task(original_task_id, config.psql_db)
    for node in original_assigned_nodes:
        await task_sql.assign_node_to_task(new_task.task_id, node, config.psql_db)

        original_expected_repo_name = await task_sql.get_expected_repo_name(original_task_id, node.hotkey, config.psql_db)
        if original_expected_repo_name:
            await task_sql.set_expected_repo_name(new_task.task_id, node, config.psql_db, original_expected_repo_name)
            logger.info(
                f"Copied node {node.hotkey} with expected_repo_name {original_expected_repo_name} to replacement task {new_task.task_id}"
            )
        else:
            logger.warning(f"No expected repo name found for node {node.hotkey} in original task {original_task_id}")

    await task_sql.delete_task(original_task_id, config.psql_db)
    logger.info(f"Deleted original task {original_task_id} from db.")

    return new_task.task_id


async def get_task_results_for_ranking(task_id: str, psql_db: PSQLDB) -> list[MinerResultsText | MinerResultsImage]:
    """
    Fetch task results from database and convert to MinerResults objects for ranking.
    """
    scores_dicts = await get_all_scores_and_losses_for_task(task_id, psql_db)

    if not scores_dicts:
        logger.warning(f"No scores found for task {task_id}")
        return []

    task_object = await get_task(task_id, psql_db)
    if not task_object:
        logger.warning(f"Could not get task object for task {task_id}")
        return []

    task_type = task_object.task_type

    miner_results = []
    for score_dict in scores_dicts:
        hotkey = score_dict[db_cst.HOTKEY]
        test_loss = score_dict.get(db_cst.TEST_LOSS)
        synth_loss = score_dict.get(db_cst.SYNTH_LOSS)

        # Skip invalid results
        if test_loss is None or np.isnan(test_loss):
            continue

        # Create appropriate MinerResults object
        if task_type in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK]:
            miner_result = MinerResultsText(
                hotkey=hotkey,
                test_loss=test_loss,
                synth_loss=synth_loss if synth_loss is not None and not np.isnan(synth_loss) else 1000.0,
                is_finetune=True,  # assume all finetuned
                task_type=task_type,
            )
        else:
            # For image tasks
            miner_result = MinerResultsImage(
                hotkey=hotkey,
                test_loss=test_loss,
                synth_loss=synth_loss if synth_loss is not None and not np.isnan(synth_loss) else 1000.0,
                is_finetune=True,
            )

        miner_results.append(miner_result)

    return miner_results


async def get_base_contestant(psql_db: PSQLDB, tournament_type: TournamentType, config: Config) -> TournamentParticipant | None:
    """Get a BASE contestant as the last tournament winner."""

    latest_winner = await get_latest_tournament_winner_participant(psql_db, tournament_type, config)
    if latest_winner:
        logger.info(f"Using latest tournament winner as BASE: {latest_winner.hotkey}")
        return latest_winner

    logger.info(
        f"No previous tournament winner found for type {tournament_type.value}, using hardcoded base winner: {EMISSION_BURN_HOTKEY}"
    )

    hardcoded_participant = TournamentParticipant(
        tournament_id="",
        hotkey=EMISSION_BURN_HOTKEY,
        training_repo=DEFAULT_PARTICIPANT_REPO,
        training_commit_hash=DEFAULT_PARTICIPANT_COMMIT,
        stake_required=0,
    )

    return hardcoded_participant


async def get_latest_tournament_winner_participant(
    psql_db: PSQLDB, tournament_type: TournamentType, config: Config
) -> TournamentParticipant | None:
    """Get the winner participant from the latest completed tournament of the given type."""
    latest_tournament = await get_latest_completed_tournament(psql_db, tournament_type)
    if not latest_tournament:
        logger.warning(f"No completed tournaments found for type {tournament_type.value}")
        return None

    winner_hotkey = latest_tournament.winner_hotkey
    if not winner_hotkey:
        logger.warning(f"Tournament {latest_tournament.tournament_id} is completed but has no winner_hotkey stored")
        return None

    logger.info(f"Found latest tournament winner: {winner_hotkey}")
    winner_participant = await get_tournament_participant(latest_tournament.tournament_id, winner_hotkey, psql_db)
    if winner_participant.hotkey == EMISSION_BURN_HOTKEY:
        winner_participant.hotkey = latest_tournament.base_winner_hotkey

    return winner_participant


def draw_knockout_bracket(rounds_data, winners_by_round):
    """Draw an ASCII art bracket diagram for knockout tournament progression."""
    logger.info("\nKNOCKOUT BRACKET:")
    logger.info("=" * 60)

    if not rounds_data:
        logger.info("No rounds data available")
        return

    knockout_rounds = [r for r in rounds_data if r.get("type") == RoundType.KNOCKOUT]
    if not knockout_rounds:
        logger.info("No knockout rounds found")
        return

    bracket_lines = []

    for round_num, round_data in enumerate(knockout_rounds):
        participants = round_data.get("participants", [])
        knockout_round_index = None
        for i, r in enumerate(rounds_data):
            if r.get("type") == RoundType.KNOCKOUT and r == round_data:
                knockout_round_index = i
                break

        winners = winners_by_round.get(knockout_round_index, []) if knockout_round_index is not None else []

        if not participants:
            continue

        round_header = f"Round {round_num + 1}"
        if round_data.get("is_final_round"):
            round_header += " 🔥 BOSS ROUND 🔥"
        bracket_lines.append(f"{round_header:>20}")

        for i in range(0, len(participants), 2):
            if i + 1 < len(participants):
                p1 = participants[i]
                p2 = participants[i + 1]

                p1_won = p1 in winners
                p2_won = p2 in winners

                indent = "  " * round_num
                if p1_won:
                    line1 = f"{indent}├─ {p1} ✓"
                else:
                    line1 = f"{indent}├─ {p1}"

                if p2_won:
                    line2 = f"{indent}├─ {p2} ✓"
                else:
                    line2 = f"{indent}├─ {p2}"

                bracket_lines.append(f"{line1:>40}")
                bracket_lines.append(f"{line2:>40}")

                if round_num < len(knockout_rounds) - 1:
                    bracket_lines.append(f"{indent}│")

        bracket_lines.append("")

    for line in bracket_lines:
        logger.info(line)


async def draw_group_stage_table(rounds_data, winners_by_round, psql_db):
    """Draw a table showing group stage results."""
    logger.info("\nGROUP STAGE RESULTS:")
    logger.info("=" * 60)

    group_round = None
    group_round_index = None
    for i, round_data in enumerate(rounds_data):
        if round_data.get("type") == RoundType.GROUP:
            group_round = round_data
            group_round_index = i
            break

    if not group_round:
        logger.info("No group stage found")
        return

    round_id = group_round.get("round_id")
    if not round_id:
        logger.info("No round ID found for group stage")
        return

    group_objs = await get_tournament_groups(round_id, psql_db)
    if not group_objs:
        logger.info("No groups found for group stage")
        return

    winners = winners_by_round.get(group_round_index, []) if group_round_index is not None else []

    logger.info(f"Group Stage: {len(group_objs)} groups")
    logger.info("")

    for group in group_objs:
        group_id = group.group_id
        members = await get_tournament_group_members(group_id, psql_db)
        hotkeys = [m.hotkey for m in members]
        logger.info(f"Group {group_id}:")
        logger.info("-" * 40)
        for i, participant in enumerate(hotkeys):
            if participant in winners:
                logger.info(f"  {i + 1:2d}. {participant} ✓ (ADVANCED)")
            else:
                logger.info(f"  {i + 1:2d}. {participant}")
        logger.info("")


async def get_knockout_winners(
    completed_round: TournamentRoundData, round_tasks: list[TournamentTask], psql_db: PSQLDB, config: Config
) -> list[str]:
    """Get winners from knockout round."""
    winners = []

    if not completed_round.is_final_round:
        # Use simple quality score comparison for regular knockout rounds
        for task in round_tasks:
            winner = await get_task_winner(task.task_id, psql_db)
            if winner:
                winners.append(winner)
    else:
        # Boss round. You need to beat the boss by 5% to win the task.
        # Best of 3 wins the round.
        boss_hotkey = EMISSION_BURN_HOTKEY
        opponent_hotkey = None
        task_winners = []

        for task in round_tasks:
            logger.info(f"Processing boss round task {task.task_id}")

            task_object = await get_task(task.task_id, psql_db)

            miner_results = await get_task_results_for_ranking(task.task_id, psql_db)
            if not miner_results:
                logger.warning(f"No valid results for boss round task {task.task_id}. Winner is base contestant.")
                task_winners.append(boss_hotkey)
                continue

            ranked_results = calculate_miner_ranking_and_scores(miner_results)

            boss_loss = None
            opponent_loss = None
            opponent_hotkey = None

            for result in ranked_results:
                if result.hotkey == boss_hotkey:
                    boss_loss = result.adjusted_loss
                else:
                    if opponent_hotkey is None:
                        opponent_hotkey = result.hotkey
                        opponent_loss = result.adjusted_loss

            if boss_loss is None or opponent_loss is None:
                logger.warning(f"Boss round task {task.task_id} missing boss or opponent loss")
                continue

            logger.info(f"Boss round task {task.task_id}: Boss loss: {boss_loss:.6f}, Opponent loss: {opponent_loss:.6f}")

            if task_object.task_type == TaskType.GRPOTASK:
                if boss_loss * 1.05 > opponent_loss:
                    task_winners.append(boss_hotkey)
                    logger.info(f"GRPO task: Boss wins (higher is better): {boss_loss:.6f} > {opponent_loss * 1.05:.6f}")
                else:
                    task_winners.append(opponent_hotkey)
                    logger.info(f"GRPO task: Opponent wins (higher is better): {opponent_loss:.6f} >= {boss_loss * 1.05:.6f}")
            else:
                if boss_loss * 0.95 < opponent_loss:
                    task_winners.append(boss_hotkey)
                    logger.info(
                        f"{task_object.task_type} task: Boss wins (lower is better): {boss_loss:.6f} < {opponent_loss * 1.05:.6f}"
                    )
                else:
                    task_winners.append(opponent_hotkey)
                    logger.info(
                        f"{task_object.task_type} task: Opponent wins (lower is better): "
                        f"{opponent_loss:.6f} <= {boss_loss * 0.95:.6f}"
                    )

        boss_round_winner = Counter(task_winners).most_common(1)[0][0]
        logger.info(f"Boss round winner: {boss_round_winner}")
        winners = [boss_round_winner]

    return winners


async def get_group_winners(
    completed_round: TournamentRoundData, round_tasks: list[TournamentTask], psql_db: PSQLDB
) -> list[str]:
    """Get winners from group round based on task wins."""
    NUM_WINNERS_TO_ADVANCE = 2
    group_tasks = {}
    for task in round_tasks:
        if task.group_id:
            if task.group_id not in group_tasks:
                group_tasks[task.group_id] = []
            group_tasks[task.group_id].append(task.task_id)

    logger.info(f"Processing {len(group_tasks)} groups in round {completed_round.round_id}")
    all_winners = []
    for group_id, task_ids in group_tasks.items():
        participants = await get_tournament_group_members(group_id, psql_db)
        participant_hotkeys = [p.hotkey for p in participants]
        logger.info(f"Group {group_id}: {len(participant_hotkeys)} participants, {len(task_ids)} tasks")

        if not participant_hotkeys or not task_ids:
            continue

        task_winners = await get_task_winners(task_ids, psql_db)
        logger.info(f"Group {group_id} task winners: {task_winners}")

        hotkey_win_counts = Counter(task_winners.values())
        logger.info(f"Group {group_id} win counts: {dict(hotkey_win_counts)}")

        if len(hotkey_win_counts) == 0:
            logger.warning(f"Group {group_id} has {len(hotkey_win_counts)} winners - proceeding with no winners")
            continue

        sorted_participants = sorted(hotkey_win_counts.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_participants) == 1:
            all_winners.append(sorted_participants[0][0])
            logger.info(f"Group {group_id}: Single winner {sorted_participants[0][0]} with {sorted_participants[0][1]} wins")
        else:
            max_wins = sorted_participants[0][1]
            tied_for_first = [hotkey for hotkey, wins in sorted_participants if wins == max_wins]

            if len(tied_for_first) == 1:
                group_winners = [hotkey for hotkey, _ in sorted_participants[:NUM_WINNERS_TO_ADVANCE]]
                logger.info(f"Group {group_id}: Advancing top {NUM_WINNERS_TO_ADVANCE}: {group_winners}")
            else:
                group_winners = tied_for_first
                logger.info(f"Group {group_id}: {len(tied_for_first)} tied for first with {max_wins} wins each: {group_winners}")

            all_winners.extend(group_winners)

    logger.info(f"Total group stage winners: {len(all_winners)} - {all_winners}")
    return all_winners


async def get_round_winners(completed_round: TournamentRoundData, psql_db: PSQLDB, config: Config) -> list[str]:
    """Get winners from the completed round."""
    round_tasks = await get_tournament_tasks(completed_round.round_id, psql_db)

    if completed_round.round_type == RoundType.KNOCKOUT:
        return await get_knockout_winners(completed_round, round_tasks, psql_db, config)
    else:
        return await get_group_winners(completed_round, round_tasks, psql_db)