#!/usr/bin/env python3


from collections import Counter

from core.models.tournament_models import RoundType
from core.models.tournament_models import TournamentParticipant
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentTask
from core.models.tournament_models import TournamentType
from validator.core import constants as cst
from validator.db import constants as db_cst
from validator.db.database import PSQLDB
from validator.db.sql.submissions_and_scoring import get_all_scores_and_losses_for_task
from validator.db.sql.submissions_and_scoring import get_task_winner
from validator.db.sql.submissions_and_scoring import get_task_winners
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament_group_members
from validator.db.sql.tournaments import get_tournament_groups
from validator.db.sql.tournaments import get_tournament_participant
from validator.db.sql.tournaments import get_tournament_tasks
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def get_base_contestant(psql_db: PSQLDB, tournament_type: TournamentType) -> TournamentParticipant | None:
    """Get a BASE contestant as the last tournament winner."""

    latest_winner = await get_latest_tournament_winner_participant(psql_db, tournament_type)
    if latest_winner:
        logger.info(f"Using latest tournament winner as BASE: {latest_winner.hotkey}")
        return latest_winner


async def get_latest_tournament_winner_participant(
    psql_db: PSQLDB, tournament_type: TournamentType
) -> TournamentParticipant | None:
    """Get the winner of the most recently completed tournament from the tournament table."""

    latest_tournament = await get_latest_completed_tournament(psql_db, tournament_type)
    if not latest_tournament:
        return None

    tournament_id = latest_tournament.tournament_id
    winner_hotkey = latest_tournament.winner_hotkey

    if not winner_hotkey:
        logger.warning(f"Tournament {tournament_id} is completed but has no winner_hotkey stored")
        return None

    logger.info(f"Found latest tournament winner: {winner_hotkey}")
    winner_participant = await get_tournament_participant(tournament_id, winner_hotkey, psql_db)
    if winner_participant.hotkey == cst.TOURNAMENT_BASE_CONTESTANT_HOTKEY:
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
    completed_round: TournamentRoundData, round_tasks: list[TournamentTask], psql_db: PSQLDB
) -> list[str]:
    """Get winners from knockout round."""
    winners = []

    if not completed_round.is_final_round:
        for task in round_tasks:
            winner = await get_task_winner(task.task_id, psql_db)
            if winner:
                winners.append(winner)
    else:
        # Boss round. You need to beat the boss by 5% to win the task.
        # Best of 3 wins the round.
        boss_hotkey = cst.TOURNAMENT_BASE_CONTESTANT_HOTKEY
        opponent_hotkey = None
        task_winners = []
        for task in round_tasks:
            logger.info(f"Processing boss round task {task.task_id}")
            boss_loss = None
            opponent_loss = None
            scores_dicts = await get_all_scores_and_losses_for_task(task.task_id, psql_db)
            logger.info(f"Boss round task {task.task_id}: Found {len(scores_dicts)} score entries")
            if scores_dicts:
                for score_dict in scores_dicts:
                    if score_dict[db_cst.HOTKEY] == boss_hotkey:
                        boss_loss = score_dict[db_cst.TEST_LOSS]
                    else:
                        opponent_loss = score_dict[db_cst.TEST_LOSS]
                        opponent_hotkey = score_dict[db_cst.HOTKEY]
            if boss_loss is not None and opponent_loss is not None:
                logger.info(f"Boss round task {task.task_id}: Boss loss: {boss_loss}, Opponent loss: {opponent_loss}")
                if opponent_loss < boss_loss * 0.95:
                    task_winners.append(opponent_hotkey)
                    logger.info(f"Opponent wins task {task.task_id}")
                else:
                    task_winners.append(boss_hotkey)
                    logger.info(f"Boss wins task {task.task_id}")

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

    all_winners = []
    for group_id, task_ids in group_tasks.items():
        participants = await get_tournament_group_members(group_id, psql_db)
        participant_hotkeys = [p.hotkey for p in participants]

        if not participant_hotkeys or not task_ids:
            continue

        task_winners = await get_task_winners(task_ids, psql_db)

        hotkey_win_counts = Counter(task_winners.values())

        if len(hotkey_win_counts) == 0:
            raise ValueError(f"Group {group_id} has {len(hotkey_win_counts)} winners")

        sorted_participants = sorted(hotkey_win_counts.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_participants) == 1:
            all_winners.append(sorted_participants[0][0])
        else:
            max_wins = sorted_participants[0][1]
            tied_for_first = [hotkey for hotkey, wins in sorted_participants if wins == max_wins]

            if len(tied_for_first) == 1:
                group_winners = [hotkey for hotkey, _ in sorted_participants[:NUM_WINNERS_TO_ADVANCE]]
            else:
                group_winners = tied_for_first

            all_winners.extend(group_winners)

    return all_winners


async def get_round_winners(completed_round: TournamentRoundData, psql_db: PSQLDB) -> list[str]:
    """Get winners from the completed round."""
    round_tasks = await get_tournament_tasks(completed_round.round_id, psql_db)

    if completed_round.round_type == RoundType.KNOCKOUT:
        return await get_knockout_winners(completed_round, round_tasks, psql_db)
    else:
        return await get_group_winners(completed_round, round_tasks, psql_db)
