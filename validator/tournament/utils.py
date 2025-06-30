#!/usr/bin/env python3


from core.models.tournament_models import RoundStatus
from core.models.tournament_models import RoundType
from core.models.tournament_models import TournamentType
from validator.db.database import PSQLDB
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament_group_members
from validator.db.sql.tournaments import get_tournament_groups
from validator.db.sql.tournaments import get_tournament_rounds
from validator.tournament.tournament_manager import get_round_winners
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def get_base_contestant(psql_db: PSQLDB, tournament_type: TournamentType) -> str:
    """Get a BASE contestant as the last tournament winner."""

    latest_winner = await get_latest_tournament_winner(psql_db, tournament_type)
    if latest_winner:
        logger.info(f"Using latest tournament winner as BASE: {latest_winner}")
        return latest_winner


async def get_latest_tournament_winner(psql_db: PSQLDB, tournament_type: TournamentType) -> str | None:
    """Get the winner of the most recently completed tournament from the final round."""

    latest_tournament = await get_latest_completed_tournament(psql_db, tournament_type)
    if not latest_tournament:
        return None

    tournament_id = latest_tournament.tournament_id

    rounds = await get_tournament_rounds(tournament_id, psql_db)
    if not rounds:
        return None

    final_round = rounds[-1]
    if not final_round.is_final_round:
        logger.error(f"Last round in tournament {tournament_id} is not the final round: {final_round.round_id}")
        return None

    if final_round.status != RoundStatus.COMPLETED:
        return None

    winners = await get_round_winners(final_round, psql_db)

    if len(winners) != 1:
        logger.error(f"Expected 1 winner in final round {final_round.round_id}, got {len(winners)}")

    if winners:
        return winners[0]
    else:
        return None


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


def get_boss_round_ascii_art() -> str:
    """Get ASCII art for the boss round."""
    return """
====================================================================
BOSS ROUND INCOMING
====================================================================

    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║                       FINAL ROUND                        ║
    ║                                                          ║
    ║              The ultimate challenge awaits!              ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝

====================================================================
"""
