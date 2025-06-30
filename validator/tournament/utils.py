#!/usr/bin/env python3

import random
from typing import List

from fiber.chain.models import Node

from core.models.tournament_models import RoundStatus
from core.models.tournament_models import RoundType
from core.models.tournament_models import TournamentType
from validator.db.database import PSQLDB
from validator.db.sql.nodes import get_all_nodes
from validator.db.sql.nodes import insert_nodes
from validator.db.sql.submissions_and_scoring import get_task_winner
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament_group_members
from validator.db.sql.tournaments import get_tournament_groups
from validator.db.sql.tournaments import get_tournament_pairs
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import get_tournament_tasks
from validator.tournament.tournament_manager import get_round_winners
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def get_best_dropped_out_contestant(tournament_id: str, current_round_id: str, psql_db: PSQLDB) -> str | None:
    """
    Logic:
    - If previous round was knockout: take one of the losers
    - If previous round was group: take random dropped out contestant
    """
    rounds = await get_tournament_rounds(tournament_id, psql_db)
    rounds.sort(key=lambda r: r.round_number)

    current_round_index = None
    for i, round_data in enumerate(rounds):
        if round_data.round_id == current_round_id:
            current_round_index = i
            break

    if current_round_index is None or current_round_index == 0:
        return None

    previous_round = rounds[current_round_index - 1]

    if previous_round.round_type == RoundType.KNOCKOUT:
        pairs = await get_tournament_pairs(previous_round.round_id, psql_db)
        tasks = await get_tournament_tasks(previous_round.round_id, psql_db)

        losers = []
        for i, pair in enumerate(pairs):
            if i < len(tasks):
                winner = await get_task_winner(tasks[i].task_id, psql_db)
                if winner:
                    loser = pair.hotkey2 if winner == pair.hotkey1 else pair.hotkey1
                    losers.append(loser)

        if losers:
            return losers[0]

    elif previous_round.round_type == RoundType.GROUP:
        groups = await get_tournament_groups(previous_round.round_id, psql_db)

        current_participants = set()
        if rounds[current_round_index].round_type == RoundType.GROUP:
            current_groups = await get_tournament_groups(current_round_id, psql_db)
            for group in current_groups:
                members = await get_tournament_group_members(group.group_id, psql_db)
                for member in members:
                    current_participants.add(member.hotkey)
        else:
            current_pairs = await get_tournament_pairs(current_round_id, psql_db)
            for pair in current_pairs:
                current_participants.add(pair.hotkey1)
                current_participants.add(pair.hotkey2)

        all_previous_participants = set()
        for group in groups:
            members = await get_tournament_group_members(group.group_id, psql_db)
            for member in members:
                all_previous_participants.add(member.hotkey)

        non_advancing = all_previous_participants - current_participants

        if non_advancing:
            return random.choice(list(non_advancing))

    return None


async def setup_mock_nodes(psql_db: PSQLDB, num_nodes: int = 8) -> List[Node]:
    """Create mock nodes and insert them into the database if no nodes exist."""

    existing_nodes = await get_all_nodes(psql_db)
    if existing_nodes:
        logger.info(f"Found {len(existing_nodes)} existing nodes in database")
        # Use existing nodes, but limit to requested number
        return existing_nodes[:num_nodes]

    logger.info(f"Setting up {num_nodes} mock nodes...")
    mock_nodes = []
    for i in range(num_nodes):
        hotkey = f"mock_hotkey_{i:03d}"
        coldkey = f"mock_coldkey_{i:03d}"

        node = Node(
            hotkey=hotkey,
            coldkey=coldkey,
            node_id=i,
            netuid=1,
            ip="127.0.0.1",
            ip_type=4,
            port=8080 + i,
            stake=100.0 + random.uniform(0, 50),
            incentive=0.0,
            alpha_stake=0.0,
            tao_stake=0.0,
            trust=random.uniform(0.5, 1.0),
            vtrust=random.uniform(0.5, 1.0),
            last_updated=0.0,
            protocol=4,
        )
        mock_nodes.append(node)

    async with await psql_db.connection() as connection:
        await insert_nodes(connection, mock_nodes)

    logger.info(f"Created and inserted {len(mock_nodes)} mock nodes")
    return mock_nodes


async def get_node_by_hotkey(psql_db: PSQLDB, hotkey: str) -> Node | None:
    """Get node by hotkey from database."""
    nodes = await get_all_nodes(psql_db)
    for node in nodes:
        if node.hotkey == hotkey:
            return node
    return None


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
