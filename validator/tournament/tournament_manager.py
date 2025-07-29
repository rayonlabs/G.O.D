import asyncio
import math
import random

from fiber.chain.models import Node

import validator.core.constants as cst
from core.models.payload_models import TrainingRepoResponse
from core.models.tournament_models import Group
from core.models.tournament_models import GroupRound
from core.models.tournament_models import KnockoutRound
from core.models.tournament_models import RespondingNode
from core.models.tournament_models import Round
from core.models.tournament_models import RoundStatus
from core.models.tournament_models import RoundType
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentParticipant
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentStatus
from core.models.tournament_models import TournamentType
from core.models.tournament_models import generate_round_id
from core.models.tournament_models import generate_tournament_id
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.db.database import PSQLDB
from validator.db.sql import tasks as task_sql
from validator.db.sql.nodes import get_all_nodes
from validator.db.sql.nodes import get_node_by_hotkey
from validator.db.sql.tournaments import add_tournament_participants
from validator.db.sql.tournaments import calculate_boosted_stake
from validator.db.sql.tournaments import count_completed_tournament_entries
from validator.db.sql.tournaments import create_tournament
from validator.db.sql.tournaments import eliminate_tournament_participants
from validator.db.sql.tournaments import get_participants_with_insufficient_stake
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import get_tournament_group_members
from validator.db.sql.tournaments import get_tournament_groups
from validator.db.sql.tournaments import get_tournament_pairs
from validator.db.sql.tournaments import get_tournament_participants
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import get_tournament_rounds_with_status
from validator.db.sql.tournaments import get_tournament_tasks
from validator.db.sql.tournaments import get_tournaments_with_status
from validator.db.sql.tournaments import insert_tournament_groups_with_members
from validator.db.sql.tournaments import insert_tournament_pairs
from validator.db.sql.tournaments import insert_tournament_round
from validator.db.sql.tournaments import update_round_status
from validator.db.sql.tournaments import update_tournament_participant_training_repo
from validator.db.sql.tournaments import update_tournament_status
from validator.db.sql.tournaments import update_tournament_winner_hotkey
from validator.tournament import constants as t_cst
from validator.tournament.boss_round_sync import _copy_task_to_general
from validator.tournament.boss_round_sync import get_synced_task_id
from validator.tournament.boss_round_sync import get_synced_task_ids
from validator.tournament.boss_round_sync import sync_boss_round_tasks_to_general
from validator.tournament.task_creator import create_image_tournament_tasks
from validator.tournament.task_creator import create_text_tournament_tasks
from validator.tournament.utils import get_base_contestant
from validator.tournament.utils import get_round_winners
from validator.tournament.utils import replace_tournament_task
from validator.utils.call_endpoint import process_non_stream_fiber_get
from validator.utils.logging import LogContext
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def organise_tournament_round(nodes: list[Node], config: Config) -> Round:
    nodes_copy = nodes.copy()
    random.shuffle(nodes_copy)

    if len(nodes_copy) <= t_cst.MAX_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND:
        hotkeys = [node.hotkey for node in nodes_copy]

        if len(hotkeys) % 2 == 1:
            if cst.EMISSION_BURN_HOTKEY not in hotkeys:
                hotkeys.append(cst.EMISSION_BURN_HOTKEY)
            else:
                hotkeys.remove(cst.EMISSION_BURN_HOTKEY)

        random.shuffle(hotkeys)
        pairs = []
        for i in range(0, len(hotkeys), 2):
            pairs.append((hotkeys[i], hotkeys[i + 1]))
        random.shuffle(pairs)
        return KnockoutRound(pairs=pairs)
    else:
        num_groups = math.ceil(len(nodes_copy) / t_cst.EXPECTED_GROUP_SIZE)
        if len(nodes_copy) / num_groups < t_cst.MIN_GROUP_SIZE:
            num_groups = math.ceil(len(nodes_copy) / t_cst.EXPECTED_GROUP_SIZE - 1)

        groups = [[] for _ in range(num_groups)]
        base_size = len(nodes_copy) // num_groups
        remainder = len(nodes_copy) % num_groups
        group_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_groups)]

        random.shuffle(nodes_copy)
        idx = 0
        for g in range(num_groups):
            group_nodes = nodes_copy[idx : idx + group_sizes[g]]
            group_hotkeys = [node.hotkey for node in group_nodes]
            groups[g] = Group(member_ids=group_hotkeys, task_ids=[])
            idx += group_sizes[g]

        random.shuffle(groups)
        return GroupRound(groups=groups)


async def _create_first_round(
    tournament_id: str, tournament_type: TournamentType, nodes: list[Node], psql_db: PSQLDB, config: Config
):
    round_id = generate_round_id(tournament_id, 1)
    with LogContext(round_id=round_id):
        round_structure = organise_tournament_round(nodes, config)

        round_type = RoundType.KNOCKOUT if isinstance(round_structure, KnockoutRound) else RoundType.GROUP

        round_data = TournamentRoundData(
            round_id=round_id,
            tournament_id=tournament_id,
            round_number=1,
            round_type=round_type,
            is_final_round=False,
            status=RoundStatus.PENDING,
        )

        await insert_tournament_round(round_data, psql_db)

        if isinstance(round_structure, GroupRound):
            await insert_tournament_groups_with_members(round_id, round_structure, psql_db)
        else:
            await insert_tournament_pairs(round_id, round_structure.pairs, psql_db)

        logger.info(f"Created first round {round_id}")


async def _create_tournament_tasks(
    tournament_id: str, round_id: str, round_structure: Round, tournament_type: TournamentType, is_final: bool, config: Config
) -> list[str]:
    if tournament_type == TournamentType.TEXT:
        tasks = await create_text_tournament_tasks(round_structure, tournament_id, round_id, config, is_final)
    else:
        tasks = await create_image_tournament_tasks(round_structure, tournament_id, round_id, config, is_final)

    return tasks


async def assign_nodes_to_tournament_tasks(tournament_id: str, round_id: str, round_structure: Round, psql_db: PSQLDB) -> None:
    """Assign nodes to tournament tasks for the given round."""

    if isinstance(round_structure, GroupRound):
        for i, group in enumerate(round_structure.groups):
            group_id = f"{round_id}_group_{i + 1:03d}"

            group_tasks = await get_tournament_tasks(round_id, psql_db)
            group_tasks = [task for task in group_tasks if task.group_id == group_id]

            for task in group_tasks:
                for hotkey in group.member_ids:
                    node = await get_node_by_hotkey(hotkey, psql_db)
                    if node:
                        await task_sql.assign_node_to_task(task.task_id, node, psql_db)

                        expected_repo_name = f"tournament-{tournament_id}-{task.task_id}-{hotkey[:8]}"
                        await task_sql.set_expected_repo_name(task.task_id, node, psql_db, expected_repo_name)

                        logger.info(
                            f"Assigned {hotkey} to group task {task.task_id} with expected_repo_name: {expected_repo_name}"
                        )
    else:
        logger.info("Processing KNOCKOUT round assignment")
        round_tasks = await get_tournament_tasks(round_id, psql_db)
        logger.info(f"Found {len(round_tasks)} tasks for round {round_id}")

        for i, pair in enumerate(round_structure.pairs):
            pair_id = f"{round_id}_pair_{i + 1:03d}"
            logger.info(f"Processing pair {i+1}/{len(round_structure.pairs)}: {pair} -> {pair_id}")

            pair_tasks = [task for task in round_tasks if task.pair_id == pair_id]
            logger.info(f"Found {len(pair_tasks)} tasks for pair {pair_id}")

            for pair_task in pair_tasks:
                logger.info(f"Assigning nodes to task {pair_task.task_id}")
                for hotkey in pair:
                    node = await get_node_by_hotkey(hotkey, psql_db)
                    if node:
                        await task_sql.assign_node_to_task(pair_task.task_id, node, psql_db)

                        expected_repo_name = f"tournament-{tournament_id}-{pair_task.task_id}-{hotkey[:8]}"
                        await task_sql.set_expected_repo_name(pair_task.task_id, node, psql_db, expected_repo_name)

                        logger.info(
                            f"Assigned {hotkey} to pair task {pair_task.task_id} with expected_repo_name: {expected_repo_name}"
                        )
                    else:
                        logger.warning(f"Could not find node for hotkey {hotkey} during task assignment")


async def create_next_round(
    tournament: TournamentData, completed_round: TournamentRoundData, winners: list[str], config, psql_db: PSQLDB
):
    """Create the next round of the tournament."""
    next_round_number = completed_round.round_number + 1
    next_round_id = generate_round_id(tournament.tournament_id, next_round_number)

    with LogContext(tournament_id=tournament.tournament_id, round_id=next_round_id):
        logger.info(f"Creating next round with {len(winners)} winners: {winners}")
        next_round_is_final = len(winners) == 1

        if len(winners) == 2:
            if cst.EMISSION_BURN_HOTKEY in winners:
                next_round_is_final = True
        elif len(winners) % 2 == 1:
            if cst.EMISSION_BURN_HOTKEY not in winners:
                winners.append(cst.EMISSION_BURN_HOTKEY)
                logger.info("Added burn hotkey to make even number of participants")
            else:
                if len(winners) == 1:
                    next_round_is_final = True
                else:
                    winners = [w for w in winners if w != cst.EMISSION_BURN_HOTKEY]
                    logger.info("Removed burn hotkey to make even number of participants")

        winner_nodes = []
        for hotkey in winners:
            node = await get_node_by_hotkey(hotkey, psql_db)
            if node:
                winner_nodes.append(node)
                logger.info(f"Found node for winner {hotkey}")
            else:
                logger.warning(f"CRITICAL: Could not find node for winner {hotkey} - this winner will be excluded from next round!")

        if not winner_nodes:
            logger.error("No winner nodes found, cannot create next round")
            return

        logger.info(f"Successfully found {len(winner_nodes)} nodes out of {len(winners)} winners")

        round_structure = organise_tournament_round(winner_nodes, config)

        round_type = RoundType.KNOCKOUT if isinstance(round_structure, KnockoutRound) else RoundType.GROUP

        round_data = TournamentRoundData(
            round_id=next_round_id,
            tournament_id=tournament.tournament_id,
            round_number=next_round_number,
            round_type=round_type,
            is_final_round=next_round_is_final,
            status=RoundStatus.PENDING,
        )

        await insert_tournament_round(round_data, psql_db)

        if isinstance(round_structure, GroupRound):
            await insert_tournament_groups_with_members(next_round_id, round_structure, psql_db)
        else:
            await insert_tournament_pairs(next_round_id, round_structure.pairs, psql_db)

        logger.info(f"Created next round {next_round_id}")


async def advance_tournament(tournament: TournamentData, completed_round: TournamentRoundData, config: Config, psql_db: PSQLDB):
    with LogContext(tournament_id=tournament.tournament_id, round_id=completed_round.round_id):
        logger.info(f"Advancing tournament {tournament.tournament_id} from round {completed_round.round_id}")

        winners = await get_round_winners(completed_round, psql_db, config)
        logger.info(f"Round winners: {winners}")

        # Get all active participants and handle eliminations
        all_participants = await get_tournament_participants(tournament.tournament_id, psql_db)
        active_participants = [p.hotkey for p in all_participants if p.eliminated_in_round_id is None]
        logger.info(f"Active participants before elimination: {len(active_participants)} - {active_participants}")

        # Eliminate losers (those who didn't win)
        losers = [p for p in active_participants if p not in winners]
        logger.info(f"Losers to be eliminated: {len(losers)} - {losers}")

        # Check stake requirements for winners
        insufficient_stake_hotkeys = await get_participants_with_insufficient_stake(tournament.tournament_id, psql_db)
        logger.info(f"Participants with insufficient stake: {len(insufficient_stake_hotkeys)} - {insufficient_stake_hotkeys}")
        winners_with_insufficient_stake = [w for w in winners if w in insufficient_stake_hotkeys]

        # Combine all eliminations
        all_eliminated = losers + winners_with_insufficient_stake
        if all_eliminated:
            await eliminate_tournament_participants(tournament.tournament_id, completed_round.round_id, all_eliminated, psql_db)
            if winners_with_insufficient_stake:
                logger.info(
                    f"Eliminated {len(winners_with_insufficient_stake)} winners for insufficient stake: {winners_with_insufficient_stake}"
                )

        # Update winners list to remove those with insufficient stake
        winners = [w for w in winners if w not in winners_with_insufficient_stake]
        logger.info(f"Final winners after stake check: {len(winners)} - {winners}")

        if len(winners) == 0:
            logger.warning(
                f"No winners found for round {completed_round.round_id}. Setting base contestant as winner of the tournament."
            )
            winner = cst.EMISSION_BURN_HOTKEY
            await update_tournament_winner_hotkey(tournament.tournament_id, winner, psql_db)
            await update_tournament_status(tournament.tournament_id, TournamentStatus.COMPLETED, psql_db)
            logger.info(f"Tournament {tournament.tournament_id} completed with winner: {winner}.")
            return

        if len(winners) == 1 and completed_round.is_final_round:
            round_tasks = await get_tournament_tasks(completed_round.round_id, psql_db)
            task_ids = [task.task_id for task in round_tasks]
            snyced_task_ids = await get_synced_task_ids(task_ids, psql_db)
            if len(snyced_task_ids) == 0:
                await sync_boss_round_tasks_to_general(tournament.tournament_id, completed_round, psql_db, config)
            elif len(snyced_task_ids) == len(task_ids):
                for synced_task_id in snyced_task_ids:
                    task = await task_sql.get_task(synced_task_id, psql_db)
                    if task.status == TaskStatus.SUCCESS or task.status == TaskStatus.FAILURE:
                        logger.info(f"Task {synced_task_id} finished with status {task.status}")
                    else:
                        logger.info(f"Tournament not completed yet. Synced task {synced_task_id} has status: {task.status}.")
                        return
                winner = winners[0]
                await update_tournament_winner_hotkey(tournament.tournament_id, winner, psql_db)
                await update_tournament_status(tournament.tournament_id, TournamentStatus.COMPLETED, psql_db)
                logger.info(f"Tournament {tournament.tournament_id} completed with winner: {winner}.")
                return
            else:
                logger.info(
                    f"Tournament not completed yet. Synced {len(snyced_task_ids)} tasks out of {len(task_ids)}."
                )
        else:
            await create_next_round(tournament, completed_round, winners, config, psql_db)


async def start_tournament(tournament_id: str, psql_db: PSQLDB):
    await update_tournament_status(tournament_id, TournamentStatus.ACTIVE, psql_db)
    logger.info(f"Started tournament {tournament_id}")


async def complete_tournament(tournament_id: str, psql_db: PSQLDB):
    await update_tournament_status(tournament_id, TournamentStatus.COMPLETED, psql_db)
    logger.info(f"Completed tournament {tournament_id}")


async def create_basic_tournament(tournament_type: TournamentType, psql_db: PSQLDB, config: Config) -> str:
    """Create a basic tournament in the database without participants or rounds."""
    tournament_id = generate_tournament_id()

    base_contestant = await get_base_contestant(psql_db, tournament_type, config)
    base_winner_hotkey = base_contestant.hotkey if base_contestant else None

    logger.info(f"Base winner hotkey: {base_winner_hotkey}")

    tournament_data = TournamentData(
        tournament_id=tournament_id,
        tournament_type=tournament_type,
        status=TournamentStatus.PENDING,
        base_winner_hotkey=base_winner_hotkey,
    )

    await create_tournament(tournament_data, psql_db)

    if base_winner_hotkey:
        base_participant = TournamentParticipant(
            tournament_id=tournament_id,
            hotkey=cst.EMISSION_BURN_HOTKEY,
            training_repo=base_contestant.training_repo,
            training_commit_hash=base_contestant.training_commit_hash,
            stake_required=0,
        )
        await add_tournament_participants([base_participant], psql_db)

    logger.info(f"Created basic tournament {tournament_id} with type {tournament_type.value}")

    return tournament_id


async def populate_tournament_participants(tournament_id: str, config: Config, psql_db: PSQLDB) -> int:
    logger.info(
        f"Populating participants for tournament {tournament_id} with minimum requirement of {cst.MIN_MINERS_FOR_TOURN} miners"
    )

    tournament = await get_tournament(tournament_id, psql_db)
    if not tournament:
        logger.error(f"Tournament {tournament_id} not found")
        return 0

    while True:
        all_nodes = await get_all_nodes(psql_db)

        # Get all nodes except base contestant
        eligible_nodes = [node for node in all_nodes if node.hotkey != cst.EMISSION_BURN_HOTKEY]

        if not eligible_nodes:
            logger.warning("No eligible nodes found for tournament")
            return 0

        logger.info(f"Found {len(eligible_nodes)} eligible nodes in database")

        # Ping all nodes to get responders
        responding_nodes = []
        batch_size = t_cst.TOURNAMENT_PARTICIPANT_PING_BATCH_SIZE

        for i in range(0, len(eligible_nodes), batch_size):
            batch = eligible_nodes[i : i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1}/{(len(eligible_nodes) + batch_size - 1) // batch_size} with {len(batch)} nodes"
            )

            batch_results = await asyncio.gather(
                *[_get_miner_training_repo(node, config, tournament.tournament_type) for node in batch],
                return_exceptions=True,
            )

            for node, result in zip(batch, batch_results):
                with LogContext(node_hotkey=node.hotkey):
                    if isinstance(result, Exception):
                        logger.warning(f"Exception pinging {node.hotkey}: {result}")
                    elif result:
                        completed_entries = await count_completed_tournament_entries(node.hotkey, psql_db)
                        boosted_stake = calculate_boosted_stake(node.alpha_stake, completed_entries)

                        responding_node = RespondingNode(
                            node=node, training_repo_response=result, boosted_stake=boosted_stake, actual_stake=node.alpha_stake
                        )
                        responding_nodes.append(responding_node)
                        logger.info(
                            f"Node responded with training repo {result.github_repo}@{result.commit_hash}, "
                            f"stake: {node.alpha_stake:.2f}, boosted: {boosted_stake:.2f}"
                        )

        logger.info(f"Got {len(responding_nodes)} responding nodes")

        # Sort by boosted stake (descending) and take top N
        responding_nodes.sort(key=lambda x: x.boosted_stake, reverse=True)
        selected_nodes = responding_nodes[: cst.TOURNAMENT_TOP_N_BY_STAKE]

        logger.info(f"Selected top {len(selected_nodes)} responders by boosted stake")

        # Add selected participants to tournament
        miners_that_accept_and_give_repos = 0
        for responding_node in selected_nodes:
            participant = TournamentParticipant(
                tournament_id=tournament_id, hotkey=responding_node.node.hotkey, stake_required=responding_node.actual_stake
            )
            await add_tournament_participants([participant], psql_db)

            await update_tournament_participant_training_repo(
                tournament_id,
                responding_node.node.hotkey,
                responding_node.training_repo_response.github_repo,
                responding_node.training_repo_response.commit_hash,
                psql_db,
            )

            miners_that_accept_and_give_repos += 1

        logger.info(f"Successfully populated {miners_that_accept_and_give_repos} participants for tournament {tournament_id}")

        if miners_that_accept_and_give_repos >= cst.MIN_MINERS_FOR_TOURN:
            logger.info(
                f"Tournament {tournament_id} has sufficient miners ({miners_that_accept_and_give_repos} >= {cst.MIN_MINERS_FOR_TOURN})"
            )
            return miners_that_accept_and_give_repos

        logger.warning(
            f"Tournament {tournament_id} only has {miners_that_accept_and_give_repos} miners that accept and give repos, need at least {cst.MIN_MINERS_FOR_TOURN}. Waiting 30 minutes and retrying..."
        )
        await asyncio.sleep(30 * 60)


async def _get_miner_training_repo(node: Node, config: Config, tournament_type: TournamentType) -> TrainingRepoResponse | None:
    """Get training repo from a miner, similar to how submissions are fetched in the main validator cycle."""
    try:
        url = f"{cst.TRAINING_REPO_ENDPOINT}/{tournament_type.value}"
        response = await process_non_stream_fiber_get(url, config, node)

        if response and isinstance(response, dict):
            return TrainingRepoResponse(**response)
        else:
            logger.warning(f"Invalid response format from {node.hotkey}: {response}")
            return None

    except Exception as e:
        logger.error(f"Failed to get training repo from {node.hotkey}: {e}")
        return None


async def create_first_round_for_active_tournament(tournament_id: str, config: Config, psql_db: PSQLDB) -> bool:
    logger.info(f"Checking if tournament {tournament_id} needs first round creation")

    existing_rounds = await get_tournament_rounds(tournament_id, psql_db)
    if existing_rounds:
        logger.info(f"Tournament {tournament_id} already has {len(existing_rounds)} rounds")
        return False

    tournament = await get_tournament(tournament_id, psql_db)
    if not tournament:
        logger.error(f"Tournament {tournament_id} not found")
        return False

    participants = await get_tournament_participants(tournament_id, psql_db)
    if not participants:
        logger.error(f"No participants found for tournament {tournament_id}")
        return False

    participant_nodes = []
    for participant in participants:
        if participant.hotkey == cst.EMISSION_BURN_HOTKEY:
            continue

        node = await get_node_by_hotkey(participant.hotkey, psql_db)
        if node:
            participant_nodes.append(node)

    if not participant_nodes:
        logger.error(f"No valid nodes found for tournament {tournament_id} participants")
        return False

    logger.info(f"Creating first round for tournament {tournament_id} with {len(participant_nodes)} participants")

    await _create_first_round(tournament_id, tournament.tournament_type, participant_nodes, psql_db, config)

    logger.info(f"Successfully created first round for tournament {tournament_id}")
    return True


async def process_pending_tournaments(config: Config) -> list[str]:
    """
    Process all pending tournaments by populating participants and activating them.
    """
    while True:
        logger.info("Processing pending tournaments...")

        try:
            pending_tournaments = await get_tournaments_with_status(TournamentStatus.PENDING, config.psql_db)

            logger.info(f"Found {len(pending_tournaments)} pending tournaments")

            activated_tournaments = []

            for tournament in pending_tournaments:
                with LogContext(tournament_id=tournament.tournament_id):
                    logger.info(f"Processing pending tournament {tournament.tournament_id}")

                    num_participants = await populate_tournament_participants(tournament.tournament_id, config, config.psql_db)

                    if num_participants > 0:
                        await update_tournament_status(tournament.tournament_id, TournamentStatus.ACTIVE, config.psql_db)
                        activated_tournaments.append(tournament.tournament_id)
                        logger.info(f"Activated tournament {tournament.tournament_id} with {num_participants} participants")
                    else:
                        logger.warning(f"Tournament {tournament.tournament_id} has no participants, skipping activation")

            logger.info(f"Activated tournaments: {activated_tournaments}")
        except Exception as e:
            logger.error(f"Error processing pending tournaments: {e}")
        finally:
            await asyncio.sleep(t_cst.TOURNAMENT_PENDING_CYCLE_INTERVAL)


async def process_pending_rounds(config: Config):
    """
    Process all pending rounds by creating tasks and assigning nodes to them.
    """
    logger.info("Processing pending rounds...")

    while True:
        try:
            pending_rounds = await get_tournament_rounds_with_status(RoundStatus.PENDING, config.psql_db)

            logger.info(f"Found {len(pending_rounds)} pending rounds")

            for round_data in pending_rounds:
                with LogContext(tournament_id=round_data.tournament_id, round_id=round_data.round_id):
                    logger.info(f"Processing pending round {round_data.round_id} (type: {round_data.round_type})")

                    try:
                        tournament = await get_tournament(round_data.tournament_id, config.psql_db)
                        logger.info(f"Found tournament {tournament.tournament_id} with status {tournament.status}")

                        if round_data.round_type == RoundType.GROUP:
                            logger.info("Processing GROUP round")
                            groups_data = await get_tournament_groups(round_data.round_id, config.psql_db)
                            logger.info(f"Found {len(groups_data)} groups")
                            groups = []
                            for group_data in groups_data:
                                members = await get_tournament_group_members(group_data.group_id, config.psql_db)
                                member_ids = [member.hotkey for member in members]
                                groups.append(Group(member_ids=member_ids))
                                logger.info(f"Group {group_data.group_id}: {len(member_ids)} members")
                            round_structure = GroupRound(groups=groups)
                        else:
                            logger.info("Processing KNOCKOUT round")
                            pairs = await get_tournament_pairs(round_data.round_id, config.psql_db)
                            logger.info(f"Found {len(pairs)} pairs: {[(p.hotkey1, p.hotkey2) for p in pairs]}")
                            round_structure = KnockoutRound(pairs=[(pair.hotkey1, pair.hotkey2) for pair in pairs])

                        logger.info(f"About to create tournament tasks for round {round_data.round_id}")
                        tasks = await _create_tournament_tasks(
                            round_data.tournament_id,
                            round_data.round_id,
                            round_structure,
                            tournament.tournament_type,
                            round_data.is_final_round,
                            config,
                        )
                        logger.info(f"Created {len(tasks)} tasks for round {round_data.round_id}")

                        logger.info("About to assign nodes to tournament tasks")
                        await assign_nodes_to_tournament_tasks(
                            round_data.tournament_id, round_data.round_id, round_structure, config.psql_db
                        )
                        logger.info("Finished assigning nodes to tournament tasks")

                        logger.info(f"Setting round {round_data.round_id} to ACTIVE status")
                        await update_round_status(round_data.round_id, RoundStatus.ACTIVE, config.psql_db)

                        logger.info(f"Successfully processed pending round {round_data.round_id} with {len(tasks)} tasks")

                    except Exception as e:
                        logger.error(f"Error processing pending round {round_data.round_id}: {e}")

        except Exception as e:
            logger.error(f"Error processing pending rounds: {e}")
        finally:
            await asyncio.sleep(t_cst.TOURNAMENT_PENDING_ROUND_CYCLE_INTERVAL)


async def process_active_tournaments(config: Config):
    """
    Process all active tournaments by advancing them if needed.
    """
    logger.info("Processing active tournaments...")

    while True:
        try:
            active_tournaments = await get_tournaments_with_status(TournamentStatus.ACTIVE, config.psql_db)
            for tournament in active_tournaments:
                with LogContext(tournament_id=tournament.tournament_id):
                    logger.info(f"Processing active tournament {tournament.tournament_id}")
                    rounds = await get_tournament_rounds(tournament.tournament_id, config.psql_db)
                    if not rounds:
                        logger.info(f"Tournament {tournament.tournament_id} has no rounds, creating first round...")
                        await create_first_round_for_active_tournament(tournament.tournament_id, config, config.psql_db)
                    else:
                        current_round = rounds[-1]

                        if current_round.status == RoundStatus.ACTIVE:
                            if await check_if_round_is_completed(current_round, config):
                                await update_round_status(current_round.round_id, RoundStatus.COMPLETED, config.psql_db)
                                logger.info(
                                    f"Tournament {tournament.tournament_id} round {current_round.round_id} is completed, advancing..."
                                )
                                await advance_tournament(tournament, current_round, config, config.psql_db)
        except Exception as e:
            logger.error(f"Error processing active tournaments: {e}", exc_info=True)
        finally:
            await asyncio.sleep(t_cst.TOURNAMENT_ACTIVE_CYCLE_INTERVAL)


async def check_if_round_is_completed(round_data: TournamentRoundData, config: Config) -> bool:
    """Check if a round should be marked as completed based on task completion."""
    logger.info(f"Checking if round {round_data.round_id} should be completed...")

    round_tasks = await get_tournament_tasks(round_data.round_id, config.psql_db)

    if not round_tasks:
        logger.info(f"No tasks found for round {round_data.round_id}")
        return False

    all_tasks_completed = True
    for task in round_tasks:
        task_obj = await task_sql.get_task(task.task_id, config.psql_db)
        if task_obj and task_obj.status not in [TaskStatus.SUCCESS.value, TaskStatus.FAILURE.value]:
            all_tasks_completed = False
            logger.info(f"Task {task.task_id} not completed yet (status: {task_obj.status})")
            break

    waiting_for_synced_tasks = False
    if not all_tasks_completed:
        logger.info(f"Round {round_data.round_id} not ready for completion yet")
        return False
    else:
        # For final rounds, we need to check ALL synced tasks, not just failed ones
        if round_data.is_final_round:
            task_ids = [task.task_id for task in round_tasks]
            synced_task_ids = await get_synced_task_ids(task_ids, config.psql_db)
            
            if synced_task_ids:
                logger.info(f"Final round has {len(synced_task_ids)} synced tasks, checking their status...")
                for synced_task_id in synced_task_ids:
                    synced_task_obj = await task_sql.get_task(synced_task_id, config.psql_db)
                    if synced_task_obj:
                        if synced_task_obj.status not in [TaskStatus.SUCCESS.value, TaskStatus.FAILURE.value]:
                            logger.info(f"Synced task {synced_task_id} not completed yet (status: {synced_task_obj.status})")
                            waiting_for_synced_tasks = True
                            break
                        else:
                            logger.info(f"Synced task {synced_task_id} completed with status: {synced_task_obj.status}")
                
                if not waiting_for_synced_tasks:
                    logger.info(f"All synced tasks for final round {round_data.round_id} are completed")
                else:
                    logger.info(f"Final round {round_data.round_id} waiting for synced tasks to complete")
                    return False
        
        # Check for failed tasks that need syncing (for all rounds)
        for task in round_tasks:
            synced_task_id = await get_synced_task_id(task.task_id, config.psql_db)
            if synced_task_id:
                synced_task_obj = await task_sql.get_task(synced_task_id, config.psql_db)
                if synced_task_obj:
                    if synced_task_obj.status == TaskStatus.SUCCESS.value:
                        logger.info(f"Synced task {synced_task_id} completed successfully")
                        continue
                    elif synced_task_obj.status == TaskStatus.FAILURE.value:
                        original_task_obj = await task_sql.get_task(task.task_id, config.psql_db)
                        if original_task_obj.status == TaskStatus.SUCCESS.value:
                            logger.info(f"Synced task {synced_task_id} failed. Original task was successful. Ignoring...")
                            continue
                        else:
                            logger.info(f"Synced task {synced_task_id} failed. Original task  also failed. Replacing...")

                        new_task_id = await replace_tournament_task(
                            original_task_id=task.task_id,
                            tournament_id=round_data.tournament_id,
                            round_id=round_data.round_id,
                            group_id=task.group_id,
                            pair_id=task.pair_id,
                            config=config,
                        )
                        logger.info(f"Successfully replaced task {task.task_id} with {new_task_id}")
                        waiting_for_synced_tasks = True
                    else:
                        logger.info(f"Synced task {synced_task_id} not completed yet (status: {synced_task_obj.status})")
                        waiting_for_synced_tasks = True
            else:
                task_obj = await task_sql.get_task(task.task_id, config.psql_db)
                if task_obj and task_obj.status == TaskStatus.FAILURE.value:
                    logger.info(f"Task {task.task_id} failed, copying to main cycle to check.")
                    await _copy_task_to_general(task.task_id, config.psql_db)
                    waiting_for_synced_tasks = True

    if waiting_for_synced_tasks:
        logger.info(f"Waiting for synced tasks to complete in round {round_data.round_id}")
        return False
    else:
        logger.info(f"All tasks in round {round_data.round_id} are completed, marking round as completed")
        return True
