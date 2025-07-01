import asyncio
import math
import random

from fiber.chain.models import Node

from core.models.tournament_models import Group
from core.models.tournament_models import GroupRound
from core.models.tournament_models import KnockoutRound
from core.models.tournament_models import Round
from core.models.tournament_models import RoundStatus
from core.models.tournament_models import RoundType
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentParticipant
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentStatus
from core.models.tournament_models import TournamentTask
from core.models.tournament_models import TournamentType
from core.models.tournament_models import generate_round_id
from core.models.tournament_models import generate_tournament_id
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.db.database import PSQLDB
from validator.db.sql.nodes import get_all_nodes
from validator.db.sql.nodes import get_node_by_hotkey
from validator.db.sql.tasks import assign_node_to_task
from validator.db.sql.tasks import get_task
from validator.db.sql.tasks import get_tasks_with_status
from validator.db.sql.tasks import update_task_status
from validator.db.sql.tournaments import add_tournament_participants
from validator.db.sql.tournaments import add_tournament_tasks
from validator.db.sql.tournaments import create_tournament
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import get_tournament_participants
from validator.db.sql.tournaments import get_tournament_rounds
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
from validator.tournament.task_creator import create_image_tournament_round
from validator.tournament.task_creator import create_text_tournament_round
from validator.tournament.utils import get_base_contestant
from validator.tournament.utils import get_round_winners
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def organise_tournament_round(nodes: list[Node], config: Config) -> Round:
    nodes_copy = nodes.copy()
    random.shuffle(nodes_copy)

    if len(nodes_copy) <= t_cst.MAX_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND:
        hotkeys = [node.hotkey for node in nodes_copy]

        if len(hotkeys) % 2 == 1:
            if config.tournament_base_contestant_hotkey not in hotkeys:
                hotkeys.append(config.tournament_base_contestant_hotkey)
            else:
                hotkeys.remove(config.tournament_base_contestant_hotkey)

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

    tasks = await _create_tournament_tasks(tournament_id, round_id, round_structure, tournament_type, False, config)
    await add_tournament_tasks(tasks, psql_db)

    await assign_nodes_to_tournament_tasks(round_id, round_structure, psql_db)

    await update_round_status(round_id, RoundStatus.ACTIVE, psql_db)

    logger.info(f"Created first round {round_id} with {len(tasks)} tasks")


async def _create_tournament_tasks(
    tournament_id: str, round_id: str, round_structure: Round, tournament_type: TournamentType, is_final: bool, config: Config
) -> list[TournamentTask]:
    if tournament_type == TournamentType.TEXT:
        tournament_round = await create_text_tournament_round(round_structure, config, is_final)
    else:
        tournament_round = await create_image_tournament_round(round_structure, config)

    tasks = []
    if isinstance(round_structure, GroupRound):
        group_task_count = t_cst.TEXT_TASKS_PER_GROUP if tournament_type == TournamentType.TEXT else t_cst.IMAGE_TASKS_PER_GROUP

        for i, group in enumerate(round_structure.groups):
            group_id = f"{round_id}_group_{i + 1:03d}"

            for j in range(group_task_count):
                task_index = i * group_task_count + j
                if task_index < len(tournament_round.tasks):
                    task = TournamentTask(
                        tournament_id=tournament_id,
                        round_id=round_id,
                        task_id=tournament_round.tasks[task_index],
                        group_id=group_id,
                        pair_id=None,
                    )
                    tasks.append(task)
    else:
        if is_final:
            for i in range(len(tournament_round.tasks)):
                pair_id = f"{round_id}_pair_{1:03d}"
                task = TournamentTask(
                    tournament_id=tournament_id,
                    round_id=round_id,
                    task_id=tournament_round.tasks[i],
                    group_id=None,
                    pair_id=pair_id,
                )
                tasks.append(task)
        else:
            for i in range(len(round_structure.pairs)):
                pair_id = f"{round_id}_pair_{i + 1:03d}"
                if i < len(tournament_round.tasks):
                    task = TournamentTask(
                        tournament_id=tournament_id,
                        round_id=round_id,
                        task_id=tournament_round.tasks[i],
                        group_id=None,
                        pair_id=pair_id,
                    )
                    tasks.append(task)

    return tasks


async def assign_nodes_to_tournament_tasks(round_id: str, round_structure: Round, psql_db: PSQLDB) -> None:
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
                        await assign_node_to_task(task.task_id, node, psql_db)
                        logger.info(f"Assigned {hotkey} to group task {task.task_id}")
    else:
        round_tasks = await get_tournament_tasks(round_id, psql_db)

        for i, pair in enumerate(round_structure.pairs):
            pair_id = f"{round_id}_pair_{i + 1:03d}"

            pair_tasks = [task for task in round_tasks if task.pair_id == pair_id]

            for pair_task in pair_tasks:
                for hotkey in pair:
                    node = await get_node_by_hotkey(hotkey, psql_db)
                    if node:
                        await assign_node_to_task(pair_task.task_id, node, psql_db)
                        logger.info(f"Assigned {hotkey} to pair task {pair_task.task_id}")


async def create_next_round(
    tournament: TournamentData, completed_round: TournamentRoundData, winners: list[str], config, psql_db: PSQLDB
):
    """Create the next round of the tournament."""
    next_round_number = completed_round.round_number + 1
    next_round_id = generate_round_id(tournament.tournament_id, next_round_number)

    next_round_is_final = len(winners) == 1

    if len(winners) == 2:
        if config.tournament_base_contestant_hotkey in winners:
            next_round_is_final = True
    elif len(winners) % 2 == 1:
        if config.tournament_base_contestant_hotkey not in winners:
            winners.append(config.tournament_base_contestant_hotkey)
        else:
            if len(winners) == 1:
                next_round_is_final = True
            else:
                winners = [w for w in winners if w != config.tournament_base_contestant_hotkey]

    winner_nodes = []
    for hotkey in winners:
        node = await get_node_by_hotkey(hotkey, psql_db)
        if node:
            winner_nodes.append(node)

    if not winner_nodes:
        logger.error("No winner nodes found, cannot create next round")
        return

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

    tasks = await _create_tournament_tasks(
        tournament.tournament_id, next_round_id, round_structure, tournament.tournament_type, next_round_is_final, config
    )
    await add_tournament_tasks(tasks, psql_db)

    await assign_nodes_to_tournament_tasks(next_round_id, round_structure, psql_db)

    await update_round_status(next_round_id, RoundStatus.ACTIVE, psql_db)

    logger.info(f"Created next round {next_round_id} with {len(tasks)} tasks")


async def advance_tournament(tournament: TournamentData, completed_round: TournamentRoundData, config, psql_db: PSQLDB):
    logger.info(f"Advancing tournament {tournament.tournament_id} from round {completed_round.round_id}")

    winners = await get_round_winners(completed_round, psql_db, config)
    logger.info(f"Round winners: {winners}")

    if len(winners) == 1 and completed_round.is_final_round:
        await update_tournament_status(tournament.tournament_id, TournamentStatus.COMPLETED, psql_db)
        winner = winners[0]
        await update_tournament_winner_hotkey(tournament.tournament_id, winner, psql_db)
        logger.info(f"Tournament {tournament.tournament_id} completed with winner: {winner}")
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
            hotkey=config.tournament_base_contestant_hotkey,
            training_repo=base_contestant.training_repo,
            training_commit_hash=base_contestant.training_commit_hash,
        )
        await add_tournament_participants([base_participant], psql_db)

    logger.info(f"Created basic tournament {tournament_id} with type {tournament_type.value}")

    return tournament_id


async def populate_tournament_participants(tournament_id: str, config: Config, psql_db: PSQLDB, max_nodes: int = 4) -> int:
    logger.info(f"Populating participants for tournament {tournament_id}")

    all_nodes = await get_all_nodes(psql_db)

    eligible_nodes = [node for node in all_nodes if node.hotkey != config.tournament_base_contestant_hotkey]

    if not eligible_nodes:
        logger.warning("No eligible nodes found for tournament")
        return 0

    logger.info(f"Found {len(eligible_nodes)} eligible nodes in database")
    eligible_nodes = eligible_nodes[:max_nodes]

    successful_participants = 0

    for node in eligible_nodes:
        try:
            # Mock the training repo fetch for now
            if successful_participants < max_nodes:  # Limit participants for testing
                mock_repo = "https://github.com/rayonlabs/G.O.D"
                mock_commit = "9d14a63e4d1f065a203f51b19c2f6066933dd3a5"

                # Add participant to tournament
                participant = TournamentParticipant(tournament_id=tournament_id, hotkey=node.hotkey)
                await add_tournament_participants([participant], psql_db)

                await update_tournament_participant_training_repo(tournament_id, node.hotkey, mock_repo, mock_commit, psql_db)

                successful_participants += 1
                logger.info(f"Added participant {node.hotkey} with training repo {mock_repo}@{mock_commit}")
            else:
                logger.info(f"Skipping {node.hotkey} - tournament participant limit reached")

        except Exception as e:
            logger.warning(f"Failed to fetch training repo from {node.hotkey}: {str(e)}")
            continue

    logger.info(f"Successfully populated {successful_participants} participants for tournament {tournament_id}")
    return successful_participants


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
        if participant.hotkey == config.tournament_base_contestant_hotkey:
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


async def process_active_tournaments(config: Config):
    """
    Process all active tournaments by advancing them if needed.
    """
    logger.info("Processing active tournaments...")

    while True:
        try:
            active_tournaments = await get_tournaments_with_status(TournamentStatus.ACTIVE, config.psql_db)
            for tournament in active_tournaments:
                logger.info(f"Processing active tournament {tournament.tournament_id}")
                rounds = await get_tournament_rounds(tournament.tournament_id, config.psql_db)
                if not rounds:
                    logger.info(f"Tournament {tournament.tournament_id} has no rounds, creating first round...")
                    await create_first_round_for_active_tournament(tournament.tournament_id, config, config.psql_db)
                else:
                    current_round = rounds[-1]

                    if current_round.status in [RoundStatus.PENDING, RoundStatus.ACTIVE]:
                        if await check_if_round_is_completed(current_round, config.psql_db):
                            await update_round_status(current_round.round_id, RoundStatus.COMPLETED, config.psql_db)
                            logger.info(
                                f"Tournament {tournament.tournament_id} round {current_round.round_id} is completed, advancing..."
                            )
                            await advance_tournament(tournament, current_round, config, config.psql_db)
        except Exception as e:
            logger.error(f"Error processing active tournaments: {e}")
        finally:
            await asyncio.sleep(t_cst.TOURNAMENT_ACTIVE_CYCLE_INTERVAL)


async def check_if_round_is_completed(round_data, psql_db: PSQLDB):
    """Check if a round should be marked as completed based on task completion."""
    logger.info(f"Checking if round {round_data.round_id} should be completed...")

    round_tasks = await get_tournament_tasks(round_data.round_id, psql_db)

    if not round_tasks:
        logger.info(f"No tasks found for round {round_data.round_id}")
        return False

    all_tasks_completed = True
    for task in round_tasks:
        task_obj = await get_task(task.task_id, psql_db)
        if task_obj and task_obj.status != TaskStatus.SUCCESS.value:
            all_tasks_completed = False
            logger.info(f"Task {task.task_id} not completed yet (status: {task_obj.status})")
            break

    if all_tasks_completed:
        logger.info(f"All tasks in round {round_data.round_id} are completed, marking round as completed")
        return True
    else:
        logger.info(f"Round {round_data.round_id} not ready for completion yet")
        return False


async def process_prepped_tournament_tasks(config: Config):
    while True:
        logger.info("Processing prepped tournament tasks...")

        try:
            prepped_tasks = await get_tasks_with_status(TaskStatus.LOOKING_FOR_NODES, config.psql_db, tournament_filter="only")

            for task in prepped_tasks:
                await update_task_status(task.task_id, TaskStatus.READY, config.psql_db)
                logger.info(f"Set task {task.task_id} status to ready")
        except Exception as e:
            logger.error(f"Error processing prepped tournament tasks: {e}")
        finally:
            await asyncio.sleep(t_cst.TOURNAMENT_PREP_TASK_CYCLE_INTERVAL)
