import asyncio
import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone

from dotenv import load_dotenv

from core.models.tournament_models import NodeWeightsResult
from core.models.tournament_models import TournamentAuditData
from core.models.tournament_models import TournamentBurnData
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentResults
from core.models.tournament_models import TournamentResultsWithWinners
from core.models.tournament_models import TournamentType
from core.models.utility_models import TaskType
from validator.db.sql.auditing import store_latest_scores_url
from validator.db.sql.tournament_performance import get_boss_round_synthetic_task_completion
from validator.db.sql.tournaments import count_champion_consecutive_wins
from validator.db.sql.tournaments import get_active_tournament_participants
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament_full_results
from validator.db.sql.tournaments import get_weekly_task_participation_data
from validator.evaluation.tournament_scoring import get_tournament_weights_from_data
from validator.tournament.performance_calculator import calculate_performance_difference


load_dotenv(os.getenv("ENV_FILE", ".vali.env"))

import json
from uuid import UUID

from fiber.chain import fetch_nodes
from fiber.chain import weights
from fiber.chain.chain_utils import query_substrate
from fiber.chain.models import Node
from substrateinterface import SubstrateInterface

import validator.core.constants as cts
from core import constants as ccst
from core.constants import BUCKET_NAME
from validator.core.config import Config
from validator.core.config import load_config
from validator.core.models import TaskResults
from validator.db.sql.nodes import get_vali_node_id
from validator.evaluation.scoring import get_period_scores_from_results
from validator.utils.logging import get_logger
from validator.utils.util import save_json_to_temp_file
from validator.utils.util import try_db_connections
from validator.utils.util import upload_file_to_minio


logger = get_logger(__name__)


TIME_PER_BLOCK: int = 500


def get_organic_proportion(task_results: list[TaskResults], task_types: TaskType | set[TaskType], days: int) -> float:
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

    if isinstance(task_types, set):
        type_set = task_types
    else:
        type_set = {task_types}

    specific_type_tasks = [i for i in task_results if i.task.created_at > cutoff_date and i.task.task_type in type_set]

    organic_count = sum(1 for task in specific_type_tasks if task.task.is_organic)
    total_count = len(specific_type_tasks)

    logger.info(f"The total count is {total_count} with organic_count {organic_count} for types {type_set}")

    organic_proportion = organic_count / total_count if total_count > 0 else 0.0
    logger.info(f"THE ORGANIC PROPORTION RIGHT NOW IS {organic_proportion}")
    return organic_proportion


def detect_suspicious_nodes(task_results: list[TaskResults], task_types: TaskType | set[TaskType], days: int = 7) -> set[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    if isinstance(task_types, set):
        type_set = task_types
    else:
        type_set = {task_types}

    period_tasks_organic = [
        task
        for task in task_results
        if task.task.task_type in type_set and task.task.is_organic and task.task.created_at > cutoff
    ]

    period_tasks_synth = [
        task
        for task in task_results
        if task.task.task_type in type_set and not task.task.is_organic and task.task.created_at > cutoff
    ]

    # Get scores for comparison
    organic_scores = get_period_scores_from_results(
        period_tasks_organic,
        weight_multiplier=1.0,  # Temporary multiplier just for comparison
    )

    synth_scores = get_period_scores_from_results(
        period_tasks_synth,
        weight_multiplier=1.0,  # Temporary multiplier just for comparison
    )

    # Count synth jobs per hotkey
    synth_job_counts = {}
    for task in period_tasks_synth:
        for node_score in task.node_scores:
            if node_score.hotkey not in synth_job_counts:
                synth_job_counts[node_score.hotkey] = 0
            synth_job_counts[node_score.hotkey] += 1

    suspicious_hotkeys = set()
    synth_by_hotkey = {score.hotkey: score for score in synth_scores}

    for organic_score in organic_scores:
        hotkey = organic_score.hotkey
        synth_job_count = synth_job_counts.get(hotkey, 0)

        min_required_synth_jobs = cts.MIN_SYNTH_JOBS_REQUIRED_PER_DAY * days
        if synth_job_count < min_required_synth_jobs:
            logger.info(
                f"Node {hotkey} has only {synth_job_count} synth jobs (requires {min_required_synth_jobs} for {days} days) "
                f"for {type_set} in {days}-day period - flagging as suspicious"
            )
            suspicious_hotkeys.add(hotkey)
        elif hotkey in synth_by_hotkey:
            synth_score = synth_by_hotkey[hotkey]
            if organic_score.average_score > (synth_score.average_score + 0.5 * synth_score.std_score):
                logger.info(
                    f"Node {hotkey} has a much higher organic vs synth score "
                    f"for {type_set} in {days}-day period - flagging as suspicious"
                )
                suspicious_hotkeys.add(hotkey)
        else:
            logger.info(
                f"Node {hotkey} has organic scores but no synth scores "
                f"for {task_types} in {days}-day period - flagging as suspicious"
            )
            suspicious_hotkeys.add(hotkey)

    return suspicious_hotkeys


def filter_tasks_by_period(tasks: list[TaskResults], cutoff_time: datetime) -> list[TaskResults]:
    return [task for task in tasks if task.task.created_at > cutoff_time]


def filter_tasks_by_type(tasks: list[TaskResults], task_type: TaskType, is_organic: bool | None = None) -> list[TaskResults]:
    if is_organic is None:
        return [task for task in tasks if task.task.task_type == task_type]
    return [task for task in tasks if task.task.task_type == task_type and task.task.is_organic == is_organic]


async def _upload_results_to_s3(config: Config, tournament_audit_data: TournamentAuditData) -> None:
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, UUID):
                return str(obj)
            return super().default(obj)

    upload_data = {
        "tournament_audit_data": tournament_audit_data.model_dump(),
    }

    scores_json = json.dumps(upload_data, indent=2, cls=DateTimeEncoder)

    temp_file, _ = await save_json_to_temp_file(scores_json, "latest_scores", dump_json=False)
    datetime_of_upload = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    presigned_url = await upload_file_to_minio(temp_file, BUCKET_NAME, f"latest_scores_{datetime_of_upload}.json")
    os.remove(temp_file)
    await store_latest_scores_url(presigned_url, config)
    return presigned_url


def get_miner_performance_breakdown(hotkey: str, task_results: list[TaskResults]) -> dict:
    """Get detailed performance breakdown for a specific miner"""

    task_type_configs = [
        {"type": {TaskType.INSTRUCTTEXTTASK, TaskType.CHATTASK}, "weight_key": "INSTRUCT_TEXT_TASK_SCORE_WEIGHT"},
        {"type": TaskType.DPOTASK, "weight_key": "DPO_TASK_SCORE_WEIGHT"},
        {"type": TaskType.IMAGETASK, "weight_key": "IMAGE_TASK_SCORE_WEIGHT"},
        {"type": TaskType.GRPOTASK, "weight_key": "GRPO_TASK_SCORE_WEIGHT"},
    ]

    periods = {
        "one_day": {"cutoff": datetime.now(timezone.utc) - timedelta(days=1), "weight": cts.ONE_DAY_SCORE_WEIGHT},
        "three_day": {"cutoff": datetime.now(timezone.utc) - timedelta(days=3), "weight": cts.THREE_DAY_SCORE_WEIGHT},
        "seven_day": {"cutoff": datetime.now(timezone.utc) - timedelta(days=7), "weight": cts.SEVEN_DAY_SCORE_WEIGHT},
    }

    organic_proportions = {}
    suspicious_hotkeys = {}

    for task_config in task_type_configs:
        raw_types = task_config["type"]
        task_type_list = raw_types if isinstance(raw_types, set) else [raw_types]

        task_types_key = str(sorted(task_type_list)) if len(task_type_list) > 1 else str(task_type_list[0])

        organic_proportions[task_types_key] = get_organic_proportion(
            task_results, set(task_type_list) if len(task_type_list) > 1 else task_type_list[0], days=7
        )
        suspicious_hotkeys[task_types_key] = detect_suspicious_nodes(
            task_results, set(task_type_list) if len(task_type_list) > 1 else task_type_list[0], days=7
        )

    breakdown = {"task_types": {}, "period_totals": {}, "all_scores": []}

    for task_config in task_type_configs:
        raw_types = task_config["type"]
        task_type_list = raw_types if isinstance(raw_types, set) else [raw_types]

        task_weight = getattr(cts, task_config["weight_key"])

        task_types_key = str(sorted(task_type_list)) if len(task_type_list) > 1 else str(task_type_list[0])

        organic_tasks = []
        synthetic_tasks = []
        for task_type in task_type_list:
            organic_tasks.extend(filter_tasks_by_type(task_results, task_type, is_organic=True))
            synthetic_tasks.extend(filter_tasks_by_type(task_results, task_type, is_organic=False))

        miner_organic_tasks = [tr for tr in organic_tasks if any(ns.hotkey == hotkey for ns in tr.node_scores)]
        miner_synthetic_tasks = [tr for tr in synthetic_tasks if any(ns.hotkey == hotkey for ns in tr.node_scores)]

        type_data = {
            "task_weight": task_weight,
            "organic_proportion": organic_proportions[task_types_key],
            "is_suspicious": hotkey in suspicious_hotkeys[task_types_key],
            "periods": {},
        }

        for period_name, period_config in periods.items():
            period_weight = period_config["weight"]
            cutoff = period_config["cutoff"]

            period_organic = filter_tasks_by_period(miner_organic_tasks, cutoff)
            period_synthetic = filter_tasks_by_period(miner_synthetic_tasks, cutoff)

            organic_mult = period_weight * task_weight * organic_proportions[task_types_key]
            synth_mult = period_weight * task_weight * (1 - organic_proportions[task_types_key])

            organic_scores = (
                get_period_scores_from_results(period_organic, weight_multiplier=organic_mult) if period_organic else []
            )
            synth_scores = (
                get_period_scores_from_results(period_synthetic, weight_multiplier=synth_mult) if period_synthetic else []
            )

            miner_organic_score = next((s for s in organic_scores if s.hotkey == hotkey), None)
            miner_synth_score = next((s for s in synth_scores if s.hotkey == hotkey), None)

            if miner_organic_score and hotkey in suspicious_hotkeys[task_types_key]:
                miner_organic_score.weight_multiplier = 0.0

            type_data["periods"][period_name] = {
                "organic": {
                    "score": miner_organic_score,
                    "task_count": len(period_organic),
                    "weighted_contribution": (miner_organic_score.normalised_score * miner_organic_score.weight_multiplier)
                    if miner_organic_score and miner_organic_score.normalised_score
                    else 0,
                },
                "synthetic": {
                    "score": miner_synth_score,
                    "task_count": len(period_synthetic),
                    "weighted_contribution": (miner_synth_score.normalised_score * miner_synth_score.weight_multiplier)
                    if miner_synth_score and miner_synth_score.normalised_score
                    else 0,
                },
            }

            breakdown["all_scores"].extend([s for s in [miner_organic_score, miner_synth_score] if s])

        type_data["total_organic_tasks"] = len(miner_organic_tasks)
        type_data["total_synthetic_tasks"] = len(miner_synthetic_tasks)

        breakdown["task_types"][task_types_key] = type_data

        for period_name in periods:
            total = sum(
                breakdown["task_types"][tt]["periods"][period_name]["organic"]["weighted_contribution"]
                + breakdown["task_types"][tt]["periods"][period_name]["synthetic"]["weighted_contribution"]
                for tt in breakdown["task_types"]
            )
            breakdown["period_totals"][period_name] = total

    return breakdown


async def check_boss_round_synthetic_tasks_complete(tournament_id: str, psql_db) -> bool:
    completion_data = await get_boss_round_synthetic_task_completion(tournament_id, psql_db)
    return completion_data.total_synth_tasks > 0 and completion_data.total_synth_tasks == completion_data.completed_synth_tasks


def calculate_emission_multiplier(performance_diff: float) -> float:
    if performance_diff <= cts.EMISSION_MULTIPLIER_THRESHOLD:
        return 0.0

    excess_performance = performance_diff - cts.EMISSION_MULTIPLIER_THRESHOLD
    emission_increase = excess_performance * cts.EMISSION_MULTIPLIER_RATE

    return emission_increase


async def get_tournament_burn_details(psql_db) -> TournamentBurnData:
    """
    Calculate detailed tournament burn data with calculations for TEXT and IMAGE tournaments.

    This function calculates burn proportions for TEXT and IMAGE tournaments,
    then applies them based on each hotkey's tournament participation.

    Returns:
        TournamentBurnData with performance metrics and weight distributions
    """
    logger.info("=== CALCULATING TOURNAMENT BURN DATA ===")

    text_performance_diff = None
    image_performance_diff = None

    for tournament_type in [TournamentType.TEXT, TournamentType.IMAGE]:
        logger.info(f"Processing {tournament_type} tournament type")
        performance_diff = None

        latest_tournament = await get_latest_completed_tournament(psql_db, tournament_type)
        if latest_tournament:
            logger.info(f"Found latest {tournament_type} tournament: {latest_tournament.tournament_id}")

            previous_tournament = await get_latest_completed_tournament(
                psql_db, tournament_type, exclude_tournament_id=latest_tournament.tournament_id
            )

            winner_changed = False
            if previous_tournament:
                if previous_tournament.winner_hotkey != latest_tournament.winner_hotkey:
                    winner_changed = True
                    logger.info(
                        f"[{tournament_type}] Winner changed: {previous_tournament.winner_hotkey} → {latest_tournament.winner_hotkey}"
                    )
                else:
                    logger.info(f"[{tournament_type}] Same winner defended: {latest_tournament.winner_hotkey}")
            else:
                winner_changed = True
                logger.info(f"[{tournament_type}] First tournament winner: {latest_tournament.winner_hotkey}")

            if winner_changed:
                performance_diff = await calculate_performance_difference(latest_tournament.tournament_id, psql_db)
                logger.info(f"NEW winner - calculated fresh performance difference for {tournament_type}: {performance_diff:.4f}")
            elif latest_tournament.winning_performance_difference is not None:
                performance_diff = latest_tournament.winning_performance_difference
                logger.info(f"SAME winner - using stored performance difference for {tournament_type}: {performance_diff:.4f}")
            elif (
                previous_tournament.winning_performance_difference is not None
                and latest_tournament.winning_performance_difference is None
            ):
                performance_diff = previous_tournament.winning_performance_difference
                logger.info(f"SAME winner - using stored performance difference for {tournament_type}: {performance_diff:.4f}")
            else:
                performance_diff = 0.0

        if performance_diff is None and latest_tournament:
            if latest_tournament.winner_hotkey == cts.EMISSION_BURN_HOTKEY:
                logger.info(
                    f"No performance data available for {tournament_type} tournament, burn account won - assuming worst performance (100% difference)"
                )
                performance_diff = 1.0
            else:
                logger.info(
                    f"No performance data available for {tournament_type} tournament, assuming perfect performance (0% difference)"
                )
                performance_diff = 0.0

        if tournament_type == TournamentType.TEXT:
            text_performance_diff = performance_diff
        elif tournament_type == TournamentType.IMAGE:
            image_performance_diff = performance_diff

    text_emission_increase = calculate_emission_multiplier(text_performance_diff) if text_performance_diff is not None else 0.0
    image_emission_increase = calculate_emission_multiplier(image_performance_diff) if image_performance_diff is not None else 0.0

    logger.info(
        f"Text emission increase (before decay): {text_emission_increase}, Image emission increase (before decay): {image_emission_increase}"
    )

    text_consecutive_wins = 0
    image_consecutive_wins = 0

    latest_text_tournament = await get_latest_completed_tournament(psql_db, TournamentType.TEXT)
    if latest_text_tournament and latest_text_tournament.winner_hotkey:
        winner_hotkey = latest_text_tournament.winner_hotkey
        if winner_hotkey == cts.EMISSION_BURN_HOTKEY:
            winner_hotkey = latest_text_tournament.base_winner_hotkey
        if winner_hotkey:
            text_consecutive_wins = await count_champion_consecutive_wins(psql_db, TournamentType.TEXT, winner_hotkey)
            text_decay = max(0, text_consecutive_wins - 1) * cts.EMISSION_BOOST_DECAY_PER_WIN
            text_emission_increase = max(0.0, text_emission_increase - text_decay)
            logger.info(
                f"Text winner {winner_hotkey} has {text_consecutive_wins} consecutive wins, decay: {text_decay:.4f}, adjusted boost: {text_emission_increase:.4f}"
            )

    latest_image_tournament = await get_latest_completed_tournament(psql_db, TournamentType.IMAGE)
    if latest_image_tournament and latest_image_tournament.winner_hotkey:
        winner_hotkey = latest_image_tournament.winner_hotkey
        if winner_hotkey == cts.EMISSION_BURN_HOTKEY:
            winner_hotkey = latest_image_tournament.base_winner_hotkey
        if winner_hotkey:
            image_consecutive_wins = await count_champion_consecutive_wins(psql_db, TournamentType.IMAGE, winner_hotkey)
            image_decay = max(0, image_consecutive_wins - 1) * cts.EMISSION_BOOST_DECAY_PER_WIN
            image_emission_increase = max(0.0, image_emission_increase - image_decay)
            logger.info(
                f"Image winner {winner_hotkey} has {image_consecutive_wins} consecutive wins, decay: {image_decay:.4f}, adjusted boost: {image_emission_increase:.4f}"
            )

    text_tournament_weight = min(cts.TOURNAMENT_TEXT_WEIGHT + text_emission_increase, cts.MAX_TEXT_TOURNAMENT_WEIGHT)
    image_tournament_weight = min(cts.TOURNAMENT_IMAGE_WEIGHT + image_emission_increase, cts.MAX_IMAGE_TOURNAMENT_WEIGHT)

    burn_weight = 1.0 - text_tournament_weight - image_tournament_weight

    text_burn_proportion = (cts.MAX_TEXT_TOURNAMENT_WEIGHT - text_tournament_weight) / cts.MAX_TEXT_TOURNAMENT_WEIGHT
    image_burn_proportion = (cts.MAX_IMAGE_TOURNAMENT_WEIGHT - image_tournament_weight) / cts.MAX_IMAGE_TOURNAMENT_WEIGHT

    logger.info(f"Weights - Text tournament: {text_tournament_weight}, Image tournament: {image_tournament_weight}")
    logger.info(f"Total burn weight: {burn_weight}")

    return TournamentBurnData(
        text_performance_diff=text_performance_diff,
        image_performance_diff=image_performance_diff,
        text_burn_proportion=text_burn_proportion,
        image_burn_proportion=image_burn_proportion,
        text_tournament_weight=text_tournament_weight,
        image_tournament_weight=image_tournament_weight,
        burn_weight=burn_weight,
    )


def apply_tournament_weights(
    text_tournament_weights: dict[str, float],
    image_tournament_weights: dict[str, float],
    hotkey_to_node_id: dict[str, int],
    all_node_weights: list[float],
    scaled_text_tournament_weight: float,
    scaled_image_tournament_weight: float,
    scaled_text_base_weight: float,
    scaled_image_base_weight: float,
    text_winner_hotkey: str | None,
    image_winner_hotkey: str | None,
) -> None:
    """Apply tournament weights with truly text and image weights."""
    logger.info("=== TOURNAMENT WEIGHT CALCULATIONS ===")

    logger.info(f"Processing {len(text_tournament_weights)} text tournament winners")
    for hotkey, weight in text_tournament_weights.items():
        node_id = hotkey_to_node_id.get(hotkey)
        if node_id is not None:
            if hotkey == text_winner_hotkey:
                text_contribution = weight * scaled_text_tournament_weight
            else:
                text_contribution = weight * scaled_text_base_weight
            all_node_weights[node_id] = all_node_weights[node_id] + text_contribution

            logger.info(
                f"Node ID {node_id} (hotkey: {hotkey[:8]}...): "
                f"TEXT TOURNAMENT - weight={weight:.6f}, "
                f"scaled_text_weight={scaled_text_tournament_weight if hotkey == text_winner_hotkey else scaled_text_base_weight:.6f}, "
                f"text_contribution={text_contribution:.6f}, "
                f"total_weight={all_node_weights[node_id]:.6f}"
            )

    logger.info(f"Processing {len(image_tournament_weights)} image tournament winners")
    for hotkey, weight in image_tournament_weights.items():
        node_id = hotkey_to_node_id.get(hotkey)
        if node_id is not None:
            if hotkey == image_winner_hotkey:
                image_contribution = weight * scaled_image_tournament_weight
            else:
                image_contribution = weight * scaled_image_base_weight
            all_node_weights[node_id] = all_node_weights[node_id] + image_contribution

            logger.info(
                f"Node ID {node_id} (hotkey: {hotkey[:8]}...): "
                f"IMAGE TOURNAMENT - weight={weight:.6f}, "
                f"scaled_image_weight={scaled_image_tournament_weight if hotkey == image_winner_hotkey else scaled_image_base_weight:.6f}, "
                f"image_contribution={image_contribution:.6f}, "
                f"total_weight={all_node_weights[node_id]:.6f}"
            )


async def get_node_weights_from_tournament_audit_data(
    substrate: SubstrateInterface,
    netuid: int,
    tournament_audit_data: TournamentAuditData,
) -> NodeWeightsResult:
    all_nodes: list[Node] = fetch_nodes.get_nodes_for_netuid(substrate, netuid)
    hotkey_to_node_id: dict[str, int] = {node.hotkey: node.node_id for node in all_nodes}

    all_node_ids: list[int] = [node.node_id for node in all_nodes]
    all_node_weights: list[float] = [0.0 for _ in all_nodes]

    logger.info("=== USING BURN DATA FROM AUDIT ===")

    logger.info(f"Text tournament weight: {tournament_audit_data.text_tournament_weight:.6f}")
    logger.info(f"Image tournament weight: {tournament_audit_data.image_tournament_weight:.6f}")
    logger.info(f"Total burn weight: {tournament_audit_data.burn_weight:.6f}")

    participants: list[str] = tournament_audit_data.participants
    participation_total: float = len(participants) * cts.TOURNAMENT_PARTICIPATION_WEIGHT
    scale_factor: float = 1.0 - participation_total if participation_total > 0 else 1.0

    scaled_text_tournament_weight: float = tournament_audit_data.text_tournament_weight * scale_factor
    scaled_image_tournament_weight: float = tournament_audit_data.image_tournament_weight * scale_factor
    scaled_burn_weight: float = tournament_audit_data.burn_weight * scale_factor

    scaled_text_base_weight: float = cts.TOURNAMENT_TEXT_WEIGHT * scale_factor
    scaled_image_base_weight: float = cts.TOURNAMENT_IMAGE_WEIGHT * scale_factor

    text_tournament_weights, image_tournament_weights = get_tournament_weights_from_data(
        tournament_audit_data.text_tournament_data, tournament_audit_data.image_tournament_data
    )

    text_winner_hotkey = None
    if tournament_audit_data.text_tournament_data:
        text_winner_hotkey = tournament_audit_data.text_tournament_data.winner_hotkey
        if text_winner_hotkey == cts.EMISSION_BURN_HOTKEY:
            text_winner_hotkey = tournament_audit_data.text_tournament_data.base_winner_hotkey

    image_winner_hotkey = None
    if tournament_audit_data.image_tournament_data:
        image_winner_hotkey = tournament_audit_data.image_tournament_data.winner_hotkey
        if image_winner_hotkey == cts.EMISSION_BURN_HOTKEY:
            image_winner_hotkey = tournament_audit_data.image_tournament_data.base_winner_hotkey

    apply_tournament_weights(
        text_tournament_weights,
        image_tournament_weights,
        hotkey_to_node_id,
        all_node_weights,
        scaled_text_tournament_weight,
        scaled_image_tournament_weight,
        scaled_text_base_weight,
        scaled_image_base_weight,
        text_winner_hotkey,
        image_winner_hotkey,
    )

    for hotkey in participants:
        node_id = hotkey_to_node_id.get(hotkey)
        if node_id is not None:
            all_node_weights[node_id] += cts.TOURNAMENT_PARTICIPATION_WEIGHT

    burn_node_id: int | None = hotkey_to_node_id.get(cts.EMISSION_BURN_HOTKEY)
    if burn_node_id is not None:
        all_node_weights[burn_node_id] = scaled_burn_weight

    logger.info(f"Number of non zero node weights: {sum(1 for weight in all_node_weights if weight != 0)}")

    return NodeWeightsResult(node_ids=all_node_ids, node_weights=all_node_weights)


async def build_tournament_audit_data(psql_db) -> TournamentAuditData:
    """
    Build TournamentAuditData with all necessary tournament information.

    This is the central function for gathering tournament data used by both
    the validator (for weight setting) and auditor (for verification).

    Args:
        psql_db: Database connection

    Returns:
        TournamentAuditData with all tournament information populated
    """
    tournament_audit_data = TournamentAuditData()

    # Fetch text tournament data
    text_tournament: TournamentData = await get_latest_completed_tournament(psql_db, TournamentType.TEXT)
    if text_tournament:
        tournament_results: TournamentResults = await get_tournament_full_results(text_tournament.tournament_id, psql_db)
        tournament_audit_data.text_tournament_data = TournamentResultsWithWinners(
            tournament_id=tournament_results.tournament_id,
            rounds=tournament_results.rounds,
            base_winner_hotkey=text_tournament.base_winner_hotkey,
            winner_hotkey=text_tournament.winner_hotkey,
        )

    # Fetch image tournament data
    image_tournament = await get_latest_completed_tournament(psql_db, TournamentType.IMAGE)
    if image_tournament:
        tournament_results = await get_tournament_full_results(image_tournament.tournament_id, psql_db)
        tournament_audit_data.image_tournament_data = TournamentResultsWithWinners(
            tournament_id=tournament_results.tournament_id,
            rounds=tournament_results.rounds,
            base_winner_hotkey=image_tournament.base_winner_hotkey,
            winner_hotkey=image_tournament.winner_hotkey,
        )

    # Fetch participants
    tournament_audit_data.participants = await get_active_tournament_participants(psql_db)

    # Fetch burn weights
    burn_data: TournamentBurnData = await get_tournament_burn_details(psql_db)
    tournament_audit_data.text_tournament_weight = burn_data.text_tournament_weight
    tournament_audit_data.image_tournament_weight = burn_data.image_tournament_weight
    tournament_audit_data.burn_weight = burn_data.burn_weight

    # Fetch weekly participation data
    tournament_audit_data.weekly_participation = await get_weekly_task_participation_data(psql_db)

    return tournament_audit_data


async def set_weights(config: Config, all_node_ids: list[int], all_node_weights: list[float], validator_node_id: int) -> bool:
    try:
        success = await asyncio.to_thread(
            weights.set_node_weights,
            substrate=config.substrate,
            keypair=config.keypair,
            node_ids=all_node_ids,
            node_weights=all_node_weights,
            netuid=config.netuid,
            version_key=ccst.VERSION_KEY,
            validator_node_id=int(validator_node_id),
            wait_for_inclusion=False,
            wait_for_finalization=False,
            max_attempts=3,
        )
    except Exception as e:
        logger.error(f"Failed to set weights: {e}")
        return False

    if success:
        logger.info("Weights set successfully.")

        return True
    else:
        logger.error("Failed to set weights :(")
        return False


async def _get_and_set_weights(config: Config, validator_node_id: int) -> bool:
    # Build tournament audit data using the centralized function
    tournament_audit_data: TournamentAuditData = await build_tournament_audit_data(config.psql_db)

    result = await get_node_weights_from_tournament_audit_data(config.substrate, config.netuid, tournament_audit_data)
    all_node_ids = result.node_ids
    all_node_weights = result.node_weights
    logger.info("Weights calculated, about to set...")

    success = await set_weights(config, all_node_ids, all_node_weights, validator_node_id)
    if success:
        # Upload both task results and tournament data
        url = await _upload_results_to_s3(config, tournament_audit_data)
        logger.info(f"Uploaded the scores and tournament data to s3 for auditing - url: {url}")

    return success


async def _set_metagraph_weights(config: Config) -> None:
    nodes: list[Node] = fetch_nodes.get_nodes_for_netuid(config.substrate, config.netuid)
    node_ids = [node.node_id for node in nodes]
    node_weights = [node.incentive for node in nodes]
    validator_node_id = await get_vali_node_id(config.substrate, config.keypair.ss58_address)
    if validator_node_id is None:
        raise ValueError("Validator node id not found")

    await asyncio.to_thread(
        weights.set_node_weights,
        substrate=config.substrate,
        keypair=config.keypair,
        node_ids=node_ids,
        node_weights=node_weights,
        netuid=config.netuid,
        version_key=ccst.VERSION_KEY,
        validator_node_id=int(validator_node_id),
        wait_for_inclusion=False,
        wait_for_finalization=False,
        max_attempts=3,
    )


# To improve: use activity cutoff & The epoch length to set weights at the perfect times
async def set_weights_periodically(config: Config, just_once: bool = False) -> None:
    substrate = config.substrate
    substrate, uid = query_substrate(
        substrate,
        "SubtensorModule",
        "Uids",
        [config.netuid, config.keypair.ss58_address],
        return_value=True,
    )

    if uid is None:
        raise ValueError(f"Can't find hotkey {config.keypair.ss58_address} for our keypair on netuid: {config.netuid}.")

    consecutive_failures = 0
    while True:
        substrate, current_block = query_substrate(substrate, "System", "Number", [], return_value=True)
        substrate, last_updated_value = query_substrate(
            substrate, "SubtensorModule", "LastUpdate", [config.netuid], return_value=False
        )
        updated: int = current_block - last_updated_value[uid]
        substrate, weights_set_rate_limit = query_substrate(
            substrate, "SubtensorModule", "WeightsSetRateLimit", [config.netuid], return_value=True
        )
        logger.info(
            f"My Validator Node ID: {uid}. Last updated {updated} blocks ago. Weights set rate limit: {weights_set_rate_limit}."
        )

        if updated < weights_set_rate_limit:
            logger.info("Sleeping for a bit as we set recently...")
            await asyncio.sleep((weights_set_rate_limit - updated + 1) * 12)
            continue

        if os.getenv("ENV", "prod").lower() == "dev":
            success = await _get_and_set_weights(config, uid)
        else:
            try:
                success = await _get_and_set_weights(config, uid)
            except Exception as e:
                logger.error(f"Failed to set weights with error: {e}")
                logger.exception(e)
                success = False

        if success:
            consecutive_failures = 0
            logger.info("Successfully set weights! Sleeping for 25 blocks before next check...")
            if just_once:
                return
            await asyncio.sleep(12 * 25)
            continue

        consecutive_failures += 1
        if just_once:
            logger.info("Failed to set weights, will try again...")
            await asyncio.sleep(12 * 1)
        else:
            logger.info(f"Failed to set weights {consecutive_failures} times in a row - sleeping for a bit...")
            await asyncio.sleep(12 * 25)  # Try again in 25 blocks

        if consecutive_failures == 1 or updated < 3000:
            continue

        if just_once or config.set_metagraph_weights_with_high_updated_to_not_dereg:
            logger.warning("Setting metagraph weights as our updated value is getting too high!")
            if just_once:
                logger.warning("Please exit if you do not want to do this!!!")
                await asyncio.sleep(4)
            try:
                success = await _set_metagraph_weights(config)
            except Exception as e:
                logger.error(f"Failed to set metagraph weights: {e}")
                success = False

            if just_once:
                return

            if success:
                consecutive_failures = 0
                continue


async def main():
    config = load_config()
    await try_db_connections(config)
    await set_weights_periodically(config)


if __name__ == "__main__":
    asyncio.run(main())
