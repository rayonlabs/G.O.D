"""
Calculates and schedules weights every SCORING_PERIOD
"""

import asyncio
import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone

from dotenv import load_dotenv

from core.models.tournament_models import TournamentAuditData
from core.models.tournament_models import TournamentResultsWithWinners
from core.models.tournament_models import TournamentType
from core.models.utility_models import TaskType
from validator.db.sql.auditing import store_latest_scores_url

from validator.db.sql.submissions_and_scoring import get_aggregate_scores_for_leaderboard_since
from validator.db.sql.submissions_and_scoring import get_aggregate_scores_since
from validator.db.sql.tournament_performance import get_boss_round_synthetic_task_completion
from validator.db.sql.tournament_performance import get_boss_round_winner_task_pairs
from validator.db.sql.tournament_performance import get_previous_completed_tournament
from validator.db.sql.tournament_performance import get_task_scores_as_models
from validator.db.sql.tournaments import get_active_tournament_participants
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament_full_results
from validator.evaluation.tournament_scoring import get_tournament_weights_from_data



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
from validator.core.models import PeriodScore
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


def get_period_scores_from_task_results(task_results: list[TaskResults]) -> list[PeriodScore]:
    """Process task results into period scores with appropriate filtering and weighting."""
    if not task_results:
        logger.info("There were no results to be scored")
        return []

    task_types = [
        {"type": {TaskType.INSTRUCTTEXTTASK, TaskType.CHATTASK}, "weight_key": "INSTRUCT_TEXT_TASK_SCORE_WEIGHT"},
        {"type": TaskType.DPOTASK, "weight_key": "DPO_TASK_SCORE_WEIGHT"},
        {"type": TaskType.IMAGETASK, "weight_key": "IMAGE_TASK_SCORE_WEIGHT"},
        {"type": TaskType.GRPOTASK, "weight_key": "GRPO_TASK_SCORE_WEIGHT"},
    ]

    organic_proportions = {}
    suspicious_hotkeys = {}

    for task_config in task_types:
        task_types_raw = task_config["type"]
        weight_key = task_config["weight_key"]

        task_type_list = task_types_raw if isinstance(task_types_raw, set) else [task_types_raw]

        task_types_key = str(sorted(task_type_list)) if len(task_type_list) > 1 else str(task_type_list[0])

        organic_proportions[task_types_key] = get_organic_proportion(
            task_results, set(task_type_list) if len(task_type_list) > 1 else task_type_list[0], days=7
        )

        suspicious_hotkeys[task_types_key] = detect_suspicious_nodes(
            task_results, set(task_type_list) if len(task_type_list) > 1 else task_type_list[0], days=7
        )

        logger.info(f"Found {len(suspicious_hotkeys[task_types_key])} suspicious nodes for {task_types_key}")

    filtered_tasks = {}

    for task_config in task_types:
        task_types_raw = task_config["type"]
        task_type_list = task_types_raw if isinstance(task_types_raw, set) else [task_types_raw]

        task_types_key = str(sorted(task_type_list)) if len(task_type_list) > 1 else str(task_type_list[0])

        organic_tasks = []
        synth_tasks = []
        for task_type in task_type_list:
            organic_tasks.extend(filter_tasks_by_type(task_results, task_type, is_organic=True))
            synth_tasks.extend(filter_tasks_by_type(task_results, task_type, is_organic=False))

        filtered_tasks[f"{task_types_key}_organic"] = organic_tasks
        filtered_tasks[f"{task_types_key}_synth"] = synth_tasks


    periods = {
        "one_day": {"cutoff": datetime.now(timezone.utc) - timedelta(days=1), "weight": cts.ONE_DAY_SCORE_WEIGHT},
        "three_day": {"cutoff": datetime.now(timezone.utc) - timedelta(days=3), "weight": cts.THREE_DAY_SCORE_WEIGHT},
        "seven_day": {"cutoff": datetime.now(timezone.utc) - timedelta(days=7), "weight": cts.SEVEN_DAY_SCORE_WEIGHT},
    }

    all_period_scores = []

    for period_name, period_config in periods.items():
        cutoff = period_config["cutoff"]
        period_weight = period_config["weight"]

        for task_config in task_types:
            raw_types = task_config["type"]
            task_type_list = raw_types if isinstance(raw_types, set) else [raw_types]

            weight_key = task_config["weight_key"]
            task_weight = getattr(cts, weight_key)

            task_types_key = str(sorted(task_type_list)) if len(task_type_list) > 1 else str(task_type_list[0])

            organic_proportion = organic_proportions[task_types_key]
            synth_proportion = 1 - organic_proportion

            if organic_proportion > 0:
                period_tasks_organic = filter_tasks_by_period(filtered_tasks[f"{task_types_key}_organic"], cutoff)
                scores_organic = get_period_scores_from_results(
                    period_tasks_organic, weight_multiplier=period_weight * task_weight * organic_proportion
                )

                for organic_score in scores_organic:
                    if organic_score.hotkey in suspicious_hotkeys[task_types_key]:
                        logger.info(
                            f"Setting {task_types_key} organic score to zero for suspicious node {organic_score.hotkey} in {period_name} period"
                        )
                        organic_score.weight_multiplier = 0.0

                all_period_scores.extend(scores_organic)

            if synth_proportion > 0:
                period_tasks_synth = filter_tasks_by_period(filtered_tasks[f"{task_types_key}_synth"], cutoff)
                scores_synth = get_period_scores_from_results(
                    period_tasks_synth, weight_multiplier=period_weight * task_weight * synth_proportion
                )

                all_period_scores.extend(scores_synth)

    return all_period_scores


def filter_tasks_by_period(tasks: list[TaskResults], cutoff_time: datetime) -> list[TaskResults]:
    return [task for task in tasks if task.task.created_at > cutoff_time]


def filter_tasks_by_type(tasks: list[TaskResults], task_type: TaskType, is_organic: bool | None = None) -> list[TaskResults]:
    if is_organic is None:
        return [task for task in tasks if task.task.task_type == task_type]
    return [task for task in tasks if task.task.task_type == task_type and task.task.is_organic == is_organic]


async def _get_weights_to_set(config: Config) -> tuple[list[PeriodScore], list[TaskResults]]:
    """
    Retrieve task results from the database and score multiple periods independently.
    This ensures a fairer ramp-up for new miners.

    In the future, as miners become more stable, we aim to encourage long-term stability.
    This means not giving new miners more weight than necessary, while still allowing them
    the potential to reach the top position without being deregistered.

    Period scores are calculated completely independently
    """
    date = datetime.now() - timedelta(days=cts.SCORING_WINDOW)
    task_results: list[TaskResults] = await get_aggregate_scores_since(date, config.psql_db)

    all_period_scores = get_period_scores_from_task_results(task_results)

    return all_period_scores, task_results


async def _get_leaderboard_data(config: Config) -> tuple[list[PeriodScore], list[TaskResults]]:
    """
    Retrieve task results from the database for leaderboard/analytics purposes.
    This includes ALL scores (including zeros) for accurate counting and statistics.
    This is separate from _get_weights_to_set which filters for weight calculations.
    """
    date = datetime.now() - timedelta(days=cts.SCORING_WINDOW)
    task_results: list[TaskResults] = await get_aggregate_scores_for_leaderboard_since(date, config.psql_db)

    all_period_scores = get_period_scores_from_task_results(task_results)

    return all_period_scores, task_results

async def _upload_results_to_s3(
    config: Config, task_results: list[TaskResults], tournament_audit_data: TournamentAuditData
) -> None:
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, UUID):
                return str(obj)
            return super().default(obj)

    upload_data = {
        "task_results": [result.model_dump() for result in task_results],
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


async def calculate_performance_difference(tournament_id: str, psql_db) -> float:
    logger.info(f"=== CALCULATING PERFORMANCE DIFFERENCE FOR TOURNAMENT {tournament_id} ===")
    task_pairs = await get_boss_round_winner_task_pairs(tournament_id, psql_db)
    logger.info(f"Found {len(task_pairs)} task pairs for performance comparison")

    if not task_pairs:
        logger.info("No task pairs found, returning 0.0 performance difference")
        return 0.0

    performance_differences = []

    for i, task_pair in enumerate(task_pairs):
        logger.info(f"Processing task pair {i+1}/{len(task_pairs)}: tournament={task_pair.tournament_task_id}, synthetic={task_pair.synthetic_task_id}, winner={task_pair.winner_hotkey}")
        
        tournament_scores = await get_task_scores_as_models(task_pair.tournament_task_id, psql_db)
        synthetic_scores = await get_task_scores_as_models(task_pair.synthetic_task_id, psql_db)
        logger.info(f"Found {len(tournament_scores)} tournament scores and {len(synthetic_scores)} synthetic scores")

        winner_tournament_score = None
        best_synthetic_score = None

        # Get the winner's score from the tournament task
        for score in tournament_scores:
            if score.hotkey == task_pair.winner_hotkey:
                winner_tournament_score = max(score.test_loss, score.synth_loss)
                logger.info(f"Winner tournament score for {task_pair.winner_hotkey}: {winner_tournament_score}")
                break

        # Get the best score from the synthetic task (from other miners)
        if synthetic_scores:
            # For lower-is-better metrics (loss), we want the minimum
            # For higher-is-better metrics (GRPO), we want the maximum
            task_type = TaskType(task_pair.task_type)
            
            if task_type == TaskType.GRPOTASK:
                # GRPO: higher is better
                best_synthetic_score = max(max(score.test_loss, score.synth_loss) for score in synthetic_scores)
                logger.info(f"Best synthetic score (GRPO - higher is better): {best_synthetic_score}")
            else:
                # Other tasks: lower is better
                best_synthetic_score = min(max(score.test_loss, score.synth_loss) for score in synthetic_scores)
                logger.info(f"Best synthetic score (lower is better): {best_synthetic_score}")

        if winner_tournament_score is not None and best_synthetic_score is not None:
            task_type = TaskType(task_pair.task_type)
            logger.info(f"Task type: {task_type}")
            if task_type == TaskType.GRPOTASK:
                # GRPO: higher is better
                # If winner scored higher than best synthetic, it's an improvement
                if best_synthetic_score > 0:
                    performance_diff = (winner_tournament_score - best_synthetic_score) / best_synthetic_score
                else:
                    performance_diff = 0.0
            else:
                # Other tasks: lower is better (loss)
                # If winner scored lower than best synthetic, it's an improvement
                if winner_tournament_score > 0:
                    performance_diff = (best_synthetic_score - winner_tournament_score) / winner_tournament_score
                else:
                    performance_diff = 0.0

            logger.info(f"Performance difference for task pair {i+1}: {performance_diff}")
            performance_differences.append(performance_diff)
        else:
            if winner_tournament_score is None and best_synthetic_score is not None:
                # Winner didn't complete the task but synthetic miners did - apply max penalty
                logger.warning(f"Winner {task_pair.winner_hotkey} has no score in tournament task but synthetic miners do - applying max burn reduction")
                performance_diff = cts.MAX_BURN_REDUCTION / cts.BURN_REDUCTION_RATE  # This will result in max burn reduction
                performance_differences.append(performance_diff)
            else:
                if winner_tournament_score is None:
                    logger.warning(f"Could not find winner {task_pair.winner_hotkey} score in tournament task for pair {i+1}")
                if best_synthetic_score is None:
                    logger.warning(f"Could not find any scores in synthetic task for pair {i+1}")

    average_performance_diff = sum(performance_differences) / len(performance_differences) if performance_differences else 0.0
    logger.info(f"Average performance difference: {average_performance_diff} from {len(performance_differences)} task pairs")
    return average_performance_diff


def calculate_burn_proportion(performance_diff: float) -> float:
    if performance_diff <= 0:
        return 0.0

    burn_reduction = min(cts.MAX_BURN_REDUCTION, performance_diff * cts.BURN_REDUCTION_RATE)
    return burn_reduction


def calculate_weight_redistribution(performance_diff: float) -> tuple[float, float, float]:
    burn_reduction = calculate_burn_proportion(performance_diff)
    tournament_burn = cts.BASE_TOURNAMENT_WEIGHT * burn_reduction

    tournament_weight = cts.BASE_TOURNAMENT_WEIGHT - tournament_burn
    regular_weight = cts.BASE_REGULAR_WEIGHT + (tournament_burn * cts.LEGACY_PERFORM_DIFF_EMISSION_GAIN_PERCENT)
    burn_weight = (1 - cts.BASE_REGULAR_WEIGHT - cts.BASE_TOURNAMENT_WEIGHT) + (tournament_burn * (1 - cts.LEGACY_PERFORM_DIFF_EMISSION_GAIN_PERCENT))

    return tournament_weight, regular_weight, burn_weight


async def get_active_tournament_burn_data(psql_db) -> tuple[float, float, float]:
    from core.models.tournament_models import TournamentType

    logger.info("=== CALCULATING TOURNAMENT BURN DATA ===")
    weighted_performance_diff = 0.0
    total_weight = 0.0

    tournament_weights = {TournamentType.TEXT: cts.TOURNAMENT_TEXT_WEIGHT, TournamentType.IMAGE: cts.TOURNAMENT_IMAGE_WEIGHT}
    logger.info(f"Tournament type weights: TEXT={cts.TOURNAMENT_TEXT_WEIGHT}, IMAGE={cts.TOURNAMENT_IMAGE_WEIGHT}")

    for tournament_type, weight in tournament_weights.items():
        logger.info(f"Processing {tournament_type} tournament type")
        performance_diff = None

        latest_tournament = await get_latest_completed_tournament(psql_db, tournament_type)
        if latest_tournament:
            logger.info(f"Found latest {tournament_type} tournament: {latest_tournament.tournament_id}")
            synth_tasks_complete = await check_boss_round_synthetic_tasks_complete(latest_tournament.tournament_id, psql_db)
            logger.info(f"Boss round synthetic tasks complete for {tournament_type}: {synth_tasks_complete}")
            
            if synth_tasks_complete:
                performance_diff = await calculate_performance_difference(latest_tournament.tournament_id, psql_db)
                logger.info(
                    f"Using latest {tournament_type} tournament {latest_tournament.tournament_id} performance: {performance_diff}"
                )
            else:
                previous_tournament_id = await get_previous_completed_tournament(
                    psql_db, tournament_type, latest_tournament.tournament_id
                )
                if previous_tournament_id:
                    if await check_boss_round_synthetic_tasks_complete(previous_tournament_id, psql_db):
                        performance_diff = await calculate_performance_difference(previous_tournament_id, psql_db)
                        logger.info(
                            f"Using previous {tournament_type} tournament {previous_tournament_id} performance: {performance_diff}"
                        )
                    else:
                        logger.info(
                            f"Previous {tournament_type} tournament {previous_tournament_id} synthetic tasks not complete"
                        )
                else:
                    logger.info(f"No previous {tournament_type} tournament found")

        if performance_diff is not None:
            weighted_performance_diff += performance_diff * weight
            total_weight += weight
        elif latest_tournament:
            # Check if burn account won this tournament
            if latest_tournament.winner_hotkey == cts.EMISSION_BURN_HOTKEY:
                logger.info(
                    f"No synthetic task data available for {tournament_type} tournaments, burn account won - assuming worst performance (100% difference)"
                )
                weighted_performance_diff += 1.0 * weight  # Maximum performance difference
            else:
                logger.info(
                    f"No synthetic task data available for {tournament_type} tournaments, assuming perfect performance (0% difference)"
                )
                weighted_performance_diff += 0.0 * weight
            total_weight += weight
        else:
            logger.info(f"No {tournament_type} tournament data available, will burn this tournament allocation")

    if total_weight == 0:
        logger.info("No tournament data available, burning entire tournament allocation")
        tournament_burn = cts.BASE_TOURNAMENT_WEIGHT
        tournament_weight = 0.0
        regular_weight = cts.BASE_REGULAR_WEIGHT + (tournament_burn * cts.LEGACY_PERFORM_DIFF_EMISSION_GAIN_PERCENT)
        burn_weight = (1 - cts.BASE_REGULAR_WEIGHT - cts.BASE_TOURNAMENT_WEIGHT) + (tournament_burn * (1 - cts.LEGACY_PERFORM_DIFF_EMISSION_GAIN_PERCENT))
        return tournament_weight, regular_weight, burn_weight

    average_performance_diff = weighted_performance_diff / total_weight
    return calculate_weight_redistribution(average_performance_diff)


async def get_node_weights_from_period_scores_with_tournament_data(
    substrate: SubstrateInterface,
    netuid: int,
    node_results: list[PeriodScore],
    tournament_weights: dict[str, float],
    participants: list[str],
    tournament_weight_multiplier: float,
    regular_weight_multiplier: float,
    burn_weight: float,
) -> tuple[list[int], list[float]]:
    """
    Get the node ids and weights from the node results with tournament data provided as arguments.
    """
    all_nodes: list[Node] = fetch_nodes.get_nodes_for_netuid(substrate, netuid)

    hotkey_to_node_id = {node.hotkey: node.node_id for node in all_nodes}

    all_node_ids = [node.node_id for node in all_nodes]
    all_node_weights = [0.0 for _ in all_nodes]

    logger.info("=== BURN CALCULATION ===")
    logger.info(
        f"Weight distribution: tournament={tournament_weight_multiplier:.6f}, regular={regular_weight_multiplier:.6f}, burn={burn_weight:.6f}"
    )

    # Calculate participation weights total and scale existing weights
    participation_total = len(participants) * cts.TOURNAMENT_PARTICIPATION_WEIGHT

    if participation_total > 0:
        scale_factor = 1.0 - participation_total
        tournament_weight_multiplier *= scale_factor
        regular_weight_multiplier *= scale_factor
        burn_weight *= scale_factor
        logger.info(f"Scaled weights for {len(participants)} participants (total participation: {participation_total:.6f})")

    logger.info(
        f"Weight distribution: tournament={tournament_weight_multiplier:.6f}, regular={regular_weight_multiplier:.6f}, burn={burn_weight:.6f}, participation={participation_total:.6f}"
    )

    logger.info("=== NODE WEIGHT CALCULATIONS ===")
    for node_result in node_results:
        if node_result.normalised_score is not None:
            node_id = hotkey_to_node_id.get(node_result.hotkey)
            if node_id is not None:
                contribution = node_result.normalised_score * node_result.weight_multiplier * regular_weight_multiplier
                all_node_weights[node_id] = all_node_weights[node_id] + contribution
                logger.info(
                    f"Node ID {node_id} (hotkey: {node_result.hotkey[:8]}...): "
                    f"normalized_score={node_result.normalised_score:.6f}, "
                    f"weight_multiplier={node_result.weight_multiplier:.6f}, "
                    f"regular_multiplier={regular_weight_multiplier:.6f}, "
                    f"contribution={contribution:.6f}, "
                    f"total_weight={all_node_weights[node_id]:.6f}"
                )

    logger.info("=== TOURNAMENT WEIGHT CALCULATIONS ===")
    logger.info(f"Tournament weights provided: {tournament_weights}")
    logger.info(f"Tournament weights length: {len(tournament_weights)}")
    if tournament_weights:
        for hotkey, weight in tournament_weights.items():
            node_id = hotkey_to_node_id.get(hotkey)
            if node_id is not None:
                tournament_contribution = weight * tournament_weight_multiplier
                all_node_weights[node_id] = all_node_weights[node_id] + tournament_contribution
                logger.info(
                    f"Node ID {node_id} (hotkey: {hotkey[:8]}...): "
                    f"tournament_weight={weight:.6f}, "
                    f"tournament_multiplier={tournament_weight_multiplier:.6f}, "
                    f"tournament_contribution={tournament_contribution:.6f}, "
                    f"total_weight={all_node_weights[node_id]:.6f}"
                )
    else:
        logger.info("No tournament weights found")

    logger.info("=== PARTICIPATION WEIGHT ALLOCATION ===")
    if participants:
        for hotkey in participants:
            node_id = hotkey_to_node_id.get(hotkey)
            if node_id is not None:
                participation_contribution = cts.TOURNAMENT_PARTICIPATION_WEIGHT
                all_node_weights[node_id] = all_node_weights[node_id] + participation_contribution
                logger.info(
                    f"Node ID {node_id} (hotkey: {hotkey[:8]}...): "
                    f"participation_weight={participation_contribution:.6f}, "
                    f"total_weight={all_node_weights[node_id]:.6f}"
                )
    else:
        logger.info("No tournament participants found")

    logger.info("=== BURN WEIGHT ALLOCATION ===")
    burn_node_id = hotkey_to_node_id.get(cts.EMISSION_BURN_HOTKEY)
    if burn_node_id is not None:
        all_node_weights[burn_node_id] = burn_weight
        logger.info(f"Burn Node ID {burn_node_id} (hotkey: {cts.EMISSION_BURN_HOTKEY[:8]}...): burn_weight={burn_weight:.6f}")
    else:
        logger.warning(f"Burn hotkey {cts.EMISSION_BURN_HOTKEY} not found in network nodes")

    logger.info("=== FINAL NODE WEIGHTS ===")
    for node_id, weight in enumerate(all_node_weights):
        if weight > 0:
            logger.info(f"Node ID {node_id}: final_weight={weight:.6f}")

    logger.info(f"Node ids: {all_node_ids}")
    logger.info(f"Node weights: {all_node_weights}")
    logger.info(f"Number of non zero node weights: {sum(1 for weight in all_node_weights if weight != 0)}")
    logger.info(f"Everything going in is {all_node_ids} {all_node_weights} {netuid} {ccst.VERSION_KEY}")
    return all_node_ids, all_node_weights


async def get_node_weights_from_period_scores(
    substrate: SubstrateInterface, netuid: int, node_results: list[PeriodScore], psql_db
) -> tuple[list[int], list[float]]:
    """
    Get the node ids and weights from the node results.
    """
    all_nodes: list[Node] = fetch_nodes.get_nodes_for_netuid(substrate, netuid)

    hotkey_to_node_id = {node.hotkey: node.node_id for node in all_nodes}

    all_node_ids = [node.node_id for node in all_nodes]
    all_node_weights = [0.0 for _ in all_nodes]


    logger.info("=== BURN CALCULATION ===")
    tournament_weight_multiplier, regular_weight_multiplier, burn_weight = await get_active_tournament_burn_data(psql_db)

    # Calculate participation weights total and scale existing weights
    participants = await get_active_tournament_participants(psql_db)
    participation_total = len(participants) * cts.TOURNAMENT_PARTICIPATION_WEIGHT

    if participation_total > 0:
        scale_factor = 1.0 - participation_total
        tournament_weight_multiplier *= scale_factor
        regular_weight_multiplier *= scale_factor
        burn_weight *= scale_factor
        logger.info(f"Scaled weights for {len(participants)} participants (total participation: {participation_total:.6f})")

    logger.info(
        f"Weight distribution: tournament={tournament_weight_multiplier:.6f}, regular={regular_weight_multiplier:.6f}, burn={burn_weight:.6f}, participation={participation_total:.6f}"
    )


    logger.info("=== NODE WEIGHT CALCULATIONS ===")
    for node_result in node_results:
        if node_result.normalised_score is not None:
            node_id = hotkey_to_node_id.get(node_result.hotkey)
            if node_id is not None:
                contribution = node_result.normalised_score * node_result.weight_multiplier * regular_weight_multiplier
                all_node_weights[node_id] = all_node_weights[node_id] + contribution
                logger.info(
                    f"Node ID {node_id} (hotkey: {node_result.hotkey[:8]}...): "
                    f"normalized_score={node_result.normalised_score:.6f}, "
                    f"weight_multiplier={node_result.weight_multiplier:.6f}, "
                    f"regular_multiplier={regular_weight_multiplier:.6f}, "
                    f"contribution={contribution:.6f}, "
                    f"total_weight={all_node_weights[node_id]:.6f}"
                )

    logger.info("=== TOURNAMENT WEIGHT CALCULATIONS ===")

    text_tournament = await get_latest_completed_tournament(psql_db, TournamentType.TEXT)
    text_tournament_data = None
    if text_tournament:
        tournament_results = await get_tournament_full_results(text_tournament.tournament_id, psql_db)
        text_tournament_data = TournamentResultsWithWinners(
            tournament_id=tournament_results.tournament_id,
            rounds=tournament_results.rounds,
            base_winner_hotkey=text_tournament.base_winner_hotkey,
            winner_hotkey=text_tournament.winner_hotkey,
        )

    image_tournament = await get_latest_completed_tournament(psql_db, TournamentType.IMAGE)
    image_tournament_data = None
    if image_tournament:
        tournament_results = await get_tournament_full_results(image_tournament.tournament_id, psql_db)
        image_tournament_data = TournamentResultsWithWinners(
            tournament_id=tournament_results.tournament_id,
            rounds=tournament_results.rounds,
            base_winner_hotkey=image_tournament.base_winner_hotkey,
            winner_hotkey=image_tournament.winner_hotkey,
        )

    tournament_weights = get_tournament_weights_from_data(text_tournament_data, image_tournament_data)
    
    # Apply tournament type weights if only one tournament type completed
    if text_tournament_data and not image_tournament_data:
        # Only text tournament - scale weights by text proportion
        tournament_weights = {hotkey: weight * cts.TOURNAMENT_TEXT_WEIGHT for hotkey, weight in tournament_weights.items()}
        logger.info(f"Only text tournament completed - scaled weights by {cts.TOURNAMENT_TEXT_WEIGHT}")
    elif image_tournament_data and not text_tournament_data:
        # Only image tournament - scale weights by image proportion  
        tournament_weights = {hotkey: weight * cts.TOURNAMENT_IMAGE_WEIGHT for hotkey, weight in tournament_weights.items()}
        logger.info(f"Only image tournament completed - scaled weights by {cts.TOURNAMENT_IMAGE_WEIGHT}")

    logger.info(f"Tournament weights returned: {tournament_weights}")
    logger.info(f"Tournament weights length: {len(tournament_weights)}")
    if tournament_weights:
        for hotkey, weight in tournament_weights.items():
            node_id = hotkey_to_node_id.get(hotkey)
            if node_id is not None:
                tournament_contribution = weight * tournament_weight_multiplier
                all_node_weights[node_id] = all_node_weights[node_id] + tournament_contribution
                logger.info(
                    f"Node ID {node_id} (hotkey: {hotkey[:8]}...): "
                    f"tournament_weight={weight:.6f}, "
                    f"tournament_multiplier={tournament_weight_multiplier:.6f}, "
                    f"tournament_contribution={tournament_contribution:.6f}, "
                    f"total_weight={all_node_weights[node_id]:.6f}"
                )
    else:
        logger.info("No tournament weights found")

    logger.info("=== PARTICIPATION WEIGHT ALLOCATION ===")
    if participants:
        for hotkey in participants:
            node_id = hotkey_to_node_id.get(hotkey)
            if node_id is not None:
                participation_contribution = cts.TOURNAMENT_PARTICIPATION_WEIGHT
                all_node_weights[node_id] = all_node_weights[node_id] + participation_contribution
                logger.info(
                    f"Node ID {node_id} (hotkey: {hotkey[:8]}...): "
                    f"participation_weight={participation_contribution:.6f}, "
                    f"total_weight={all_node_weights[node_id]:.6f}"
                )
    else:
        logger.info("No tournament participants found")

    logger.info("=== BURN WEIGHT ALLOCATION ===")
    burn_node_id = hotkey_to_node_id.get(cts.EMISSION_BURN_HOTKEY)
    if burn_node_id is not None:
        all_node_weights[burn_node_id] = burn_weight
        logger.info(f"Burn Node ID {burn_node_id} (hotkey: {cts.EMISSION_BURN_HOTKEY[:8]}...): burn_weight={burn_weight:.6f}")
    else:
        logger.warning(f"Burn hotkey {cts.EMISSION_BURN_HOTKEY} not found in network nodes")


    logger.info("=== FINAL NODE WEIGHTS ===")
    for node_id, weight in enumerate(all_node_weights):
        if weight > 0:
            logger.info(f"Node ID {node_id}: final_weight={weight:.6f}")

    logger.info(f"Node ids: {all_node_ids}")
    logger.info(f"Node weights: {all_node_weights}")
    logger.info(f"Number of non zero node weights: {sum(1 for weight in all_node_weights if weight != 0)}")
    logger.info(f"Everything going in is {all_node_ids} {all_node_weights} {netuid} {ccst.VERSION_KEY}")
    return all_node_ids, all_node_weights


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
    node_results, task_results = await _get_weights_to_set(config)
    if node_results is None:
        logger.info("No weights to set. Skipping weight setting.")
        return False
    if len(node_results) == 0:
        logger.info("No nodes to set weights for. Skipping weight setting.")
        return False

    tournament_audit_data = TournamentAuditData()

    text_tournament = await get_latest_completed_tournament(config.psql_db, TournamentType.TEXT)
    if text_tournament:
        tournament_results = await get_tournament_full_results(text_tournament.tournament_id, config.psql_db)
        tournament_audit_data.text_tournament_data = TournamentResultsWithWinners(
            tournament_id=tournament_results.tournament_id,
            rounds=tournament_results.rounds,
            base_winner_hotkey=text_tournament.base_winner_hotkey,
            winner_hotkey=text_tournament.winner_hotkey,
        )

    image_tournament = await get_latest_completed_tournament(config.psql_db, TournamentType.IMAGE)
    if image_tournament:
        tournament_results = await get_tournament_full_results(image_tournament.tournament_id, config.psql_db)
        tournament_audit_data.image_tournament_data = TournamentResultsWithWinners(
            tournament_id=tournament_results.tournament_id,
            rounds=tournament_results.rounds,
            base_winner_hotkey=image_tournament.base_winner_hotkey,
            winner_hotkey=image_tournament.winner_hotkey,
        )

    tournament_audit_data.participants = await get_active_tournament_participants(config.psql_db)

    (
        tournament_audit_data.tournament_weight_multiplier,
        tournament_audit_data.regular_weight_multiplier,
        tournament_audit_data.burn_weight,
    ) = await get_active_tournament_burn_data(config.psql_db)

    all_node_ids, all_node_weights = await get_node_weights_from_period_scores(
        config.substrate, config.netuid, node_results, config.psql_db
    )
    logger.info("Weights calculated, about to set...")

    success = await set_weights(config, all_node_ids, all_node_weights, validator_node_id)
    if success and task_results:
        # Upload both task results and tournament data
        url = await _upload_results_to_s3(config, task_results, tournament_audit_data)
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


#


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
