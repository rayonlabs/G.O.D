import asyncio
import json
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Dict

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

import validator.core.constants as cts
from core.models.payload_models import GpuRequirementSummary
from core.models.payload_models import TournamentGpuRequirementsResponse
from core.models.tournament_models import ActiveTournamentInfo
from core.models.tournament_models import ActiveTournamentParticipant
from core.models.tournament_models import ActiveTournamentsResponse
from core.models.tournament_models import DetailedTournamentRoundResult
from core.models.tournament_models import DetailedTournamentTaskScore
from core.models.tournament_models import NextTournamentDates
from core.models.tournament_models import NextTournamentInfo
from core.models.tournament_models import LatestTournamentsDetailsResponse
from core.models.tournament_models import TournamentBurnData
from core.models.tournament_models import TournamentDetailsResponse
from core.models.tournament_models import TournamentHistoryEntry
from core.models.tournament_models import TournamentHistoryResponse
from core.models.tournament_models import TournamentResultsWithWinners
from core.models.tournament_models import TournamentStatus
from core.models.tournament_models import TournamentType
from core.models.tournament_models import get_tournament_gpu_requirement
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.tournament.performance_calculator import calculate_boss_round_performance_differences
from validator.tournament.performance_calculator import get_tournament_performance_data
from validator.core.weight_setting import get_tournament_burn_details
from validator.tournament.tournament_manager import get_tournament_completion_time
from validator.tournament.tournament_manager import should_start_new_tournament_after_interval

from validator.core.constants import LATEST_TOURNAMENTS_CACHE_KEY
from validator.core.constants import LATEST_TOURNAMENTS_CACHE_TTL
from validator.core.dependencies import get_api_key
from validator.core.dependencies import get_config
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournaments as tournament_sql
from validator.evaluation.tournament_scoring import calculate_tournament_type_scores_from_data
from validator.utils.logging import get_logger


logger = get_logger(__name__)

GET_TOURNAMENT_DETAILS_ENDPOINT = "/v1/tournaments/{tournament_id}/details"
GET_LATEST_TOURNAMENTS_DETAILS_ENDPOINT = "/v1/tournaments/latest/details"
GET_TOURNAMENT_GPU_REQUIREMENTS_ENDPOINT = "/v1/tournaments/gpu-requirements"
GET_NEXT_TOURNAMENT_DATES_ENDPOINT = "/v1/tournaments/next-dates"
GET_ACTIVE_TOURNAMENTS_ENDPOINT = "/v1/tournaments/active"
GET_TOURNAMENT_HISTORY_ENDPOINT = "/v1/tournaments/history"


async def get_tournament_details(
    tournament_id: str,
    config: Config = Depends(get_config),
) -> TournamentDetailsResponse:
    try:
        # Fetch tournament, participants, and rounds in parallel
        tournament_task = tournament_sql.get_tournament(tournament_id, config.psql_db)
        participants_task = tournament_sql.get_tournament_participants(tournament_id, config.psql_db)
        rounds_task = tournament_sql.get_tournament_rounds(tournament_id, config.psql_db)
        
        tournament, participants, rounds = await asyncio.gather(
            tournament_task, participants_task, rounds_task
        )
        
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")

        detailed_rounds = []
        for round_data in rounds:
            tasks = await tournament_sql.get_tournament_tasks(round_data.round_id, config.psql_db)

            round_participants = []
            if round_data.round_type == "group":
                groups = await tournament_sql.get_tournament_groups(round_data.round_id, config.psql_db)
                for group in groups:
                    group_members = await tournament_sql.get_tournament_group_members(group.group_id, config.psql_db)
                    round_participants.extend([member.hotkey for member in group_members])
            else:
                pairs = await tournament_sql.get_tournament_pairs(round_data.round_id, config.psql_db)
                for pair in pairs:
                    round_participants.extend([pair.hotkey1, pair.hotkey2])

            detailed_tasks = []
            
            # Collect all task IDs for batch operations
            task_ids = [task.task_id for task in tasks]
            
            # Fetch all task details, scores, and winners in parallel
            if task_ids:
                task_details_tasks = [task_sql.get_task(task_id, config.psql_db) for task_id in task_ids]
                scores_tasks = [tournament_sql.get_all_scores_and_losses_for_task(task_id, config.psql_db) for task_id in task_ids]
                winners_task = tournament_sql.get_task_winners(task_ids, config.psql_db)
                
                task_details_results = await asyncio.gather(*task_details_tasks)
                scores_results = await asyncio.gather(*scores_tasks)
                task_winners = await winners_task
                
                for i, task in enumerate(tasks):
                    task_details = task_details_results[i]
                    participant_scores = scores_results[i]
                    winner = task_winners.get(str(task.task_id))

                    detailed_task = DetailedTournamentTaskScore(
                        task_id=str(task.task_id),
                        group_id=task.group_id,
                        pair_id=task.pair_id,
                        winner=winner,
                        participant_scores=participant_scores,
                        task_type=task_details.task_type if task_details else None,
                    )
                    detailed_tasks.append(detailed_task)

            detailed_round = DetailedTournamentRoundResult(
                round_id=round_data.round_id,
                round_number=round_data.round_number,
                round_type=round_data.round_type,
                is_final_round=round_data.is_final_round,
                status=round_data.status,
                participants=list(set(round_participants)),
                tasks=detailed_tasks,
            )
            detailed_rounds.append(detailed_round)

        tournament_results_with_winners = TournamentResultsWithWinners(
            tournament_id=tournament.tournament_id,
            rounds=detailed_rounds,
            base_winner_hotkey=tournament.base_winner_hotkey,
            winner_hotkey=tournament.winner_hotkey,
        )
        tournament_type_result = calculate_tournament_type_scores_from_data(
            TournamentType(tournament.tournament_type), tournament_results_with_winners
        )

        boss_round_performance = None
        sync_performance = None
        
        if tournament.status == TournamentStatus.COMPLETED:
            final_round = next((r for r in detailed_rounds if r.is_final_round), None)
            if final_round:
                try:
                    boss_round_performance = await calculate_boss_round_performance_differences(
                        tournament.tournament_id, final_round.round_id, config.psql_db
                    )
                except Exception as e:
                    logger.warning(f"Failed to get boss round performance data: {e}")
            
            try:
                sync_performance = await get_tournament_performance_data(tournament.tournament_id, config.psql_db)
            except Exception as e:
                logger.warning(f"Failed to get sync performance data: {e}")
        
        response = TournamentDetailsResponse(
            tournament_id=tournament.tournament_id,
            tournament_type=tournament.tournament_type,
            status=tournament.status,
            base_winner_hotkey=tournament.base_winner_hotkey,
            winner_hotkey=tournament.winner_hotkey,
            participants=participants,
            rounds=detailed_rounds,
            final_scores=tournament_type_result.scores,
            text_tournament_weight=cts.TOURNAMENT_TEXT_WEIGHT,
            image_tournament_weight=cts.TOURNAMENT_IMAGE_WEIGHT,
            boss_round_performance=boss_round_performance,
            sync_performance=sync_performance,
        )

        logger.info(f"Retrieved tournament details for {tournament_id}")
        return response

    except Exception as e:
        logger.error(f"Error retrieving tournament details for {tournament_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def get_latest_tournaments_details(
    config: Config = Depends(get_config),
) -> LatestTournamentsDetailsResponse:
    try:
        cached_data = await config.redis_db.get(LATEST_TOURNAMENTS_CACHE_KEY)
        if cached_data:
            logger.info("Returning cached latest tournament details")
            cached_dict = json.loads(cached_data)
            return LatestTournamentsDetailsResponse.model_validate(cached_dict)
        
        latest_text = await tournament_sql.get_latest_completed_tournament(config.psql_db, TournamentType.TEXT)
        latest_image = await tournament_sql.get_latest_completed_tournament(config.psql_db, TournamentType.IMAGE)

        text_details = None
        image_details = None

        if latest_text:
            text_details = await get_tournament_details(latest_text.tournament_id, config)

        if latest_image:
            image_details = await get_tournament_details(latest_image.tournament_id, config)

        burn_data = await get_tournament_burn_details(config.psql_db)
        
        result = LatestTournamentsDetailsResponse(
            text=text_details,
            image=image_details,
            burn_data=burn_data
        )

        cache_data = result.model_dump()
        await config.redis_db.set(LATEST_TOURNAMENTS_CACHE_KEY, json.dumps(cache_data), ex=LATEST_TOURNAMENTS_CACHE_TTL)
        logger.info(f"Cached latest tournament details for {LATEST_TOURNAMENTS_CACHE_TTL} seconds")

        logger.info(
            f"Retrieved latest tournament details: text={latest_text.tournament_id if latest_text else None}, "
            f"image={latest_image.tournament_id if latest_image else None}, "
            f"burn_weight={burn_data.burn_weight:.4f}"
        )
        return result

    except Exception as e:
        logger.error(f"Error retrieving latest tournament details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def get_tournament_gpu_requirements(
    config: Config = Depends(get_config),
) -> TournamentGpuRequirementsResponse:
    try:
        unfinished_statuses = [
            TaskStatus.PENDING,
            TaskStatus.PREPARING_DATA,
            TaskStatus.LOOKING_FOR_NODES,
            TaskStatus.READY,
            TaskStatus.TRAINING,
        ]

        unfinished_tasks = []
        for status in unfinished_statuses:
            tasks = await task_sql.get_tasks_with_status(status=status, psql_db=config.psql_db, tournament_filter="only")
            unfinished_tasks.extend(tasks)

        logger.info(f"Found {len(unfinished_tasks)} unfinished tournament tasks")

        gpu_requirements: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "total_hours": 0.0})

        for task in unfinished_tasks:
            gpu_req = get_tournament_gpu_requirement(task.task_type, task.model_params_count)
            gpu_type = gpu_req.value

            hours = float(task.hours_to_complete) if task.hours_to_complete else 1.0

            gpu_requirements[gpu_type]["count"] += 1
            gpu_requirements[gpu_type]["total_hours"] += hours

        gpu_summaries = []
        total_tasks = 0
        total_hours = 0.0

        for gpu_type, data in gpu_requirements.items():
            count = data["count"]
            hours = data["total_hours"]

            gpu_summaries.append(GpuRequirementSummary(gpu_type=gpu_type, count=count, total_hours=hours))

            total_tasks += count
            total_hours += hours

        gpu_summaries.sort(key=lambda x: x.gpu_type)

        response = TournamentGpuRequirementsResponse(
            gpu_requirements=gpu_summaries, total_tasks=total_tasks, total_hours=total_hours
        )

        logger.info(
            f"Retrieved GPU requirements: {len(gpu_summaries)} GPU types, {total_tasks} total tasks, {total_hours:.0f} total hours"
        )
        return response

    except Exception as e:
        logger.error(f"Error retrieving tournament GPU requirements: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def get_next_tournament_dates(
    config: Config = Depends(get_config),
) -> NextTournamentDates:
    """Get the next tournament info - either countdown to next tournament or current round number."""
    try:
        async def get_tournament_info_for_type(tournament_type: TournamentType) -> NextTournamentInfo:
            # Check if there's an active tournament first
            active_tournament = await tournament_sql.get_active_tournament(config.psql_db, tournament_type)
            if active_tournament:
                # Get current round number
                rounds = await tournament_sql.get_tournament_rounds(active_tournament.tournament_id, config.psql_db)
                current_round = len(rounds) if rounds else 1
                
                return NextTournamentInfo(
                    tournament_type=tournament_type,
                    current_round_number=current_round,
                    tournament_status="active",
                    interval_hours=cts.TOURNAMENT_INTERVAL_HOURS,
                )
            
            # Check if there's a pending tournament
            pending_tournaments = await tournament_sql.get_tournaments_with_status(TournamentStatus.PENDING, config.psql_db)
            pending_of_type = [t for t in pending_tournaments if t.tournament_type == tournament_type]
            if pending_of_type:
                return NextTournamentInfo(
                    tournament_type=tournament_type,
                    current_round_number=1,
                    tournament_status="pending",
                    interval_hours=cts.TOURNAMENT_INTERVAL_HOURS,
                )
            
            # No active/pending tournament, calculate next start time using same logic as scheduler
            tournament, created_at = await tournament_sql.get_latest_tournament_with_created_at(
                config.psql_db, tournament_type
            )

            current_time = datetime.now(timezone.utc)
            
            if not tournament:
                # No previous tournament, would start on next scheduler check
                # Round up to next 15-minute interval
                minutes_to_next_check = 15 - (current_time.minute % 15)
                if minutes_to_next_check == 0:
                    minutes_to_next_check = 15
                next_start = current_time + timedelta(minutes=minutes_to_next_check)
                next_start = next_start.replace(second=0, microsecond=0)
            else:
                # Check completion time like the scheduler does
                if tournament.status == TournamentStatus.COMPLETED:
                    completed_at = await get_tournament_completion_time(tournament.tournament_id, config.psql_db)
                    time_reference = completed_at or created_at
                else:
                    time_reference = created_at
                
                # Check if we should start a new tournament
                if await should_start_new_tournament_after_interval(time_reference):
                    # Tournament can start on next scheduler check
                    minutes_to_next_check = 15 - (current_time.minute % 15)
                    if minutes_to_next_check == 0:
                        minutes_to_next_check = 15
                    next_start = current_time + timedelta(minutes=minutes_to_next_check)
                    next_start = next_start.replace(second=0, microsecond=0)
                else:
                    # Calculate when 24 hours will have passed
                    if time_reference.tzinfo is None:
                        time_reference = time_reference.replace(tzinfo=timezone.utc)
                    
                    next_start = time_reference + timedelta(hours=cts.TOURNAMENT_INTERVAL_HOURS)
                    
                    # Round to next 15-minute scheduler check after that time
                    minutes = next_start.minute
                    remainder = minutes % 15
                    if remainder != 0:
                        next_start = next_start + timedelta(minutes=(15 - remainder))
                    next_start = next_start.replace(second=0, microsecond=0)

            return NextTournamentInfo(
                tournament_type=tournament_type,
                next_start_date=next_start,
                next_end_date=None,
                interval_hours=cts.TOURNAMENT_INTERVAL_HOURS,
                tournament_status="waiting",
            )

        response = NextTournamentDates(
            text=await get_tournament_info_for_type(TournamentType.TEXT),
            image=await get_tournament_info_for_type(TournamentType.IMAGE),
        )

        logger.info("Retrieved tournament info")
        return response

    except Exception as e:
        logger.error(f"Error retrieving tournament info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def get_active_tournaments(
    config: Config = Depends(get_config),
) -> ActiveTournamentsResponse:
    """Get currently active tournaments with participants and their stake requirements."""
    try:
        async def get_active_tournament_info(tournament_type: TournamentType) -> ActiveTournamentInfo | None:
            tournament = await tournament_sql.get_active_tournament(config.psql_db, tournament_type)
            if not tournament:
                return None

            _, created_at = await tournament_sql.get_tournament_with_created_at(
                tournament.tournament_id, config.psql_db
            )
            participants = await tournament_sql.get_tournament_participants(tournament.tournament_id, config.psql_db)

            active_participants = [
                ActiveTournamentParticipant(
                    hotkey=p.hotkey,
                    stake_requirement=p.stake_required,
                )
                for p in participants
                if p.stake_required is not None
            ]

            return ActiveTournamentInfo(
                tournament_id=tournament.tournament_id,
                tournament_type=tournament_type,
                status=tournament.status,
                participants=active_participants,
                created_at=created_at,
            )

        text_info = await get_active_tournament_info(TournamentType.TEXT)
        image_info = await get_active_tournament_info(TournamentType.IMAGE)

        logger.info(
            f"Retrieved active tournaments: text={text_info.tournament_id if text_info else None}, "
            f"image={image_info.tournament_id if image_info else None}"
        )

        return ActiveTournamentsResponse(text=text_info, image=image_info)

    except Exception as e:
        logger.error(f"Error retrieving active tournaments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def get_tournament_history(
    config: Config = Depends(get_config),
) -> TournamentHistoryResponse:
    """Get the history of all active and completed tournaments."""
    try:
        # Get all active tournaments
        active_tournaments = await tournament_sql.get_tournaments_with_status(TournamentStatus.ACTIVE, config.psql_db)
        
        # Get all completed tournaments
        completed_tournaments = await tournament_sql.get_tournaments_with_status(TournamentStatus.COMPLETED, config.psql_db)
        
        # Combine and sort by created_at (newest first)
        all_tournaments = active_tournaments + completed_tournaments
        
        # Get created_at dates for sorting
        tournament_entries = []
        for tournament in all_tournaments:
            _, created_at = await tournament_sql.get_tournament_with_created_at(
                tournament.tournament_id, config.psql_db
            )
            
            tournament_entries.append(
                TournamentHistoryEntry(
                    tournament_id=tournament.tournament_id,
                    tournament_type=tournament.tournament_type,
                    status=tournament.status,
                    winner_hotkey=tournament.winner_hotkey,
                    base_winner_hotkey=tournament.base_winner_hotkey,
                    created_at=created_at,
                )
            )
        
        # Sort by created_at (newest first)
        tournament_entries.sort(key=lambda x: x.created_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        
        response = TournamentHistoryResponse(
            tournaments=tournament_entries
        )
        
        logger.info(f"Retrieved tournament history: {len(tournament_entries)} tournaments")
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving tournament history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def factory_router() -> APIRouter:
    router = APIRouter(tags=["Tournament Analytics"], dependencies=[Depends(get_api_key)])
    router.add_api_route(GET_LATEST_TOURNAMENTS_DETAILS_ENDPOINT, get_latest_tournaments_details, methods=["GET"])
    router.add_api_route(GET_TOURNAMENT_DETAILS_ENDPOINT, get_tournament_details, methods=["GET"])
    router.add_api_route(GET_TOURNAMENT_GPU_REQUIREMENTS_ENDPOINT, get_tournament_gpu_requirements, methods=["GET"])
    router.add_api_route(GET_NEXT_TOURNAMENT_DATES_ENDPOINT, get_next_tournament_dates, methods=["GET"])
    router.add_api_route(GET_ACTIVE_TOURNAMENTS_ENDPOINT, get_active_tournaments, methods=["GET"])
    router.add_api_route(GET_TOURNAMENT_HISTORY_ENDPOINT, get_tournament_history, methods=["GET"])
    return router