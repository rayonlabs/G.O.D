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
from core.models.tournament_models import TournamentDetailsResponse
from core.models.tournament_models import TournamentResultsWithWinners
from core.models.tournament_models import TournamentType
from core.models.tournament_models import get_tournament_gpu_requirement
from core.models.utility_models import TaskStatus
from validator.core.config import Config
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


async def get_tournament_details(
    tournament_id: str,
    config: Config = Depends(get_config),
) -> TournamentDetailsResponse:
    try:
        tournament = await tournament_sql.get_tournament(tournament_id, config.psql_db)
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")

        participants = await tournament_sql.get_tournament_participants(tournament_id, config.psql_db)
        rounds = await tournament_sql.get_tournament_rounds(tournament_id, config.psql_db)

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
            for task in tasks:
                task_details = await task_sql.get_task(task.task_id, config.psql_db)
                participant_scores = await tournament_sql.get_all_scores_and_losses_for_task(task.task_id, config.psql_db)
                task_winners = await tournament_sql.get_task_winners([task.task_id], config.psql_db)
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
        )

        logger.info(f"Retrieved tournament details for {tournament_id}")
        return response

    except Exception as e:
        logger.error(f"Error retrieving tournament details for {tournament_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def get_latest_tournaments_details(
    config: Config = Depends(get_config),
) -> dict[str, TournamentDetailsResponse | None]:
    try:
        latest_text = await tournament_sql.get_latest_completed_tournament(config.psql_db, TournamentType.TEXT)
        latest_image = await tournament_sql.get_latest_completed_tournament(config.psql_db, TournamentType.IMAGE)

        result = {}

        if latest_text:
            result["text"] = await get_tournament_details(latest_text.tournament_id, config)
        else:
            result["text"] = None

        if latest_image:
            result["image"] = await get_tournament_details(latest_image.tournament_id, config)
        else:
            result["image"] = None

        logger.info(
            f"Retrieved latest tournament details: text={latest_text.tournament_id if latest_text else None}, image={latest_image.tournament_id if latest_image else None}"
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
    """Get the next tournament start and end dates for both text and image tournaments."""
    try:
        async def get_next_dates_for_type(tournament_type: TournamentType) -> NextTournamentInfo:
            tournament, created_at = await tournament_sql.get_latest_tournament_with_created_at(
                config.psql_db, tournament_type
            )

            if created_at is None:
                # No previous tournament, start from now
                next_start = datetime.now(timezone.utc)
            else:
                # Next tournament starts TOURNAMENT_INTERVAL_DAYS after the last one started
                next_start = created_at + timedelta(days=cts.TOURNAMENT_INTERVAL_DAYS)

                # If the calculated start date is in the past, use current time
                current_time = datetime.now(timezone.utc)
                if next_start < current_time:
                    next_start = current_time

            # Tournament ends TOURNAMENT_INTERVAL_DAYS after it starts
            next_end = next_start + timedelta(days=cts.TOURNAMENT_INTERVAL_DAYS)

            return NextTournamentInfo(
                tournament_type=tournament_type,
                next_start_date=next_start,
                next_end_date=next_end,
                interval_days=cts.TOURNAMENT_INTERVAL_DAYS,
            )

        response = NextTournamentDates(
            text=await get_next_dates_for_type(TournamentType.TEXT),
            image=await get_next_dates_for_type(TournamentType.IMAGE),
        )

        logger.info("Retrieved next tournament dates")
        return response

    except Exception as e:
        logger.error(f"Error retrieving next tournament dates: {str(e)}")
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


def factory_router() -> APIRouter:
    router = APIRouter(tags=["Tournament Analytics"], dependencies=[Depends(get_api_key)])
    router.add_api_route(GET_LATEST_TOURNAMENTS_DETAILS_ENDPOINT, get_latest_tournaments_details, methods=["GET"])
    router.add_api_route(GET_TOURNAMENT_DETAILS_ENDPOINT, get_tournament_details, methods=["GET"])
    router.add_api_route(GET_TOURNAMENT_GPU_REQUIREMENTS_ENDPOINT, get_tournament_gpu_requirements, methods=["GET"])
    router.add_api_route(GET_NEXT_TOURNAMENT_DATES_ENDPOINT, get_next_tournament_dates, methods=["GET"])
    router.add_api_route(GET_ACTIVE_TOURNAMENTS_ENDPOINT, get_active_tournaments, methods=["GET"])
    return router
