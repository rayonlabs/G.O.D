from datetime import datetime
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query

from core.models.tournament_models import TournamentDetailsResponse, DetailedTournamentRoundResult, DetailedTournamentTaskScore, TournamentScore, TournamentType, TournamentSummary
from validator.core.config import Config
from validator.core.dependencies import get_api_key
from validator.core.dependencies import get_config
from validator.db.sql import tournaments as tournament_sql
from validator.db.sql import tasks as task_sql
from validator.utils.logging import get_logger
import validator.core.constants as cts


logger = get_logger(__name__)

GET_TOURNAMENT_DETAILS_ENDPOINT = "/v1/tournaments/{tournament_id}/details"
GET_LATEST_TOURNAMENTS_DETAILS_ENDPOINT = "/v1/tournaments/latest/details"
GET_ALL_TOURNAMENTS_ENDPOINT = "/v1/tournaments"


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
                    task_type=task_details.task_type if task_details else None
                )
                detailed_tasks.append(detailed_task)
            
            detailed_round = DetailedTournamentRoundResult(
                round_id=round_data.round_id,
                round_number=round_data.round_number,
                round_type=round_data.round_type,
                is_final_round=round_data.is_final_round,
                status=round_data.status,
                participants=list(set(round_participants)),
                tasks=detailed_tasks
            )
            detailed_rounds.append(detailed_round)
        
        from validator.evaluation.tournament_scoring import calculate_tournament_type_scores
        
        tournament_type_result = await calculate_tournament_type_scores(
            TournamentType(tournament.tournament_type), 
            config.psql_db
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
            image_tournament_weight=cts.TOURNAMENT_IMAGE_WEIGHT
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
            result['text'] = await get_tournament_details(latest_text.tournament_id, config)
        else:
            result['text'] = None
            
        if latest_image:
            result['image'] = await get_tournament_details(latest_image.tournament_id, config)
        else:
            result['image'] = None
        
        logger.info(f"Retrieved latest tournament details: text={latest_text.tournament_id if latest_text else None}, image={latest_image.tournament_id if latest_image else None}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving latest tournament details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def get_all_tournaments(
    config: Config = Depends(get_config),
    from_date: datetime | None = Query(None, description="Filter tournaments from this date (inclusive)"),
    to_date: datetime | None = Query(None, description="Filter tournaments to this date (inclusive)")
) -> list[TournamentSummary]:
    try:
        tournaments = await tournament_sql.get_all_tournaments_summary(config.psql_db, from_date, to_date)
        logger.info(f"Retrieved {len(tournaments)} tournaments")
        return tournaments
        
    except Exception as e:
        logger.error(f"Error retrieving tournaments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def factory_router() -> APIRouter:
    router = APIRouter(tags=["Tournament Analytics"], dependencies=[Depends(get_api_key)])
    router.add_api_route(GET_ALL_TOURNAMENTS_ENDPOINT, get_all_tournaments, methods=["GET"])
    router.add_api_route(GET_LATEST_TOURNAMENTS_DETAILS_ENDPOINT, get_latest_tournaments_details, methods=["GET"])
    router.add_api_route(GET_TOURNAMENT_DETAILS_ENDPOINT, get_tournament_details, methods=["GET"])
    return router