from fastapi import APIRouter
from fastapi import Depends

import validator.core.constants as cts
from core.models.tournament_models import TournamentBurnData
from core.models.tournament_models import TournamentType
from core.models.tournament_models import TournamentWeightsResponse
from core.models.tournament_models import WeightProjectionResponse
from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.core.weight_setting import build_tournament_audit_data
from validator.core.weight_setting import get_tournament_burn_details
from validator.evaluation.tournament_scoring import get_tournament_weights_from_data
from validator.tournament.performance_utils import calculate_tournament_projection
from validator.tournament.performance_utils import get_top_ranked_miners


router = APIRouter(tags=["Performance Data"])


@router.get("/performance/latest-tournament-weights")
async def get_latest_tournament_weights(config: Config = Depends(get_config)) -> TournamentWeightsResponse:
    burn_data: TournamentBurnData = await get_tournament_burn_details(config.psql_db)

    tournament_audit_data = await build_tournament_audit_data(config.psql_db)

    text_tournament_weights, image_tournament_weights = get_tournament_weights_from_data(
        tournament_audit_data.text_tournament_data, tournament_audit_data.image_tournament_data
    )

    text_base_winner_hotkey = None
    if tournament_audit_data.text_tournament_data:
        text_base_winner_hotkey = tournament_audit_data.text_tournament_data.base_winner_hotkey

    image_base_winner_hotkey = None
    if tournament_audit_data.image_tournament_data:
        image_base_winner_hotkey = tournament_audit_data.image_tournament_data.base_winner_hotkey

    text_top_miners = get_top_ranked_miners(text_tournament_weights, text_base_winner_hotkey, limit=5)
    image_top_miners = get_top_ranked_miners(image_tournament_weights, image_base_winner_hotkey, limit=5)

    return TournamentWeightsResponse(
        burn_data=burn_data,
        text_top_miners=text_top_miners,
        image_top_miners=image_top_miners,
    )


@router.get("/performance/weight-projection")
async def get_weight_projection(
    percentage_improvement: float,
    config: Config = Depends(get_config),
) -> WeightProjectionResponse:
    text_projection = await calculate_tournament_projection(
        config.psql_db,
        TournamentType.TEXT,
        percentage_improvement,
        cts.TOURNAMENT_TEXT_WEIGHT,
        cts.MAX_TEXT_TOURNAMENT_WEIGHT,
    )

    image_projection = await calculate_tournament_projection(
        config.psql_db,
        TournamentType.IMAGE,
        percentage_improvement,
        cts.TOURNAMENT_IMAGE_WEIGHT,
        cts.MAX_IMAGE_TOURNAMENT_WEIGHT,
    )

    return WeightProjectionResponse(
        percentage_improvement=percentage_improvement,
        text_projection=text_projection,
        image_projection=image_projection,
    )


def factory_router():
    return router

