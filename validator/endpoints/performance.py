from fastapi import APIRouter
from fastapi import Depends

from core.models.payload_models import MinerEmissionWeight
from core.models.payload_models import PerformanceResponse
from core.models.payload_models import TournamentEmissionWeights
from core.models.tournament_models import TournamentBurnData
from core.models.tournament_models import TournamentType
from validator.core.constants import EMISSION_BURN_HOTKEY
from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.core.weight_setting import build_tournament_audit_data
from validator.core.weight_setting import get_tournament_burn_details
from validator.evaluation.tournament_scoring import get_tournament_weights_from_data


router = APIRouter(tags=["performance"])


def _get_top_ranked_miners(
    weights: dict[str, float],
    tournament_type: str,
    base_winner_hotkey: str | None = None,
    limit: int = 5,
) -> TournamentEmissionWeights:
    real_hotkey_weights = {}
    for hotkey, weight in weights.items():
        if hotkey == EMISSION_BURN_HOTKEY and base_winner_hotkey:
            real_hotkey = base_winner_hotkey
        else:
            real_hotkey = hotkey
        real_hotkey_weights[real_hotkey] = weight

    sorted_miners = sorted(real_hotkey_weights.items(), key=lambda x: x[1], reverse=True)[:limit]

    top_miners = [
        MinerEmissionWeight(hotkey=hotkey, rank=idx + 1, weight=weight) for idx, (hotkey, weight) in enumerate(sorted_miners)
    ]

    return TournamentEmissionWeights(tournament_type=tournament_type, top_miners=top_miners)


@router.get("/performance/latest-tournament-weights")
async def get_latest_tournament_weights(config: Config = Depends(get_config)) -> PerformanceResponse:
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

    text_emission_weights = _get_top_ranked_miners(
        text_tournament_weights, TournamentType.TEXT.value, text_base_winner_hotkey, limit=5
    )
    image_emission_weights = _get_top_ranked_miners(
        image_tournament_weights, TournamentType.IMAGE.value, image_base_winner_hotkey, limit=5
    )

    return PerformanceResponse(
        burn_data=burn_data,
        text_tournament_weights=text_emission_weights,
        image_tournament_weights=image_emission_weights,
    )


def factory_router():
    return router

