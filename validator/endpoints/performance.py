from fastapi import APIRouter
from fastapi import Depends

from core.models.tournament_models import MinerEmissionWeight
from core.models.tournament_models import TournamentBurnData
from core.models.tournament_models import TournamentWeightsResponse
from validator.core.constants import EMISSION_BURN_HOTKEY
from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.core.weight_setting import build_tournament_audit_data
from validator.core.weight_setting import get_tournament_burn_details
from validator.evaluation.tournament_scoring import get_tournament_weights_from_data


router = APIRouter(tags=["Performance Data"])


def _get_top_ranked_miners(
    weights: dict[str, float],
    base_winner_hotkey: str | None = None,
    limit: int = 5,
) -> list[MinerEmissionWeight]:
    real_hotkey_weights = {}
    for hotkey, weight in weights.items():
        if hotkey == EMISSION_BURN_HOTKEY and base_winner_hotkey:
            real_hotkey = base_winner_hotkey
        else:
            real_hotkey = hotkey
        real_hotkey_weights[real_hotkey] = weight

    sorted_miners = sorted(real_hotkey_weights.items(), key=lambda x: x[1], reverse=True)[:limit]

    return [
        MinerEmissionWeight(hotkey=hotkey, rank=idx + 1, weight=weight) for idx, (hotkey, weight) in enumerate(sorted_miners)
    ]


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

    text_top_miners = _get_top_ranked_miners(text_tournament_weights, text_base_winner_hotkey, limit=5)
    image_top_miners = _get_top_ranked_miners(image_tournament_weights, image_base_winner_hotkey, limit=5)

    return TournamentWeightsResponse(
        burn_data=burn_data,
        text_top_miners=text_top_miners,
        image_top_miners=image_top_miners,
    )


def factory_router():
    return router

