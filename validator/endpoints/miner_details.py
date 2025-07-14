from fastapi import APIRouter, Depends, HTTPException
from fiber.chain import fetch_nodes
from fiber.chain.models import Node

from validator.core.config import Config
from validator.core.constants import (
    MINER_PERFORMANCE_CACHE_TTL,
    MINER_PERFORMANCE_CACHE_KEY_PREFIX,
    ONE_DAY_SCORE_WEIGHT,
    THREE_DAY_SCORE_WEIGHT,
    SEVEN_DAY_SCORE_WEIGHT,
    INSTRUCT_TEXT_TASK_SCORE_WEIGHT,
    DPO_TASK_SCORE_WEIGHT,
    IMAGE_TASK_SCORE_WEIGHT,
    GRPO_TASK_SCORE_WEIGHT,
)
from validator.core.dependencies import get_config
from validator.core.miner_models import MinerDetailsResponse, WeightingDetails
from validator.core.weight_setting import _get_leaderboard_data
from validator.utils.logging import get_logger
from validator.utils.miner_analytics import build_miner_details_response

logger = get_logger(__name__)

MINER_DETAILS_ENDPOINT = "/miner/details/{hotkey}"


async def get_miner_details(
    hotkey: str,
    config: Config = Depends(get_config)
) -> MinerDetailsResponse:
    
    cache_key = f"{MINER_PERFORMANCE_CACHE_KEY_PREFIX}{hotkey}"
    
    cached_data = await config.redis_db.get(cache_key)
    if cached_data:
        logger.info(f"Returning cached data for hotkey {hotkey}")
        return MinerDetailsResponse.model_validate_json(cached_data)
    
    all_nodes = fetch_nodes.get_nodes_for_netuid(config.substrate, config.netuid)
    target_node = next((node for node in all_nodes if node.hotkey == hotkey), None)
    if not target_node:
        raise HTTPException(status_code=404, detail=f"Node not found for hotkey {hotkey}")
    
    period_scores, task_results = await _get_leaderboard_data(config)
    
    weighting_details = WeightingDetails(
        one_day_weight=ONE_DAY_SCORE_WEIGHT,
        three_day_weight=THREE_DAY_SCORE_WEIGHT,
        seven_day_weight=SEVEN_DAY_SCORE_WEIGHT,
        instruct_text_weight=INSTRUCT_TEXT_TASK_SCORE_WEIGHT,
        dpo_weight=DPO_TASK_SCORE_WEIGHT,
        image_weight=IMAGE_TASK_SCORE_WEIGHT,
        grpo_weight=GRPO_TASK_SCORE_WEIGHT
    )
    
    response = await build_miner_details_response(
        hotkey, target_node, all_nodes, task_results, weighting_details
    )
    
    await config.redis_db.set(cache_key, response.model_dump_json(), ex=MINER_PERFORMANCE_CACHE_TTL)
    logger.info(f"Cached performance data for hotkey {hotkey}")
    
    return response



def factory_router() -> APIRouter:
    router = APIRouter()
    
    router.add_api_route(
        MINER_DETAILS_ENDPOINT,
        get_miner_details,
        response_model=MinerDetailsResponse,
        tags=["miner_performance"],
        methods=["GET"],
    )
    
    return router