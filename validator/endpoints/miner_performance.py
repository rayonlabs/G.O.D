from datetime import datetime, timedelta, timezone
import json

from fastapi import APIRouter, Depends, HTTPException
from fiber.chain import fetch_nodes
from fiber.chain.models import Node

from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.core.miner_models import (
    MinerDetailsResponse,
    OnChainIncentive,
    CalculatedPerformanceWeight,
    TaskTypeBreakdown,
    TaskTypePerformance,
    PeriodScore,
    TaskSourcePerformance,
    PeriodTotals,
    TaskSubmissionResult,
    MinerPerformanceMetrics
)
from validator.core.weight_setting import (
    _get_weights_to_set, 
    get_node_weights_from_period_scores,
    get_miner_performance_breakdown
)
from validator.utils.logging import get_logger
from core.models.utility_models import TaskType

logger = get_logger(__name__)

CACHE_TTL = 3600  # 1 hour in seconds
CACHE_KEY_PREFIX = "miner_performance:"


async def get_miner_details(
    hotkey: str,
    config: Config = Depends(get_config)
) -> MinerDetailsResponse:
    """Get detailed information about a miner's performance and weights"""
    
    cache_key = f"{CACHE_KEY_PREFIX}{hotkey}"
    
    cached_data = await config.redis_db.get(cache_key)
    if cached_data:
        logger.info(f"Returning cached data for hotkey {hotkey}")
        return MinerDetailsResponse.model_validate_json(cached_data)
    
    all_nodes: list[Node] = fetch_nodes.get_nodes_for_netuid(config.substrate, config.netuid)
    hotkey_to_node = {node.hotkey: node for node in all_nodes}
    
    target_node = hotkey_to_node.get(hotkey)
    if not target_node:
        raise HTTPException(status_code=404, detail=f"Node not found for hotkey {hotkey}")
    
    period_scores, task_results = await _get_weights_to_set(config)
    all_node_ids, all_node_weights = await get_node_weights_from_period_scores(
        config.substrate, config.netuid, period_scores
    )
    
    total_incentive = sum(node.incentive for node in all_nodes)
    current_incentive = OnChainIncentive(
        raw_value=target_node.incentive,
        normalized=target_node.incentive / 65535,
        network_share_percent=(target_node.incentive / total_incentive * 100) if total_incentive > 0 else 0
    )
    
    calculated_weight = all_node_weights[target_node.node_id]
    total_weight_sum = sum(all_node_weights)
    performance_weight = CalculatedPerformanceWeight(
        weight_value=calculated_weight,
        network_share_percent=(calculated_weight / total_weight_sum * 100) if total_weight_sum > 0 else 0
    )
    
    breakdown = get_miner_performance_breakdown(hotkey, task_results)
    
    task_type_map = {
        str(TaskType.INSTRUCTTEXTTASK): "instruct_text",
        str(TaskType.DPOTASK): "dpo",
        str(TaskType.IMAGETASK): "image",
        str(TaskType.GRPOTASK): "grpo"
    }
    
    task_type_breakdown = TaskTypeBreakdown()
    
    for task_type_str, field_name in task_type_map.items():
        if task_type_str in breakdown["task_types"]:
            type_data = breakdown["task_types"][task_type_str]
            performance = TaskTypePerformance(
                weight_contribution=type_data["task_weight"],
                total_submissions=type_data["total_organic_tasks"] + type_data["total_synthetic_tasks"]
            )
            
            for period_name in ["one_day", "three_day", "seven_day"]:
                period_data = type_data["periods"][period_name]
                organic_data = period_data["organic"]
                synth_data = period_data["synthetic"]
                
                combined_score = PeriodScore()
                score_count = 0
                
                if organic_data["score"]:
                    combined_score.average_score += organic_data["score"].average_score
                    combined_score.normalized_score += organic_data["score"].normalised_score or 0
                    combined_score.weight_multiplier += organic_data["score"].weight_multiplier
                    combined_score.weighted_contribution += organic_data["weighted_contribution"]
                    score_count += 1
                
                if synth_data["score"]:
                    combined_score.average_score += synth_data["score"].average_score
                    combined_score.normalized_score += synth_data["score"].normalised_score or 0
                    combined_score.weight_multiplier += synth_data["score"].weight_multiplier
                    combined_score.weighted_contribution += synth_data["weighted_contribution"]
                    score_count += 1
                
                if score_count > 0:
                    combined_score.average_score /= score_count
                
                setattr(performance, period_name, combined_score)
            
            performance.organic_performance = TaskSourcePerformance(
                task_count=type_data["total_organic_tasks"]
            )
            performance.synthetic_performance = TaskSourcePerformance(
                task_count=type_data["total_synthetic_tasks"]
            )
            
            if breakdown["all_scores"]:
                task_scores = [s for s in breakdown["all_scores"] if s]
                if task_scores:
                    performance.average_score = sum(s.average_score for s in task_scores) / len(task_scores)
            
            setattr(task_type_breakdown, field_name, performance)
    
    period_totals = PeriodTotals(
        one_day_total=breakdown["period_totals"].get("one_day", 0),
        three_day_total=breakdown["period_totals"].get("three_day", 0),
        seven_day_total=breakdown["period_totals"].get("seven_day", 0)
    )
    
    recent_submissions = get_recent_submissions(hotkey, task_results, limit=20)
    performance_metrics = calculate_performance_metrics(hotkey, task_results)
    
    response = MinerDetailsResponse(
        hotkey=hotkey,
        node_id=target_node.node_id,
        current_incentive=current_incentive,
        performance_weight=performance_weight,
        task_type_breakdown=task_type_breakdown,
        period_totals=period_totals,
        recent_submissions=recent_submissions,
        performance_metrics=performance_metrics
    )
    
    await config.redis_db.set(cache_key, response.model_dump_json(), ex=CACHE_TTL)
    logger.info(f"Cached performance data for hotkey {hotkey}")
    
    return response


def get_recent_submissions(hotkey: str, task_results: list, limit: int = 20) -> list[TaskSubmissionResult]:
    """Get recent task submissions for the hotkey"""
    hotkey_tasks = [
        tr for tr in task_results 
        if any(ns.hotkey == hotkey for ns in tr.node_scores)
    ]
    
    hotkey_tasks.sort(key=lambda x: x.task.created_at, reverse=True)
    
    submissions = []
    for task_result in hotkey_tasks[:limit]:
        task_scores = [(ns.hotkey, ns.quality_score) for ns in task_result.node_scores]
        task_scores_sorted = sorted(task_scores, key=lambda x: x[1], reverse=True)
        
        for i, (h, score) in enumerate(task_scores_sorted):
            if h == hotkey:
                rank = i + 1
                total_participants = len(task_scores_sorted)
                percentile = ((total_participants - rank) / total_participants * 100) if total_participants > 0 else 0
                
                submissions.append(TaskSubmissionResult(
                    task_id=task_result.task.task_id,
                    task_type=task_result.task.task_type,
                    is_organic=task_result.task.is_organic,
                    created_at=task_result.task.created_at,
                    score=score,
                    rank=rank,
                    total_participants=total_participants,
                    percentile=percentile
                ))
                break
    
    return submissions


def calculate_performance_metrics(hotkey: str, task_results: list) -> MinerPerformanceMetrics:
    """Calculate overall performance metrics"""
    hotkey_tasks = [
        tr for tr in task_results 
        if any(ns.hotkey == hotkey for ns in tr.node_scores)
    ]
    
    now = datetime.now(timezone.utc)
    tasks_24h = [tr for tr in hotkey_tasks if tr.task.created_at > now - timedelta(days=1)]
    tasks_7d = [tr for tr in hotkey_tasks if tr.task.created_at > now - timedelta(days=7)]
    
    all_scores = [
        ns.quality_score for tr in hotkey_tasks 
        for ns in tr.node_scores if ns.hotkey == hotkey
    ]
    positive_scores = sum(1 for s in all_scores if s > 0)
    positive_score_rate = (positive_scores / len(all_scores) * 100) if all_scores else 0
    
    percentiles = []
    for tr in hotkey_tasks:
        task_scores = [(ns.hotkey, ns.quality_score) for ns in tr.node_scores]
        task_scores_sorted = sorted(task_scores, key=lambda x: x[1], reverse=True)
        
        for i, (h, _) in enumerate(task_scores_sorted):
            if h == hotkey:
                rank = i + 1
                total = len(task_scores_sorted)
                percentile = ((total - rank) / total * 100) if total > 0 else 0
                percentiles.append(percentile)
                break
    
    avg_percentile = sum(percentiles) / len(percentiles) if percentiles else 0
    
    task_counts = {}
    for task_type in TaskType:
        count = sum(1 for tr in hotkey_tasks if tr.task.task_type == task_type)
        task_counts[task_type.value] = count
    
    total_tasks = sum(task_counts.values())
    task_distribution = {
        k: (v / total_tasks * 100) if total_tasks > 0 else 0 
        for k, v in task_counts.items()
    }
    
    return MinerPerformanceMetrics(
        total_tasks_participated=len(hotkey_tasks),
        tasks_last_24h=len(tasks_24h),
        tasks_last_7d=len(tasks_7d),
        positive_score_rate=positive_score_rate,
        average_percentile_rank=avg_percentile,
        task_type_distribution=task_distribution
    )


def factory_router() -> APIRouter:
    router = APIRouter()
    
    router.add_api_route(
        "/miner/details/{hotkey}",
        get_miner_details,
        response_model=MinerDetailsResponse,
        tags=["miners"],
        methods=["GET"],
    )
    
    return router