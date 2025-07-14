import asyncio
from datetime import datetime, timezone, timedelta
from typing import List
import re

from fiber.chain import fetch_nodes
from fiber.chain.models import Node

from validator.core.config import Config
from validator.core.constants import (
    DEFAULT_RECENT_SUBMISSIONS_LIMIT,
    CHAIN_WEIGHT_DIVISOR,
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
from validator.core.miner_models import (
    MinerDetailsResponse,
    OnChainIncentive,
    WeightingDetails,
    TaskTypeBreakdown,
    TaskTypePerformance,
    PeriodScore,
    TaskSourcePerformance,
    PeriodTotals,
    TaskSubmissionResult,
    MinerPerformanceMetrics
)
from validator.core.weight_setting import get_miner_performance_breakdown, _get_leaderboard_data
from validator.evaluation.scoring import get_task_work_score, calculate_adjusted_task_score
from validator.utils.logging import get_logger
from core.models.utility_models import TaskType

logger = get_logger(__name__)

CACHE_UPDATE_INTERVAL = 900


async def build_miner_details_response(
    hotkey: str,
    target_node: Node,
    all_nodes: List[Node],
    task_results: list,
    weighting_details: WeightingDetails
) -> MinerDetailsResponse:
    
    total_incentive = sum(node.incentive for node in all_nodes)
    current_incentive = OnChainIncentive(
        raw_value=target_node.incentive,
        normalized=target_node.incentive / CHAIN_WEIGHT_DIVISOR,
        network_share_percent=(target_node.incentive / total_incentive) if total_incentive > 0 else 0
    )
    
    breakdown = get_miner_performance_breakdown(hotkey, task_results)
    
    task_type_map = {
        str(TaskType.INSTRUCTTEXTTASK): ("instruct_text", TaskType.INSTRUCTTEXTTASK),
        str(TaskType.DPOTASK): ("dpo", TaskType.DPOTASK),
        str(TaskType.IMAGETASK): ("image", TaskType.IMAGETASK),
        str(TaskType.GRPOTASK): ("grpo", TaskType.GRPOTASK)
    }
    
    task_type_breakdown = TaskTypeBreakdown()
    
    for task_type_str, (field_name, task_type_enum) in task_type_map.items():
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
            
            type_tasks = [tr for tr in task_results if tr.task.task_type == task_type_enum]
            miner_type_tasks = [
                tr for tr in type_tasks 
                if any(ns.hotkey == hotkey for ns in tr.node_scores)
            ]
            
            if miner_type_tasks:
                work_scores = [get_task_work_score(tr.task) for tr in miner_type_tasks]
                adjusted_scores = []
                
                for tr in miner_type_tasks:
                    work_score = get_task_work_score(tr.task)
                    miner_score = next((ns.quality_score for ns in tr.node_scores if ns.hotkey == hotkey), 0)
                    adjusted_scores.append(calculate_adjusted_task_score(miner_score, work_score))
                
                performance.average_work_score = sum(work_scores) / len(work_scores)
                performance.total_work_score = sum(work_scores)
                performance.average_adjusted_score = sum(adjusted_scores) / len(adjusted_scores)
                performance.total_adjusted_score = sum(adjusted_scores)
            
            setattr(task_type_breakdown, field_name, performance)
    
    period_totals = PeriodTotals(
        one_day_total=breakdown["period_totals"].get("one_day", 0),
        three_day_total=breakdown["period_totals"].get("three_day", 0),
        seven_day_total=breakdown["period_totals"].get("seven_day", 0)
    )
    
    recent_submissions = get_recent_submissions(hotkey, task_results)
    performance_metrics = calculate_performance_metrics(hotkey, task_results)
    
    return MinerDetailsResponse(
        hotkey=hotkey,
        node_id=target_node.node_id,
        current_incentive=current_incentive,
        weighting_details=weighting_details,
        task_type_breakdown=task_type_breakdown,
        period_totals=period_totals,
        recent_submissions=recent_submissions,
        performance_metrics=performance_metrics
    )


def get_recent_submissions(hotkey: str, task_results: list, limit: int = DEFAULT_RECENT_SUBMISSIONS_LIMIT) -> list[TaskSubmissionResult]:
    hotkey_tasks = [
        tr for tr in task_results 
        if any(ns.hotkey == hotkey for ns in tr.node_scores)
    ]
    
    hotkey_tasks.sort(key=lambda x: x.task.created_at, reverse=True)
    submissions = []
    for task_result in hotkey_tasks[:limit]:
        submitted_scores = {ns.hotkey: ns.quality_score for ns in task_result.node_scores}
        
        all_participants = []
        if task_result.task.assigned_miners:
            for node_id in task_result.task.assigned_miners:
                pass
        
        task_scores = [(hotkey, score) for hotkey, score in submitted_scores.items()]
        task_scores_sorted = sorted(task_scores, key=lambda x: x[1], reverse=True)
        
        for i, (h, score) in enumerate(task_scores_sorted):
            if h == hotkey:
                rank = i + 1
                total_participants = len(task_scores_sorted)
                percentile = ((total_participants - rank) / total_participants) if total_participants > 0 else 0
                
                work_score = get_task_work_score(task_result.task)
                adjusted_score = calculate_adjusted_task_score(score, work_score)
                
                model_size_billions = (
                    min(40, max(1, task_result.task.model_params_count // 1_000_000_000))
                    if getattr(task_result.task, "model_params_count", 0) > 0
                    else min(8, int(model_size.group(1)) if (model_size := re.search(r"(\d+)(?=[bB])", task_result.task.model_id)) else 1)
                )
                
                submissions.append(TaskSubmissionResult(
                    task_id=task_result.task.task_id,
                    task_type=task_result.task.task_type,
                    is_organic=task_result.task.is_organic,
                    created_at=task_result.task.created_at,
                    score=score,
                    rank=rank,
                    total_participants=total_participants,
                    percentile=percentile,
                    work_score=work_score,
                    adjusted_score=adjusted_score,
                    hours_to_complete=task_result.task.hours_to_complete,
                    model_size_billions=float(model_size_billions)
                ))
                break
    return submissions


def calculate_performance_metrics(hotkey: str, task_results: list) -> MinerPerformanceMetrics:
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
    positive_scores = sum(1 for s in all_scores if s > -1)
    positive_score_rate = (positive_scores / len(all_scores)) if all_scores else 0
    
    percentiles = []
    work_scores = []
    adjusted_scores = []
    
    for tr in hotkey_tasks:
        submitted_scores = {ns.hotkey: ns.quality_score for ns in tr.node_scores}
        
        task_scores = [(hotkey, score) for hotkey, score in submitted_scores.items()]
        task_scores_sorted = sorted(task_scores, key=lambda x: x[1], reverse=True)
        
        work_score = get_task_work_score(tr.task)
        work_scores.append(work_score)
        
        for i, (h, score) in enumerate(task_scores_sorted):
            if h == hotkey:
                rank = i + 1
                total = len(task_scores_sorted)
                percentile = ((total - rank) / total) if total > 0 else 0
                percentiles.append(percentile)
                adjusted_scores.append(calculate_adjusted_task_score(score, work_score))
                break
    
    avg_percentile = sum(percentiles) / len(percentiles) if percentiles else 0
    avg_work_score = sum(work_scores) / len(work_scores) if work_scores else 0
    total_work_score = sum(work_scores)
    avg_adjusted_score = sum(adjusted_scores) / len(adjusted_scores) if adjusted_scores else 0
    total_adjusted_score = sum(adjusted_scores)
    
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
        average_work_score=avg_work_score,
        total_work_score=total_work_score,
        average_adjusted_score=avg_adjusted_score,
        total_adjusted_score=total_adjusted_score,
        task_type_distribution=task_distribution
    )


async def cache_all_miner_performance(config: Config) -> None:
    logger.info("Starting miner performance cache update")
    
    try:
        all_nodes: List[Node] = fetch_nodes.get_nodes_for_netuid(config.substrate, config.netuid)
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
        
        cached_count = 0
        for node in all_nodes:
            try:
                response = await build_miner_details_response(
                    node.hotkey, node, all_nodes, task_results, weighting_details
                )
                
                cache_key = f"{MINER_PERFORMANCE_CACHE_KEY_PREFIX}{node.hotkey}"
                await config.redis_db.set(cache_key, response.model_dump_json(), ex=MINER_PERFORMANCE_CACHE_TTL)
                cached_count += 1
                
                if cached_count % 50 == 0:
                    logger.info(f"Cached performance data for {cached_count} miners")
                    
            except Exception as e:
                logger.error(f"Failed to cache performance data for hotkey {node.hotkey}: {e}")
        
        logger.info(f"Successfully cached performance data for {cached_count}/{len(all_nodes)} miners")
        
    except Exception as e:
        logger.error(f"Failed to update miner performance cache: {e}")


async def miner_performance_cache_worker(config: Config) -> None:
    logger.info("Starting miner performance cache worker")
    
    while True:
        try:
            await cache_all_miner_performance(config)
            logger.info(f"Sleeping for {CACHE_UPDATE_INTERVAL} seconds until next cache update")
            await asyncio.sleep(CACHE_UPDATE_INTERVAL)
        except Exception as e:
            logger.error(f"Error in miner performance cache worker: {e}")
            await asyncio.sleep(300)