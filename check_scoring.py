#!/usr/bin/env python3
"""
Simple script to check scoring for a hotkey using existing codebase functions
"""
import sys
import asyncio
from datetime import datetime, timedelta

# Import all the scoring logic from the codebase
from validator.core.config import Config, load_config
from validator.core.weight_setting import _get_weights_to_set, get_node_weights_from_period_scores
from validator.db.sql.nodes import get_all_nodes
from fiber.chain import fetch_nodes
from validator.utils.logging import get_logger
from validator.utils.util import try_db_connections

logger = get_logger(__name__)

async def check_hotkey_scoring(hotkey: str):
    """Check the scoring for a specific hotkey"""
    
    # Load config
    config = load_config()
    
    # Connect to database
    await try_db_connections(config)
    
    print(f"\nChecking scoring for hotkey: {hotkey}")
    print("="*80)
    
    # Get weights calculation
    print("\nCalculating weights...")
    period_scores, task_results = await _get_weights_to_set(config)
    
    # Find scores for our hotkey
    hotkey_scores = [score for score in period_scores if score.hotkey == hotkey]
    
    if not hotkey_scores:
        print(f"No scores found for hotkey {hotkey}")
        return
    
    # Get node weights
    all_node_ids, all_node_weights = await get_node_weights_from_period_scores(
        config.substrate, config.netuid, period_scores
    )
    
    # Get all nodes to map hotkey to node_id
    all_nodes = fetch_nodes.get_nodes_for_netuid(config.substrate, config.netuid)
    hotkey_to_node = {node.hotkey: node for node in all_nodes}
    
    target_node = hotkey_to_node.get(hotkey)
    if not target_node:
        print(f"Node not found for hotkey {hotkey}")
        return
    
    # Display results
    print(f"\nNode ID: {target_node.node_id}")
    print(f"Current chain weight (raw): {target_node.incentive}")
    print(f"Current chain weight (normalized): {target_node.incentive / 65535:.6f}")
    
    # Analyze task results by period and type
    print(f"\nTask breakdown by period and type for {hotkey}:")
    
    from datetime import datetime, timedelta, timezone
    from collections import defaultdict
    
    periods = {
        "7-day": (datetime.now(timezone.utc) - timedelta(days=7), 0.4),
        "3-day": (datetime.now(timezone.utc) - timedelta(days=3), 0.3),
        "1-day": (datetime.now(timezone.utc) - timedelta(days=1), 0.3)
    }
    
    task_type_weights = {
        'INSTRUCTTEXTTASK': 0.2,
        'IMAGETASK': 0.325,
        'DPOTASK': 0.225,
        'GRPOTASK': 0.25
    }
    
    # Collect scores by period and task type
    for period_name, (cutoff, period_weight) in periods.items():
        print(f"\n  {period_name} (weight: {period_weight:.0%}):")
        
        period_has_tasks = False
        for task_type, type_weight in task_type_weights.items():
            # Find tasks for this period and type
            type_scores = []
            organic_scores = []
            synthetic_scores = []
            
            for task_result in task_results:
                if task_result.task.created_at > cutoff and str(task_result.task.task_type) == f"TaskType.{task_type}":
                    for node_score in task_result.node_scores:
                        if node_score.hotkey == hotkey:
                            type_scores.append(node_score.quality_score)
                            if task_result.task.is_organic:
                                organic_scores.append(node_score.quality_score)
                            else:
                                synthetic_scores.append(node_score.quality_score)
                            break
            
            if type_scores:
                period_has_tasks = True
                avg_score = sum(type_scores) / len(type_scores)
                positive = sum(1 for s in type_scores if s > 0)
                negative = sum(1 for s in type_scores if s < 0)
                
                # Collect task IDs
                task_ids = []
                for task_result in task_results:
                    if task_result.task.created_at > cutoff and str(task_result.task.task_type) == f"TaskType.{task_type}":
                        for node_score in task_result.node_scores:
                            if node_score.hotkey == hotkey:
                                task_ids.append(str(task_result.task.task_id))
                                break
                
                print(f"    {task_type} (weight: {type_weight:.1%}):")
                print(f"      Tasks: {len(type_scores)} ({len(organic_scores)} organic, {len(synthetic_scores)} synthetic)")
                print(f"      Scores: {positive} positive, {negative} negative")
                print(f"      Average: {avg_score:.3f}")
                print(f"      Task IDs: {task_ids}")
        
        if not period_has_tasks:
            print(f"    No tasks found")
    
    # Show the normalized scores
    print(f"\nNormalized period scores (from weight calculation):")
    total_weighted_score = 0
    for i, score in enumerate(hotkey_scores):
        weighted = score.normalised_score * score.weight_multiplier if score.normalised_score else 0
        total_weighted_score += weighted
        
        norm_str = f"{score.normalised_score:.3f}" if score.normalised_score is not None else "None"
        print(f"  Score {i+1}: avg={score.average_score:6.3f}, norm={norm_str}, mult={score.weight_multiplier:.4f}, contrib={weighted:.6f}")
    
    # Get calculated weight
    calculated_weight = all_node_weights[target_node.node_id]
    
    # Calculate sum of all weights
    total_weight_sum = sum(all_node_weights)
    
    print(f"\nTotal weighted score: {total_weighted_score:.6f}")
    print(f"Calculated weight: {calculated_weight:.6f}")
    print(f"Sum of ALL node weights: {total_weight_sum:.6f}")
    print(f"This node's share: {calculated_weight/total_weight_sum:.6f} ({calculated_weight/total_weight_sum*100:.2f}%)")
    
    # Convert chain weight from raw to normalized
    chain_weight_normalized = target_node.incentive / 65535
    print(f"Current chain weight (raw): {target_node.incentive}")
    print(f"Current chain weight (normalized): {chain_weight_normalized:.6f}")
    print(f"Difference: {(calculated_weight - chain_weight_normalized):.6f} ({((calculated_weight - chain_weight_normalized)/chain_weight_normalized*100 if chain_weight_normalized > 0 else 0):+.1f}%)")
    
    # Show some task results for this hotkey with ranking info
    print(f"\nRecent task results with rankings:")
    hotkey_tasks = [tr for tr in task_results if any(ns.hotkey == hotkey for ns in tr.node_scores)]
    
    for task_result in hotkey_tasks[-10:]:  # Last 10 tasks
        # Get all scores for this task
        task_scores = [(ns.hotkey, ns.quality_score) for ns in task_result.node_scores]
        # Sort by score descending
        task_scores_sorted = sorted(task_scores, key=lambda x: x[1], reverse=True)
        
        # Find our node's position
        for i, (h, score) in enumerate(task_scores_sorted):
            if h == hotkey:
                rank = i + 1
                total_participants = len(task_scores_sorted)
                
                # Count scores by category
                positive_scores = sum(1 for _, s in task_scores_sorted if s > 0)
                zero_scores = sum(1 for _, s in task_scores_sorted if s == 0)
                negative_scores = sum(1 for _, s in task_scores_sorted if s < 0)
                
                print(f"\n  Task {task_result.task.task_id}:")
                print(f"    Type: {task_result.task.task_type}, Organic: {task_result.task.is_organic}")
                print(f"    Your Score: {score:.3f} (Rank: {rank}/{total_participants})")
                print(f"    Score distribution: {positive_scores} positive, {zero_scores} zero, {negative_scores} negative")
                
                # Show top 3 and bottom 3 if relevant
                if total_participants > 3:
                    print(f"    Top 3 scores: {task_scores_sorted[0][1]:.1f}, {task_scores_sorted[1][1]:.1f}, {task_scores_sorted[2][1]:.1f}")
                    if rank > 3:
                        print(f"    Bottom 3 scores: {task_scores_sorted[-3][1]:.1f}, {task_scores_sorted[-2][1]:.1f}, {task_scores_sorted[-1][1]:.1f}")
                break

async def main():
    if len(sys.argv) < 2:
        print("Usage: python check_scoring.py <hotkey>")
        sys.exit(1)
    
    hotkey = sys.argv[1]
    await check_hotkey_scoring(hotkey)

if __name__ == "__main__":
    asyncio.run(main())