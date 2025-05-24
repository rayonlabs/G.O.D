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

logger = get_logger(__name__)

async def check_hotkey_scoring(hotkey: str):
    """Check the scoring for a specific hotkey"""
    
    # Load config
    config = load_config()
    
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
    print(f"Current chain weight: {target_node.incentive:.6f}")
    
    # Show period scores breakdown
    print(f"\nPeriod scores for {hotkey}:")
    total_weighted_score = 0
    for score in hotkey_scores:
        weighted = score.normalised_score * score.weight_multiplier if score.normalised_score else 0
        total_weighted_score += weighted
        print(f"  Average: {score.average_score:.3f}, "
              f"Normalized: {score.normalised_score:.3f} if score.normalised_score else 'None', "
              f"Weight multiplier: {score.weight_multiplier:.3f}, "
              f"Weighted contribution: {weighted:.6f}")
    
    # Get calculated weight
    calculated_weight = all_node_weights[target_node.node_id]
    
    print(f"\nTotal weighted score: {total_weighted_score:.6f}")
    print(f"Calculated weight: {calculated_weight:.6f}")
    print(f"Current chain weight: {target_node.incentive:.6f}")
    print(f"Difference: {(calculated_weight - target_node.incentive):.6f}")
    
    # Show some task results for this hotkey
    print(f"\nRecent task results:")
    hotkey_tasks = [tr for tr in task_results if any(r.hotkey == hotkey for r in tr.results)]
    
    for task_result in hotkey_tasks[-10:]:  # Last 10 tasks
        for result in task_result.results:
            if result.hotkey == hotkey:
                print(f"  Task {task_result.task.task_id}: "
                      f"Type={task_result.task.task_type}, "
                      f"Organic={task_result.task.is_organic}, "
                      f"Score={result.score}, "
                      f"Reason={result.score_reason}")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python check_scoring.py <hotkey>")
        sys.exit(1)
    
    hotkey = sys.argv[1]
    await check_hotkey_scoring(hotkey)

if __name__ == "__main__":
    asyncio.run(main())