#!/usr/bin/env python3
"""
Script to identify image vs text miners and compare their weight distributions
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator.core.weight_setting import _get_weights_to_set, get_node_weights_from_period_scores
from validator.core.constants import IMAGE_TASK_SCORE_WEIGHT, INSTRUCT_TEXT_TASK_SCORE_WEIGHT
from validator.db.src.sql.scores import ScoreTable
from validator.db.src.database import get_database
from validator.core.config import Config
from collections import defaultdict
from datetime import datetime, timedelta

async def check_image_vs_text_weights():
    """Check actual weight distribution between image and text miners"""
    
    print("=== Image vs Text Miner Weight Analysis ===")
    print(f"Expected IMAGE_TASK_SCORE_WEIGHT: {IMAGE_TASK_SCORE_WEIGHT} (25%)")
    print(f"Expected INSTRUCT_TEXT_TASK_SCORE_WEIGHT: {INSTRUCT_TEXT_TASK_SCORE_WEIGHT} (40%)")
    print()
    
    # Get current weights using the actual weight setting functions
    config = Config()
    period_scores, task_results = await _get_weights_to_set(config)
    
    print(f"Found {len(period_scores)} period scores")
    print(f"Found {len(task_results)} task results")
    
    # Convert period scores to weights
    all_node_ids, all_node_weights = await get_node_weights_from_period_scores(
        config.substrate, config.netuid, period_scores
    )
    
    # Create hotkey to weight mapping
    from validator.core.fetch_nodes import get_nodes_for_netuid
    all_nodes = get_nodes_for_netuid(config.substrate, config.netuid)
    node_id_to_hotkey = {node.node_id: node.hotkey for node in all_nodes}
    
    weights = {}
    for node_id, weight in zip(all_node_ids, all_node_weights):
        if node_id in node_id_to_hotkey:
            weights[node_id_to_hotkey[node_id]] = weight
    
    total_weight = sum(weights.values())
    print(f"Total miners with weights: {len(weights)}")
    print(f"Total weight distributed: {total_weight:.6f}")
    print()
    
    # Get recent tasks to categorize miners
    db = get_database()
    cutoff = datetime.now() - timedelta(days=7)  # Last 7 days
    
    with db.session() as session:
        # Get all scores from last 7 days
        scores = session.query(ScoreTable).filter(
            ScoreTable.created_at >= cutoff
        ).all()
        
        print(f"Found {len(scores)} scores in last 7 days")
        
        # Categorize miners by task type
        miner_task_counts = defaultdict(lambda: defaultdict(int))
        
        for score in scores:
            hotkey = score.hotkey
            task_type = score.task_type
            miner_task_counts[hotkey][task_type] += 1
        
        print(f"Found {len(miner_task_counts)} unique miners in recent tasks")
        
        # Classify miners as image vs text based on their primary task type
        image_miners = set()
        text_miners = set()
        other_miners = set()
        
        for hotkey, task_counts in miner_task_counts.items():
            # Find the most common task type for this miner
            primary_task = max(task_counts.items(), key=lambda x: x[1])[0]
            
            if primary_task == 'image':
                image_miners.add(hotkey)
            elif primary_task in ['instruct', 'text']:
                text_miners.add(hotkey)
            else:
                other_miners.add(hotkey)
        
        print(f"\nMiner classification:")
        print(f"  Image miners: {len(image_miners)}")
        print(f"  Text miners: {len(text_miners)}")
        print(f"  Other miners: {len(other_miners)}")
        
        # Calculate weight sums
        image_weight = sum(weights.get(hotkey, 0) for hotkey in image_miners)
        text_weight = sum(weights.get(hotkey, 0) for hotkey in text_miners)
        other_weight = sum(weights.get(hotkey, 0) for hotkey in other_miners)
        
        # Account for miners with weights but no recent tasks
        miners_with_weights = set(weights.keys())
        miners_with_tasks = set(miner_task_counts.keys())
        inactive_miners = miners_with_weights - miners_with_tasks
        inactive_weight = sum(weights.get(hotkey, 0) for hotkey in inactive_miners)
        
        print(f"\n=== Weight Distribution ===")
        print(f"Image miners weight: {image_weight:.6f} ({image_weight/total_weight*100:.2f}%)")
        print(f"Text miners weight: {text_weight:.6f} ({text_weight/total_weight*100:.2f}%)")
        print(f"Other miners weight: {other_weight:.6f} ({other_weight/total_weight*100:.2f}%)")
        print(f"Inactive miners weight: {inactive_weight:.6f} ({inactive_weight/total_weight*100:.2f}%)")
        print(f"Total: {image_weight + text_weight + other_weight + inactive_weight:.6f}")
        
        print(f"\n=== Comparison to Expected ===")
        expected_image_pct = IMAGE_TASK_SCORE_WEIGHT * 100
        actual_image_pct = image_weight/total_weight*100
        print(f"Expected image percentage: {expected_image_pct:.1f}%")
        print(f"Actual image percentage: {actual_image_pct:.2f}%")
        print(f"Difference: {actual_image_pct - expected_image_pct:.2f} percentage points")
        
        if actual_image_pct > expected_image_pct:
            print(f"⚠️  Image miners are getting {actual_image_pct - expected_image_pct:.2f}% MORE than expected!")
        else:
            print(f"✅ Image miners are within expected range")
        
        # Show top miners by category
        print(f"\n=== Top Miners by Category ===")
        
        if image_miners:
            print("Top image miners:")
            image_weights = [(hotkey, weights.get(hotkey, 0)) for hotkey in image_miners]
            image_weights.sort(key=lambda x: x[1], reverse=True)
            for i, (hotkey, weight) in enumerate(image_weights[:5]):
                print(f"  {i+1}. {hotkey}: {weight:.6f} ({weight/total_weight*100:.2f}%)")
        
        if text_miners:
            print("\nTop text miners:")
            text_weights = [(hotkey, weights.get(hotkey, 0)) for hotkey in text_miners]
            text_weights.sort(key=lambda x: x[1], reverse=True)
            for i, (hotkey, weight) in enumerate(text_weights[:5]):
                print(f"  {i+1}. {hotkey}: {weight:.6f} ({weight/total_weight*100:.2f}%)")
        
        # Show task type distribution
        print(f"\n=== Task Type Distribution ===")
        task_type_counts = defaultdict(int)
        for task_counts in miner_task_counts.values():
            for task_type, count in task_counts.items():
                task_type_counts[task_type] += count
        
        total_tasks = sum(task_type_counts.values())
        for task_type, count in sorted(task_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {task_type}: {count} tasks ({count/total_tasks*100:.1f}%)")

if __name__ == "__main__":
    check_image_vs_text_weights()