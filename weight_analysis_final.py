#!/usr/bin/env python3
"""
Script to analyze actual weight distribution between image and text miners
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator.core.config import load_config
from validator.core.weight_setting import _get_weights_to_set, get_node_weights_from_period_scores
from fiber.chain import fetch_nodes
from validator.core.constants import IMAGE_TASK_SCORE_WEIGHT, INSTRUCT_TEXT_TASK_SCORE_WEIGHT

async def analyze_weight_distribution():
    """Analyze actual weight distribution between image and text miners"""
    
    print("=== Weight Distribution Analysis ===")
    print(f"Expected IMAGE_TASK_SCORE_WEIGHT: {IMAGE_TASK_SCORE_WEIGHT} ({IMAGE_TASK_SCORE_WEIGHT*100:.1f}%)")
    print(f"Expected INSTRUCT_TEXT_TASK_SCORE_WEIGHT: {INSTRUCT_TEXT_TASK_SCORE_WEIGHT} ({INSTRUCT_TEXT_TASK_SCORE_WEIGHT*100:.1f}%)")
    print()
    
    try:
        # Get the actual weights using the same method as the validator
        config = load_config()
        
        # Connect to the database
        await config.psql_db.connect()
        
        period_scores, task_results = await _get_weights_to_set(config)
        
        print(f"Found {len(period_scores)} period scores")
        print(f"Found {len(task_results)} task results")
        
        # Convert to actual node weights
        all_node_ids, all_node_weights = await get_node_weights_from_period_scores(
            config.substrate, config.netuid, period_scores
        )
        
        # Get node mapping
        all_nodes = fetch_nodes.get_nodes_for_netuid(config.substrate, config.netuid)
        node_id_to_hotkey = {node.node_id: node.hotkey for node in all_nodes}
        
        # Create hotkey to weight mapping
        weights = {}
        for node_id, weight in zip(all_node_ids, all_node_weights):
            if node_id in node_id_to_hotkey:
                weights[node_id_to_hotkey[node_id]] = weight
        
        total_weight = sum(weights.values())
        print(f"Total weight distributed: {total_weight:.6f}")
        print(f"Active miners: {len(weights)}")
        print()
        
        # Analyze task results to categorize miners
        miner_task_counts = defaultdict(lambda: defaultdict(int))
        
        for task_result in task_results:
            task_type = task_result.task.task_type
            for node_score in task_result.node_scores:
                hotkey = node_score.hotkey
                miner_task_counts[hotkey][task_type] += 1
        
        print(f"Miners with recent tasks: {len(miner_task_counts)}")
        
        # Analyze miner specialization
        print(f"\n=== Miner Specialization Analysis ===")
        multi_task_miners = 0
        specialized_miners = 0
        task_type_breakdown = defaultdict(lambda: defaultdict(int))
        
        for hotkey, task_counts in miner_task_counts.items():
            if len(task_counts) > 1:
                multi_task_miners += 1
                print(f"Multi-task miner {hotkey}: {dict(task_counts)}")
            else:
                specialized_miners += 1
            
            # Track what each miner does
            for task_type, count in task_counts.items():
                task_type_breakdown[task_type][hotkey] = count
        
        print(f"Multi-task miners: {multi_task_miners}")
        print(f"Specialized miners: {specialized_miners}")
        
        # Show task type breakdown
        print(f"\n=== Task Type Breakdown by Miners ===")
        for task_type, miners in task_type_breakdown.items():
            print(f"{task_type}: {len(miners)} unique miners")
            # Show top miners for this task type
            top_miners = sorted(miners.items(), key=lambda x: x[1], reverse=True)[:3]
            for hotkey, count in top_miners:
                print(f"  {hotkey}: {count} tasks")
        
        # Classify miners by their primary task type
        image_miners = set()
        text_miners = set()
        dpo_miners = set()
        grpo_miners = set()
        other_miners = set()
        
        print(f"\n=== Miner Classification Details ===")
        for hotkey, task_counts in miner_task_counts.items():
            # Find the most common task type for this miner
            if not task_counts:
                continue
                
            primary_task = max(task_counts.items(), key=lambda x: x[1])[0]
            
            # Show classification reasoning for first few miners
            if len(image_miners) + len(text_miners) + len(dpo_miners) + len(grpo_miners) + len(other_miners) < 10:
                print(f"Miner {hotkey}: {dict(task_counts)} -> classified as {primary_task}")
            
            if primary_task == 'ImageTask':
                image_miners.add(hotkey)
            elif primary_task in ['InstructTextTask', 'ChatTask']:
                text_miners.add(hotkey)
            elif primary_task == 'DpoTask':
                dpo_miners.add(hotkey)
            elif primary_task == 'GrpoTask':
                grpo_miners.add(hotkey)
            else:
                other_miners.add(hotkey)
        
        # Create combined text miners (instruct + dpo + grpo + chat)
        all_text_miners = text_miners | dpo_miners | grpo_miners
        
        print(f"\nMiner Classification:")
        print(f"  Image miners: {len(image_miners)}")
        print(f"  Instruct/Chat miners: {len(text_miners)}")
        print(f"  DPO miners: {len(dpo_miners)}")
        print(f"  GRPO miners: {len(grpo_miners)}")
        print(f"  Combined text miners: {len(all_text_miners)}")
        print(f"  Other miners: {len(other_miners)}")
        
        # Calculate weight distribution
        image_weight = sum(weights.get(hotkey, 0) for hotkey in image_miners)
        text_weight = sum(weights.get(hotkey, 0) for hotkey in text_miners)
        dpo_weight = sum(weights.get(hotkey, 0) for hotkey in dpo_miners)
        grpo_weight = sum(weights.get(hotkey, 0) for hotkey in grpo_miners)
        combined_text_weight = sum(weights.get(hotkey, 0) for hotkey in all_text_miners)
        other_weight = sum(weights.get(hotkey, 0) for hotkey in other_miners)
        
        # Weight for miners without recent tasks
        inactive_miners = set(weights.keys()) - set(miner_task_counts.keys())
        inactive_weight = sum(weights.get(hotkey, 0) for hotkey in inactive_miners)
        
        print(f"\n=== Actual Weight Distribution (Individual) ===")
        print(f"Image miners: {image_weight:.6f} ({image_weight/total_weight*100:.2f}%)")
        print(f"Instruct/Chat miners: {text_weight:.6f} ({text_weight/total_weight*100:.2f}%)")
        print(f"DPO miners: {dpo_weight:.6f} ({dpo_weight/total_weight*100:.2f}%)")
        print(f"GRPO miners: {grpo_weight:.6f} ({grpo_weight/total_weight*100:.2f}%)")
        print(f"Other miners: {other_weight:.6f} ({other_weight/total_weight*100:.2f}%)")
        print(f"Inactive miners: {inactive_weight:.6f} ({inactive_weight/total_weight*100:.2f}%)")
        
        print(f"\n=== Combined Weight Distribution ===")
        print(f"Image miners: {image_weight:.6f} ({image_weight/total_weight*100:.2f}%)")
        print(f"All text miners: {combined_text_weight:.6f} ({combined_text_weight/total_weight*100:.2f}%)")
        print(f"Other miners: {other_weight:.6f} ({other_weight/total_weight*100:.2f}%)")
        print(f"Inactive miners: {inactive_weight:.6f} ({inactive_weight/total_weight*100:.2f}%)")
        print(f"Total: {image_weight + combined_text_weight + other_weight + inactive_weight:.6f}")
        
        print(f"\n=== Expected vs Actual ===")
        expected_image = IMAGE_TASK_SCORE_WEIGHT * 100
        actual_image = image_weight/total_weight*100
        difference = actual_image - expected_image
        
        print(f"Expected image: {expected_image:.1f}%")
        print(f"Actual image: {actual_image:.2f}%")
        print(f"Difference: {difference:+.2f} percentage points")
        
        if abs(difference) > 1:
            if difference > 0:
                print(f"🔴 Image miners getting {difference:.1f}% MORE than expected!")
            else:
                print(f"🔴 Image miners getting {-difference:.1f}% LESS than expected!")
        else:
            print(f"✅ Image miners within 1% of expected allocation")
        
        # Calculate expected combined text percentage (instruct + dpo + grpo)
        expected_combined_text = (INSTRUCT_TEXT_TASK_SCORE_WEIGHT + 0.15 + (1 - INSTRUCT_TEXT_TASK_SCORE_WEIGHT - IMAGE_TASK_SCORE_WEIGHT - 0.15)) * 100  # instruct + dpo + grpo
        actual_combined_text = combined_text_weight/total_weight*100
        combined_text_difference = actual_combined_text - expected_combined_text
        
        print(f"\nExpected combined text (instruct+dpo+grpo): {expected_combined_text:.1f}%")
        print(f"Actual combined text: {actual_combined_text:.2f}%")
        print(f"Difference: {combined_text_difference:+.2f} percentage points")
        
        # Show top miners by category
        print(f"\n=== Top Miners by Category ===")
        
        categories = [
            ("Image", image_miners),
            ("Text", text_miners),
            ("DPO", dpo_miners),
            ("GRPO", grpo_miners)
        ]
        
        for category_name, miner_set in categories:
            if miner_set:
                category_weights = [(hotkey, weights.get(hotkey, 0)) for hotkey in miner_set]
                category_weights.sort(key=lambda x: x[1], reverse=True)
                
                print(f"\nTop {category_name} miners:")
                for i, (hotkey, weight) in enumerate(category_weights[:3]):
                    print(f"  {i+1}. {hotkey}: {weight:.6f} ({weight/total_weight*100:.2f}%)")
        
        # Task distribution analysis
        print(f"\n=== Task Distribution ===")
        task_type_counts = defaultdict(int)
        organic_counts = defaultdict(int)
        synthetic_counts = defaultdict(int)
        
        for task_result in task_results:
            task_type = task_result.task.task_type
            task_type_counts[task_type] += 1
            
            if task_result.task.is_organic:
                organic_counts[task_type] += 1
            else:
                synthetic_counts[task_type] += 1
        
        total_tasks = sum(task_type_counts.values())
        
        for task_type in sorted(task_type_counts.keys()):
            count = task_type_counts[task_type]
            organic = organic_counts[task_type]
            synthetic = synthetic_counts[task_type]
            
            print(f"{task_type}: {count} tasks ({count/total_tasks*100:.1f}%)")
            print(f"  Organic: {organic} ({organic/count*100:.1f}%)")
            print(f"  Synthetic: {synthetic} ({synthetic/count*100:.1f}%)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(analyze_weight_distribution())