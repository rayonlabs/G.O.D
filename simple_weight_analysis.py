#!/usr/bin/env python3
"""
Simple script to analyze task distribution and infer weight patterns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator.db.sql.submissions_and_scoring import TaskResult, Submission
from validator.db.database import get_database
from validator.core.constants import IMAGE_TASK_SCORE_WEIGHT, INSTRUCT_TEXT_TASK_SCORE_WEIGHT
from collections import defaultdict
from datetime import datetime, timedelta

def analyze_task_distribution():
    """Analyze task distribution and scoring patterns"""
    
    print("=== Task Distribution Analysis ===")
    print(f"Expected IMAGE_TASK_SCORE_WEIGHT: {IMAGE_TASK_SCORE_WEIGHT} (25%)")
    print(f"Expected INSTRUCT_TEXT_TASK_SCORE_WEIGHT: {INSTRUCT_TEXT_TASK_SCORE_WEIGHT} (40%)")
    print()
    
    # Get recent tasks from database
    db = get_database()
    cutoff = datetime.now() - timedelta(days=7)  # Last 7 days
    
    with db.session() as session:
        # Get all task results from last 7 days
        task_results = session.query(TaskResult).filter(
            TaskResult.created_at >= cutoff
        ).all()
        
        print(f"Found {len(task_results)} task results in last 7 days")
        
        # Analyze by task type
        task_type_data = defaultdict(lambda: {
            'miners': set(),
            'total_scores': [],
            'organic_count': 0,
            'synth_count': 0,
            'total_tasks': 0
        })
        
        miner_task_counts = defaultdict(lambda: defaultdict(int))
        
        for task_result in task_results:
            task_type = task_result.task_type
            hotkey = task_result.hotkey
            
            task_type_data[task_type]['miners'].add(hotkey)
            task_type_data[task_type]['total_scores'].append(task_result.score)
            task_type_data[task_type]['total_tasks'] += 1
            
            if task_result.is_organic:
                task_type_data[task_type]['organic_count'] += 1
            else:
                task_type_data[task_type]['synth_count'] += 1
                
            miner_task_counts[hotkey][task_type] += 1
        
        print(f"Found {len(miner_task_counts)} unique miners")
        print()
        
        # Task type breakdown
        print("=== Task Type Breakdown ===")
        total_tasks = sum(data['total_tasks'] for data in task_type_data.values())
        
        for task_type, data in sorted(task_type_data.items(), key=lambda x: x[1]['total_tasks'], reverse=True):
            unique_miners = len(data['miners'])
            total_task_count = data['total_tasks']
            organic_count = data['organic_count']
            synth_count = data['synth_count']
            
            print(f"{task_type}:")
            print(f"  Total tasks: {total_task_count} ({total_task_count/total_tasks*100:.1f}%)")
            print(f"  Unique miners: {unique_miners}")
            print(f"  Organic: {organic_count} ({organic_count/total_task_count*100:.1f}%)")
            print(f"  Synthetic: {synth_count} ({synth_count/total_task_count*100:.1f}%)")
            if data['total_scores']:
                avg_score = sum(data['total_scores']) / len(data['total_scores'])
                print(f"  Average score: {avg_score:.3f}")
            print()
        
        # Classify miners by primary task type
        print("=== Miner Classification ===")
        image_miners = set()
        text_miners = set()
        other_miners = set()
        multi_type_miners = set()
        
        for hotkey, task_counts in miner_task_counts.items():
            if len(task_counts) > 1:
                multi_type_miners.add(hotkey)
            
            # Find primary task type
            primary_task = max(task_counts.items(), key=lambda x: x[1])[0]
            
            if primary_task == 'image':
                image_miners.add(hotkey)
            elif primary_task in ['instruct', 'text']:
                text_miners.add(hotkey)
            else:
                other_miners.add(hotkey)
        
        print(f"Image-focused miners: {len(image_miners)}")
        print(f"Text-focused miners: {len(text_miners)}")
        print(f"Other-focused miners: {len(other_miners)}")
        print(f"Multi-type miners: {len(multi_type_miners)}")
        print()
        
        # Show some examples
        if image_miners:
            print("Sample image miners:")
            for i, hotkey in enumerate(list(image_miners)[:5]):
                task_breakdown = dict(miner_task_counts[hotkey])
                print(f"  {hotkey}: {task_breakdown}")
        
        if text_miners:
            print("\nSample text miners:")
            for i, hotkey in enumerate(list(text_miners)[:5]):
                task_breakdown = dict(miner_task_counts[hotkey])
                print(f"  {hotkey}: {task_breakdown}")
        
        # Analyze scoring patterns
        print(f"\n=== Scoring Analysis ===")
        
        # Get high-scoring miners by task type
        for task_type, data in task_type_data.items():
            if not data['total_scores']:
                continue
                
            # Get task results for this task type
            task_type_results = session.query(TaskResult).filter(
                TaskResult.created_at >= cutoff,
                TaskResult.task_type == task_type
            ).all()
            
            # Group by miner
            miner_scores = defaultdict(list)
            for result in task_type_results:
                miner_scores[result.hotkey].append(result.score)
            
            # Calculate average scores per miner
            miner_avg_scores = {}
            for hotkey, scores in miner_scores.items():
                miner_avg_scores[hotkey] = sum(scores) / len(scores)
            
            # Top performers
            top_performers = sorted(miner_avg_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            print(f"\nTop {task_type} performers:")
            for hotkey, avg_score in top_performers:
                task_count = len(miner_scores[hotkey])
                print(f"  {hotkey}: {avg_score:.3f} avg ({task_count} tasks)")
        
        # Check for potential issues
        print(f"\n=== Potential Issues ===")
        
        # Check if any task type is over-represented
        for task_type, data in task_type_data.items():
            percentage = data['total_tasks'] / total_tasks * 100
            if task_type == 'image' and percentage > 25:
                print(f"⚠️  Image tasks are {percentage:.1f}% of total (expected ~25%)")
            elif task_type == 'instruct' and percentage > 40:
                print(f"⚠️  Instruct tasks are {percentage:.1f}% of total (expected ~40%)")

if __name__ == "__main__":
    analyze_task_distribution()