#!/usr/bin/env python3
"""
Emission calculation script for Gradients subnet.
Adapted from subnet 19 to work with Gradients' scoring system.
"""

import asyncio
import os
import json
import httpx
import argparse
import csv
from datetime import datetime, timedelta
from typing import List, Optional, Dict

from fiber.logging_utils import get_logger
from core.constants import NETUID, DEFAULT_NETUID
from validator.core.constants import (
    ONE_DAY_SCORE_WEIGHT,
    THREE_DAY_SCORE_WEIGHT,
    SEVEN_DAY_SCORE_WEIGHT,
    INSTRUCT_TEXT_TASK_SCORE_WEIGHT,
    IMAGE_TASK_SCORE_WEIGHT,
    DPO_TASK_SCORE_WEIGHT,
    GRPO_TASK_SCORE_WEIGHT
)

logger = get_logger(__name__)

# Constants
SECONDS_PER_BLOCK_EMISSION = 12
MINER_ALPHA_EMISSION_PER_EPOCH = 147.6  # Adjust based on Gradients emission rate
EXPECTED_EPOCH_INTERVAL_MINUTES = 72
TOLERANCE_MINUTES = 8

# API configuration
VALIDATOR_API_URL = os.getenv("VALIDATOR_API_URL", "http://localhost:9000/")


def load_epoch_steps_csv(filepath: str) -> List[Dict]:
    """Load epoch steps from CSV file."""
    epoch_steps = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch_steps.append({
                'block_number': int(row['block_number']),
                'timestamp': datetime.strptime(row['timestamp'].replace('+00', ''), '%Y-%m-%d %H:%M:%S.%f' if '.' in row['timestamp'] else '%Y-%m-%d %H:%M:%S'),
                'netuid': int(row['netuid']),
                'blocks_since_last_step': int(row['blocks_since_last_step'])
            })
    return epoch_steps


def identify_missing_emission_windows(epoch_steps: List[Dict]) -> List[Dict]:
    """Identify missing emission windows from epoch steps data."""
    missing_windows = []
    
    for i in range(1, len(epoch_steps)):
        current = epoch_steps[i]['timestamp']
        previous = epoch_steps[i-1]['timestamp']
        time_diff_minutes = (current - previous).total_seconds() / 60
        
        if time_diff_minutes > EXPECTED_EPOCH_INTERVAL_MINUTES + TOLERANCE_MINUTES:
            missed_epochs = int(time_diff_minutes / EXPECTED_EPOCH_INTERVAL_MINUTES) - 1
            if missed_epochs > 0:
                missing_windows.append({
                    'start': previous,
                    'end': current,
                    'duration_hours': time_diff_minutes / 60,
                    'missed_epochs': missed_epochs
                })
                logger.info(f"Missing window: {previous} to {current} ({missed_epochs} epochs)")
    
    return missing_windows


async def get_tasks_in_window(client: httpx.AsyncClient, start_time: datetime, end_time: datetime) -> List[Dict]:
    """Get all tasks within a time window from the API."""
    all_tasks = []
    offset = 0
    limit = 100
    
    while True:
        params = {
            'created_at_lower': start_time.isoformat(),
            'created_at_upper': end_time.isoformat(),
            'offset': offset,
            'limit': limit
        }
        
        try:
            response = await client.get(f"{VALIDATOR_API_URL}auditing/tasks", params=params)
            if response.status_code != 200:
                logger.error(f"Failed to get tasks: {response.status_code} {response.text}")
                break
                
            tasks = response.json()
            if not tasks:
                break
                
            all_tasks.extend(tasks)
            offset += limit
            
            if len(tasks) < limit:
                break
        except Exception as e:
            logger.error(f"Error fetching tasks: {e}")
            break
    
    logger.info(f"Retrieved {len(all_tasks)} tasks for window {start_time} to {end_time}")
    return all_tasks


async def get_task_details(client: httpx.AsyncClient, task_id: str) -> Optional[Dict]:
    """Get detailed information for a specific task."""
    try:
        response = await client.get(f"{VALIDATOR_API_URL}auditing/tasks/{task_id}")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.error(f"Error getting task details for {task_id}: {e}")
    return None


async def calculate_weights_from_tasks(tasks: List[Dict], reference_time: datetime = None) -> Dict[str, float]:
    """Calculate weights based on task results using Gradients' scoring system."""
    # Initialize score tracking
    hotkey_scores = {}
    
    # Use reference time or current time for period calculations
    now = reference_time or datetime.utcnow()
    one_day_ago = now - timedelta(days=1)
    three_days_ago = now - timedelta(days=3)
    seven_days_ago = now - timedelta(days=7)
    
    for task in tasks:
        # Extract relevant fields
        submissions = task.get('submissions', [])
        task_type = task.get('task_type', 'instruct')  # Default to instruct
        created_at_str = task.get('created_at')
        
        if not created_at_str:
            continue
            
        # Parse datetime
        try:
            if created_at_str.endswith('Z'):
                created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
            else:
                created_at = datetime.fromisoformat(created_at_str)
        except:
            continue
        
        # Process each submission
        for submission in submissions:
            hotkey = submission.get('hotkey')
            score = submission.get('score', 0)
            
            if not hotkey or score <= 0:
                continue
                
            if hotkey not in hotkey_scores:
                hotkey_scores[hotkey] = {
                    'one_day': {'instruct': 0, 'image': 0, 'dpo': 0, 'grpo': 0},
                    'three_day': {'instruct': 0, 'image': 0, 'dpo': 0, 'grpo': 0},
                    'seven_day': {'instruct': 0, 'image': 0, 'dpo': 0, 'grpo': 0}
                }
            
            # Map task type
            task_category = 'instruct'  # default
            if 'image' in task_type.lower() or 'diffusion' in task_type.lower():
                task_category = 'image'
            elif 'dpo' in task_type.lower():
                task_category = 'dpo'
            elif 'grpo' in task_type.lower():
                task_category = 'grpo'
            
            # Categorize by time period
            if created_at >= one_day_ago:
                hotkey_scores[hotkey]['one_day'][task_category] += score
                hotkey_scores[hotkey]['three_day'][task_category] += score
                hotkey_scores[hotkey]['seven_day'][task_category] += score
            elif created_at >= three_days_ago:
                hotkey_scores[hotkey]['three_day'][task_category] += score
                hotkey_scores[hotkey]['seven_day'][task_category] += score
            elif created_at >= seven_days_ago:
                hotkey_scores[hotkey]['seven_day'][task_category] += score
    
    # Calculate weighted scores
    final_weights = {}
    
    for hotkey, periods in hotkey_scores.items():
        # Calculate task type weighted scores for each period
        one_day_score = (
            periods['one_day']['instruct'] * INSTRUCT_TEXT_TASK_SCORE_WEIGHT +
            periods['one_day']['image'] * IMAGE_TASK_SCORE_WEIGHT +
            periods['one_day']['dpo'] * DPO_TASK_SCORE_WEIGHT +
            periods['one_day']['grpo'] * GRPO_TASK_SCORE_WEIGHT
        )
        
        three_day_score = (
            periods['three_day']['instruct'] * INSTRUCT_TEXT_TASK_SCORE_WEIGHT +
            periods['three_day']['image'] * IMAGE_TASK_SCORE_WEIGHT +
            periods['three_day']['dpo'] * DPO_TASK_SCORE_WEIGHT +
            periods['three_day']['grpo'] * GRPO_TASK_SCORE_WEIGHT
        )
        
        seven_day_score = (
            periods['seven_day']['instruct'] * INSTRUCT_TEXT_TASK_SCORE_WEIGHT +
            periods['seven_day']['image'] * IMAGE_TASK_SCORE_WEIGHT +
            periods['seven_day']['dpo'] * DPO_TASK_SCORE_WEIGHT +
            periods['seven_day']['grpo'] * GRPO_TASK_SCORE_WEIGHT
        )
        
        # Apply time period weights
        weighted_score = (
            one_day_score * ONE_DAY_SCORE_WEIGHT +
            three_day_score * THREE_DAY_SCORE_WEIGHT +
            seven_day_score * SEVEN_DAY_SCORE_WEIGHT
        )
        
        if weighted_score > 0:
            final_weights[hotkey] = weighted_score
    
    # Normalize to sum to 1
    total_weight = sum(final_weights.values())
    if total_weight > 0:
        for hotkey in final_weights:
            final_weights[hotkey] /= total_weight
    
    return final_weights


async def calculate_weights_for_window(client: httpx.AsyncClient, start_time: datetime, end_time: datetime) -> Dict[str, float]:
    """Calculate weights for a specific time window."""
    logger.info(f"Calculating weights for window {start_time} to {end_time}")
    
    # Get all tasks in the wider window (7 days before end_time) to properly calculate time-based weights
    extended_start = end_time - timedelta(days=7)
    tasks = await get_tasks_in_window(client, extended_start, end_time)
    
    if not tasks:
        logger.warning(f"No tasks found for window")
        return {}
    
    # Get detailed task information if needed
    detailed_tasks = []
    for task in tasks:
        # If task doesn't have submissions, get details
        if 'submissions' not in task:
            details = await get_task_details(client, task['task_id'])
            if details:
                detailed_tasks.append(details)
        else:
            detailed_tasks.append(task)
    
    # Calculate weights using the end_time as reference
    weights = await calculate_weights_from_tasks(detailed_tasks, reference_time=end_time)
    
    logger.info(f"Calculated weights for {len(weights)} miners in window")
    return weights


async def calculate_missing_emissions(client: httpx.AsyncClient, epoch_steps_file: str) -> Dict[str, float]:
    """Calculate missing emissions for all miners based on epoch steps data."""
    epoch_steps = load_epoch_steps_csv(epoch_steps_file)
    
    # Filter for Gradients subnet
    gradients_steps = [step for step in epoch_steps if step['netuid'] == DEFAULT_NETUID]
    
    missing_windows = identify_missing_emission_windows(gradients_steps)
    
    logger.info(f"Found {len(missing_windows)} missing emission windows for netuid {DEFAULT_NETUID}")
    
    emissions_owed = {}
    
    for window in missing_windows:
        logger.info(f"\nProcessing missing window: {window['start']} to {window['end']} ({window['missed_epochs']} epochs)")
        
        weights = await calculate_weights_for_window(client, window['start'], window['end'])
        
        if not weights:
            logger.warning(f"No weights found for window, skipping")
            continue
        
        window_emissions = window['missed_epochs'] * MINER_ALPHA_EMISSION_PER_EPOCH
        
        for hotkey, weight in weights.items():
            if hotkey not in emissions_owed:
                emissions_owed[hotkey] = 0
            emissions_owed[hotkey] += weight * window_emissions
            
        logger.info(f"Distributed {window_emissions} alpha across {len(weights)} miners")
    
    return emissions_owed


async def analyze_historical_weights(client: httpx.AsyncClient, datetime_lower: datetime, datetime_upper: datetime) -> Dict:
    """Analyze historical weight distribution for a given time period."""
    
    # Calculate daily windows within the range
    current_date = datetime_lower
    daily_weights = []
    
    while current_date < datetime_upper:
        next_date = min(current_date + timedelta(days=1), datetime_upper)
        
        logger.info(f"Analyzing weights for {current_date.date()}")
        weights = await calculate_weights_for_window(client, current_date, next_date)
        
        if weights:
            daily_weights.append({
                'date': current_date.isoformat(),
                'weights': weights,
                'top_miners': sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]
            })
        
        current_date = next_date
    
    # Calculate average weights across the period
    all_hotkeys = set()
    for day in daily_weights:
        all_hotkeys.update(day['weights'].keys())
    
    average_weights = {}
    for hotkey in all_hotkeys:
        weights = [day['weights'].get(hotkey, 0) for day in daily_weights]
        average_weights[hotkey] = sum(weights) / len(weights) if weights else 0
    
    # Get task type distribution
    tasks = await get_tasks_in_window(client, datetime_lower, datetime_upper)
    task_type_counts = {}
    for task in tasks:
        task_type = task.get('task_type', 'unknown')
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
    
    return {
        'daily_weights': daily_weights,
        'average_weights': average_weights,
        'task_type_distribution': task_type_counts,
        'total_tasks': len(tasks),
        'period': {
            'start': datetime_lower.isoformat(),
            'end': datetime_upper.isoformat()
        }
    }


async def main(datetime_lower: datetime, datetime_upper: datetime, epoch_steps_file: Optional[str] = None):
    logger.info(f"Gradients Subnet Weight Analysis")
    logger.info(f"API URL: {VALIDATOR_API_URL}")
    logger.info(f"Subnet: {NETUID}")
    logger.info("=" * 50)
    
    httpx_limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
    timeout = httpx.Timeout(timeout=600.0, connect=60.0)  # 10 minutes
    
    async with httpx.AsyncClient(limits=httpx_limits, timeout=timeout) as client:
        try:
            if epoch_steps_file:
                emissions_owed = await calculate_missing_emissions(client, epoch_steps_file)
                
                logger.info("\n=== MISSING EMISSIONS SUMMARY ===")
                total_emissions = sum(emissions_owed.values())
                logger.info(f"Total emissions owed: {total_emissions:.2f} alpha")
                logger.info(f"Number of miners affected: {len(emissions_owed)}")
                
                sorted_emissions = sorted(emissions_owed.items(), key=lambda x: x[1], reverse=True)
                
                logger.info("\nTop 20 miners by emissions owed:")
                for i, (hotkey, amount) in enumerate(sorted_emissions[:20]):
                    logger.info(f"{i+1:2d}. {hotkey}: {amount:10.6f} alpha")
                
                output_file = f"missing_emissions_gradients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_file, 'w') as f:
                    json.dump({
                        'netuid': NETUID,
                        'total_emissions_owed': total_emissions,
                        'miners_affected': len(emissions_owed),
                        'emissions_by_miner': emissions_owed,
                        'calculation_time': datetime.now().isoformat()
                    }, f, indent=2)
                logger.info(f"\nDetailed results saved to {output_file}")
                
            else:
                analysis = await analyze_historical_weights(client, datetime_lower, datetime_upper)
                
                logger.info("\n=== HISTORICAL WEIGHTS ANALYSIS ===")
                logger.info(f"Period: {datetime_lower.date()} to {datetime_upper.date()}")
                logger.info(f"Days analyzed: {len(analysis['daily_weights'])}")
                logger.info(f"Total tasks: {analysis['total_tasks']}")
                
                # Show task type distribution
                logger.info("\nTask type distribution:")
                for task_type, count in sorted(analysis['task_type_distribution'].items()):
                    percentage = (count / analysis['total_tasks'] * 100) if analysis['total_tasks'] > 0 else 0
                    logger.info(f"  {task_type:20s}: {count:5d} ({percentage:5.1f}%)")
                
                # Show top miners by average weight
                sorted_avg = sorted(analysis['average_weights'].items(), key=lambda x: x[1], reverse=True)
                logger.info(f"\nTop 20 miners by average weight:")
                for i, (hotkey, weight) in enumerate(sorted_avg[:20]):
                    logger.info(f"{i+1:2d}. {hotkey}: {weight:.6f}")
                
                output_file = f"weight_analysis_gradients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_file, 'w') as f:
                    json.dump(analysis, f, indent=2)
                logger.info(f"\nDetailed results saved to {output_file}")

        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze Gradients subnet (netuid 56) weights and emissions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze weights for a date range
  python calculate_emissions.py --datetime-lower 2025-05-15_00:00:00 --datetime-upper 2025-05-22_00:00:00
  
  # Calculate missing emissions from epoch steps CSV
  python calculate_emissions.py --datetime-lower 2025-05-15_00:00:00 --datetime-upper 2025-05-22_00:00:00 --epoch-steps epochsteps.csv
  
Environment variables:
  VALIDATOR_API_URL - Set the validator API endpoint (default: http://localhost:9000/)
        """
    )
    parser.add_argument(
        "--datetime-lower",
        required=True,
        help="Lower datetime bound in format YYYY-MM-DD_HH:MM:SS"
    )
    parser.add_argument(
        "--datetime-upper",
        required=True,
        help="Upper datetime bound in format YYYY-MM-DD_HH:MM:SS"
    )
    parser.add_argument(
        "--epoch-steps",
        required=False,
        help="Path to epoch steps CSV file for calculating missing emissions"
    )
    args = parser.parse_args()

    try:
        dt_lower = datetime.strptime(args.datetime_lower, "%Y-%m-%d_%H:%M:%S")
        dt_upper = datetime.strptime(args.datetime_upper, "%Y-%m-%d_%H:%M:%S")
    except ValueError as ve:
        logger.error(f"Invalid datetime format: {ve}")
        logger.error("Please use format: YYYY-MM-DD_HH:MM:SS")
        exit(1)

    asyncio.run(main(dt_lower, dt_upper, args.epoch_steps))