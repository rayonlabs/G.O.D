#!/usr/bin/env python3
"""
Direct emission calculation script for Gradients subnet.
Uses direct database access like check_scoring.py instead of API.
"""

import asyncio
import os
import json
import argparse
import csv
from datetime import datetime, timedelta
from typing import List, Optional, Dict

# Import all the scoring logic from the codebase
from validator.core.config import Config, load_config
from validator.core.weight_setting import _get_weights_to_set, get_node_weights_from_period_scores
from validator.db.sql.submissions_and_scoring import get_aggregate_scores_since
from fiber.chain import fetch_nodes
from validator.utils.logging import get_logger
from validator.utils.util import try_db_connections
from core.constants import NETUID, DEFAULT_NETUID

logger = get_logger(__name__)

# Constants
SECONDS_PER_BLOCK_EMISSION = 12
MINER_ALPHA_EMISSION_PER_EPOCH = 147.6  # Adjust based on Gradients emission rate
EXPECTED_EPOCH_INTERVAL_MINUTES = 72
TOLERANCE_MINUTES = 8


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


async def calculate_weights_for_window(config: Config, start_time: datetime, end_time: datetime) -> Dict[str, float]:
    """Calculate weights for a specific time window using direct database access."""
    logger.info(f"Calculating weights for window {start_time} to {end_time}")
    
    # Get task results from database
    task_results = await get_aggregate_scores_since(end_time - timedelta(days=7), config.psql_db)
    
    if not task_results:
        logger.warning(f"No task results found for window")
        return {}
    
    # Get period scores
    period_scores, _ = await _get_weights_to_set(config)
    
    # Calculate node weights
    all_node_ids, all_node_weights = await get_node_weights_from_period_scores(
        config.substrate, config.netuid, period_scores
    )
    
    # Get all nodes to create hotkey mapping
    all_nodes = fetch_nodes.get_nodes_for_netuid(config.substrate, config.netuid)
    node_id_to_hotkey = {node.node_id: node.hotkey for node in all_nodes}
    
    # Create weights dictionary with hotkeys
    weights = {}
    total_weight = sum(all_node_weights)
    
    for node_id, weight in enumerate(all_node_weights):
        if weight > 0 and node_id in node_id_to_hotkey:
            normalized_weight = weight / total_weight if total_weight > 0 else 0
            weights[node_id_to_hotkey[node_id]] = normalized_weight
    
    logger.info(f"Calculated weights for {len(weights)} miners in window")
    return weights


async def calculate_missing_emissions(config: Config, epoch_steps_file: str, emission_per_epoch: float = None) -> Dict[str, float]:
    """Calculate missing emissions for all miners based on epoch steps data."""
    epoch_steps = load_epoch_steps_csv(epoch_steps_file)
    
    # Filter for Gradients subnet
    gradients_steps = [step for step in epoch_steps if step['netuid'] == config.netuid]
    
    missing_windows = identify_missing_emission_windows(gradients_steps)
    
    logger.info(f"Found {len(missing_windows)} missing emission windows for netuid {config.netuid}")
    
    emissions_owed = {}
    
    for window in missing_windows:
        logger.info(f"\nProcessing missing window: {window['start']} to {window['end']} ({window['missed_epochs']} epochs)")
        
        weights = await calculate_weights_for_window(config, window['start'], window['end'])
        
        if not weights:
            logger.warning(f"No weights found for window, skipping")
            continue
        
        window_emissions = window['missed_epochs'] * (emission_per_epoch or MINER_ALPHA_EMISSION_PER_EPOCH)
        
        for hotkey, weight in weights.items():
            if hotkey not in emissions_owed:
                emissions_owed[hotkey] = 0
            emissions_owed[hotkey] += weight * window_emissions
            
        logger.info(f"Distributed {window_emissions} alpha across {len(weights)} miners")
    
    return emissions_owed


async def analyze_historical_weights(config: Config, datetime_lower: datetime, datetime_upper: datetime) -> Dict:
    """Analyze historical weight distribution for a given time period."""
    
    # Calculate daily windows within the range
    current_date = datetime_lower
    daily_weights = []
    
    while current_date < datetime_upper:
        next_date = min(current_date + timedelta(days=1), datetime_upper)
        
        logger.info(f"Analyzing weights for {current_date.date()}")
        weights = await calculate_weights_for_window(config, current_date, next_date)
        
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
    
    # Get task statistics from the period
    task_results = await get_aggregate_scores_since(datetime_lower, config.psql_db)
    task_type_counts = {}
    for result in task_results:
        task_type = str(result.task.task_type)
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
    
    return {
        'daily_weights': daily_weights,
        'average_weights': average_weights,
        'task_type_distribution': task_type_counts,
        'total_tasks': len(task_results),
        'period': {
            'start': datetime_lower.isoformat(),
            'end': datetime_upper.isoformat()
        }
    }


async def verify_emission_rate(config: Config):
    """Verify the emission rate from the chain."""
    from fiber.chain.chain_utils import query_substrate
    
    # Get current emission info for our subnet
    substrate = config.substrate
    
    logger.info(f"\n=== CHAIN EMISSION DATA FOR NETUID {config.netuid} ===")
    
    # Query all emission-related storage items
    substrate, alpha_out = query_substrate(
        substrate, "SubtensorModule", "SubnetAlphaOutEmission", [config.netuid], return_value=True
    )
    
    substrate, alpha_in = query_substrate(
        substrate, "SubtensorModule", "SubnetAlphaInEmission", [config.netuid], return_value=True
    )
    
    substrate, tao_in = query_substrate(
        substrate, "SubtensorModule", "SubnetTaoInEmission", [config.netuid], return_value=True
    )
    
    # Get current block number to understand context
    substrate, current_block = query_substrate(substrate, "System", "Number", [], return_value=True)
    
    logger.info(f"Current block: {current_block}")
    logger.info(f"SubnetAlphaOutEmission: {alpha_out}")
    logger.info(f"SubnetAlphaInEmission: {alpha_in}")
    logger.info(f"SubnetTaoInEmission: {tao_in}")
    
    if alpha_out is not None:
        # This might be cumulative, let's see if we can figure out the rate
        logger.info(f"\nIf SubnetAlphaOutEmission ({alpha_out}) is:")
        logger.info(f"  - Cumulative total: {alpha_out / current_block:.6f} alpha per block average")
        logger.info(f"  - Per block: {alpha_out} alpha per block (seems too high)")
        logger.info(f"  - Per epoch: {alpha_out / 360:.6f} alpha per block")
        
        # Let's assume it's cumulative and calculate average per block
        if current_block > 0:
            avg_per_block = alpha_out / current_block
            return avg_per_block * 360  # Per epoch
    
    logger.info(f"\nUsing default emission rate: {MINER_ALPHA_EMISSION_PER_EPOCH} alpha per epoch")
    return None


async def main(datetime_lower: datetime, datetime_upper: datetime, epoch_steps_file: Optional[str] = None):
    # Load config
    config = load_config()
    
    # Connect to database
    await try_db_connections(config)
    
    logger.info(f"Gradients Subnet Weight Analysis (Direct DB Access)")
    logger.info(f"Subnet: {config.netuid}")
    logger.info("=" * 50)
    
    # Verify emission rate from chain
    actual_emission_per_epoch = await verify_emission_rate(config)
    if actual_emission_per_epoch:
        logger.info(f"Verified emission rate from chain: {actual_emission_per_epoch:.2f} alpha per epoch")
    else:
        logger.info(f"Using default emission rate: {MINER_ALPHA_EMISSION_PER_EPOCH} alpha per epoch")
    
    try:
        if epoch_steps_file:
            emissions_owed = await calculate_missing_emissions(config, epoch_steps_file, actual_emission_per_epoch)
            
            logger.info("\n=== MISSING EMISSIONS SUMMARY ===")
            total_emissions = sum(emissions_owed.values())
            logger.info(f"Total emissions owed: {total_emissions:.2f} alpha")
            logger.info(f"Number of miners affected: {len(emissions_owed)}")
            emission_rate = actual_emission_per_epoch or MINER_ALPHA_EMISSION_PER_EPOCH
            logger.info(f"Expected total (24 epochs × {emission_rate:.2f}): {24 * emission_rate:.2f} alpha")
            logger.info(f"Difference: {abs(total_emissions - (24 * emission_rate)):.6f} alpha")
            
            sorted_emissions = sorted(emissions_owed.items(), key=lambda x: x[1], reverse=True)
            
            logger.info("\nTop 20 miners by emissions owed:")
            for i, (hotkey, amount) in enumerate(sorted_emissions[:20]):
                logger.info(f"{i+1:2d}. {hotkey}: {amount:10.6f} alpha")
            
            output_file = f"missing_emissions_gradients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'netuid': config.netuid,
                    'total_emissions_owed': total_emissions,
                    'miners_affected': len(emissions_owed),
                    'emissions_by_miner': emissions_owed,
                    'calculation_time': datetime.now().isoformat()
                }, f, indent=2)
            logger.info(f"\nDetailed results saved to {output_file}")
            
        else:
            analysis = await analyze_historical_weights(config, datetime_lower, datetime_upper)
            
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
            total_weight = sum(analysis['average_weights'].values())
            logger.info(f"Total weight sum: {total_weight:.6f}")
            
            # Calculate alpha emissions based on average weights
            # This is a simulation - actual emissions depend on epoch timing
            logger.info(f"\nIf 24 epochs were distributed based on these average weights:")
            total_alpha = 24 * 147.6  # 3542.4 alpha
            logger.info(f"Total alpha to distribute: {total_alpha:.2f}")
            
            alpha_sum = 0
            for i, (hotkey, weight) in enumerate(sorted_avg[:20]):
                alpha_amount = weight * total_alpha
                alpha_sum += alpha_amount
                logger.info(f"{i+1:2d}. {hotkey}: {weight:.6f} (≈ {alpha_amount:.2f} alpha)")
            
            output_file = f"weight_analysis_gradients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"\nDetailed results saved to {output_file}")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close database connections
        await config.psql_db.close()
        if hasattr(config, 'redis_db'):
            await config.redis_db.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze Gradients subnet (netuid 56) weights and emissions using direct database access.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze weights for a date range
  python calculate_emissions_direct.py --datetime-lower 2025-05-15_00:00:00 --datetime-upper 2025-05-22_00:00:00
  
  # Calculate missing emissions from epoch steps CSV
  python calculate_emissions_direct.py --datetime-lower 2025-05-15_00:00:00 --datetime-upper 2025-05-22_00:00:00 --epoch-steps epochsteps.csv
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