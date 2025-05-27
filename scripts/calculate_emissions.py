import asyncio
import os
import re
import json
import httpx
import argparse
import csv
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from fiber.chain import chain_utils
from fiber.logging_utils import get_logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.utils import download_s3_file
from core.constants import PROD_NETUID, TEST_NETUID
from core.models.config_models import AuditConfig
from validator.db.sql.submissions_and_scoring import get_aggregate_scores_since
from validator.db.sql.nodes import get_node_stats
from validator.core.weight_setting import (
    ONE_DAY_SCORE_WEIGHT,
    THREE_DAY_SCORE_WEIGHT,
    SEVEN_DAY_SCORE_WEIGHT,
    INSTRUCT_TEXT_TASK_SCORE_WEIGHT,
    IMAGE_TASK_SCORE_WEIGHT,
    DPO_TASK_SCORE_WEIGHT,
    GRPO_TASK_SCORE_WEIGHT,
    calculate_final_scores
)
from validator.evaluation.scoring import (
    FIRST_PLACE_SCORE,
    SIGMOID_STEEPNESS,
    SIGMOID_SHIFT,
    SIGMOID_POWER,
    LINEAR_WEIGHT,
    SIGMOID_WEIGHT,
    sigmoid_normalize
)


SECONDS_PER_BLOCK_EMISSION = 12
# Miner alpha emission per epoch (this should be adjusted based on actual emission rate)
MINER_ALPHA_EMISSION_PER_EPOCH = 147.6  # Adjust based on Gradients subnet emission rate
EXPECTED_EPOCH_INTERVAL_MINUTES = 72
TOLERANCE_MINUTES = 8


logger = get_logger(__name__)


_config = None

def load_config() -> AuditConfig:
    global _config
    if _config is None:
        subtensor_network = os.getenv("SUBTENSOR_NETWORK") or 'finney'
        subtensor_address = os.getenv("SUBTENSOR_ADDRESS") or 'wss://entrypoint-finney.opentensor.ai:443'
        wallet_name = os.getenv("WALLET_NAME") or None
        hotkey_name = os.getenv("HOTKEY_NAME") or None
        netuid = os.getenv("NETUID")
        if netuid is None:
            netuid = TEST_NETUID if subtensor_network == "test" else PROD_NETUID
            logger.warning(f"NETUID not set, using {netuid}")
        else:
            netuid = int(netuid)

        substrate = None
        
        keypair = None 
        if wallet_name and hotkey_name:
            keypair = chain_utils.load_hotkey_keypair(wallet_name=wallet_name, hotkey_name=hotkey_name)
        logger.info(f"This is my own keypair {keypair}")

        httpx_limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
        timeout = httpx.Timeout(timeout=600.0, connect=60.0)  # 10 minutes
        httpx_client = httpx.AsyncClient(limits=httpx_limits, timeout=timeout)

        # Database connection
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        engine = create_engine(database_url)
        Session = sessionmaker(bind=engine)

        _config = AuditConfig(
            substrate=substrate,
            keypair=keypair,
            netuid=netuid,
            httpx_client=httpx_client,
            db_session=Session
        )
    return _config


def extract_timestamp(url: str) -> Optional[datetime]:
    """
    Extracts a timestamp from a given S3 URL with filename format like:
    audit_data_YYYY-MM-DD_HH-MM-SS.json

    Args:
        url (str): The input S3 URL.

    Returns:
        Optional[datetime]: A datetime object if a timestamp is found, otherwise None.
    """
    pattern = r'audit_data_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.json'
    match = re.search(pattern, url)
    if match:
        timestamp_str = match.group(1)
        return datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
    return None


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


async def get_task_results_in_time_window(config: AuditConfig, start_time: datetime, end_time: datetime) -> Dict[str, Dict]:
    """Get task results for a specific time window from the database."""
    with config.db_session() as session:
        # Get aggregate scores for the time window
        results = await asyncio.to_thread(
            get_aggregate_scores_since,
            session,
            start_time,
            end_time
        )
        
        # Get node stats for quality metrics
        node_stats = await asyncio.to_thread(
            get_node_stats,
            session
        )
        
        return {
            'task_results': results,
            'node_stats': node_stats
        }


async def calculate_weights_from_db_data(config: AuditConfig, task_data: Dict) -> Dict[str, float]:
    """Calculate weights based on database task results."""
    task_results = task_data['task_results']
    node_stats = task_data['node_stats']
    
    # Initialize score tracking
    hotkey_scores = {}
    
    # Process results by time period
    now = datetime.utcnow()
    one_day_ago = now - timedelta(days=1)
    three_days_ago = now - timedelta(days=3)
    seven_days_ago = now - timedelta(days=7)
    
    for result in task_results:
        hotkey = result['hotkey']
        task_type = result['task_type']
        score = result['score']
        created_at = result['created_at']
        
        if hotkey not in hotkey_scores:
            hotkey_scores[hotkey] = {
                'one_day': {'instruct': 0, 'image': 0, 'dpo': 0, 'grpo': 0},
                'three_day': {'instruct': 0, 'image': 0, 'dpo': 0, 'grpo': 0},
                'seven_day': {'instruct': 0, 'image': 0, 'dpo': 0, 'grpo': 0}
            }
        
        # Categorize by time period
        if created_at >= one_day_ago:
            hotkey_scores[hotkey]['one_day'][task_type] += score
            hotkey_scores[hotkey]['three_day'][task_type] += score
            hotkey_scores[hotkey]['seven_day'][task_type] += score
        elif created_at >= three_days_ago:
            hotkey_scores[hotkey]['three_day'][task_type] += score
            hotkey_scores[hotkey]['seven_day'][task_type] += score
        elif created_at >= seven_days_ago:
            hotkey_scores[hotkey]['seven_day'][task_type] += score
    
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
        
        # Apply sigmoid normalization
        if weighted_score > 0:
            normalized_score = sigmoid_normalize(
                weighted_score,
                steepness=SIGMOID_STEEPNESS,
                shift=SIGMOID_SHIFT,
                power=SIGMOID_POWER,
                linear_weight=LINEAR_WEIGHT,
                sigmoid_weight=SIGMOID_WEIGHT
            )
            final_weights[hotkey] = normalized_score
    
    # Normalize to sum to 1
    total_weight = sum(final_weights.values())
    if total_weight > 0:
        for hotkey in final_weights:
            final_weights[hotkey] /= total_weight
    
    return final_weights


async def calculate_weights_for_window(config: AuditConfig, start_time: datetime, end_time: datetime) -> Dict[str, float]:
    """Calculate weights for a specific time window."""
    logger.info(f"Calculating weights for window {start_time} to {end_time}")
    
    task_data = await get_task_results_in_time_window(config, start_time, end_time)
    
    if not task_data['task_results']:
        logger.warning(f"No task results found for window {start_time} to {end_time}")
        return {}
    
    weights = await calculate_weights_from_db_data(config, task_data)
    
    logger.info(f"Calculated weights for {len(weights)} miners in window")
    return weights


async def calculate_missing_emissions(config: AuditConfig, epoch_steps_file: str) -> Dict[str, float]:
    """Calculate missing emissions for all miners based on epoch steps data."""
    epoch_steps = load_epoch_steps_csv(epoch_steps_file)
    
    missing_windows = identify_missing_emission_windows(epoch_steps)
    
    logger.info(f"Found {len(missing_windows)} missing emission windows")
    
    emissions_owed = {}
    
    for window in missing_windows:
        logger.info(f"\nProcessing missing window: {window['start']} to {window['end']} ({window['missed_epochs']} epochs)")
        
        weights = await calculate_weights_for_window(config, window['start'], window['end'])
        
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


async def analyze_historical_weights(config: AuditConfig, datetime_lower: datetime, datetime_upper: datetime) -> Dict:
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
    
    return {
        'daily_weights': daily_weights,
        'average_weights': average_weights,
        'period': {
            'start': datetime_lower.isoformat(),
            'end': datetime_upper.isoformat()
        }
    }


async def main(datetime_lower: datetime, datetime_upper: datetime, epoch_steps_file: Optional[str] = None):
    config = load_config()

    try:
        if epoch_steps_file:
            emissions_owed = await calculate_missing_emissions(config, epoch_steps_file)
            
            logger.info("\n=== MISSING EMISSIONS SUMMARY ===")
            total_emissions = sum(emissions_owed.values())
            logger.info(f"Total emissions owed: {total_emissions} alpha")
            logger.info(f"Number of miners affected: {len(emissions_owed)}")
            
            sorted_emissions = sorted(emissions_owed.items(), key=lambda x: x[1], reverse=True)
            
            logger.info("\nTop 10 miners by emissions owed:")
            for i, (hotkey, amount) in enumerate(sorted_emissions[:10]):
                logger.info(f"{i+1}. {hotkey}: {amount:.6f} alpha")
            
            output_file = f"missing_emissions_gradients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'total_emissions_owed': total_emissions,
                    'miners_affected': len(emissions_owed),
                    'emissions_by_miner': emissions_owed,
                    'calculation_time': datetime.now().isoformat(),
                    'netuid': config.netuid
                }, f, indent=2)
            logger.info(f"\nResults saved to {output_file}")
            
        else:
            analysis = await analyze_historical_weights(config, datetime_lower, datetime_upper)
            
            logger.info("\n=== HISTORICAL WEIGHTS ANALYSIS ===")
            logger.info(f"Period: {datetime_lower} to {datetime_upper}")
            logger.info(f"Days analyzed: {len(analysis['daily_weights'])}")
            
            # Show top miners by average weight
            sorted_avg = sorted(analysis['average_weights'].items(), key=lambda x: x[1], reverse=True)
            logger.info("\nTop 10 miners by average weight:")
            for i, (hotkey, weight) in enumerate(sorted_avg[:10]):
                logger.info(f"{i+1}. {hotkey}: {weight:.6f}")
            
            output_file = f"weight_analysis_gradients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"\nResults saved to {output_file}")

    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await config.httpx_client.aclose()
        logger.info("Gracefully shut down HTTP client")


# Example: `python scripts/calculate_emissions.py --datetime-lower 2025-05-15_00:00:00 --datetime-upper 2025-05-22_00:00:00`
# Example with epoch steps: `python scripts/calculate_emissions.py --datetime-lower 2025-05-15_00:00:00 --datetime-upper 2025-05-22_00:00:00 --epoch-steps epochsteps_gradients.csv`
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Gradients subnet weights and emissions between two datetimes.")
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
        print(f"Invalid datetime format: {ve}")
        exit(1)

    asyncio.run(main(dt_lower, dt_upper, args.epoch_steps))