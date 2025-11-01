import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv

from core.models.tournament_models import NodeWeightsResult
from core.models.tournament_models import TournamentAuditData
from core.models.tournament_models import TournamentBurnData
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentResults
from core.models.tournament_models import TournamentResultsWithWinners
from core.models.tournament_models import TournamentType
from validator.db.sql.auditing import store_latest_scores_url
from validator.db.sql.tournaments import count_champion_consecutive_wins
from validator.db.sql.tournaments import get_active_tournament_participants
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament_full_results
from validator.db.sql.tournaments import get_tournament_where_champion_first_won
from validator.db.sql.tournaments import get_weekly_task_participation_data
from validator.evaluation.tournament_scoring import get_tournament_weights_from_data
from validator.tournament.performance_calculator import calculate_performance_difference


load_dotenv(os.getenv("ENV_FILE", ".vali.env"))

import json
from uuid import UUID

from fiber.chain import fetch_nodes
from fiber.chain import weights
from fiber.chain.chain_utils import query_substrate
from fiber.chain.models import Node
from substrateinterface import SubstrateInterface

import validator.core.constants as cts
from core import constants as ccst
from core.constants import BUCKET_NAME
from validator.core.config import Config
from validator.core.config import load_config
from validator.db.sql.nodes import get_vali_node_id
from validator.utils.logging import get_logger
from validator.utils.util import save_json_to_temp_file
from validator.utils.util import try_db_connections
from validator.utils.util import upload_file_to_minio


logger = get_logger(__name__)


TIME_PER_BLOCK: int = 500


async def _upload_results_to_s3(config: Config, tournament_audit_data: TournamentAuditData) -> None:
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, UUID):
                return str(obj)
            return super().default(obj)

    upload_data = {
        "tournament_audit_data": tournament_audit_data.model_dump(),
    }

    scores_json = json.dumps(upload_data, indent=2, cls=DateTimeEncoder)

    temp_file, _ = await save_json_to_temp_file(scores_json, "latest_scores", dump_json=False)
    datetime_of_upload = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    presigned_url = await upload_file_to_minio(temp_file, BUCKET_NAME, f"latest_scores_{datetime_of_upload}.json")
    os.remove(temp_file)
    await store_latest_scores_url(presigned_url, config)
    return presigned_url


def calculate_emission_multiplier(performance_diff: float) -> float:
    if performance_diff <= cts.EMISSION_MULTIPLIER_THRESHOLD:
        return 0.0

    excess_performance = performance_diff - cts.EMISSION_MULTIPLIER_THRESHOLD
    emission_increase = excess_performance * cts.EMISSION_MULTIPLIER_RATE

    return emission_increase


async def get_tournament_burn_details(psql_db) -> TournamentBurnData:
    """
    Calculate detailed tournament burn data with calculations for TEXT and IMAGE tournaments.

    This function calculates burn proportions for TEXT and IMAGE tournaments,
    then applies them based on each hotkey's tournament participation.

    Returns:
        TournamentBurnData with performance metrics and weight distributions
    """
    logger.info("=== CALCULATING TOURNAMENT BURN DATA ===")

    text_performance_diff = None
    image_performance_diff = None

    for tournament_type in [TournamentType.TEXT, TournamentType.IMAGE]:
        logger.info(f"Processing {tournament_type} tournament type")
        performance_diff = None

        latest_tournament = await get_latest_completed_tournament(psql_db, tournament_type)
        if latest_tournament:
            logger.info(f"Found latest {tournament_type} tournament: {latest_tournament.tournament_id}")

            previous_tournament = await get_latest_completed_tournament(
                psql_db, tournament_type, exclude_tournament_id=latest_tournament.tournament_id
            )

            winner_changed = False
            if previous_tournament:
                # If latest winner is not EMISSION_BURN_HOTKEY, a new challenger won
                # If it IS EMISSION_BURN_HOTKEY, the defending champion won (use stored performance)
                if previous_tournament.winner_hotkey != latest_tournament.winner_hotkey and latest_tournament.winner_hotkey != cts.EMISSION_BURN_HOTKEY:
                    winner_changed = True
                    logger.info(
                        f"[{tournament_type}] Winner changed: {previous_tournament.winner_hotkey} → {latest_tournament.winner_hotkey}"
                    )
                else:
                    logger.info(f"[{tournament_type}] Same winner defended: {latest_tournament.winner_hotkey}")
            else:
                winner_changed = True
                logger.info(f"[{tournament_type}] First tournament winner: {latest_tournament.winner_hotkey}")

            if winner_changed:
                performance_diff = await calculate_performance_difference(latest_tournament.tournament_id, psql_db)
                logger.info(f"NEW winner - calculated fresh performance difference for {tournament_type}: {performance_diff:.4f}")
            else:
                # Champion defended - get performance from when they ACTUALLY won (not from a defense)
                champion_hotkey = latest_tournament.base_winner_hotkey
                if champion_hotkey:
                    champion_win_tournament = await get_tournament_where_champion_first_won(
                        psql_db, tournament_type, champion_hotkey
                    )
                    if champion_win_tournament and champion_win_tournament.winning_performance_difference is not None:
                        performance_diff = champion_win_tournament.winning_performance_difference
                        logger.info(
                            f"SAME winner - using stored performance difference from when {champion_hotkey} first won "
                            f"(tournament {champion_win_tournament.tournament_id}): {performance_diff:.4f}"
                        )
                    else:
                        logger.warning(f"Could not find tournament where {champion_hotkey} first won for {tournament_type}")
                        performance_diff = 0.0
                else:
                    logger.warning(f"No base_winner_hotkey found for defending champion in {tournament_type}")
                    performance_diff = 0.0

        if performance_diff is None and latest_tournament:
            if latest_tournament.winner_hotkey == cts.EMISSION_BURN_HOTKEY:
                logger.info(
                    f"No performance data available for {tournament_type} tournament, burn account won - assuming worst performance (100% difference)"
                )
                performance_diff = 1.0
            else:
                logger.info(
                    f"No performance data available for {tournament_type} tournament, assuming perfect performance (0% difference)"
                )
                performance_diff = 0.0

        if tournament_type == TournamentType.TEXT:
            text_performance_diff = performance_diff
        elif tournament_type == TournamentType.IMAGE:
            image_performance_diff = performance_diff

    text_emission_increase = calculate_emission_multiplier(text_performance_diff) if text_performance_diff is not None else 0.0
    image_emission_increase = calculate_emission_multiplier(image_performance_diff) if image_performance_diff is not None else 0.0

    logger.info(
        f"Text emission increase (before decay): {text_emission_increase}, Image emission increase (before decay): {image_emission_increase}"
    )

    text_consecutive_wins = 0
    image_consecutive_wins = 0

    latest_text_tournament = await get_latest_completed_tournament(psql_db, TournamentType.TEXT)
    if latest_text_tournament and latest_text_tournament.winner_hotkey:
        winner_hotkey = latest_text_tournament.winner_hotkey
        if winner_hotkey == cts.EMISSION_BURN_HOTKEY:
            winner_hotkey = latest_text_tournament.base_winner_hotkey
        if winner_hotkey:
            text_consecutive_wins = await count_champion_consecutive_wins(psql_db, TournamentType.TEXT, winner_hotkey)
            text_decay = max(0, text_consecutive_wins - 1) * cts.EMISSION_BOOST_DECAY_PER_WIN
            text_emission_increase = max(0.0, text_emission_increase - text_decay)
            logger.info(
                f"Text winner {winner_hotkey} has {text_consecutive_wins} consecutive wins, decay: {text_decay:.4f}, adjusted boost: {text_emission_increase:.4f}"
            )

    latest_image_tournament = await get_latest_completed_tournament(psql_db, TournamentType.IMAGE)
    if latest_image_tournament and latest_image_tournament.winner_hotkey:
        winner_hotkey = latest_image_tournament.winner_hotkey
        if winner_hotkey == cts.EMISSION_BURN_HOTKEY:
            winner_hotkey = latest_image_tournament.base_winner_hotkey
        if winner_hotkey:
            image_consecutive_wins = await count_champion_consecutive_wins(psql_db, TournamentType.IMAGE, winner_hotkey)
            image_decay = max(0, image_consecutive_wins - 1) * cts.EMISSION_BOOST_DECAY_PER_WIN
            image_emission_increase = max(0.0, image_emission_increase - image_decay)
            logger.info(
                f"Image winner {winner_hotkey} has {image_consecutive_wins} consecutive wins, decay: {image_decay:.4f}, adjusted boost: {image_emission_increase:.4f}"
            )

    text_tournament_weight = min(cts.TOURNAMENT_TEXT_WEIGHT + text_emission_increase, cts.MAX_TEXT_TOURNAMENT_WEIGHT)
    image_tournament_weight = min(cts.TOURNAMENT_IMAGE_WEIGHT + image_emission_increase, cts.MAX_IMAGE_TOURNAMENT_WEIGHT)

    burn_weight = 1.0 - text_tournament_weight - image_tournament_weight

    text_burn_proportion = (cts.MAX_TEXT_TOURNAMENT_WEIGHT - text_tournament_weight) / cts.MAX_TEXT_TOURNAMENT_WEIGHT
    image_burn_proportion = (cts.MAX_IMAGE_TOURNAMENT_WEIGHT - image_tournament_weight) / cts.MAX_IMAGE_TOURNAMENT_WEIGHT

    logger.info(f"Weights - Text tournament: {text_tournament_weight}, Image tournament: {image_tournament_weight}")
    logger.info(f"Total burn weight: {burn_weight}")

    return TournamentBurnData(
        text_performance_diff=text_performance_diff,
        image_performance_diff=image_performance_diff,
        text_burn_proportion=text_burn_proportion,
        image_burn_proportion=image_burn_proportion,
        text_tournament_weight=text_tournament_weight,
        image_tournament_weight=image_tournament_weight,
        burn_weight=burn_weight,
    )


def apply_tournament_weights(
    text_tournament_weights: dict[str, float],
    image_tournament_weights: dict[str, float],
    hotkey_to_node_id: dict[str, int],
    all_node_weights: list[float],
    scaled_text_tournament_weight: float,
    scaled_image_tournament_weight: float,
    scaled_text_base_weight: float,
    scaled_image_base_weight: float,
    text_winner_hotkey: str | None,
    image_winner_hotkey: str | None,
) -> float:
    """Apply tournament weights. Returns the total undistributed weight that should go to burn."""
    logger.info("=== TOURNAMENT WEIGHT CALCULATIONS ===")

    text_distributed = 0.0
    logger.info(f"Processing {len(text_tournament_weights)} text tournament winners")
    for hotkey, weight in text_tournament_weights.items():
        node_id = hotkey_to_node_id.get(hotkey)
        if node_id is not None:
            if hotkey == text_winner_hotkey:
                text_contribution = weight * scaled_text_tournament_weight
            else:
                text_contribution = weight * scaled_text_base_weight
            all_node_weights[node_id] = all_node_weights[node_id] + text_contribution
            text_distributed += text_contribution

            logger.info(
                f"Node ID {node_id} (hotkey: {hotkey[:8]}...): "
                f"TEXT TOURNAMENT - weight={weight:.6f}, "
                f"scaled_text_weight={scaled_text_tournament_weight if hotkey == text_winner_hotkey else scaled_text_base_weight:.6f}, "
                f"text_contribution={text_contribution:.6f}, "
                f"total_weight={all_node_weights[node_id]:.6f}"
            )

    text_undistributed = scaled_text_tournament_weight - text_distributed
    logger.info(f"Text tournament: allocated={scaled_text_tournament_weight:.10f}, distributed={text_distributed:.10f}, undistributed={text_undistributed:.10f}")

    image_distributed = 0.0
    logger.info(f"Processing {len(image_tournament_weights)} image tournament winners")
    for hotkey, weight in image_tournament_weights.items():
        node_id = hotkey_to_node_id.get(hotkey)
        if node_id is not None:
            if hotkey == image_winner_hotkey:
                image_contribution = weight * scaled_image_tournament_weight
            else:
                image_contribution = weight * scaled_image_base_weight
            all_node_weights[node_id] = all_node_weights[node_id] + image_contribution
            image_distributed += image_contribution

            logger.info(
                f"Node ID {node_id} (hotkey: {hotkey[:8]}...): "
                f"IMAGE TOURNAMENT - weight={weight:.6f}, "
                f"scaled_image_weight={scaled_image_tournament_weight if hotkey == image_winner_hotkey else scaled_image_base_weight:.6f}, "
                f"image_contribution={image_contribution:.6f}, "
                f"total_weight={all_node_weights[node_id]:.6f}"
            )

    image_undistributed = scaled_image_tournament_weight - image_distributed
    logger.info(f"Image tournament: allocated={scaled_image_tournament_weight:.10f}, distributed={image_distributed:.10f}, undistributed={image_undistributed:.10f}")

    total_undistributed = text_undistributed + image_undistributed
    logger.info(f"Total undistributed weight to add to burn: {total_undistributed:.10f}")

    return total_undistributed


async def get_node_weights_from_tournament_audit_data(
    substrate: SubstrateInterface,
    netuid: int,
    tournament_audit_data: TournamentAuditData,
) -> NodeWeightsResult:
    all_nodes: list[Node] = fetch_nodes.get_nodes_for_netuid(substrate, netuid)
    hotkey_to_node_id: dict[str, int] = {node.hotkey: node.node_id for node in all_nodes}

    all_node_ids: list[int] = [node.node_id for node in all_nodes]
    all_node_weights: list[float] = [0.0 for _ in all_nodes]

    logger.info("=== USING BURN DATA FROM AUDIT ===")

    logger.info(f"Text tournament weight: {tournament_audit_data.text_tournament_weight:.6f}")
    logger.info(f"Image tournament weight: {tournament_audit_data.image_tournament_weight:.6f}")
    logger.info(f"Total burn weight: {tournament_audit_data.burn_weight:.6f}")

    # Check that base weights sum to 1.0
    base_weight_sum = tournament_audit_data.text_tournament_weight + tournament_audit_data.image_tournament_weight + tournament_audit_data.burn_weight
    logger.info(f"Base weights sum (text + image + burn): {base_weight_sum:.10f}")
    logger.info(f"Base weights sum to 1.0? {abs(base_weight_sum - 1.0) < 0.0001}")

    participants: list[str] = tournament_audit_data.participants
    participation_total: float = len(participants) * cts.TOURNAMENT_PARTICIPATION_WEIGHT
    scale_factor: float = 1.0 - participation_total if participation_total > 0 else 1.0

    logger.info(f"Number of participants: {len(participants)}")
    logger.info(f"Participation total weight: {participation_total:.10f}")
    logger.info(f"Scale factor (1.0 - participation_total): {scale_factor:.10f}")

    scaled_text_tournament_weight: float = tournament_audit_data.text_tournament_weight * scale_factor
    scaled_image_tournament_weight: float = tournament_audit_data.image_tournament_weight * scale_factor
    scaled_burn_weight: float = tournament_audit_data.burn_weight * scale_factor

    scaled_text_base_weight: float = cts.TOURNAMENT_TEXT_WEIGHT * scale_factor
    scaled_image_base_weight: float = cts.TOURNAMENT_IMAGE_WEIGHT * scale_factor

    # Check that scaled weights + participation still sum to 1.0
    scaled_weight_sum = scaled_text_tournament_weight + scaled_image_tournament_weight + scaled_burn_weight + participation_total
    logger.info(f"Scaled weights sum (scaled_text + scaled_image + scaled_burn + participation): {scaled_weight_sum:.10f}")
    logger.info(f"Scaled weights sum to 1.0? {abs(scaled_weight_sum - 1.0) < 0.0001}")

    text_tournament_weights, image_tournament_weights = get_tournament_weights_from_data(
        tournament_audit_data.text_tournament_data, tournament_audit_data.image_tournament_data
    )

    text_winner_hotkey = None
    if tournament_audit_data.text_tournament_data:
        text_winner_hotkey = tournament_audit_data.text_tournament_data.winner_hotkey
        if text_winner_hotkey == cts.EMISSION_BURN_HOTKEY:
            text_winner_hotkey = tournament_audit_data.text_tournament_data.base_winner_hotkey

    image_winner_hotkey = None
    if tournament_audit_data.image_tournament_data:
        image_winner_hotkey = tournament_audit_data.image_tournament_data.winner_hotkey
        if image_winner_hotkey == cts.EMISSION_BURN_HOTKEY:
            image_winner_hotkey = tournament_audit_data.image_tournament_data.base_winner_hotkey

    undistributed_weight = apply_tournament_weights(
        text_tournament_weights,
        image_tournament_weights,
        hotkey_to_node_id,
        all_node_weights,
        scaled_text_tournament_weight,
        scaled_image_tournament_weight,
        scaled_text_base_weight,
        scaled_image_base_weight,
        text_winner_hotkey,
        image_winner_hotkey,
    )

    # Check sum after tournament weights applied
    weight_sum_after_tournament = sum(all_node_weights)
    logger.info(f"Weight sum after tournament weights applied: {weight_sum_after_tournament:.10f}")

    for hotkey in participants:
        node_id = hotkey_to_node_id.get(hotkey)
        if node_id is not None:
            all_node_weights[node_id] += cts.TOURNAMENT_PARTICIPATION_WEIGHT

    # Check sum after participation weights added
    weight_sum_after_participation = sum(all_node_weights)
    logger.info(f"Weight sum after participation weights added: {weight_sum_after_participation:.10f}")

    # Add undistributed tournament weight to burn.
    # Undistributed weight comes from the gap between the boosted allocation and what's
    # actually distributed (winner gets boost, non-winners capped at base weight).
    # This ensures total weights sum to exactly 1.0.
    burn_node_id: int | None = hotkey_to_node_id.get(cts.EMISSION_BURN_HOTKEY)
    if burn_node_id is not None:
        all_node_weights[burn_node_id] = scaled_burn_weight + undistributed_weight
        logger.info(f"Burn weight: base={scaled_burn_weight:.10f} + undistributed={undistributed_weight:.10f} = total={all_node_weights[burn_node_id]:.10f}")

    # Final weight sum check
    final_weight_sum = sum(all_node_weights)
    logger.info(f"=== FINAL WEIGHT SUM CHECK ===")
    logger.info(f"Total weight sum (before normalization): {final_weight_sum:.10f}")
    logger.info(f"Expected: 1.0")
    logger.info(f"Difference from 1.0: {abs(final_weight_sum - 1.0):.10f}")
    logger.info(f"Weights sum to 1.0? {abs(final_weight_sum - 1.0) < 0.0001}")
    logger.info(f"Number of non zero node weights: {sum(1 for weight in all_node_weights if weight != 0)}")

    if abs(final_weight_sum - 1.0) >= 0.0001:
        logger.warning(f"⚠️  WARNING: Weights DO NOT sum to 1.0! Sum is {final_weight_sum:.10f}")
    else:
        logger.info(f"✅ Weights correctly sum to 1.0")

    return NodeWeightsResult(node_ids=all_node_ids, node_weights=all_node_weights)


async def build_tournament_audit_data(psql_db) -> TournamentAuditData:
    """
    Build TournamentAuditData with all necessary tournament information.

    This is the central function for gathering tournament data used by both
    the validator (for weight setting) and auditor (for verification).

    Args:
        psql_db: Database connection

    Returns:
        TournamentAuditData with all tournament information populated
    """
    tournament_audit_data = TournamentAuditData()

    # Fetch text tournament data
    text_tournament: TournamentData = await get_latest_completed_tournament(psql_db, TournamentType.TEXT)
    if text_tournament:
        tournament_results: TournamentResults = await get_tournament_full_results(text_tournament.tournament_id, psql_db)
        tournament_audit_data.text_tournament_data = TournamentResultsWithWinners(
            tournament_id=tournament_results.tournament_id,
            rounds=tournament_results.rounds,
            base_winner_hotkey=text_tournament.base_winner_hotkey,
            winner_hotkey=text_tournament.winner_hotkey,
        )

    # Fetch image tournament data
    image_tournament = await get_latest_completed_tournament(psql_db, TournamentType.IMAGE)
    if image_tournament:
        tournament_results = await get_tournament_full_results(image_tournament.tournament_id, psql_db)
        tournament_audit_data.image_tournament_data = TournamentResultsWithWinners(
            tournament_id=tournament_results.tournament_id,
            rounds=tournament_results.rounds,
            base_winner_hotkey=image_tournament.base_winner_hotkey,
            winner_hotkey=image_tournament.winner_hotkey,
        )

    # Fetch participants
    tournament_audit_data.participants = await get_active_tournament_participants(psql_db)

    # Fetch burn weights
    burn_data: TournamentBurnData = await get_tournament_burn_details(psql_db)
    tournament_audit_data.text_tournament_weight = burn_data.text_tournament_weight
    tournament_audit_data.image_tournament_weight = burn_data.image_tournament_weight
    tournament_audit_data.burn_weight = burn_data.burn_weight

    # Fetch weekly participation data
    tournament_audit_data.weekly_participation = await get_weekly_task_participation_data(psql_db)

    return tournament_audit_data


async def set_weights(config: Config, all_node_ids: list[int], all_node_weights: list[float], validator_node_id: int) -> bool:
    try:
        success = await asyncio.to_thread(
            weights.set_node_weights,
            substrate=config.substrate,
            keypair=config.keypair,
            node_ids=all_node_ids,
            node_weights=all_node_weights,
            netuid=config.netuid,
            version_key=ccst.VERSION_KEY,
            validator_node_id=int(validator_node_id),
            wait_for_inclusion=False,
            wait_for_finalization=False,
            max_attempts=3,
        )
    except Exception as e:
        logger.error(f"Failed to set weights: {e}")
        return False

    if success:
        logger.info("Weights set successfully.")

        return True
    else:
        logger.error("Failed to set weights :(")
        return False


async def _get_and_set_weights(config: Config, validator_node_id: int) -> bool:
    # Build tournament audit data using the centralized function
    tournament_audit_data: TournamentAuditData = await build_tournament_audit_data(config.psql_db)

    result = await get_node_weights_from_tournament_audit_data(config.substrate, config.netuid, tournament_audit_data)
    all_node_ids = result.node_ids
    all_node_weights = result.node_weights
    logger.info("Weights calculated, about to set...")

    success = await set_weights(config, all_node_ids, all_node_weights, validator_node_id)
    if success:
        # Upload both task results and tournament data
        url = await _upload_results_to_s3(config, tournament_audit_data)
        logger.info(f"Uploaded the scores and tournament data to s3 for auditing - url: {url}")

    return success


async def _set_metagraph_weights(config: Config) -> None:
    nodes: list[Node] = fetch_nodes.get_nodes_for_netuid(config.substrate, config.netuid)
    node_ids = [node.node_id for node in nodes]
    node_weights = [node.incentive for node in nodes]
    validator_node_id = await get_vali_node_id(config.substrate, config.keypair.ss58_address)
    if validator_node_id is None:
        raise ValueError("Validator node id not found")

    await asyncio.to_thread(
        weights.set_node_weights,
        substrate=config.substrate,
        keypair=config.keypair,
        node_ids=node_ids,
        node_weights=node_weights,
        netuid=config.netuid,
        version_key=ccst.VERSION_KEY,
        validator_node_id=int(validator_node_id),
        wait_for_inclusion=False,
        wait_for_finalization=False,
        max_attempts=3,
    )


# To improve: use activity cutoff & The epoch length to set weights at the perfect times
async def set_weights_periodically(config: Config, just_once: bool = False) -> None:
    substrate = config.substrate
    substrate, uid = query_substrate(
        substrate,
        "SubtensorModule",
        "Uids",
        [config.netuid, config.keypair.ss58_address],
        return_value=True,
    )

    if uid is None:
        raise ValueError(f"Can't find hotkey {config.keypair.ss58_address} for our keypair on netuid: {config.netuid}.")

    consecutive_failures = 0
    while True:
        substrate, current_block = query_substrate(substrate, "System", "Number", [], return_value=True)
        substrate, last_updated_value = query_substrate(
            substrate, "SubtensorModule", "LastUpdate", [config.netuid], return_value=False
        )
        updated: int = current_block - last_updated_value[uid]
        substrate, weights_set_rate_limit = query_substrate(
            substrate, "SubtensorModule", "WeightsSetRateLimit", [config.netuid], return_value=True
        )
        logger.info(
            f"My Validator Node ID: {uid}. Last updated {updated} blocks ago. Weights set rate limit: {weights_set_rate_limit}."
        )

        if updated < weights_set_rate_limit:
            logger.info("Sleeping for a bit as we set recently...")
            await asyncio.sleep((weights_set_rate_limit - updated + 1) * 12)
            continue

        if os.getenv("ENV", "prod").lower() == "dev":
            success = await _get_and_set_weights(config, uid)
        else:
            try:
                success = await _get_and_set_weights(config, uid)
            except Exception as e:
                logger.error(f"Failed to set weights with error: {e}")
                logger.exception(e)
                success = False

        if success:
            consecutive_failures = 0
            logger.info("Successfully set weights! Sleeping for 25 blocks before next check...")
            if just_once:
                return
            await asyncio.sleep(12 * 25)
            continue

        consecutive_failures += 1
        if just_once:
            logger.info("Failed to set weights, will try again...")
            await asyncio.sleep(12 * 1)
        else:
            logger.info(f"Failed to set weights {consecutive_failures} times in a row - sleeping for a bit...")
            await asyncio.sleep(12 * 25)  # Try again in 25 blocks

        if consecutive_failures == 1 or updated < 3000:
            continue

        if just_once or config.set_metagraph_weights_with_high_updated_to_not_dereg:
            logger.warning("Setting metagraph weights as our updated value is getting too high!")
            if just_once:
                logger.warning("Please exit if you do not want to do this!!!")
                await asyncio.sleep(4)
            try:
                success = await _set_metagraph_weights(config)
            except Exception as e:
                logger.error(f"Failed to set metagraph weights: {e}")
                success = False

            if just_once:
                return

            if success:
                consecutive_failures = 0
                continue


async def main():
    config = load_config()
    await try_db_connections(config)
    await set_weights_periodically(config)


if __name__ == "__main__":
    asyncio.run(main())
