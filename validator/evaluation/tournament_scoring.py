from typing import Optional

import validator.core.constants as cts
from core.models.tournament_models import TournamentResultsWithWinners
from core.models.tournament_models import TournamentScore
from core.models.tournament_models import TournamentTaskScore
from core.models.tournament_models import TournamentType
from core.models.tournament_models import TournamentTypeResult
from core.models.utility_models import TaskType
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def calculate_final_round_winner(task: TournamentTaskScore, prev_winner_hotkey: str, task_type: TaskType) -> str | None:
    if len(task.participant_scores) < 2:
        return None

    prev_winner_score = None
    contender_score = None
    contender_hotkey = None

    for score_data in task.participant_scores:
        hotkey = score_data.get("hotkey")
        test_loss = score_data.get("test_loss")
        synth_loss = score_data.get("synth_loss")

        if not test_loss or not synth_loss:
            continue

        if hotkey == prev_winner_hotkey:
            prev_winner_score = (test_loss, synth_loss)
        elif contender_hotkey is None:
            contender_hotkey = hotkey
            contender_score = (test_loss, synth_loss)
    if prev_winner_score and not contender_score:
        return prev_winner_hotkey
    elif contender_score and not prev_winner_score:
        return contender_hotkey
    elif not (prev_winner_score and contender_score and contender_hotkey):
        return None

    prev_test, prev_synth = prev_winner_score
    cont_test, cont_synth = contender_score

    prev_loss = max(prev_test, prev_synth)
    cont_loss = max(cont_test, cont_synth)

    if task_type == TaskType.GRPOTASK:
        if cont_loss > prev_loss * 1.05:
            return contender_hotkey
        else:
            return prev_winner_hotkey
    else:
        if cont_loss * 1.05 < prev_loss:
            return contender_hotkey
        else:
            return prev_winner_hotkey


def calculate_tournament_type_scores_from_data(
    tournament_type: TournamentType, tournament_data: TournamentResultsWithWinners | None
) -> TournamentTypeResult:
    """Calculate tournament scores from tournament data without database access."""
    if not tournament_data:
        return TournamentTypeResult(scores=[], prev_winner_hotkey=None, prev_winner_won_final=False)

    type_weight = cts.TOURNAMENT_TEXT_WEIGHT if tournament_type == TournamentType.TEXT else cts.TOURNAMENT_IMAGE_WEIGHT
    score_dict = {}
    prev_winner_won_final = False

    for round_result in tournament_data.rounds:
        round_number = round_result.round_number
        is_final_round = round_result.is_final_round

        for task in round_result.tasks:
            winner = task.winner

            if is_final_round and tournament_data.winner_hotkey and winner == tournament_data.winner_hotkey:
                prev_winner_won_final = True

            if winner and winner != tournament_data.winner_hotkey:
                if winner not in score_dict:
                    score_dict[winner] = 0
                score_dict[winner] += round_number * type_weight

    scores = [TournamentScore(hotkey=hotkey, score=score) for hotkey, score in score_dict.items()]

    return TournamentTypeResult(
        scores=scores, prev_winner_hotkey=tournament_data.winner_hotkey, prev_winner_won_final=prev_winner_won_final
    )


def exponential_decline_mapping(total_participants: int, rank: float) -> float:
    """Exponential weight decay based on rank."""
    if total_participants <= 1:
        return 1.0
    
    decay_factor = (rank - 1) / (total_participants - 1)
    return 1.0 / (cts.TOURNAMENT_WEIGHT_DECAY_RATE ** decay_factor)



def tournament_scores_to_weights(
    tournament_scores: list[TournamentScore], prev_winner_hotkey: str | None, prev_winner_won_final: bool
) -> dict[str, float]:
    if not tournament_scores and not prev_winner_hotkey:
        return {}

    # Filter out zero scores
    non_zero_scores = [score for score in tournament_scores if score.score > 0]

    # If we have a previous winner, place them appropriately
    if prev_winner_hotkey:
        if prev_winner_won_final:
            # Previous winner won final round, place them 1st
            prev_winner_score = TournamentScore(hotkey=prev_winner_hotkey, score=float("inf"))
            non_zero_scores.insert(0, prev_winner_score)
        else:
            # Check if prev_winner is in the scores (meaning they participated and lost)
            # vs won by default (not in scores, won because others failed)
            prev_winner_in_scores = any(score.hotkey == prev_winner_hotkey for score in non_zero_scores)
            
            if prev_winner_in_scores:
                # Previous winner participated but lost final round, place them 2nd
                if len(non_zero_scores) > 0:
                    max_score = max(score.score for score in non_zero_scores)
                    prev_winner_score = TournamentScore(hotkey=prev_winner_hotkey, score=max_score - 0.1)
                    non_zero_scores.append(prev_winner_score)
            else:
                # Previous winner won by default (not in scores), place them 1st
                prev_winner_score = TournamentScore(hotkey=prev_winner_hotkey, score=float("inf"))
                non_zero_scores.insert(0, prev_winner_score)

    if not non_zero_scores:
        return {}

    # Group by score to handle ties
    score_groups = {}
    for tournament_score in non_zero_scores:
        score = tournament_score.score
        if score not in score_groups:
            score_groups[score] = []
        score_groups[score].append(tournament_score.hotkey)

    # Sort scores in descending order
    sorted_scores = sorted(score_groups.keys(), reverse=True)

    # Calculate weights
    total_participants = len(non_zero_scores)
    weights = {}

    current_rank = 1
    for score in sorted_scores:
        hotkeys_with_score = score_groups[score]

        # Calculate average rank for tied participants
        if len(hotkeys_with_score) == 1:
            avg_rank = current_rank
        else:
            avg_rank = current_rank + (len(hotkeys_with_score) - 1) / 2

        weight = exponential_decline_mapping(total_participants, avg_rank)

        # Assign same weight to all tied participants
        for hotkey in hotkeys_with_score:
            weights[hotkey] = weight

        current_rank += len(hotkeys_with_score)

    return weights


def get_tournament_weights_from_data(
    text_tournament_data: Optional[TournamentResultsWithWinners],
    image_tournament_data: Optional[TournamentResultsWithWinners],
) -> dict[str, float]:
    text_result = calculate_tournament_type_scores_from_data(TournamentType.TEXT, text_tournament_data)
    image_result = calculate_tournament_type_scores_from_data(TournamentType.IMAGE, image_tournament_data)

    logger.info(f"Text tournament scores: {text_result.scores}")
    logger.info(f"Image tournament scores: {image_result.scores}")
    logger.info(f"Text tournament prev winner: {text_result.prev_winner_hotkey}")
    logger.info(f"Image tournament prev winner: {image_result.prev_winner_hotkey}")

    combined_scores = {}
    all_tournament_scores = text_result.scores + image_result.scores

    for tournament_score in all_tournament_scores:
        if tournament_score.hotkey not in combined_scores:
            combined_scores[tournament_score.hotkey] = 0
        combined_scores[tournament_score.hotkey] += tournament_score.score

    combined_score_list = [TournamentScore(hotkey=hotkey, score=score) for hotkey, score in combined_scores.items()]

    prev_winner_hotkey = text_result.prev_winner_hotkey or image_result.prev_winner_hotkey
    prev_winner_won_final = text_result.prev_winner_won_final or image_result.prev_winner_won_final

    logger.info(f"Combined tournament scores: {combined_score_list}")
    logger.info(f"Overall prev winner: {prev_winner_hotkey}")
    logger.info(f"Prev winner won final: {prev_winner_won_final}")

    result = tournament_scores_to_weights(combined_score_list, prev_winner_hotkey, prev_winner_won_final)
    logger.info(f"Final tournament weights: {result}")
    return result
