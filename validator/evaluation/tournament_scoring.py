from core.models.utility_models import TaskType
from core.models.tournament_models import TournamentType, TournamentTaskScore
from validator.db.sql.tournaments import get_latest_completed_tournament, get_tournament_full_results
import validator.core.constants as cts


def calculate_final_round_winner(task: TournamentTaskScore, prev_winner_hotkey: str, task_type: TaskType) -> str | None:
    if len(task.participant_scores) < 2:
        return None
    
    prev_winner_score = None
    contender_score = None
    contender_hotkey = None
    
    for score_data in task.participant_scores:
        hotkey = score_data.get('hotkey')
        test_loss = score_data.get('test_loss')
        synth_loss = score_data.get('synth_loss')
        
        if not test_loss or not synth_loss:
            continue
            
        if hotkey == prev_winner_hotkey:
            prev_winner_score = (test_loss, synth_loss)
        elif contender_hotkey is None:
            contender_hotkey = hotkey
            contender_score = (test_loss, synth_loss)
    
    if not (prev_winner_score and contender_score and contender_hotkey):
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


async def calculate_tournament_type_scores(tournament_type: TournamentType, psql_db) -> dict[str, float]:
    latest_tournament = await get_latest_completed_tournament(psql_db, tournament_type)
    if not latest_tournament:
        return {}
    
    tournament_data = await get_tournament_full_results(latest_tournament.tournament_id, psql_db)
    prev_winner_hotkey = latest_tournament.base_winner_hotkey
    
    type_weight = cts.TOURNAMENT_TEXT_WEIGHT if tournament_type == TournamentType.TEXT else cts.TOURNAMENT_IMAGE_WEIGHT
    scores = {}
    
    for round_result in tournament_data.rounds:
        round_number = round_result.round_number
        is_final_round = round_result.is_final_round
        
        for task in round_result.tasks:
            if is_final_round and prev_winner_hotkey:
                from validator.db.sql.tasks import get_task
                task_obj = await get_task(task.task_id, psql_db)
                if task_obj:
                    winner = calculate_final_round_winner(task, prev_winner_hotkey, task_obj.task_type)
                else:
                    winner = task.winner
            else:
                winner = task.winner
            
            if winner and winner != prev_winner_hotkey:
                if winner not in scores:
                    scores[winner] = 0
                scores[winner] += round_number * type_weight
    
    return scores


def linear_decline_mapping(total_participants: int, rank: float) -> float:
    if total_participants <= 1:
        return 1.0
    return 1.0 - (rank - 1) / (total_participants - 1)


def tournament_scores_to_weights(tournament_scores: dict[str, float]) -> dict[str, float]:
    if not tournament_scores:
        return {}
    
    # Filter out zero scores
    non_zero_scores = {hotkey: score for hotkey, score in tournament_scores.items() if score > 0}
    if not non_zero_scores:
        return {}
    
    # Group by score to handle ties
    score_groups = {}
    for hotkey, score in non_zero_scores.items():
        if score not in score_groups:
            score_groups[score] = []
        score_groups[score].append(hotkey)
    
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
        
        weight = linear_decline_mapping(total_participants, avg_rank)
        
        # Assign same weight to all tied participants
        for hotkey in hotkeys_with_score:
            weights[hotkey] = weight
        
        current_rank += len(hotkeys_with_score)
    
    return weights


async def get_tournament_scores(psql_db) -> dict[str, float]:
    all_scores = {}
    
    for tournament_type in [TournamentType.TEXT, TournamentType.IMAGE]:
        type_scores = await calculate_tournament_type_scores(tournament_type, psql_db)
        
        for hotkey, score in type_scores.items():
            if hotkey not in all_scores:
                all_scores[hotkey] = 0
            all_scores[hotkey] += score
    
    return all_scores


async def get_tournament_weights(psql_db) -> dict[str, float]:
    tournament_scores = await get_tournament_scores(psql_db)
    return tournament_scores_to_weights(tournament_scores)