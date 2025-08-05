from validator.db.database import PSQLDB
from core.models.tournament_models import BossRoundTaskCompletion, BossRoundTaskPair, TaskScore
from validator.db.sql.submissions_and_scoring import get_all_scores_and_losses_for_task
from validator.db import constants as cst


async def get_boss_round_synthetic_task_completion(tournament_id: str, psql_db: PSQLDB) -> BossRoundTaskCompletion:
    async with await psql_db.connection() as connection:
        query = """
            SELECT COUNT(*) as total_synth_tasks,
                   COUNT(CASE WHEN t.status = 'success' THEN 1 END) as completed_synth_tasks
            FROM boss_round_synced_tasks brst
            JOIN tasks t ON t.task_id = brst.general_task_id
            JOIN tasks tournament_task ON tournament_task.task_id = brst.tournament_task_id
            JOIN tournament_tasks tt ON tt.task_id = tournament_task.task_id
            JOIN tournament_rounds tr ON tr.round_id = tt.round_id
            WHERE tr.tournament_id = $1 AND tr.is_final_round = true
        """
        result = await connection.fetchrow(query, tournament_id)
        return BossRoundTaskCompletion(
            total_synth_tasks=result['total_synth_tasks'],
            completed_synth_tasks=result['completed_synth_tasks']
        )


async def get_boss_round_winner_task_pairs(tournament_id: str, psql_db: PSQLDB) -> list[BossRoundTaskPair]:
    async with await psql_db.connection() as connection:
        query = """
            SELECT t.task_id, t.task_type, tourn.winner_hotkey, brst.general_task_id
            FROM tasks t
            JOIN tournament_tasks tt ON tt.task_id = t.task_id
            JOIN tournament_rounds tr ON tr.round_id = tt.round_id
            JOIN tournaments tourn ON tourn.tournament_id = tr.tournament_id
            JOIN boss_round_synced_tasks brst ON brst.tournament_task_id = t.task_id
            WHERE tr.tournament_id = $1 AND tr.is_final_round = true
        """
        results = await connection.fetch(query, tournament_id)
        return [
            BossRoundTaskPair(
                tournament_task_id=str(row['task_id']),
                synthetic_task_id=str(row['general_task_id']),
                winner_hotkey=row['winner_hotkey'],
                task_type=row['task_type']
            )
            for row in results
        ]


async def get_task_scores_as_models(task_id: str, psql_db: PSQLDB) -> list[TaskScore]:
    raw_scores = await get_all_scores_and_losses_for_task(task_id, psql_db)
    return [
        TaskScore(
            hotkey=score[cst.HOTKEY],
            test_loss=score[cst.TEST_LOSS],
            synth_loss=score[cst.SYNTH_LOSS],
            quality_score=score[cst.TASK_NODE_QUALITY_SCORE]
        )
        for score in raw_scores
        if (score[cst.TEST_LOSS] is not None and not (isinstance(score[cst.TEST_LOSS], float) and score[cst.TEST_LOSS] != score[cst.TEST_LOSS])) and 
           (score[cst.SYNTH_LOSS] is not None and not (isinstance(score[cst.SYNTH_LOSS], float) and score[cst.SYNTH_LOSS] != score[cst.SYNTH_LOSS]))
    ]


async def get_task_scores_batch(task_ids: list[str], psql_db: PSQLDB) -> dict[str, list[TaskScore]]:
    """Fetch task scores for multiple tasks in a single query to avoid N+1 problem."""
    if not task_ids:
        return {}
    
    async with await psql_db.connection() as connection:
        # Convert task_ids to UUIDs for query
        query = f"""
            SELECT task_id, hotkey, test_loss, synth_loss, quality_score
            FROM {cst.TASK_NODES_TABLE}
            WHERE task_id = ANY($1::uuid[])
        """
        results = await connection.fetch(query, task_ids)
        
        # Group results by task_id
        task_scores = {}
        for row in results:
            task_id = str(row['task_id'])
            if task_id not in task_scores:
                task_scores[task_id] = []
            
            # Filter out NaN values
            test_loss = row['test_loss']
            synth_loss = row['synth_loss']
            
            if (test_loss is not None and not (isinstance(test_loss, float) and test_loss != test_loss)) and \
               (synth_loss is not None and not (isinstance(synth_loss, float) and synth_loss != synth_loss)):
                task_scores[task_id].append(TaskScore(
                    hotkey=row['hotkey'],
                    test_loss=test_loss,
                    synth_loss=synth_loss,
                    quality_score=row['quality_score']
                ))
        
        return task_scores


async def get_previous_completed_tournament(psql_db: PSQLDB, tournament_type: str, exclude_tournament_id: str = None) -> str | None:
    async with await psql_db.connection() as connection:
        if exclude_tournament_id:
            query = """
                SELECT tournament_id
                FROM tournaments
                WHERE tournament_type = $1
                AND status = 'completed'
                AND tournament_id != $2
                ORDER BY created_at DESC
                LIMIT 1
            """
            result = await connection.fetchrow(query, tournament_type, exclude_tournament_id)
        else:
            query = """
                SELECT tournament_id
                FROM tournaments
                WHERE tournament_type = $1
                AND status = 'completed'
                ORDER BY created_at DESC
                LIMIT 1
            """
            result = await connection.fetchrow(query, tournament_type)
        return result['tournament_id'] if result else None