from fiber.chain.models import Node

from core.models.tournament_models import GroupRound
from core.models.tournament_models import KnockoutRound
from core.models.tournament_models import Round
from core.models.tournament_models import RoundStatus
from core.models.tournament_models import RoundType
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentParticipant
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentStatus
from core.models.tournament_models import TournamentTask
from core.models.tournament_models import TournamentType
from core.models.tournament_models import generate_round_id
from core.models.tournament_models import generate_tournament_id
from core.models.tournament_models import get_tournament_gpu_requirement
from validator.core.config import Config
from validator.core.constants import IMAGE_TASKS_PER_GROUP
from validator.core.constants import TEXT_TASKS_PER_GROUP
from validator.db.database import PSQLDB
from validator.db.sql.tournaments import add_tournament_participants
from validator.db.sql.tournaments import add_tournament_tasks
from validator.db.sql.tournaments import create_tournament
from validator.db.sql.tournaments import create_tournament_groups_with_members
from validator.db.sql.tournaments import create_tournament_pairs
from validator.db.sql.tournaments import create_tournament_round
from validator.db.sql.tournaments import get_active_tournaments
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import update_tournament_status
from validator.db.sql.tasks import get_task
from validator.tournament.organiser import organise_tournament_round
from validator.tournament.task_creator import create_image_tournament_round
from validator.tournament.task_creator import create_text_tournament_round
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def create_new_tournament(
    tournament_type: TournamentType,
    eligible_nodes: list[Node],
    psql_db: PSQLDB,
    config: Config
) -> str:
    tournament_id = generate_tournament_id()

    tournament_data = TournamentData(
        tournament_id=tournament_id,
        tournament_type=tournament_type,
        status=TournamentStatus.PENDING
    )

    await create_tournament(tournament_data, psql_db)
    logger.info(f"Created tournament {tournament_id} with {len(eligible_nodes)} nodes")

    participants = [
        TournamentParticipant(tournament_id=tournament_id, hotkey=node.hotkey)
        for node in eligible_nodes
    ]
    await add_tournament_participants(participants, psql_db)

    await _create_first_round(tournament_id, tournament_type, eligible_nodes, psql_db, config)

    return tournament_id


async def _create_first_round(
    tournament_id: str,
    tournament_type: TournamentType,
    nodes: list[Node],
    psql_db: PSQLDB,
    config: Config
):
    round_id = generate_round_id(tournament_id, 1)
    round_structure = organise_tournament_round(nodes)

    is_final = len(nodes) <= 2
    round_type = RoundType.KNOCKOUT if isinstance(round_structure, KnockoutRound) else RoundType.GROUP

    round_data = TournamentRoundData(
        round_id=round_id,
        tournament_id=tournament_id,
        round_number=1,
        round_type=round_type,
        is_final_round=is_final,
        status=RoundStatus.PENDING
    )

    await create_tournament_round(round_data, psql_db)

    if isinstance(round_structure, GroupRound):
        await create_tournament_groups_with_members(round_id, round_structure, psql_db)
    else:
        await create_tournament_pairs(round_id, round_structure, psql_db)

    tasks = await _create_tournament_tasks(tournament_id, round_id, round_structure, tournament_type, is_final, config)
    await add_tournament_tasks(tasks, psql_db)

    logger.info(f"Created first round {round_id} with {len(tasks)} tasks")


async def _create_tournament_tasks(
    tournament_id: str,
    round_id: str,
    round_structure: Round,
    tournament_type: TournamentType,
    is_final: bool,
    config: Config
) -> list[TournamentTask]:
    if tournament_type == TournamentType.TEXT:
        tournament_round = await create_text_tournament_round(round_structure, config, is_final)
    else:
        tournament_round = await create_image_tournament_round(round_structure, config)

    tasks = []
    if isinstance(round_structure, GroupRound):
        group_task_count = TEXT_TASKS_PER_GROUP if tournament_type == TournamentType.TEXT else IMAGE_TASKS_PER_GROUP

        for i, group in enumerate(round_structure.groups):
            group_id = f"{round_id}_group_{i+1:03d}"

            for j in range(group_task_count):
                task_index = i * group_task_count + j
                if task_index < len(tournament_round.tasks):
                    task_id = tournament_round.tasks[task_index]
                    task_data = await get_task(task_id, config.psql_db)
                    gpu_requirement = get_tournament_gpu_requirement(task_data.task_type, task_data.model_params_count)
                    
                    task = TournamentTask(
                        tournament_id=tournament_id,
                        round_id=round_id,
                        task_id=task_id,
                        group_id=group_id,
                        pair_id=None,
                        gpu_requirement=gpu_requirement
                    )
                    tasks.append(task)
    else:
        for i, pair in enumerate(round_structure.pairs):
            pair_id = f"{round_id}_pair_{i+1:03d}"
            if i < len(tournament_round.tasks):
                task_id = tournament_round.tasks[i]
                task_data = await get_task(task_id, config.psql_db)
                gpu_requirement = get_tournament_gpu_requirement(task_data.task_type, task_data.model_params_count)
                
                task = TournamentTask(
                    tournament_id=tournament_id,
                    round_id=round_id,
                    task_id=task_id,
                    group_id=None,
                    pair_id=pair_id,
                    gpu_requirement=gpu_requirement
                )
                tasks.append(task)

    return tasks


async def start_tournament(tournament_id: str, psql_db: PSQLDB):
    await update_tournament_status(tournament_id, TournamentStatus.ACTIVE, psql_db)
    logger.info(f"Started tournament {tournament_id}")


async def complete_tournament(tournament_id: str, psql_db: PSQLDB):
    await update_tournament_status(tournament_id, TournamentStatus.COMPLETED, psql_db)
    logger.info(f"Completed tournament {tournament_id}")


async def get_tournament_status(tournament_id: str, psql_db: PSQLDB) -> TournamentData | None:
    return await get_tournament(tournament_id, psql_db)


async def list_active_tournaments(psql_db: PSQLDB) -> list[TournamentData]:
    return await get_active_tournaments(psql_db)


async def create_text_tournament_with_database(
    eligible_nodes: list[Node],
    psql_db: PSQLDB,
    config: Config
) -> str:
    tournament_id = await create_new_tournament(TournamentType.TEXT, eligible_nodes, psql_db, config)
    await start_tournament(tournament_id, psql_db)
    return tournament_id


async def create_image_tournament_with_database(
    eligible_nodes: list[Node],
    psql_db: PSQLDB,
    config: Config
) -> str:
    tournament_id = await create_new_tournament(TournamentType.IMAGE, eligible_nodes, psql_db, config)
    await start_tournament(tournament_id, psql_db)
    return tournament_id
