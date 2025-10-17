from fastapi import Depends
from fastapi.routing import APIRouter
from fiber.miner.dependencies import blacklist_low_stake
from fiber.miner.dependencies import verify_get_request

from core.models.payload_models import TrainingRepoResponse
from core.models.tournament_models import TournamentType


async def get_training_repo(task_type: TournamentType) -> TrainingRepoResponse:
    return TrainingRepoResponse(
        github_repo="https://github.com/rayonlabs/G.O.D", commit_hash="076e87fc746985e272015322cc91fb3bbbca2f26"
    )


def factory_router() -> APIRouter:
    router = APIRouter()

    router.add_api_route(
        "/training_repo/{task_type}",
        get_training_repo,
        tags=["Subnet"],
        methods=["GET"],
        response_model=TrainingRepoResponse,
        summary="Get Training Repo",
        description="Retrieve the training repository and commit hash for the tournament.",
        dependencies=[Depends(blacklist_low_stake), Depends(verify_get_request)],
    )

    return router