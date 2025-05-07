import uuid
from datetime import datetime

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import status

from core.models.payload_models import ModelCheckRequest
from core.models.payload_models import ModelCheckSubmissionResponse
from core.models.utility_models import ModelCheckStatus
from validator.core.config import Config
from validator.core.dependencies import get_api_key
from validator.core.dependencies import get_config
from validator.core.models import ModelCheckQueueEntry
from validator.db.sql import model_checks as mc_sql
from validator.model_check.utils import check_model_in_content_service
from validator.utils.logging import get_logger


MODEL_CHECKS_REQUEST_ENDPOINT = "/request"
MODEL_CHECKS_STATUS_ENDPOINT = "/{request_id}"

logger = get_logger(__name__)


async def request_model_check(
    request: ModelCheckRequest,
    config: Config = Depends(get_config),
) -> ModelCheckSubmissionResponse:
    """
    Submits a request to check if a Hugging Face model is suitable for training.
    The model will be queued for loading, parameter count, and basic inference testing.
    """
    new_check_entry = ModelCheckQueueEntry(
        model_id=request.model_id,
    )

    try:
        content_service_result = await check_model_in_content_service(new_check_entry.model_id, config.keypair)

        if content_service_result.model_found and content_service_result.is_suitable is not None:
            if content_service_result.is_suitable:
                new_check_entry.status = ModelCheckStatus.SUCCESS
                logger.info(f"Model {new_check_entry.model_id} marked as SUCCESS based on content service.")
            else:
                new_check_entry.status = ModelCheckStatus.FAILURE
                new_check_entry.error_message = "Model marked as unsuitable by content service."
                logger.info(
                    f"Model {new_check_entry.model_id} marked as FAILURE based on content service: {new_check_entry.error_message}"
                )

            new_check_entry.processed_at = datetime.utcnow()
            new_check_entry.parameter_count = content_service_result.parameter_count
        else:
            logger.info(
                f"Model {new_check_entry.model_id} will be queued for a full check (Content service: found={content_service_result.model_found}, suitable={content_service_result.is_suitable})."
            )

        created_entry = await mc_sql.add_model_check_request(new_check_entry, config.psql_db)

        logger.info(
            f"Model check request for {request.model_id} (ID {created_entry.id}) processed. Status: {created_entry.status}"
        )

        message = "Model check request successfully submitted and queued."
        if created_entry.status == ModelCheckStatus.SUCCESS:
            message = "Model check successful based on content service."
        elif created_entry.status == ModelCheckStatus.FAILURE:
            message = "Model check failed based on content service."
            if created_entry.error_message:
                message += f" Reason: {created_entry.error_message}"

        return ModelCheckSubmissionResponse(
            request_id=created_entry.id,
            model_id=created_entry.model_id,
            status=created_entry.status,
            message=message,
        )
    except Exception as e:
        logger.error(f"Failed to submit model check request for {request.model_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit model check request: {str(e)}",
        )


async def get_model_check_status(
    request_id: uuid.UUID,
    config: Config = Depends(get_config),
) -> ModelCheckQueueEntry:
    """
    Retrieves the status and results of a specific model check request.
    """
    db_entry = await mc_sql.get_model_check_by_id(request_id, config.psql_db)
    if not db_entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model check request not found.")
    return db_entry


def factory_router() -> APIRouter:
    router = APIRouter(
        prefix="/model_checks",
        tags=["Model Checks"],
        dependencies=[Depends(get_api_key)],
    )
    router.add_api_route(
        MODEL_CHECKS_REQUEST_ENDPOINT,
        request_model_check,
        methods=["POST"],
        response_model=ModelCheckSubmissionResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    router.add_api_route(
        MODEL_CHECKS_STATUS_ENDPOINT,
        get_model_check_status,
        methods=["GET"],
        response_model=ModelCheckQueueEntry,
    )
    return router
