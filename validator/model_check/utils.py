import httpx
from fiber import Keypair

import validator.core.constants as cst
from validator.core.models import ContentServiceCheckResult
from validator.utils.call_endpoint import call_content_service
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def check_model_in_content_service(model_id: str, keypair: Keypair) -> ContentServiceCheckResult:
    """
    Check if a model exists in the content service and get its suitability status
    using the signed call_content_service utility.

    Returns:
        ContentServiceCheckResult: An object containing the check results.
    """
    result = ContentServiceCheckResult()

    try:
        endpoint_url = cst.GET_MODEL_INFO_ENDPOINT.format(model=model_id)
        logger.debug(f"Checking content service URL via call_content_service: {endpoint_url}")

        data = await call_content_service(endpoint=endpoint_url, keypair=keypair)

        result.model_found = True

        if "is_suitable" in data:
            result.is_suitable = data.get("is_suitable")

        if "parameter_count" in data and data.get("parameter_count") is not None:
            try:
                result.parameter_count = int(data["parameter_count"])
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not parse parameter_count '{data['parameter_count']}' from content service for {model_id}."
                )

        logger.info(
            f"Model {model_id} found in content service. Suitable: {result.is_suitable}, Parameters: {result.parameter_count}"
        )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.info(f"Model {model_id} not found in content service (status: 404) via call_content_service.")
        else:
            logger.warning(f"call_content_service returned status {e.response.status_code} for model {model_id}.")
    except Exception as e:
        logger.error(
            f"Unexpected error checking model {model_id} with call_content_service: {type(e).__name__} - {str(e)}",
            exc_info=True,
        )

    return result
