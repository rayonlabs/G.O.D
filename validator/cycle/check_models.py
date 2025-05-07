import asyncio
from typing import Any

from fiber import Keypair

import validator.core.constants as cst
from core.models.utility_models import ModelCheckStatus
from validator.core.config import Config
from validator.core.gpu_management import GPUQueueManager
from validator.db.sql import model_checks as mc_sql
from validator.evaluation.utils import compute_required_gpus
from validator.model_check.text_model_check import run_model_loading_and_inference_check
from validator.utils.call_endpoint import post_to_content_service
from validator.utils.logging import LogContext
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def update_content_service_model_info(
    model_id: str,
    is_suitable: bool,
    keypair: Keypair,
    parameter_count: int | None = None,
) -> dict[str, Any] | None:
    """
    Update the model information (suitability, parameter count) in the content service
    by calling the generic post_to_content_service function.
    """
    payload: dict[str, Any] = {
        "model_id": model_id,
        "is_suitable": is_suitable,
    }
    if parameter_count is not None:
        payload["parameter_count"] = parameter_count

    logger.info(f"Updating content service for model {model_id} with payload: {payload}")

    return await post_to_content_service(
        endpoint=cst.POST_MODEL_INFO_ENDPOINT.format(model=model_id),
        keypair=keypair,
        payload=payload,
    )


async def model_testing_loop(config: Config, gpu_manager: GPUQueueManager):
    """Loop to fetch model check requests from queue and test them."""
    while True:
        model_check_task = None
        log_context_details = {"loop_name": "model_testing_loop"}
        try:
            model_check_task = await mc_sql.get_oldest_pending_model_check_and_set_processing(config.psql_db)

            if not model_check_task:
                await asyncio.sleep(60)
                continue

            log_context_details["model_id"] = model_check_task.model_id
            log_context_details["task_id"] = str(model_check_task.id)
            with LogContext(**log_context_details):
                logger.info(f"Processing model check request for: {model_check_task.model_id} (ID: {model_check_task.id})")

                required_gpus = compute_required_gpus(model_check_task.model_id, None)
                async with gpu_manager.with_gpus(required_gpus) as gpu_ids:
                    if not gpu_ids:
                        logger.warning(
                            f"Could not acquire {required_gpus} GPU(s) for model {model_check_task.model_id}. Re-queuing task."
                        )
                        await mc_sql.update_model_check_result(
                            request_id=model_check_task.id,
                            status=ModelCheckStatus.PENDING,
                            psql_db=config.psql_db,
                            error_message="Temporarily unable to acquire GPU, re-queued.",
                            parameter_count=None,
                        )
                        continue

                    logger.info(f"Acquired GPU(s): {gpu_ids} for model {model_check_task.model_id}")
                    check_results = await run_model_loading_and_inference_check(model_check_task.model_id, gpu_ids)

                    status_str = check_results.get("status", "Failure")
                    params_raw = check_results.get("parameter_count")
                    error_msg = check_results.get("error_message")

                    final_status = ModelCheckStatus.SUCCESS if status_str == "Success" else ModelCheckStatus.FAILURE
                    is_suitable_for_cs = final_status == ModelCheckStatus.SUCCESS

                    parameter_count = None
                    if isinstance(params_raw, (int, float)):
                        parameter_count = int(params_raw)
                    elif isinstance(params_raw, str) and params_raw.isdigit():
                        parameter_count = int(params_raw)
                    else:
                        if params_raw is not None and params_raw != "N/A":
                            logger.warning(
                                f"Could not parse parameter_count '{params_raw}' for {model_check_task.model_id}. Storing as None."
                            )

                    await mc_sql.update_model_check_result(
                        request_id=model_check_task.id,
                        status=final_status,
                        parameter_count=parameter_count,
                        error_message=error_msg,
                        psql_db=config.psql_db,
                    )

                    if final_status == ModelCheckStatus.SUCCESS:
                        logger.info(f"Model check successful for {model_check_task.model_id}: Params={parameter_count}")
                    else:
                        logger.warning(
                            f"Model check failed for {model_check_task.model_id}: Params={parameter_count}, Error={error_msg}"
                        )

                    try:
                        logger.info(f"Attempting to update content service for model {model_check_task.model_id}...")
                        await update_content_service_model_info(
                            model_id=model_check_task.model_id,
                            is_suitable=is_suitable_for_cs,
                            keypair=config.keypair,
                            parameter_count=parameter_count,
                        )
                    except Exception as cs_update_e:
                        logger.error(
                            f"Failed to update content service for model {model_check_task.model_id} after check. "
                            + f"Error: {cs_update_e}",
                            exc_info=True,
                        )

            await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Error in model testing loop: {str(e)}", exc_info=True)
            if model_check_task and model_check_task.id:
                try:
                    await mc_sql.update_model_check_result(
                        request_id=model_check_task.id,
                        status=ModelCheckStatus.FAILURE,
                        error_message=f"Outer loop error: {str(e)}",
                        psql_db=config.psql_db,
                    )
                    logger.info(f"Marked model check task {model_check_task.id} as FAILURE due to loop error.")
                except Exception as db_err:
                    logger.error(
                        f"Failed to update model check task {model_check_task.id} status after loop error: {db_err}",
                        exc_info=True,
                    )
            await asyncio.sleep(60)
