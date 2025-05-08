import uuid
from datetime import datetime
from typing import Optional

import validator.core.constants as cst
from core.models.utility_models import ModelCheckStatus
from validator.core.models import ModelCheckQueueEntry
from validator.db.database import PSQLDB
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def add_model_check_request(entry_to_save: ModelCheckQueueEntry, psql_db: PSQLDB) -> ModelCheckQueueEntry:
    async with await psql_db.connection() as connection:
        query = f"""
            INSERT INTO {cst.MODEL_CHECKS_QUEUE_TABLE} (
                {cst.MCQ_ID}, {cst.MCQ_MODEL_ID}, {cst.MCQ_STATUS},
                {cst.MCQ_REQUESTED_AT}, {cst.MCQ_PROCESSED_AT}, 
                {cst.MCQ_PARAMETER_COUNT}, {cst.MCQ_ERROR_MESSAGE}
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING *
        """
        record = await connection.fetchrow(
            query,
            entry_to_save.id,
            entry_to_save.model_id,
            entry_to_save.status.value,
            entry_to_save.requested_at,
            entry_to_save.processed_at,
            entry_to_save.parameter_count,
            entry_to_save.error_message,
        )

    if record:
        return ModelCheckQueueEntry.model_validate(dict(record))
    raise Exception("Failed to retrieve record after insert in add_model_check_request")


async def get_oldest_pending_model_check_and_set_processing(
    psql_db: PSQLDB,
) -> Optional[ModelCheckQueueEntry]:
    """
    Atomically fetches the oldest PENDING model check request,
    updates its status to PROCESSING, and returns it.
    """
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            query_select_lock = f"""
                SELECT {cst.MCQ_ID}
                FROM {cst.MODEL_CHECKS_QUEUE_TABLE}
                WHERE {cst.MCQ_STATUS} = $1
                ORDER BY {cst.MCQ_REQUESTED_AT} ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            """
            locked_record_id_row = await connection.fetchrow(query_select_lock, ModelCheckStatus.PENDING.value)

            if not locked_record_id_row:
                return None

            locked_id = locked_record_id_row[cst.MCQ_ID]

            query_update = f"""
                UPDATE {cst.MODEL_CHECKS_QUEUE_TABLE}
                SET {cst.MCQ_STATUS} = $1, {cst.MCQ_PROCESSED_AT} = $2
                WHERE {cst.MCQ_ID} = $3
                RETURNING *
            """
            record = await connection.fetchrow(query_update, ModelCheckStatus.PROCESSING.value, datetime.utcnow(), locked_id)
            return ModelCheckQueueEntry.model_validate(dict(record)) if record else None


async def update_model_check_result(
    request_id: uuid.UUID,
    status: ModelCheckStatus,
    psql_db: PSQLDB,
    parameter_count: Optional[int] = None,
    error_message: Optional[str] = None,
):
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.MODEL_CHECKS_QUEUE_TABLE}
            SET {cst.MCQ_STATUS} = $1,
                {cst.MCQ_PARAMETER_COUNT} = $2,
                {cst.MCQ_ERROR_MESSAGE} = $3,
                {cst.MCQ_PROCESSED_AT} = $4
            WHERE {cst.MCQ_ID} = $5
        """
        await connection.execute(
            query,
            status.value,
            parameter_count,
            error_message,
            datetime.utcnow(),
            request_id,
        )


async def get_model_check_by_id(request_id: uuid.UUID, psql_db: PSQLDB) -> Optional[ModelCheckQueueEntry]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT * FROM {cst.MODEL_CHECKS_QUEUE_TABLE}
            WHERE {cst.MCQ_ID} = $1
        """
        record = await connection.fetchrow(query, request_id)
        return ModelCheckQueueEntry.model_validate(dict(record)) if record else None
