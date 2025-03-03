import json
import os
import tempfile

from validator.core.config import Config
from validator.utils.logging import get_logger
from validator.utils.minio import async_minio_client


logger = get_logger(__name__)


async def try_db_connections(config: Config) -> None:
    logger.info("Attempting to connect to PostgreSQL...")
    await config.psql_db.connect()
    await config.psql_db.pool.execute("SELECT 1=1 as one")
    logger.info("PostgreSQL connected successfully")

    logger.info("Attempting to connect to Redis")
    await config.redis_db.ping()
    logger.info("Redis connected successfully")


async def save_json_to_temp_file(data: list[dict], prefix: str, dump_json: bool = True) -> tuple[str, int]:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", prefix=prefix)
    if dump_json:
        with open(temp_file.name, "w") as f:
            json.dump(data, f)
    else:
        with open(temp_file.name, "w") as f:
            f.write(data)
    file_size = os.path.getsize(temp_file.name)
    return temp_file.name, file_size


async def upload_file_to_minio(file_path: str, bucket_name: str, object_name: str) -> str | None:
    """
    Uploads a file to MinIO and returns the presigned URL for the uploaded file.
    """
    result = await async_minio_client.upload_file(bucket_name, object_name, file_path)
    if result:
        return await async_minio_client.get_presigned_url(bucket_name, object_name)
    else:
        return None
