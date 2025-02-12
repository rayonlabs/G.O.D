import asyncio
import datetime
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from minio import Minio

from validator.core.config import load_config
from validator.utils.logging import get_logger


logger = get_logger(__name__)
config = load_config()

# NOTE: This all needs rewriting to be honest
# TODO: TODO: TODO: BIN ALL OF THIS PLZ :PRAY:
## WW: Looks like the best GPT code to date


class AsyncMinioClient:
    def __init__(
        self,
        endpoint,
        access_key,
        secret_key,
        region,
        secure=True,
    ):
        self.endpoint = endpoint
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region,
        )
        self.executor = ThreadPoolExecutor()

    async def upload_file(self, bucket_name, object_name, file_path):
        func = self.client.fput_object
        args = (bucket_name, object_name, file_path)
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, func, *args)
            logger.info(f"Successfully uploaded args={args}")
            return result
        except Exception as e:
            logger.info(f"There was an issue with uploading file to s3. {e}")
            return False

    async def download_file(self, bucket_name, object_name, file_path):
        func = self.client.fget_object
        args = (bucket_name, object_name, file_path)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def delete_file(self, bucket_name, object_name):
        func = self.client.remove_object
        args = (bucket_name, object_name)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def list_objects(self, bucket_name, prefix=None, recursive=True):
        func = self.client.list_objects
        args = (bucket_name, prefix, recursive)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def get_stats(self, bucket_name: str, object_name: str):
        """Get stats for an object in MinIO storage."""
        func = self.client.stat_object
        args = (bucket_name, object_name)
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(self.executor, func, *args)
        return stats

    async def ensure_bucket_exists(self, bucket_name):
        exists = await self.client.bucket_exists(bucket_name)
        if not exists:
            await self.client.make_bucket(bucket_name)

    async def get_presigned_url(self, bucket_name, object_name, expires=604800):
        expires_duration = datetime.timedelta(seconds=expires)
        func = self.client.presigned_get_object
        args = (bucket_name, object_name, expires_duration)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    def get_public_url(self, bucket_name, object_name):
        return f"https://{self.endpoint}/{bucket_name}/{object_name}"

    def parse_s3_url(self, url: str) -> tuple[str, str]:
        """Extract bucket name and object name from S3 URL."""
        parsed_url = urlparse(url)
        bucket_name = parsed_url.hostname.split(".")[0]
        object_name = parsed_url.path.lstrip("/").split("?")[0]
        return bucket_name, object_name

    def __del__(self):
        self.executor.shutdown(wait=False)


async_minio_client = AsyncMinioClient(
    endpoint=config.s3_compatible_endpoint,
    access_key=config.s3_compatible_access_key,
    secret_key=config.s3_compatible_secret_key,
    region=config.s3_region,
    secure=not config.debug,
)
