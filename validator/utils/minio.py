import asyncio
import datetime
import os
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from fiber.logging_utils import get_logger
from minio import Minio


logger = get_logger(__name__)


class AsyncMinioClient:
    def __init__(self, endpoint, access_key, secret_key, secure=True, region="us-east-1"):
        self.endpoint = endpoint
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region,
        )
        self.loop = asyncio.get_event_loop()
        self.executor = ThreadPoolExecutor()

    async def upload_file(self, bucket_name, object_name, file_path):
        func = self.client.fput_object
        args = (bucket_name, object_name, file_path)
        logger.info("Attempting to upload")
        try:
            result = await self.loop.run_in_executor(self.executor, func, *args)
            logger.info(
                f"The bucket_name is {bucket_name} and the result was {result}")
            return result
        except Exception as e:
            logger.info(f"There was an issue with uploading {e}")
            return False

    async def download_file(self, bucket_name, object_name, file_path):
        func = self.client.fget_object
        args = (bucket_name, object_name, file_path)
        logger.info("Attempting to download")
        return await self.loop.run_in_executor(self.executor, func, *args)

    async def delete_file(self, bucket_name, object_name):
        func = self.client.remove_object
        args = (bucket_name, object_name)
        return await self.loop.run_in_executor(self.executor, func, *args)

    async def list_objects(self, bucket_name, prefix=None, recursive=True):
        func = self.client.list_objects
        args = (bucket_name, prefix, recursive)
        logger.info("Listing objects")
        return await self.loop.run_in_executor(self.executor, func, *args)

    async def get_stats(self, bucket_name: str, object_name: str):
        """Get stats for an object in MinIO storage."""
        func = self.client.stat_object
        args = (bucket_name, object_name)
        logger.info("Getting stats for object")
        stats = await self.loop.run_in_executor(self.executor, func, *args)
        return stats

    async def ensure_bucket_exists(self, bucket_name):
        exists = await self.loop.run_in_executor(self.executor, self.client.bucket_exists, bucket_name)
        if not exists:
            await self.loop.run_in_executor(self.executor, self.client.make_bucket, bucket_name)

    async def get_presigned_url(self, bucket_name, object_name, expires=604800):
        expires_duration = datetime.timedelta(seconds=expires)
        func = self.client.presigned_get_object
        args = (bucket_name, object_name, expires_duration)
        return await self.loop.run_in_executor(self.executor, func, *args)

    def get_public_url(self, bucket_name, object_name):
        return f"https://{self.endpoint}/{bucket_name}/{object_name}"

    def parse_s3_url(self, url: str) -> tuple[str, str]:
        """Extract bucket name and object name from S3 URL."""
        parsed_url = urlparse(url)
        bucket_name = parsed_url.hostname.split('.')[0]
        object_name = parsed_url.path.lstrip('/').split('?')[0]
        return bucket_name, object_name

    def __del__(self):
        self.executor.shutdown(wait=False)


S3_COMPATIBLE_ENDPOINT = os.getenv("S3_COMPATIBLE_ENDPOINT", "localhost:9000")
S3_COMPATIBLE_ACCESS_KEY = os.getenv("S3_COMPATIBLE_ACCESS_KEY", "minioadmin")
S3_COMPATIBLE_SECRET_KEY = os.getenv("S3_COMPATIBLE_SECRET_KEY", "minioadmin")

async_minio_client = AsyncMinioClient(
    endpoint=S3_COMPATIBLE_ENDPOINT,
    access_key=S3_COMPATIBLE_ACCESS_KEY,
    secret_key=S3_COMPATIBLE_SECRET_KEY
)
