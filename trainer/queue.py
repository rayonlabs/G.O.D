import asyncio
import docker
from core.models.payload_models import TrainerProxyRequestImage
from trainer.image_manager import build_docker_image, run_trainer_container
from trainer import constants as cst
from validator.utils.logging import get_logger

logger = get_logger(__name__)


class TrainerQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.worker_task = asyncio.create_task(self.worker())

    async def submit(self, job: TrainerProxyRequestImage):
        await self.queue.put(job)
        logger.info(f"Job queued: {job.task_id}")

    async def worker(self):
        while True:
            job = await self.queue.get()
            logger.info(f"Starting job: {job.task_id}")
            try:
                local_repo_path = job.local_repo_path
                tag = await build_docker_image(
                    dockerfile_path=f"{local_repo_path}/{cst.DEFAULT_IMAGE_DOCKERFILE_PATH}",
                    context_path=local_repo_path,
                )

                timeout_seconds = job.hours_to_complete * 3600

                try:
                    container = await asyncio.wait_for(
                        run_trainer_container(
                            task_id=job.task_id,
                            tag=tag,
                            model=job.model,
                            dataset_zip=job.dataset_zip,
                            model_type=job.model_type,
                            hours_to_complete=job.hours_to_complete,
                        ),
                        timeout=60
                    )

                    await asyncio.sleep(timeout_seconds)

                    logger.info(f"Timeout reached for job {job.task_id}. Killing container...")
                    container.kill()
                    logger.info(f"Container killed: {container.name}")

                except asyncio.TimeoutError:
                    logger.info(f"Container for job {job.task_id} failed to start in time.")

                logger.info(f"Finished job: {job.task_id}")

            except Exception as e:
                logger.error(f"Job {job.task_id} failed: {e}")

            self.queue.task_done()
