import asyncio
import docker
import pynvml
from core.models.payload_models import TrainerProxyJobImage
from core.models.utility_models import GPUType
from trainer.image_manager import build_docker_image, run_trainer_container
from trainer import constants as cst
from trainer.gpupool import GPUPool
from validator.utils.logging import get_logger

logger = get_logger(__name__)


class TrainerQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.worker_task = asyncio.create_task(self.worker())

    async def submit(self, job: TrainerProxyJobImage):
        await self.queue.put(job)
        logger.info(f"Job queued: {job.task_id}")

    async def worker(self):
        while True:
            job = await self.queue.get()
            logger.info(f"Starting job: {job.task_id}")
            gpu_ids = []

            try:
                if job.gpu_type and job.num_gpus:
                    available_gpus = await GPUPool.get_free_gpu_indices()
                    matched = []

                    pynvml.nvmlInit()
                    for idx in available_gpus:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                        name = pynvml.nvmlDeviceGetName(handle).decode("utf-8").upper()
                        if job.gpu_type.value in name:
                            matched.append(idx)
                        if len(matched) >= job.num_gpus:
                            break
                    pynvml.nvmlShutdown()
                    gpu_ids = matched
                else:
                    available_gpus = await GPUPool.get_free_gpu_indices()
                    gpu_ids = [available_gpus[0]]

                await GPUPool.mark_busy(gpu_ids)

                tag = await asyncio.to_thread(
                    build_docker_image,
                    dockerfile_path=f"{job.local_repo_path}/{cst.DEFAULT_IMAGE_DOCKERFILE_PATH}",
                    context_path=job.local_repo_path,
                )

                timeout_seconds = job.hours_to_complete * 3600

                container = await asyncio.wait_for(
                    run_trainer_container(
                        task_id=job.task_id,
                        tag=tag,
                        model=job.model,
                        dataset_zip=job.dataset_zip,
                        model_type=job.model_type,
                        hours_to_complete=job.hours_to_complete,
                        gpu_ids=gpu_ids,
                    ),
                    timeout=60
                )

                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(container.wait),
                        timeout=timeout_seconds
                    )
                    logger.info(f"Job {job.task_id} finished with status code {result['StatusCode']}")
                except asyncio.TimeoutError:
                    logger.info(f"Timeout reached for job {job.task_id}. Killing container...")
                    container.kill()
                    container.remove(force=True)
                    logger.info(f"Container killed and removed: {container.name}")

            except Exception as e:
                logger.error(f"Job {job.task_id} failed: {e}")

            finally:
                if gpu_ids:
                    await GPUPool.mark_free(gpu_ids)
                self.queue.task_done()
