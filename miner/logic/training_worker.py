import queue
import threading
from uuid import UUID

import docker
from fiber.logging_utils import get_logger

from core.models.utility_models import DiffusionJob
from core.models.utility_models import Job
from core.models.utility_models import JobStatus
from core.models.utility_models import TextJob
from miner.logic.job_handler import start_tuning_container
from miner.logic.job_handler import start_tuning_container_diffusion


logger = get_logger(__name__)


class TrainingWorker:
    def __init__(self):
        logger.info("=" * 80)
        logger.info("STARTING A TRAINING WORKER")
        logger.info("=" * 80)

        self.job_queue: queue.Queue[Job] = queue.Queue()
        self.job_store: dict[str, Job] = {}
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.docker_client = docker.from_env()

    def _worker(self):
        while True:
            job = self.job_queue.get()
            if job is None:
                break
            try:
                if isinstance(job, TextJob):
                    start_tuning_container(job)
                elif isinstance(job, DiffusionJob):
                    start_tuning_container_diffusion(job)
                job.status = JobStatus.COMPLETED
            except Exception as e:
                logger.error(f"Error processing job {job.job_id}: {str(e)}")
                job.status = JobStatus.FAILED
                job.error_message = str(e)
            finally:
                self.job_queue.task_done()

    def enqueue_job(self, job: Job):
        self.job_queue.put(job)
        self.job_store[job.job_id] = job

    def get_status(self, job_id: UUID) -> JobStatus:
        job = self.job_store.get(str(job_id))

        return job.status if job else JobStatus.NOT_FOUND

    def shutdown(self):
        self.thread.join()
        self.docker_client.close()
