
import httpx
from trainer.endpoints import GET_GPU_AVAILABILITY_ENDPOINT

from core.models.utility_models import GPUInfo
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def fetch_trainer_gpus(trainer_ip: str) -> list[GPUInfo]:
    """
    Fetch GPU availability information from a trainer.

    Args:
        trainer_ip: IP address of the trainer to contact

    Returns:
        List of GPUInfo objects from the trainer
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        url = f"http://{trainer_ip}{GET_GPU_AVAILABILITY_ENDPOINT}"
        logger.info(f"Contacting trainer at {url}")

        response = await client.get(url)
        response.raise_for_status()

        gpu_data = response.json()
        gpu_infos = [GPUInfo.model_validate(gpu_info) for gpu_info in gpu_data]

        logger.info(f"Retrieved {len(gpu_infos)} GPUs from trainer {trainer_ip}")
        return gpu_infos
