import asyncio

from core.models.utility_models import FileFormat
from core.models.utility_models import InstructTextDatasetType
from validator.evaluation.docker_evaluation import run_evaluation_docker_text
from validator.utils.logging import get_logger

logger = get_logger(__name__)


async def test():
    custom_dataset_type = InstructTextDatasetType(
        field_instruction = "instruct",
        field_output = "output"
    )

    results = await run_evaluation_docker_text(
        dataset="/root/test.json",
        models=['johngreendr1/fa273251-16c7-46b3-bc5d-9a763b8afbbb'],
        original_model="unsloth/Llama-3.2-3B-Instruct",
        dataset_type=custom_dataset_type,
        file_format=FileFormat.JSON,
        gpu_ids=[0]
    )
    logger.info(f"Evaluation results: {results}")


if __name__ == "__main__":
    asyncio.run(test())