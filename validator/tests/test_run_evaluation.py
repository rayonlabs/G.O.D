from core.models.utility_models import CustomDatasetType
from core.models.utility_models import FileFormat
from core.utils import download_s3_file
from validator.evaluation.docker_evaluation import run_evaluation_docker
from validator.utils.logging import get_logger


logger = get_logger(__name__)
s3_dataset = "https://gradients.s3.eu-north-1.amazonaws.com/a93ea2cf7f944480_test_data.json"


async def test_run_evaluation():
    custom_dataset = CustomDatasetType(
        field_instruction="raw_text",
        field_output="clean_text",
    )

    dataset = await download_s3_file(s3_dataset)
    logger.info(f"Downloaded dataset to {dataset}")

    result = await run_evaluation_docker(
        dataset=dataset,
        models=["samoline/7c929f55-f88d-4c10-9953-16c7dab4efc7"],
        original_model="facebook/opt-125m",
        dataset_type=custom_dataset,
        file_format=FileFormat.JSON,
        gpu_ids=[0],
    )
    logger.info(f"Evaluation result: {result}")
