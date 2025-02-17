import asyncio

from core.models.utility_models import FileFormat
from validator.core.config import load_config
from validator.tasks.task_prep import prepare_task
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def main():
    dataset_name = "OpenSafetyLab/Salad-Data"
    columns_to_sample = ["augq", "baseq"]

    config = load_config()

    try:
        test_data, synth_data, train_data = await prepare_task(
            dataset_name=dataset_name, file_format=FileFormat.HF, columns_to_sample=columns_to_sample, keypair=config.keypair
        )

        logger.info(f"Test data URL: {test_data}")
        logger.info(f"Synthetic data URL: {synth_data}")
        logger.info(f"Training data URL: {train_data}")

    except Exception as e:
        logger.error(f"Error in task preparation: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
