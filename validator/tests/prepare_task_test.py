import asyncio

from validator.core.config import load_config
from validator.core.models import FileFormat
from validator.tasks.task_prep import prepare_task
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def main():
    config = load_config()
    dataset_name = "mhenrichsen/alpaca_2k_test"
    columns_to_sample = ["input", "output", "instruction", "text"]

    datasets = await prepare_task(dataset_name, FileFormat.HF, columns_to_sample, config.keypair)

    logger.info(f"Test dataset size: {datasets.testing.num_rows}")
    logger.info(f"Synthetic data size: {datasets.synthetic.num_rows}")


if __name__ == "__main__":
    asyncio.run(main())
