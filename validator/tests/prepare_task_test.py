import asyncio

from validator.tasks.task_prep import prepare_task
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def main():
    dataset_name = "mhenrichsen/alpaca_2k_test"
    columns_to_sample = ["input", "output", "instruction", "text"]
    repo_name = "cwaud/test_ds"

    test_dataset, synthetic_data = await prepare_task(dataset_name, columns_to_sample, repo_name)

    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"Synthetic data size: {len(synthetic_data)}")


if __name__ == "__main__":
    asyncio.run(main())
