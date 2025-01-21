import random

from validator.augmentation.augmentation import generate_augmented_dataset
from validator.augmentation.augmentation import load_and_sample_dataset
from validator.core.config import Config
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def test_synth_generation():
    dataset_name = "mhenrichsen/alpaca_2k_test"
    columns_to_sample = ["instruction", "input", "output"]

    sampled_data = load_and_sample_dataset(
        dataset_name,
        columns_to_sample,
    )

    assert await generate_augmented_dataset(
        sampled_data,
        random.choice(columns_to_sample),
        Config.keypair,
    ), "Synth generation returned no data"
