import logging

from core.models.utility_models import FileFormat
from validator.core.config import Config
from validator.tasks.task_prep import prepare_task


logger = logging.getLogger(__name__)


async def test_prepare_task():
    assert all(
        await prepare_task(
            "mhenrichsen/alpaca_2k_test",
            FileFormat.HF,
            ["input", "output", "instruction", "text"],
            Config.keypair,
        )
    )
