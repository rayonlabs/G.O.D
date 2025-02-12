import logging

import pytest

from core.models.utility_models import FileFormat
from validator.tasks.task_prep import prepare_task


logger = logging.getLogger(__name__)


@pytest.mark.usefixtures("mock_nineteen_api")
async def test_prepare_task(test_config):
    assert all(
        await prepare_task(
            "mhenrichsen/alpaca_2k_test",
            FileFormat.HF,
            ["input", "output", "instruction", "text"],
            test_config.keypair,
        )
    )
