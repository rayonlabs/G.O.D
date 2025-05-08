import asyncio
import contextlib

import validator.core.constants as cst
from validator.evaluation.utils import get_model_num_params


class GPUQueueManager:
    def __init__(self, gpu_ids=None):
        self.gpu_queue = asyncio.Queue()
        self.gpu_acquisition_lock = asyncio.Lock()
        self._initialized = False
        self._gpu_ids = gpu_ids or cst.GPU_IDS

    async def initialize(self):
        if not self._initialized:
            for gpu_id in self._gpu_ids:
                await self.gpu_queue.put(gpu_id)
            self._initialized = True
        return self

    async def acquire_gpus(self, count=1):
        gpu_ids = []
        async with self.gpu_acquisition_lock:
            for _ in range(count):
                gpu_ids.append(await self.gpu_queue.get())
        return gpu_ids

    async def release_gpus(self, gpu_ids):
        for gpu_id in gpu_ids:
            await self.gpu_queue.put(gpu_id)

    @contextlib.asynccontextmanager
    async def with_gpus(self, count=1):
        """Context manager for GPU acquisition and release."""
        gpu_ids = await self.acquire_gpus(count)
        try:
            yield gpu_ids
        finally:
            await self.release_gpus(gpu_ids)


def compute_required_gpus(model_id: str, model_params_count: int | None) -> int:
    num_params = model_params_count if model_params_count is not None else get_model_num_params(model_id)

    if num_params and num_params > cst.MODEL_SIZE_REQUIRING_2_GPUS:
        return 2
    return 1
