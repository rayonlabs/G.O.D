import pynvml
from asyncio import Lock
from core.models.utility_models import GPUType


class GPUPool:
    busy_gpus: set[int] = set()
    lock = Lock()

    @classmethod
    async def mark_busy(cls, gpu_indices: list[int]):
        async with cls.lock:
            cls.busy_gpus.update(gpu_indices)

    @classmethod
    async def mark_free(cls, gpu_indices: list[int]):
        async with cls.lock:
            cls.busy_gpus.difference_update(gpu_indices)

    @classmethod
    async def get_free_gpu_indices(cls) -> list[int]:
        pynvml.nvmlInit()
        total = pynvml.nvmlDeviceGetCount()
        async with cls.lock:
            free = [i for i in range(total) if i not in cls.busy_gpus]
        pynvml.nvmlShutdown()
        return free


async def get_available_gpu_types() -> dict[GPUType, int]:
    pynvml.nvmlInit()

    device_count = pynvml.nvmlDeviceGetCount()
    free_indices = await GPUPool.get_free_gpu_indices()

    existing_gpu_types: set[GPUType] = set()
    index_to_type: dict[int, GPUType] = {}

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle).decode("utf-8").upper()

        for gpu_type in GPUType:
            if gpu_type.value in name:
                existing_gpu_types.add(gpu_type)
                index_to_type[i] = gpu_type
                break

    gpu_counts: dict[GPUType, int] = {gpu_type: 0 for gpu_type in existing_gpu_types}

    for idx in free_indices:
        gpu_type = index_to_type.get(idx)
        if gpu_type:
            gpu_counts[gpu_type] += 1

    pynvml.nvmlShutdown()
    return gpu_counts
