from git import Repo
import os
import pynvml
from urllib.parse import urlparse

from core.models.utility_models import GPUType
from trainer.tasks import get_running_tasks


def clone_repo(repo_url: str, parent_dir: str, branch: str = None) -> str:
    repo_name = os.path.basename(urlparse(repo_url).path)
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    repo_dir = os.path.join(parent_dir, repo_name)

    if os.path.exists(repo_dir):
        print(f"Repository already exists at {repo_dir}. Skipping clone.")
        return repo_dir

    try:
        if branch:
            Repo.clone_from(repo_url, repo_dir, branch=branch)
        else:
            Repo.clone_from(repo_url, repo_dir)
        print(f"Repository cloned to {repo_dir}")
        return repo_dir
    except Exception as e:
        raise RuntimeError(f"Failed to clone repository: {e}")


async def get_available_gpu_types() -> dict[GPUType, int]:
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    index_to_type: dict[int, GPUType] = {}
    gpu_counts: dict[GPUType, int] = {}

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle).decode("utf-8").upper()
        for gpu_type in GPUType:
            if gpu_type.value in name:
                index_to_type[i] = gpu_type
                gpu_counts[gpu_type] = gpu_counts.get(gpu_type, 0) + 1
                break

    running_tasks = get_running_tasks()

    if running_tasks:
        for task in running_tasks:
            for gpu_id in task.gpu_ids:
                gpu_type = index_to_type.get(gpu_id)
                if gpu_type and gpu_counts.get(gpu_type, 0) > 0:
                    gpu_counts[gpu_type] -= 1

    pynvml.nvmlShutdown()
    return gpu_counts
