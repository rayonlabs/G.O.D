import os
import shutil
import tempfile

from fiber.logging_utils import get_logger
from huggingface_hub import hf_api
from huggingface_hub import hf_hub_download

from core import constants as cst


logger = get_logger(__name__)

def download_from_huggingface(repo_id: str, filename: str, local_dir: str) -> str:
    try:
        local_dir = os.path.expanduser(local_dir)
        local_filename = f"flux_unet_{repo_id.replace('/', '_')}.safetensors"
        final_path = os.path.join(local_dir, local_filename)
        os.makedirs(local_dir, exist_ok=True)
        if os.path.exists(final_path):
            logger.info(f"File {filename} already exists. Skipping download.")
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=temp_dir)
                shutil.move(temp_file_path, final_path)
            logger.info(f"File {filename} downloaded successfully")
        return final_path
    except Exception as e:
        logger.error(f"Error downloading file: {e}")

def download_flux_unet(repo_id: str) -> str:
    files_metadata = hf_api.list_repo_tree(repo_id=repo_id, repo_type="model")
    file_path = None
    for file in files_metadata:
        if hasattr(file, "size") and file.size is not None:
            if file.path.endswith(".safetensors") and file.size > 10 * 1024 * 1024 * 1024:
                file_path = file.path
                local_path = download_from_huggingface(repo_id, file_path, cst.CACHE_DIR)
    if not file_path:
        raise FileNotFoundError(f"No valid file found in root of repo '{repo_id}'.")

    return local_path
