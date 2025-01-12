import glob
import os
import shutil
import tempfile

from validator.utils.logging import get_logger


logger = get_logger(__name__)


def cleanup_temp_files():
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.endswith(".json") and (
            any(prefix in filename for prefix in ["train_data_", "test_data_", "synth_data_"])
            or any(suffix in filename for suffix in ["_test_data.json", "_train_data.json", "_synth_data.json"])
        ):
            try:
                os.remove(os.path.join(temp_dir, filename))
            except OSError:
                pass


def delete_dataset_from_cache(dataset_name):
    """
    Delete dataset and associated lock files from HuggingFace cache.
    Case-insensitive matching for dataset names.
    """
    dataset_name = dataset_name.lower().replace("/", "___")
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")

    all_datasets = os.listdir(cache_dir) if os.path.exists(cache_dir) else []
    matching_datasets = [d for d in all_datasets if d.lower() == dataset_name]

    deleted = False

    for dataset_dir in matching_datasets:
        dataset_path = os.path.join(cache_dir, dataset_dir)
        try:
            shutil.rmtree(dataset_path)
            logger.info(f"Deleted dataset directory: {dataset_dir}")
            deleted = True
        except Exception as e:
            logger.error(f"Error deleting dataset directory '{dataset_dir}': {str(e)}")

    try:
        all_lock_files = glob.glob(os.path.join(cache_dir, "*.lock"))
        matching_locks = [
            f for f in all_lock_files if f"cache_huggingface_datasets_{dataset_name}" in os.path.basename(f).lower()
        ]

        for lock_file in matching_locks:
            try:
                os.remove(lock_file)
                logger.info(f"Deleted lock file: {os.path.basename(lock_file)}")
                deleted = True
            except Exception as e:
                logger.error(f"Error deleting lock file {os.path.basename(lock_file)}: {str(e)}")
    except Exception as e:
        logger.error(f"Error searching for lock files: {str(e)}")

    if not deleted:
        logger.info(f"No files found for dataset '{dataset_name}'")


def delete_model_from_cache(model_name):
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_path = os.path.join(cache_dir, model_name)

    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        logger.info(f"Deleted model '{model_name}' from cache.")
    else:
        logger.info(f"Model '{model_name}' not found in cache.")


def clean_all_hf_datasets_cache():
    """Clean the entire Huggingface datasets cache directory."""
    try:
        hf_cache_path = os.path.expanduser("~/.cache/huggingface/datasets/")
        if os.path.exists(hf_cache_path):
            shutil.rmtree(hf_cache_path)
            logger.info(f"Cleaned Huggingface datasets cache at {hf_cache_path}")
    except Exception as e:
        logger.error(f"Error cleaning Huggingface datasets cache: {e}")
