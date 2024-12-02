import os
import shutil
import glob
import tempfile 

def cleanup_temp_files():
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.endswith('.json') and (
            any(prefix in filename for prefix in ['train_data_', 'test_data_', 'synth_data_']) or
            any(suffix in filename for suffix in ['_test_data.json', '_train_data.json', '_synth_data.json'])
        ):
            try:
                os.remove(os.path.join(temp_dir, filename))
            except OSError:
                pass

def delete_dataset_from_cache(dataset_name):
    """
    Delete dataset and associated lock files from HuggingFace cache.
    """
    dataset_name = dataset_name.replace("/", "___")
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    dataset_path = os.path.join(cache_dir, dataset_name)
    lock_pattern = os.path.join(cache_dir, f"*cache_huggingface_datasets_{dataset_name}*.lock")
    
    deleted = False
    cleanup_temp_files()
    
    if os.path.exists(dataset_path):
        try:
            shutil.rmtree(dataset_path)
            print(f"Deleted dataset directory: {dataset_name}")
            deleted = True
        except Exception as e:
            print(f"Error deleting dataset directory '{dataset_name}': {str(e)}")
    
    try:
        lock_files = glob.glob(lock_pattern)
        for lock_file in lock_files:
            try:
                os.remove(lock_file)
                print(f"Deleted lock file: {os.path.basename(lock_file)}")
                deleted = True
            except Exception as e:
                print(f"Error deleting lock file {os.path.basename(lock_file)}: {str(e)}")
    except Exception as e:
        print(f"Error searching for lock files: {str(e)}")
    
    if not deleted:
        print(f"No files found for dataset '{dataset_name}'")


def delete_model_from_cache(model_name):
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    model_path = os.path.join(cache_dir, model_name)

    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        print(f"Deleted model '{model_name}' from cache.")
    else:
        print(f"Model '{model_name}' not found in cache.")
