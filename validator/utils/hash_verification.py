import hashlib
import os
import shutil
from typing import Optional, List
from huggingface_hub import snapshot_download

from validator.utils.logging import get_logger

logger = get_logger(__name__)


def calculate_model_hash(repo_id: str, cleanup_cache: bool = True) -> Optional[str]:
    if not repo_id or not isinstance(repo_id, str):
        logger.error(f"Invalid repo_id provided: {repo_id}")
        return None
        
    local_path = None
    try:
        logger.info(f"Calculating hash for model repo: {repo_id}")
        
        local_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=None,
            token=os.environ.get("HUGGINGFACE_TOKEN"),
        )
            
        hasher = hashlib.sha256()
        
        specific_files = [
            "adapter_model.bin",
            "adapter_model.safetensors",
            "pytorch_model.bin",
            "pytorch_model.safetensors", 
            "model.safetensors",
            "adapter_config.json",
            "config.json",
        ]
        
        all_files = os.listdir(local_path)
        logger.info(f"All files in {repo_id}: {all_files}")
        
        subdirs = [f for f in all_files if os.path.isdir(os.path.join(local_path, f))]
        logger.info(f"Subdirectories found: {subdirs}")
        for subdir in subdirs:
            subdir_files = os.listdir(os.path.join(local_path, subdir))
            logger.info(f"Files in {subdir}/: {subdir_files}")
        
        import glob
        sharded_patterns = [
            "model-*.safetensors",
            "pytorch_model-*.bin", 
            "pytorch_model-*.safetensors",
            "checkpoint/last.safetensors",
            "checkpoint/last-*.safetensors",
            "checkpoint/checkpoint-*.safetensors",
            "last-*.safetensors",
            "diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/pytorch_model.bin",
            "text_encoder_2/pytorch_model.bin",
        ]
        
        files_to_hash = specific_files.copy()
        for pattern in sharded_patterns:
            pattern_path = os.path.join(local_path, pattern)
            matched_files = glob.glob(pattern_path)
            logger.info(f"Pattern '{pattern}' matched: {[os.path.relpath(f, local_path) for f in matched_files]}")
            for file_path in matched_files:
                relative_path = os.path.relpath(file_path, local_path)
                files_to_hash.append(relative_path)
        
        files_to_hash = sorted(list(set(files_to_hash)))
        
        hashed_files = []
        for filename in files_to_hash:
            file_path = os.path.join(local_path, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                        if len(file_content) == 0:
                            logger.warning(f"Empty file found: {filename} in {repo_id}")
                            continue
                        file_hash = hashlib.sha256(file_content).hexdigest()
                        hasher.update(f"{filename}:{file_hash}".encode('utf-8'))
                        hashed_files.append(filename)
                except Exception as e:
                    logger.warning(f"Failed to hash file {filename} in {repo_id}: {e}")
                    continue
        
        if not hashed_files:
            logger.error(f"No valid model files found to hash in {repo_id}")
            return None
            
        model_hash = hasher.hexdigest()
        logger.info(f"Model hash calculated from files: {hashed_files}")
        logger.info(f"Model hash: {model_hash}")
        
        return model_hash
        
    except Exception as e:
        logger.error(f"Failed to calculate model hash for {repo_id}: {e}")
        return None
    finally:
        if cleanup_cache and local_path and os.path.exists(local_path):
            try:
                cache_root = os.path.dirname(local_path)
                if "models--" in os.path.basename(local_path):
                    shutil.rmtree(local_path)
                    logger.debug(f"Cleaned up cache for {repo_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up cache for {repo_id}: {e}")


def verify_model_hash(repo_id: str, expected_hash: str, cleanup_cache: bool = True) -> bool:
    if not is_valid_model_hash(expected_hash):
        logger.error(f"Invalid hash format provided for {repo_id}: {expected_hash}")
        return False
        
    logger.info(f"Starting hash verification for {repo_id}")
    actual_hash = calculate_model_hash(repo_id, cleanup_cache)
    
    if actual_hash is None:
        logger.error(f"Could not calculate hash for verification: {repo_id}")
        return False
        
    matches = actual_hash == expected_hash
    
    if matches:
        logger.info(f"Hash verification PASSED for {repo_id}")
    else:
        logger.warning(f"Hash verification FAILED for {repo_id}")
        logger.warning(f"Expected: {expected_hash}")
        logger.warning(f"Actual: {actual_hash}")
        
    return matches


def is_valid_model_hash(model_hash: str) -> bool:
    if not isinstance(model_hash, str):
        return False
        
    if len(model_hash) != 64:
        return False
        
    try:
        int(model_hash, 16)
        return True
    except ValueError:
        return False