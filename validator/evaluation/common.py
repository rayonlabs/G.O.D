import json
import os
import re
import time
from math import ceil

import psutil
import torch
import yaml
from axolotl.utils.dict import DictDefault
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TrainerCallback

from core.config.config_handler import create_dataset_entry
from validator.core import constants as cst
from validator.core.models import EvaluationArgs
from validator.utils.logging import get_logger
from validator.utils.retry_utils import retry_on_5xx


logger = get_logger(__name__)


def log_memory_stats():
    """Log detailed memory statistics for debugging."""
    logger.info("===== MEMORY STATS =====")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**2
            logger.info(
                f"GPU {i} Memory: Allocated: {allocated:.2f} MB, "
                f"Reserved: {reserved:.2f} MB, "
                f"Max Allocated: {max_allocated:.2f} MB"
            )
    else:
        logger.info("No CUDA devices available")

    ram = psutil.Process().memory_info()
    system_memory = psutil.virtual_memory()
    logger.info(f"RAM Usage: RSS: {ram.rss / 1024**2:.2f} MB, VMS: {ram.vms / 1024**2:.2f} MB")
    logger.info(
        f"System Memory: Total: {system_memory.total / 1024**2:.2f} MB, "
        f"Available: {system_memory.available / 1024**2:.2f} MB, "
        f"Used: {(system_memory.total - system_memory.available) / 1024**2:.2f} MB "
        f"({system_memory.percent}%)"
    )
    logger.info("========================")


class ProgressLoggerCallback(TrainerCallback):
    """A callback that logs the progress of the evaluation every log_interval_seconds seconds."""

    def __init__(self, log_interval_seconds):
        self.step = 0
        self.last_log_time = time.time()
        self.log_interval_seconds = log_interval_seconds
        logger.info(f"Initialized ProgressLoggerCallback with log interval of {log_interval_seconds} seconds")

    def on_prediction_step(self, args, state, control, **kwargs):
        self.step += 1
        current_time = time.time()

        if current_time - self.last_log_time >= self.log_interval_seconds:
            self.last_log_time = current_time
            logger.info(f"Evaluation step: {self.step}")

        return control


def create_finetuned_cache_dir():
    """Create and return a dedicated cache directory for finetuned models."""
    finetuned_cache_dir = os.path.join(cst.DOCKER_EVAL_HF_CACHE_DIR, "finetuned_repos")
    os.makedirs(finetuned_cache_dir, exist_ok=True)
    return finetuned_cache_dir


@retry_on_5xx()
def load_model(model_name_or_path: str, is_base_model: bool = False) -> AutoModelForCausalLM:
    try:
        # Only use default cache for the base model
        cache_dir = None if is_base_model else create_finetuned_cache_dir()

        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            token=os.environ.get("HUGGINGFACE_TOKEN"),
            device_map="auto",
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
        )
    except RuntimeError as e:
        error_msg = str(e)
        if "size mismatch for" in error_msg and ("lm_head.weight" in error_msg or "model.embed_tokens.weight" in error_msg):
            pattern = re.search(r"shape torch\.Size\(\[(\d+), (\d+)\]\).*shape.*torch\.Size\(\[(\d+), \2\]\)", error_msg)
            if pattern and abs(int(pattern.group(1)) - int(pattern.group(3))) == 1:
                logger.info("Detected vocabulary size off-by-one error, attempting to load with ignore_mismatched_sizes=True")
                return AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    token=os.environ.get("HUGGINGFACE_TOKEN"),
                    ignore_mismatched_sizes=True,
                    device_map="auto",
                    cache_dir=cache_dir,
                    torch_dtype=torch.bfloat16,
                )
        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise  # Re-raise the exception to trigger retry


@retry_on_5xx()
def load_tokenizer(original_model: str) -> AutoTokenizer:
    try:
        return AutoTokenizer.from_pretrained(original_model, token=os.environ.get("HUGGINGFACE_TOKEN"))
    except Exception as e:
        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise  # Re-raise the exception to trigger retry


@retry_on_5xx()
def load_finetuned_model(repo: str) -> AutoPeftModelForCausalLM:
    try:
        cache_dir = create_finetuned_cache_dir()
        return AutoPeftModelForCausalLM.from_pretrained(
            repo,
            is_trainable=False,
            device_map="auto",
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
        )
    except RuntimeError as e:
        error_msg = str(e)
        if "size mismatch for" in error_msg and ("lm_head.weight" in error_msg or "model.embed_tokens.weight" in error_msg):
            pattern = re.search(r"shape torch\.Size\(\[(\d+), (\d+)\]\).*shape.*torch\.Size\(\[(\d+), \2\]\)", error_msg)
            if pattern and abs(int(pattern.group(1)) - int(pattern.group(3))) == 1:
                logger.info("Detected vocabulary size off-by-one error, attempting to load with ignore_mismatched_sizes=True")
                return AutoPeftModelForCausalLM.from_pretrained(
                    repo,
                    is_trainable=False,
                    ignore_mismatched_sizes=True,
                    device_map="auto",
                    cache_dir=cache_dir,
                    torch_dtype=torch.bfloat16,
                )

        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise  # Re-raise the exception to trigger retry


def count_model_parameters(model):
    """Count the total number of parameters in a model."""
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception as e:
        logger.error(f"Failed to count model parameters: {e}")
        return 0


def load_results_dict():
    """Load existing evaluation results or create an empty dict if not found."""
    results_dict = {}
    output_dir = os.path.dirname(cst.CONTAINER_EVAL_RESULTS_PATH)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(cst.CONTAINER_EVAL_RESULTS_PATH):
        try:
            with open(cst.CONTAINER_EVAL_RESULTS_PATH, "r") as f:
                results_dict = json.load(f)
        except Exception as e:
            logger.error(f"Could not read existing results from {cst.CONTAINER_EVAL_RESULTS_PATH}, starting fresh: {e}")

    return results_dict


def save_results_dict(results_dict, model_id=None):
    """Save evaluation results to file."""
    with open(cst.CONTAINER_EVAL_RESULTS_PATH, "w") as f:
        json.dump(results_dict, f, indent=2)

    msg = "Saved evaluation results"
    if model_id:
        msg += f" for {model_id}"

    logger.info(msg)
    logger.info(json.dumps(results_dict, indent=2))


def check_env_variables(required_vars):
    """Check for required environment variables."""
    env_vars = {var: os.environ.get(var, "") for var in required_vars}
    missing = [var for var, value in env_vars.items() if not value]

    if missing:
        logger.error("Missing required environment variables: " + ", ".join(missing))
        return False, env_vars

    return True, env_vars


def _log_dataset_and_model_info(
    eval_dataset: Dataset,
    language_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> None:
    logger.info(f"Eval dataset sample: {eval_dataset[0]}")
    logger.info(f"Model type: {type(language_model)}")
    logger.info(f"Model config: {language_model.config}")
    logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
    logger.info(f"Model vocabulary size: {language_model.config.vocab_size}")


def log_cuda_info():
    """Log information about CUDA availability and devices."""
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA not available")


def _load_and_update_evaluation_config(
    evaluation_args: EvaluationArgs,
    finetuned_model: AutoModelForCausalLM,
    config_path: str,
) -> DictDefault:
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    dataset_entry = create_dataset_entry(
        dataset=evaluation_args.dataset,
        dataset_type=evaluation_args.dataset_type,
        file_format=evaluation_args.file_format,
        is_eval=True,
    )
    config_dict["datasets"] = [dataset_entry]

    max_embeddings = getattr(finetuned_model.config, "max_position_embeddings", None)

    if max_embeddings and max_embeddings < 2 * config_dict["sequence_len"]:
        config_dict["sequence_len"] = ceil(max_embeddings / 2)

    return DictDefault(config_dict)


def check_and_log_base_model_size(original_model: str) -> None:
    """Check if base model size is logged in results, if not load and log it."""
    results_dict = load_results_dict()

    if "model_params_count" not in results_dict:
        logger.info("Base model size not logged, loading base model to calculate size")
        base_model = load_model(original_model, is_base_model=True)
        results_dict["model_params_count"] = count_model_parameters(base_model)
        save_results_dict(results_dict)
        logger.info(f"Logged base model size: {results_dict['model_params_count']} parameters")
    else:
        logger.info(f"Base model size already logged: {results_dict['model_params_count']} parameters")
