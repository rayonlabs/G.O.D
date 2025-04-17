import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
import traceback
from math import ceil

import psutil
import torch
import yaml
from accelerate.utils import find_executable_batch_size
from axolotl.utils.dict import DictDefault
from datasets import Dataset
from datasets import load_dataset
from peft import PeftModel
from requests.exceptions import HTTPError
from tenacity import retry
from tenacity import retry_if_exception
from tenacity import stop_after_attempt
from tenacity import wait_exponential
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TrainerCallback
from trl import DPOConfig
from trl import DPOTrainer

from core.config.config_handler import create_dataset_entry
from core.models.utility_models import DPODatasetType
from core.models.utility_models import FileFormat
from validator.core import constants as cst
from validator.evaluation.utils import model_is_a_finetune
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def _adapt_dpo_columns_to_trl(dataset: Dataset, dataset_type: DPODatasetType) -> Dataset:
    """
    Transform a DPO dataset to match trl's expected column names.

    Args:
        dataset: Hugging Face dataset object
        dataset_type: DPODatasetType with field mappings
    """
    logger.info("Adapting DPO columns to standard format")

    column_mapping = {
        dataset_type.field_prompt: cst.TRL_DPO_FIELD_PROMPT,
        dataset_type.field_chosen: cst.TRL_DPO_FIELD_CHOSEN,
        dataset_type.field_rejected: cst.TRL_DPO_FIELD_REJECTED
    }
    for src_col, dst_col in column_mapping.items():
        if src_col in dataset.column_names and src_col != dst_col:
            dataset = dataset.rename_column(src_col, dst_col)

    columns_to_keep = [cst.TRL_DPO_FIELD_PROMPT, cst.TRL_DPO_FIELD_CHOSEN, cst.TRL_DPO_FIELD_REJECTED]
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
    for col in columns_to_remove:
        dataset = dataset.remove_columns(col)

    return dataset


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
            log_memory_stats()

        return control


def _load_and_update_evaluation_config(dataset_name, dataset_type, file_format, finetuned_model, config_path):
    logger.info(f"Loading evaluation config from {config_path}")
    try:
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)
        logger.info(f"Successfully loaded config: {config_dict.keys()}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        logger.error(traceback.format_exc())
        raise

    try:
        logger.info(f"Creating dataset entry for {dataset_name}, type={dataset_type}, format={file_format}")
        dataset_entry = create_dataset_entry(
            dataset=dataset_name,
            dataset_type=dataset_type,
            file_format=file_format,
        )
        logger.info(f"Created dataset entry: {dataset_entry}")

        config_dict["datasets"] = [dataset_entry]

        max_embeddings = getattr(finetuned_model.config, "max_position_embeddings", None)
        logger.info(
            f"Model max position embeddings: {max_embeddings}, "
            f"config sequence_len: {config_dict.get('sequence_len')}"
        )

        if max_embeddings and max_embeddings < 2 * config_dict.get("sequence_len", 0):
            logger.info(f"Adjusting sequence_len from {config_dict['sequence_len']} to {ceil(max_embeddings / 2)}")
            config_dict["sequence_len"] = ceil(max_embeddings / 2)

        logger.info(f"Final config: {config_dict}")
        return DictDefault(config_dict)
    except Exception as e:
        logger.error(f"Error updating evaluation config: {e}")
        logger.error(traceback.format_exc())
        raise


def _load_evaluation_dataset(evaluation_config, tokenizer):
    logger.info("Starting to load evaluation dataset")
    try:
        dataset_path = evaluation_config.datasets[0]["path"]
        logger.info(f"Loading dataset directly from: {dataset_path}")

        eval_dataset = load_dataset("json", data_files=dataset_path, split="train")

        logger.info(f"Dataset loaded with {len(eval_dataset)} samples")

        if "prompt_ids" in eval_dataset[0]:
            logger.info("Sorting dataset by prompt length")
            eval_dataset = sorted(eval_dataset, key=lambda x: len(x["prompt_ids"]))
            logger.info(
                f"Dataset sorted. Shortest prompt length: {len(eval_dataset[0]['prompt_ids'])}, "
                f"Longest: {len(eval_dataset[-1]['prompt_ids'])}"
            )

        return eval_dataset
    except Exception as e:
        logger.error(f"Error loading evaluation dataset: {e}")
        logger.error(traceback.format_exc())
        raise


def _log_dataset_and_model_info(eval_dataset, language_model, tokenizer):
    logger.info("=== DATASET AND MODEL INFO ===")

    # Log dataset info
    try:
        sample = eval_dataset[0]
        keys = list(sample.keys())
        logger.info(f"Dataset keys: {keys}")

        for key in keys:
            if isinstance(sample[key], list):
                logger.info(f"Sample {key} length: {len(sample[key])}")
            else:
                logger.info(f"Sample {key} type: {type(sample[key])}")

        logger.info(f"Total dataset samples: {len(eval_dataset)}")
    except Exception as e:
        logger.error(f"Error logging dataset info: {e}")

    # Log model info
    try:
        logger.info(f"Model type: {type(language_model)}")
        logger.info(f"Model class: {language_model.__class__.__name__}")
        logger.info(f"Model config: {language_model.config.__class__.__name__}")

        # Log model architecture details
        if hasattr(language_model.config, "model_type"):
            logger.info(f"Model architecture: {language_model.config.model_type}")
        if hasattr(language_model.config, "vocab_size"):
            logger.info(f"Model vocabulary size: {language_model.config.vocab_size}")
        if hasattr(language_model.config, "hidden_size"):
            logger.info(f"Model hidden size: {language_model.config.hidden_size}")
        if hasattr(language_model.config, "num_hidden_layers"):
            logger.info(f"Model layers: {language_model.config.num_hidden_layers}")
    except Exception as e:
        logger.error(f"Error logging model info: {e}")

    # Log tokenizer info
    try:
        logger.info(f"Tokenizer class: {tokenizer.__class__.__name__}")
        logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
        special_tokens = {
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
            "bos_token": tokenizer.bos_token if hasattr(tokenizer, "bos_token") else None,
            "unk_token": tokenizer.unk_token if hasattr(tokenizer, "unk_token") else None,
        }
        logger.info(f"Special tokens: {special_tokens}")
    except Exception as e:
        logger.error(f"Error logging tokenizer info: {e}")

    logger.info("==============================")


def _collate_dpo_batch(batch, tokenizer):
    logger.debug(f"Collating batch of size {len(batch)}")
    try:
        prompt_ids = [torch.tensor(item["prompt_ids"]) for item in batch]
        prompt_attention_mask = [torch.tensor(item["prompt_attention_mask"]) for item in batch]
        chosen_ids = [torch.tensor(item["chosen_ids"]) for item in batch]
        chosen_attention_mask = [torch.tensor(item["chosen_attention_mask"]) for item in batch]
        rejected_ids = [torch.tensor(item["rejected_ids"]) for item in batch]
        rejected_attention_mask = [torch.tensor(item["rejected_attention_mask"]) for item in batch]

        # Log tensors shape before padding
        if logger.isEnabledFor(10):  # DEBUG level
            shapes = {
                "prompt_ids": [t.shape for t in prompt_ids],
                "prompt_attention_mask": [t.shape for t in prompt_attention_mask],
                "chosen_ids": [t.shape for t in chosen_ids],
                "chosen_attention_mask": [t.shape for t in chosen_attention_mask],
                "rejected_ids": [t.shape for t in rejected_ids],
                "rejected_attention_mask": [t.shape for t in rejected_attention_mask],
            }
            logger.debug(f"Tensor shapes before padding: {shapes}")

        prompt_ids = pad_sequence(prompt_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        prompt_attention_mask = pad_sequence(prompt_attention_mask, batch_first=True, padding_value=0)
        chosen_ids = pad_sequence(chosen_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        chosen_attention_mask = pad_sequence(chosen_attention_mask, batch_first=True, padding_value=0)
        rejected_ids = pad_sequence(rejected_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        rejected_attention_mask = pad_sequence(rejected_attention_mask, batch_first=True, padding_value=0)

        # Log tensors shape after padding
        if logger.isEnabledFor(10):  # DEBUG level
            shapes = {
                "prompt_ids": prompt_ids.shape,
                "prompt_attention_mask": prompt_attention_mask.shape,
                "chosen_ids": chosen_ids.shape,
                "chosen_attention_mask": chosen_attention_mask.shape,
                "rejected_ids": rejected_ids.shape,
                "rejected_attention_mask": rejected_attention_mask.shape,
            }
            logger.debug(f"Tensor shapes after padding: {shapes}")

        return {
            "prompt_ids": prompt_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "chosen_ids": chosen_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_ids": rejected_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }
    except Exception as e:
        logger.error(f"Error in collate function: {e}")
        logger.error(traceback.format_exc())
        raise


def evaluate_dpo_model(evaluation_config, finetuned_model, reference_model, tokenizer, dataset_type):
    evaluation_config.tokenizer_config = tokenizer.name_or_path
    logger.info(f"Set tokenizer config to {tokenizer.name_or_path}")

    logger.info("Loading evaluation dataset")
    # eval_dataset = _load_evaluation_dataset(evaluation_config, tokenizer)
    dataset_path = evaluation_config.datasets[0]["path"]
    eval_dataset = load_dataset("json", data_files=dataset_path, split="train")
    eval_dataset = _adapt_dpo_columns_to_trl(eval_dataset, dataset_type)

    _log_dataset_and_model_info(eval_dataset, finetuned_model, tokenizer)

    log_memory_stats()

    def custom_data_collator(features):
        logger.debug(f"Collating {len(features)} features")
        return _collate_dpo_batch(features, tokenizer)

    logger.info(f"Setting up evaluation with starting batch size: {evaluation_config.starting_batch_size}")

    @find_executable_batch_size(starting_batch_size=evaluation_config.starting_batch_size)
    def evaluate_dpo_with_batch_size(batch_size):
        beta = evaluation_config.get("dpo_beta", 0.1)
        training_args = DPOConfig(
            output_dir=evaluation_config.output_dir,
            per_device_eval_batch_size=batch_size,
            report_to="none",
            fp16=torch.cuda.is_available(),
            beta=beta,
        )
        dpo_trainer = DPOTrainer(
            model=finetuned_model,
            ref_model=reference_model,
            args=training_args,
            train_dataset=Dataset.from_dict({"prompt": [], "chosen": [], "rejected": []}),
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=[ProgressLoggerCallback(log_interval_seconds=evaluation_config.log_interval_seconds)],
        )

        results = dpo_trainer.evaluate()
        return results

    eval_results = evaluate_dpo_with_batch_size()
    logger.info(f"Final DPO evaluation results: {eval_results}")
    evaluation_results = {
        "eval_loss": eval_results["eval_loss"],
    }
    return evaluation_results


def evaluate_finetuned_dpo_model(dataset_name, finetuned_model, dataset_type, file_format, tokenizer, reference_model):
    logger.info(f"Evaluating finetuned DPO model for dataset: {dataset_name}")
    try:
        logger.info("Loading and updating evaluation config")
        evaluation_config = _load_and_update_evaluation_config(
            dataset_name, dataset_type, file_format, finetuned_model, cst.VALI_CONFIG_PATH
        )
        logger.info(f"Evaluation config loaded: {evaluation_config}")

        logger.info("Starting DPO model evaluation")
        return evaluate_dpo_model(
            evaluation_config, finetuned_model, reference_model, tokenizer, dataset_type
        )
    except Exception as e:
        logger.error(f"Error in evaluate_finetuned_dpo_model: {e}")
        logger.error(traceback.format_exc())
        raise


def has_status_code_5xx(e):
    logger.debug(f"Checking if exception is a 5xx error: {e}")
    while e is not None:
        if isinstance(e, HTTPError) and 500 <= e.response.status_code < 600:
            logger.debug(f"Found 5xx error: {e.response.status_code}")
            return True
        e = e.__cause__
    return False


def retry_on_5xx():
    logger.info("Creating retry decorator for 5xx errors")
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2.5, min=30, max=600),
        retry=retry_if_exception(has_status_code_5xx),
        reraise=True,
    )


def create_finetuned_cache_dir():
    logger.info(f"Creating finetuned cache directory at {cst.DOCKER_EVAL_HF_CACHE_DIR}/finetuned_repos")
    finetuned_cache_dir = os.path.join(cst.DOCKER_EVAL_HF_CACHE_DIR, "finetuned_repos")
    os.makedirs(finetuned_cache_dir, exist_ok=True)
    return finetuned_cache_dir


@retry_on_5xx()
def load_model(model_name_or_path, is_base_model=False):
    logger.info(f"Loading model: {model_name_or_path} (is_base_model={is_base_model})")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    cache_dir = None
    if not is_base_model:
        cache_dir = create_finetuned_cache_dir()
        logger.info(f"Using cache directory: {cache_dir}")

    try:
        logger.info(f"Calling AutoModelForCausalLM.from_pretrained for {model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device,
            cache_dir=cache_dir
        )
        logger.info(f"Successfully loaded model: {model.__class__.__name__}")
        return model
    except RuntimeError as e:
        error_msg = str(e)
        logger.error(f"RuntimeError loading model: {error_msg}")

        if "size mismatch for" in error_msg and ("lm_head.weight" in error_msg or "model.embed_tokens.weight" in error_msg):
            pattern = re.search(r'shape torch\.Size\(\[(\d+), (\d+)\]\).*shape.*torch\.Size\(\[(\d+), \2\]\)', error_msg)
            if pattern and abs(int(pattern.group(1)) - int(pattern.group(3))) == 1:
                logger.info("Detected vocabulary size off-by-one error, attempting to load with ignore_mismatched_sizes=True")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    ignore_mismatched_sizes=True,
                    device_map=device,
                    cache_dir=cache_dir
                )
                logger.info("Successfully loaded model with ignore_mismatched_sizes=True")
                return model

        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        logger.error(f"Exception loading model: {type(e)}, message: {str(e)}")
        logger.error(traceback.format_exc())
        raise


@retry_on_5xx()
def load_tokenizer(original_model):
    logger.info(f"Loading tokenizer for model: {original_model}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(original_model)
        logger.info(f"Successfully loaded tokenizer: {tokenizer.__class__.__name__}")
        logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")

        # Log special tokens
        special_tokens = {
            "pad_token": getattr(tokenizer, "pad_token", None),
            "eos_token": getattr(tokenizer, "eos_token", None),
            "bos_token": getattr(tokenizer, "bos_token", None),
            "unk_token": getattr(tokenizer, "unk_token", None),
        }
        logger.info(f"Tokenizer special tokens: {special_tokens}")

        return tokenizer
    except Exception as e:
        logger.error(f"Exception loading tokenizer: {type(e)}, message: {str(e)}")
        logger.error(traceback.format_exc())
        raise


@retry_on_5xx()
def load_finetuned_model(base_model, repo):
    logger.info(f"Loading finetuned model: {repo} with base model: {base_model.__class__.__name__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    try:
        cache_dir = create_finetuned_cache_dir()
        logger.info(f"Using cache directory: {cache_dir}")

        logger.info(f"Calling PeftModel.from_pretrained for {repo}")
        model = PeftModel.from_pretrained(
            base_model,
            repo,
            is_trainable=False,
            device_map=device,
            cache_dir=cache_dir
        )
        logger.info(f"Successfully loaded PeftModel: {model.__class__.__name__}")
        return model
    except RuntimeError as e:
        error_msg = str(e)
        logger.error(f"RuntimeError loading finetuned model: {error_msg}")

        if "size mismatch for" in error_msg and ("lm_head.weight" in error_msg or "model.embed_tokens.weight" in error_msg):
            pattern = re.search(r'shape torch\.Size\(\[(\d+), (\d+)\]\).*shape.*torch\.Size\(\[(\d+), \2\]\)', error_msg)
            if pattern and abs(int(pattern.group(1)) - int(pattern.group(3))) == 1:
                logger.info("Detected vocabulary size off-by-one error, attempting to load with ignore_mismatched_sizes=True")
                model = PeftModel.from_pretrained(
                    base_model,
                    repo,
                    is_trainable=False,
                    ignore_mismatched_sizes=True,
                    device_map=device,
                    cache_dir=cache_dir
                )
                logger.info("Successfully loaded model with ignore_mismatched_sizes=True")
                return model

        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        logger.error(f"Exception loading finetuned model: {type(e)}, message: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def _count_model_parameters(model):
    logger.info(f"Counting parameters for model: {model.__class__.__name__}")
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: Total={total_params}, Trainable={trainable_params}")
        return total_params
    except Exception as e:
        logger.error(f"Failed to count model parameters: {e}")
        logger.error(traceback.format_exc())
        return 0


def evaluate_dpo_repo(repo, dataset, original_model, dataset_type_str, file_format_str):
    """Evaluate a single model repository and save results directly to file."""
    output_dir = os.path.dirname(cst.CONTAINER_EVAL_RESULTS_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load existing results
    results_dict = {}
    if os.path.exists(cst.CONTAINER_EVAL_RESULTS_PATH):
        try:
            with open(cst.CONTAINER_EVAL_RESULTS_PATH, "r") as f:
                results_dict = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not read existing results from {cst.CONTAINER_EVAL_RESULTS_PATH}, starting fresh")

    # Skip if duplicate
    if repo in results_dict:
        logger.info(f"Skipping {repo} as it's already evaluated")
        return

    file_format = FileFormat(file_format_str)
    try:
        dataset_type = DPODatasetType.model_validate_json(dataset_type_str)
    except Exception as e:
        logger.error(f"Invalid dataset type: {dataset_type_str}, error: {e}")

    tokenizer = load_tokenizer(original_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        log_memory_stats()
        logger.info(f"Loading reference model: {original_model}")
        reference_model = load_model(original_model, is_base_model=True)
        if "model_params_count" not in results_dict:
            results_dict["model_params_count"] = _count_model_parameters(reference_model)
        try:
            finetuned_model = load_finetuned_model(reference_model, repo)
            is_finetune = True
        except Exception as lora_error:
            logger.info(f"Failed to load as LoRA adapter: {lora_error}")
            logger.info(f"Loading finetuned model as full model: {repo}")
            finetuned_model = load_model(repo, is_base_model=False)
            try:
                is_finetune = model_is_a_finetune(original_model, finetuned_model)
            except Exception as e:
                logger.warning(f"Problem with detection of finetune for {repo}: {e}")
                is_finetune = False
        log_memory_stats()
        finetuned_model.eval()
        reference_model.eval()

        results = evaluate_finetuned_dpo_model(
            dataset_name=dataset,
            finetuned_model=finetuned_model,
            dataset_type=dataset_type,
            file_format=file_format,
            tokenizer=tokenizer,
            reference_model=reference_model,
        )
        results["is_finetune"] = is_finetune
        results_dict[repo] = results
    except Exception as e:
        logger.error(f"Error evaluating {repo}: {e}", exc_info=True)
        results_dict[repo] = str(e)
    finally:
        with open(cst.CONTAINER_EVAL_RESULTS_PATH, "w") as f:
            json.dump(results_dict, f, indent=2)
        logger.info(f"Saved DPO evaluation results for {repo}")
        logger.info(json.dumps(results_dict, indent=2))
        log_memory_stats()


def main():
    dataset = os.environ.get("DATASET")
    original_model = os.environ.get("ORIGINAL_MODEL")
    dataset_type_str = os.environ.get("DATASET_TYPE", "")
    file_format_str = os.environ.get("FILE_FORMAT")
    models_str = os.environ.get("MODELS", "")

    if not all([dataset, original_model, file_format_str, models_str]):
        logger.error("Missing required environment variables.")
        for var, value in {
            "DATASET": dataset,
            "ORIGINAL_MODEL": original_model,
            "FILE_FORMAT": file_format_str,
            "MODELS": models_str
        }.items():
            if not value:
                logger.error(f"Missing {var}")
        exit(1)

    # Log environment variables
    env_vars = {
        "DATASET": os.environ.get("DATASET"),
        "ORIGINAL_MODEL": os.environ.get("ORIGINAL_MODEL"),
        "DATASET_TYPE": os.environ.get("DATASET_TYPE", ""),
        "FILE_FORMAT": os.environ.get("FILE_FORMAT"),
        "MODELS": os.environ.get("MODELS", ""),
        "HUGGINGFACE_TOKEN": "***" if os.environ.get("HUGGINGFACE_TOKEN") else None,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    logger.info(f"Environment variables: {env_vars}")

    # Log CUDA information
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA not available")

    repos = [m.strip() for m in models_str.split(",") if m.strip()]
    logger.info(f"Models to evaluate: {repos}")
    for repo in repos:
        try:
            subprocess.run([
                "python",
                "-m",
                "validator.evaluation.eval_dpo_single",
                repo,
                dataset,
                original_model,
                json.dumps(json.loads(dataset_type_str)),
                file_format_str,
            ], check=True)
            logger.info(f"Subprocess completed for {repo}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running subprocess for {repo}: {e}")

    logger.info("All DPO evaluations completed")


if __name__ == "__main__":
    logger.info("=== DPO EVALUATION SCRIPT STARTING ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers version: {importlib.metadata.version('transformers')}")
    logger.info(f"PEFT version: {importlib.metadata.version('peft')}")

    main()

    logger.info("=== DPO EVALUATION SCRIPT COMPLETED SUCCESSFULLY ===")
