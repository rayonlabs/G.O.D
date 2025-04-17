import json
import os
import re
import subprocess
import time
from math import ceil
from pathlib import Path

import psutil
import torch
import yaml
from accelerate.utils import find_executable_batch_size
from axolotl.utils.data import load_tokenized_prepared_datasets
from axolotl.utils.dict import DictDefault
from peft import PeftModel
from requests.exceptions import HTTPError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments
from trl import DPOTrainer

from core.config.config_handler import create_dataset_entry
from core.models.utility_models import FileFormat, DPODatasetType
from validator.core import constants as cst
from validator.evaluation.utils import model_is_a_finetune
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def log_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU Memory: Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

    ram = psutil.Process().memory_info()
    logger.info(f"RAM Usage: RSS (Resident Set Size): {ram.rss / 1024**2:.2f} MB")


class ProgressLoggerCallback(TrainerCallback):
    def __init__(self, log_interval_seconds):
        self.step = 0
        self.last_log_time = time.time()
        self.log_interval_seconds = log_interval_seconds

    def on_prediction_step(self, args, state, control, **kwargs):
        self.step += 1
        current_time = time.time()

        if current_time - self.last_log_time >= self.log_interval_seconds:
            self.last_log_time = current_time
            logger.info(f"Evaluation step: {self.step}")

        return control


def _load_and_update_evaluation_config(dataset_name, dataset_type, file_format, finetuned_model, config_path):
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    dataset_entry = create_dataset_entry(
        dataset=dataset_name,
        dataset_type=dataset_type,
        file_format=file_format,
    )
    config_dict["datasets"] = [dataset_entry]

    max_embeddings = getattr(finetuned_model.config, "max_position_embeddings", None)
    if max_embeddings and max_embeddings < 2 * config_dict["sequence_len"]:
        config_dict["sequence_len"] = ceil(max_embeddings / 2)

    return DictDefault(config_dict)


def _load_evaluation_dataset(evaluation_config, tokenizer):
    prepared_path = Path(evaluation_config.output_dir) / "prepared"
    eval_dataset, _ = load_tokenized_prepared_datasets(tokenizer, evaluation_config, prepared_path)

    if "prompt_ids" in eval_dataset[0]:
        eval_dataset = sorted(eval_dataset, key=lambda x: len(x["prompt_ids"]))

    logger.info(f"Loaded DPO evaluation dataset with {len(eval_dataset)} samples")
    return eval_dataset


def _log_dataset_and_model_info(eval_dataset, language_model, tokenizer):
    logger.info(f"Eval dataset sample: {eval_dataset[0]}")
    logger.info(f"Model type: {type(language_model)}")
    logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
    logger.info(f"Model vocabulary size: {language_model.config.vocab_size}")


def _collate_dpo_batch(batch, tokenizer):
    prompt_ids = [torch.tensor(item["prompt_ids"]) for item in batch]
    prompt_attention_mask = [torch.tensor(item["prompt_attention_mask"]) for item in batch]
    chosen_ids = [torch.tensor(item["chosen_ids"]) for item in batch]
    chosen_attention_mask = [torch.tensor(item["chosen_attention_mask"]) for item in batch]
    rejected_ids = [torch.tensor(item["rejected_ids"]) for item in batch]
    rejected_attention_mask = [torch.tensor(item["rejected_attention_mask"]) for item in batch]

    prompt_ids = pad_sequence(prompt_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    prompt_attention_mask = pad_sequence(prompt_attention_mask, batch_first=True, padding_value=0)
    chosen_ids = pad_sequence(chosen_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    chosen_attention_mask = pad_sequence(chosen_attention_mask, batch_first=True, padding_value=0)
    rejected_ids = pad_sequence(rejected_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    rejected_attention_mask = pad_sequence(rejected_attention_mask, batch_first=True, padding_value=0)

    return {
        "prompt_ids": prompt_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "chosen_ids": chosen_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "rejected_ids": rejected_ids,
        "rejected_attention_mask": rejected_attention_mask,
    }


def evaluate_dpo_model(evaluation_config, finetuned_model, reference_model, tokenizer):
    evaluation_config.tokenizer_config = tokenizer.name_or_path

    eval_dataset = _load_evaluation_dataset(evaluation_config, tokenizer)
    _log_dataset_and_model_info(eval_dataset, finetuned_model, tokenizer)
    log_memory_stats()

    def custom_data_collator(features):
        return _collate_dpo_batch(features, tokenizer)

    @find_executable_batch_size(starting_batch_size=evaluation_config.starting_batch_size)
    def evaluate_dpo_with_batch_size(batch_size):
        training_args = TrainingArguments(
            output_dir=evaluation_config.output_dir,
            per_device_eval_batch_size=batch_size,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        beta = evaluation_config.get("dpo_beta", 0.1)
        logger.info(f"Using DPO beta: {beta}")

        dpo_trainer = DPOTrainer(
            model=finetuned_model,
            ref_model=reference_model,
            args=training_args,
            train_dataset=None,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            beta=beta,
            callbacks=[ProgressLoggerCallback(log_interval_seconds=evaluation_config.log_interval_seconds)],
        )

        return dpo_trainer.evaluate()

    eval_results = evaluate_dpo_with_batch_size()
    logger.info(f"Final DPO evaluation results: {eval_results}")

    return {
        "eval_loss": eval_results["eval_loss"],
    }


def evaluate_finetuned_dpo_model(dataset_name, finetuned_model, dataset_type, file_format, tokenizer, reference_model):
    evaluation_config = _load_and_update_evaluation_config(
        dataset_name, dataset_type, file_format, finetuned_model, cst.VALI_CONFIG_PATH
    )
    return evaluate_dpo_model(
        evaluation_config, finetuned_model, reference_model, tokenizer
    )


def has_status_code_5xx(e):
    while e is not None:
        if isinstance(e, HTTPError) and 500 <= e.response.status_code < 600:
            return True
        e = e.__cause__
    return False


def retry_on_5xx():
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2.5, min=30, max=600),
        retry=retry_if_exception(has_status_code_5xx),
        reraise=True,
    )


def create_finetuned_cache_dir():
    finetuned_cache_dir = os.path.join(cst.DOCKER_EVAL_HF_CACHE_DIR, "finetuned_repos")
    os.makedirs(finetuned_cache_dir, exist_ok=True)
    return finetuned_cache_dir


@retry_on_5xx()
def load_model(model_name_or_path, is_base_model=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        cache_dir = None if is_base_model else create_finetuned_cache_dir()

        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            token=os.environ.get("HUGGINGFACE_TOKEN"),
            device_map=device,
            cache_dir=cache_dir
        )
    except RuntimeError as e:
        error_msg = str(e)
        if "size mismatch for" in error_msg and ("lm_head.weight" in error_msg or "model.embed_tokens.weight" in error_msg):
            pattern = re.search(r'shape torch\.Size\(\[(\d+), (\d+)\]\).*shape.*torch\.Size\(\[(\d+), \2\]\)', error_msg)
            if pattern and abs(int(pattern.group(1)) - int(pattern.group(3))) == 1:
                logger.info("Detected vocabulary size off-by-one error, attempting to load with ignore_mismatched_sizes=True")
                return AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    token=os.environ.get("HUGGINGFACE_TOKEN"),
                    ignore_mismatched_sizes=True,
                    device_map=device,
                    cache_dir=cache_dir
                )
        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise


@retry_on_5xx()
def load_tokenizer(original_model):
    try:
        return AutoTokenizer.from_pretrained(original_model, token=os.environ.get("HUGGINGFACE_TOKEN"))
    except Exception as e:
        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise


@retry_on_5xx()
def load_finetuned_model(base_model, repo):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        cache_dir = create_finetuned_cache_dir()
        return PeftModel.from_pretrained(
            base_model,
            repo,
            is_trainable=False,
            device_map=device,
            cache_dir=cache_dir
        )
    except RuntimeError as e:
        error_msg = str(e)
        if "size mismatch for" in error_msg and ("lm_head.weight" in error_msg or "model.embed_tokens.weight" in error_msg):
            pattern = re.search(r'shape torch\.Size\(\[(\d+), (\d+)\]\).*shape.*torch\.Size\(\[(\d+), \2\]\)', error_msg)
            if pattern and abs(int(pattern.group(1)) - int(pattern.group(3))) == 1:
                logger.info("Detected vocabulary size off-by-one error, attempting to load with ignore_mismatched_sizes=True")
                return PeftModel.from_pretrained(
                    base_model,
                    repo,
                    is_trainable=False,
                    ignore_mismatched_sizes=True,
                    device_map=device,
                    cache_dir=cache_dir
                )

        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise


def _count_model_parameters(model):
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception as e:
        logger.error(f"Failed to count model parameters: {e}")
        return 0


def evaluate_dpo_repo(repo, dataset, original_model, dataset_type_str, file_format_str):
    output_dir = os.path.dirname(cst.CONTAINER_EVAL_RESULTS_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_dict = {}
    if os.path.exists(cst.CONTAINER_EVAL_RESULTS_PATH):
        try:
            with open(cst.CONTAINER_EVAL_RESULTS_PATH, "r") as f:
                results_dict = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not read existing results from {cst.CONTAINER_EVAL_RESULTS_PATH}, starting fresh")

    if repo in results_dict:
        logger.info(f"Skipping {repo} as it's already evaluated")
        return

    file_format = FileFormat(file_format_str)

    try:
        # Parse the DPO dataset type
        if isinstance(dataset_type_str, str):
            dataset_type = DPODatasetType.model_validate_json(dataset_type_str)
        else:
            dataset_type = DPODatasetType(**dataset_type_str)
    except Exception as e:
        logger.error(f"Failed to parse dataset type: {e}")
        results_dict[repo] = f"Failed to parse dataset type: {str(e)}"
        with open(cst.CONTAINER_EVAL_RESULTS_PATH, "w") as f:
            json.dump(results_dict, f, indent=2)
        return

    tokenizer = load_tokenizer(original_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        try:
            reference_model = load_model(original_model, is_base_model=True)
            if "model_params_count" not in results_dict:
                results_dict["model_params_count"] = _count_model_parameters(reference_model)

            finetuned_model = load_finetuned_model(reference_model, repo)
            is_finetune = True
        except Exception as lora_error:
            logger.info(f"Loading full model... failed to load as LoRA: {lora_error}")

            reference_model.to('cpu')
            reference_model_copy = reference_model
            del reference_model
            torch.cuda.empty_cache()

            finetuned_model = load_model(repo, is_base_model=False)

            try:
                is_finetune = model_is_a_finetune(original_model, finetuned_model)
            except Exception as e:
                logger.info(f"Problem with detection of finetune for {repo}: {e}")
                logger.info("Assuming False")
                is_finetune = False

            reference_model = reference_model_copy if reference_model_copy else load_model(original_model, is_base_model=True)

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
        log_memory_stats()


def main():
    dataset = os.environ.get("DATASET")
    original_model = os.environ.get("ORIGINAL_MODEL")
    dataset_type_str = os.environ.get("DATASET_TYPE", "")
    file_format_str = os.environ.get("FILE_FORMAT")
    models_str = os.environ.get("MODELS", "")

    if not all([dataset, original_model, file_format_str, models_str]):
        logger.error("Missing required environment variables.")
        exit(1)

    repos = [m.strip() for m in models_str.split(",") if m.strip()]

    for repo in repos:
        try:
            subprocess.run([
                "python",
                "-m",
                "validator.evaluation.eval_dpo",
                repo,
                dataset,
                original_model,
                dataset_type_str,
                file_format_str,
            ], check=True)
            logger.info(f"DPO subprocess completed for {repo}")
            log_memory_stats()
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running DPO subprocess for {repo}: {e}")

    logger.info("All DPO evaluations completed")


if __name__ == "__main__":
    main()
