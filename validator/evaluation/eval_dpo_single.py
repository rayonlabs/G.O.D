import json
import os
import re
import sys
import gc
import traceback
from math import ceil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer
from transformers import TrainerCallback, TrainingArguments
from accelerate.utils import find_executable_batch_size
from torch.nn.utils.rnn import pad_sequence

import yaml
from axolotl.utils.data import load_tokenized_prepared_datasets
from axolotl.utils.dict import DictDefault

from core.config.config_handler import create_dataset_entry
from core.models.utility_models import FileFormat, DPODatasetType
from validator.core import constants as cst
from validator.evaluation.utils import model_is_a_finetune, check_for_lora
from validator.utils.logging import get_logger

logger = get_logger(__name__)


def log_memory_stats():
    """Log GPU/CPU memory information"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            logger.info(f"GPU {i} Memory: Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

    import psutil
    ram = psutil.Process().memory_info()
    logger.info(f"RAM Usage: RSS (Resident Set Size): {ram.rss / 1024**2:.2f} MB")


class ProgressLoggerCallback(TrainerCallback):
    """Callback to log evaluation progress"""
    def __init__(self, log_interval_seconds):
        self.step = 0
        self.last_log_time = 0
        self.log_interval_seconds = log_interval_seconds

    def on_prediction_step(self, args, state, control, **kwargs):
        import time
        self.step += 1
        current_time = time.time()

        if current_time - self.last_log_time >= self.log_interval_seconds:
            self.last_log_time = current_time
            logger.info(f"Evaluation step: {self.step}")

        return control


def _load_and_update_evaluation_config(dataset_name, dataset_type, file_format, finetuned_model, config_path):
    logger.info(f"Loading evaluation config from {config_path}")
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
    logger.info("Loading evaluation dataset")
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
    logger.info("Starting DPO model evaluation")

    evaluation_config.tokenizer_config = tokenizer.name_or_path

    eval_dataset = _load_evaluation_dataset(evaluation_config, tokenizer)
    _log_dataset_and_model_info(eval_dataset, finetuned_model, tokenizer)
    log_memory_stats()

    def custom_data_collator(features):
        return _collate_dpo_batch(features, tokenizer)

    @find_executable_batch_size(starting_batch_size=evaluation_config.starting_batch_size)
    def evaluate_dpo_with_batch_size(batch_size):
        logger.info(f"Trying batch size: {batch_size}")

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
    logger.info(f"Evaluating finetuned DPO model for dataset: {dataset_name}")

    evaluation_config = _load_and_update_evaluation_config(
        dataset_name, dataset_type, file_format, finetuned_model, cst.VALI_CONFIG_PATH
    )
    return evaluate_dpo_model(
        evaluation_config, finetuned_model, reference_model, tokenizer
    )


def load_model(model_name_or_path, is_base_model=False):
    logger.info(f"Loading model: {model_name_or_path} (is_base_model={is_base_model})")

    try:
        cache_dir = None
        if not is_base_model:
            cache_dir = os.path.join(cst.DOCKER_EVAL_HF_CACHE_DIR, "finetuned_repos")
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Using cache directory: {cache_dir}")

        logger.info("Using device_map='auto' and torch_dtype=torch.bfloat16")
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            token=os.environ.get("HUGGINGFACE_TOKEN"),
            device_map="auto",
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
        )
    except RuntimeError as e:
        error_msg = str(e)
        logger.error(f"RuntimeError loading model: {error_msg}")

        if "size mismatch for" in error_msg and ("lm_head.weight" in error_msg or "model.embed_tokens.weight" in error_msg):
            pattern = re.search(r'shape torch\.Size\(\[(\d+), (\d+)\]\).*shape.*torch\.Size\(\[(\d+), \2\]\)', error_msg)
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
        raise
    except Exception as e:
        logger.error(f"Exception loading model: {type(e)}, message: {str(e)}")
        raise


def load_tokenizer(original_model):
    logger.info(f"Loading tokenizer for model: {original_model}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(original_model, token=os.environ.get("HUGGINGFACE_TOKEN"))
        logger.info(f"Successfully loaded tokenizer: {tokenizer.__class__.__name__}")
        logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
        return tokenizer
    except Exception as e:
        logger.error(f"Exception loading tokenizer: {type(e)}, message: {str(e)}")
        raise


def load_finetuned_model(base_model, repo):
    logger.info(f"Loading finetuned model: {repo} with base model: {base_model.__class__.__name__}")

    try:
        cache_dir = os.path.join(cst.DOCKER_EVAL_HF_CACHE_DIR, "finetuned_repos")
        os.makedirs(cache_dir, exist_ok=True)

        logger.info(f"Using device_map='auto' and torch_dtype=torch.bfloat16")
        return PeftModel.from_pretrained(
            base_model,
            repo,
            is_trainable=False,
            device_map="auto",
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
        )
    except RuntimeError as e:
        error_msg = str(e)
        logger.error(f"RuntimeError loading finetuned model: {error_msg}")

        if "size mismatch for" in error_msg and ("lm_head.weight" in error_msg or "model.embed_tokens.weight" in error_msg):
            pattern = re.search(r'shape torch\.Size\(\[(\d+), (\d+)\]\).*shape.*torch\.Size\(\[(\d+), \2\]\)', error_msg)
            if pattern and abs(int(pattern.group(1)) - int(pattern.group(3))) == 1:
                logger.info("Detected vocabulary size off-by-one error, attempting to load with ignore_mismatched_sizes=True")
                return PeftModel.from_pretrained(
                    base_model,
                    repo,
                    is_trainable=False,
                    ignore_mismatched_sizes=True,
                    device_map="auto",
                    cache_dir=cache_dir,
                    torch_dtype=torch.bfloat16,
                )
        raise
    except Exception as e:
        logger.error(f"Exception loading finetuned model: {type(e)}, message: {str(e)}")
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
        return 0


def evaluate_single_repo(repo, dataset, original_model, dataset_type_str, file_format_str):
    """Main function to evaluate a single repository."""
    logger.info(f"Starting evaluation for repo: {repo}")
    logger.info(f"Dataset: {dataset}, Original model: {original_model}")
    logger.info(f"Dataset type: {dataset_type_str}, File format: {file_format_str}")

    # Make sure the output directory exists
    output_dir = os.path.dirname(cst.CONTAINER_EVAL_RESULTS_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load existing results
    results_dict = {}
    if os.path.exists(cst.CONTAINER_EVAL_RESULTS_PATH):
        try:
            with open(cst.CONTAINER_EVAL_RESULTS_PATH, "r") as f:
                results_dict = json.load(f)
            logger.info(f"Loaded existing results for {len(results_dict)} repos")
        except json.JSONDecodeError:
            logger.warning(f"Could not read existing results from {cst.CONTAINER_EVAL_RESULTS_PATH}, starting fresh")

    if repo in results_dict:
        logger.info(f"Skipping {repo} as it's already evaluated")
        return

    try:
        # Parse input formats
        file_format = FileFormat(file_format_str)

        if isinstance(dataset_type_str, str):
            dataset_type = DPODatasetType.model_validate_json(dataset_type_str)
        else:
            dataset_type = DPODatasetType(**dataset_type_str)

        logger.info(f"File format: {file_format}, Dataset type: {dataset_type}")

        # Load tokenizer
        tokenizer = load_tokenizer(original_model)
        if tokenizer.pad_token_id is None:
            logger.info("Setting pad_token to eos_token")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Check if the model is a LoRA adapter
        is_lora = check_for_lora(repo)
        logger.info(f"Is LoRA adapter: {is_lora}")

        try:
            # Load models based on the type (LoRA or full model)
            if is_lora:
                # Load base model first
                logger.info(f"Loading reference model: {original_model}")
                reference_model = load_model(original_model, is_base_model=True)

                # Record base model params count if not already done
                if "model_params_count" not in results_dict:
                    results_dict["model_params_count"] = _count_model_parameters(reference_model)

                # Load LoRA adapter on top of the base model
                logger.info(f"Loading LoRA adapter: {repo}")
                finetuned_model = load_finetuned_model(reference_model, repo)
                is_finetune = True
            else:
                # Load as a full model
                logger.info(f"Loading as full model: {repo}")
                finetuned_model = load_model(repo, is_base_model=False)

                # Load reference model
                logger.info(f"Loading reference model: {original_model}")
                reference_model = load_model(original_model, is_base_model=True)

                # Check if it's a finetune
                try:
                    is_finetune = model_is_a_finetune(original_model, finetuned_model)
                except Exception as e:
                    logger.warning(f"Problem with detection of finetune: {e}")
                    is_finetune = False

                # Record base model params count if not already done
                if "model_params_count" not in results_dict:
                    results_dict["model_params_count"] = _count_model_parameters(reference_model)

            # Set models to evaluation mode
            logger.info("Setting models to evaluation mode")
            finetuned_model.eval()
            reference_model.eval()

            # Run evaluation
            logger.info("Starting evaluation")
            results = evaluate_finetuned_dpo_model(
                dataset_name=dataset,
                finetuned_model=finetuned_model,
                dataset_type=dataset_type,
                file_format=file_format,
                tokenizer=tokenizer,
                reference_model=reference_model,
            )

            # Record results
            results["is_finetune"] = is_finetune
            results_dict[repo] = results
            logger.info(f"Evaluation successful: {results}")

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            logger.error(traceback.format_exc())
            results_dict[repo] = str(e)
    except Exception as outer_e:
        logger.error(f"Outer error: {outer_e}")
        logger.error(traceback.format_exc())
        results_dict[repo] = str(outer_e)
    finally:
        # Save results regardless of success/failure
        logger.info(f"Saving results to {cst.CONTAINER_EVAL_RESULTS_PATH}")
        with open(cst.CONTAINER_EVAL_RESULTS_PATH, "w") as f:
            json.dump(results_dict, f, indent=2)

        # Clean up
        logger.info("Cleaning up memory")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log_memory_stats()


if __name__ == "__main__":
    try:
        # Check command line arguments
        if len(sys.argv) != 6:
            logger.error(f"Expected 5 arguments, got {len(sys.argv)-1}")
            logger.error("Usage: python -m validator.evaluation.eval_dpo_single <repo> <dataset> <original_model> <dataset_type> <file_format>")
            sys.exit(1)

        # Get command line arguments
        repo = sys.argv[1]
        dataset = sys.argv[2]
        original_model = sys.argv[3]
        dataset_type_str = sys.argv[4]
        file_format_str = sys.argv[5]

        logger.info(f"Starting evaluation for {repo}")
        logger.info(f"Arguments: dataset={dataset}, original_model={original_model}")
        logger.info(f"dataset_type={dataset_type_str}, file_format={file_format_str}")

        # Run evaluation
        evaluate_single_repo(repo, dataset, original_model, dataset_type_str, file_format_str)

        logger.info(f"Evaluation for {repo} completed")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
