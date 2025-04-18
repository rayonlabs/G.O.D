import os
import subprocess
from pathlib import Path

import torch
from accelerate.utils import find_executable_batch_size
from axolotl.utils.data import load_tokenized_prepared_datasets
from axolotl.utils.dict import DictDefault
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

from core.models.utility_models import FileFormat
from core.models.utility_models import InstructDatasetType
from validator.core import constants as cst
from validator.evaluation.common import ProgressLoggerCallback
from validator.evaluation.common import _load_and_update_evaluation_config
from validator.evaluation.common import _log_dataset_and_model_info
from validator.evaluation.common import count_model_parameters
from validator.evaluation.common import load_finetuned_model
from validator.evaluation.common import load_model
from validator.evaluation.common import load_results_dict
from validator.evaluation.common import load_tokenizer
from validator.evaluation.common import log_memory_stats
from validator.evaluation.common import save_results_dict
from validator.evaluation.utils import check_for_lora
from validator.evaluation.utils import model_is_a_finetune
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def _load_evaluation_dataset(evaluation_config: DictDefault, tokenizer: AutoTokenizer) -> Dataset:
    prepared_path = Path(evaluation_config.output_dir) / "prepared"
    eval_dataset, _ = load_tokenized_prepared_datasets(tokenizer, evaluation_config, prepared_path)

    original_length = len(eval_dataset)
    eval_dataset = [sample for sample in eval_dataset if any(label != -100 for label in sample["labels"])]
    filtered_length = len(eval_dataset)

    logger.info(f"Filtered out {original_length - filtered_length} samples with empty outputs")
    eval_dataset = sorted(eval_dataset, key=lambda x: len(x["input_ids"]))
    logger.info(f"Loaded evaluation dataset with {filtered_length} samples")
    return eval_dataset


def _collate_evaluation_batch(batch: list[dict[str, list[int]]], tokenizer: AutoTokenizer) -> dict[str, torch.Tensor]:
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def evaluate_language_model_loss(
    evaluation_config: DictDefault,
    language_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> dict[str, float]:
    evaluation_config.tokenizer_config = tokenizer.name_or_path
    logger.info(f"Config: {evaluation_config}")

    eval_dataset = _load_evaluation_dataset(evaluation_config, tokenizer)

    _log_dataset_and_model_info(eval_dataset, language_model, tokenizer)

    def custom_data_collator(features):
        return _collate_evaluation_batch(features, tokenizer)

    @find_executable_batch_size(starting_batch_size=evaluation_config.starting_batch_size)
    def evaluate_with_batch_size(batch_size):
        training_args = TrainingArguments(
            output_dir=evaluation_config.output_dir,
            per_device_eval_batch_size=batch_size,
            report_to="none",
        )

        trainer = Trainer(
            model=language_model,
            args=training_args,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            data_collator=custom_data_collator,
            callbacks=[ProgressLoggerCallback(log_interval_seconds=evaluation_config.log_interval_seconds)],
        )

        eval_results = trainer.evaluate()
        return eval_results

    eval_results = evaluate_with_batch_size()
    logger.info(f"Final evaluation results: {eval_results}")
    evaluation_results = {
        "eval_loss": eval_results["eval_loss"],
    }
    return evaluation_results


def evaluate_finetuned_model(
    dataset_name: str,
    finetuned_model: AutoModelForCausalLM,
    dataset_type: InstructDatasetType,
    file_format: FileFormat,
    tokenizer: AutoTokenizer,
) -> dict[str, float]:
    evaluation_config = _load_and_update_evaluation_config(
        dataset_name, dataset_type, file_format, finetuned_model, cst.VALI_CONFIG_PATH
    )
    return evaluate_language_model_loss(evaluation_config, finetuned_model, tokenizer)


def evaluate_repo(repo: str, dataset: str, original_model: str, dataset_type_str: str, file_format_str: str) -> None:
    """Evaluate a single model repository and save results directly to file."""
    results_dict = load_results_dict()

    # Skip if duplicate
    if repo in results_dict:
        logger.info(f"Skipping {repo} as it's already evaluated")
        return

    file_format = FileFormat(file_format_str)
    try:
        dataset_type = InstructDatasetType.model_validate_json(dataset_type_str)
    except ValueError:
        logger.error(f"Invalid dataset type: {dataset_type_str}")

    tokenizer = load_tokenizer(original_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        if check_for_lora(repo):
            logger.info("LoRA adapter detected. Loading as with Peft")
            base_model = load_model(original_model, is_base_model=True)
            if "model_params_count" not in results_dict:
                results_dict["model_params_count"] = count_model_parameters(base_model)
            finetuned_model = load_finetuned_model(base_model, repo)
            is_finetune = True
        else:
            logger.info("No LoRA adapter detected. Loading full model")
            finetuned_model = load_model(repo, is_base_model=False)
            try:
                is_finetune = model_is_a_finetune(original_model, finetuned_model)
            except Exception as e:
                logger.info(f"Problem with detection of finetune for {repo}: {e}")
                logger.info("Assuming False")
                is_finetune = False
        log_memory_stats()
        finetuned_model.eval()

        results = evaluate_finetuned_model(
            dataset_name=dataset,
            finetuned_model=finetuned_model,
            dataset_type=dataset_type,
            file_format=file_format,
            tokenizer=tokenizer,
        )
        results["is_finetune"] = is_finetune
        results_dict[repo] = results
    except Exception as e:
        logger.error(f"Error evaluating {repo}: {e}", exc_info=True)
        results_dict[repo] = str(e)
    finally:
        save_results_dict(results_dict, repo)
        log_memory_stats()


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


def main():
    logger.info("=== INSTRUCT TEXT EVALUATION SCRIPT STARTING ===")
    dataset = os.environ.get("DATASET")
    original_model = os.environ.get("ORIGINAL_MODEL")
    dataset_type_str = os.environ.get("DATASET_TYPE", "")
    file_format_str = os.environ.get("FILE_FORMAT")
    models_str = os.environ.get("MODELS", "")  # Comma-separated list of LoRA repos
    if not all([dataset, original_model, file_format_str, models_str]):
        logger.error("Missing required environment variables.")
        exit(1)

    lora_repos = [m.strip() for m in models_str.split(",") if m.strip()]

    for repo in lora_repos:
        try:
            # Launching subprocess to purge memory: https://github.com/huggingface/transformers/issues/26571
            subprocess.run([
                "python",
                "-m",
                "validator.evaluation.single_eval_instruct_text",
                repo,
                dataset,
                original_model,
                dataset_type_str,
                file_format_str
            ], check=True)
            logger.info(f"Subprocess completed for {repo}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running subprocess for {repo}: {e}")
    try:
        check_and_log_base_model_size(original_model)
    except Exception as e:
        logger.error(f"Error checking and logging base model size: {e}")

    logger.info("=== INSTRUCT TEXT EVALUATION SCRIPT COMPLETED ===")


if __name__ == "__main__":
    main()
