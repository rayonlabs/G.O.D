import json
import os
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
import yaml
from axolotl.utils.data import load_tokenized_prepared_datasets
from axolotl.utils.dict import DictDefault
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from core.config.config_handler import create_dataset_entry
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import DatasetType
from core.models.utility_models import FileFormat
from validator.core import constants as cst
from validator.evaluation.utils import model_is_a_finetune
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def _load_and_update_evaluation_config(
    dataset_name: str,
    dataset_type: Union[DatasetType, CustomDatasetType],
    file_format: FileFormat,
    config_path: str,
) -> DictDefault:
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    dataset_entry = create_dataset_entry(
        dataset=dataset_name,
        dataset_type=dataset_type,
        file_format=file_format,
    )
    config_dict["datasets"] = [dataset_entry]
    return DictDefault(config_dict)


def _load_evaluation_dataset(evaluation_config: DictDefault, tokenizer: AutoTokenizer) -> Dataset:
    prepared_path = Path(evaluation_config.output_dir) / "prepared"
    eval_dataset, _ = load_tokenized_prepared_datasets(tokenizer, evaluation_config, prepared_path)
    logger.info(f"Loaded evaluation dataset with {len(eval_dataset)} samples")
    return eval_dataset


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


def _create_evaluation_dataloader(eval_dataset: Dataset, evaluation_config: DictDefault, tokenizer: AutoTokenizer) -> DataLoader:
    return DataLoader(
        eval_dataset,
        batch_size=evaluation_config.micro_batch_size,
        collate_fn=lambda batch: _collate_evaluation_batch(batch, tokenizer),
        shuffle=False,
    )


def _collate_evaluation_batch(batch: list[dict[str, list[int]]], tokenizer: AutoTokenizer) -> dict[str, torch.Tensor]:
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _process_evaluation_batches(
    language_model: AutoModelForCausalLM,
    eval_dataloader: DataLoader,
    device: torch.device,
    evaluation_config: DictDefault,
) -> tuple[list[float], int]:
    batch_losses = []
    num_batches = 0
    consecutive_nans = 0
    max_consecutive_nans = evaluation_config.get('max_consecutive_nans')

    language_model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            logger.info(f"Processing batch {batch_idx + 1}")
            batch_loss = _compute_batch_loss(language_model, batch, device)
            logger.info(f"Batch {batch_idx + 1} loss: {batch_loss}")

            if torch.isnan(torch.tensor(batch_loss)):
                consecutive_nans += 1
                if consecutive_nans >= max_consecutive_nans:
                    logger.error(f"Stopping evaluation early: {max_consecutive_nans} consecutive NaN losses detected")
                    return [float('nan')], 1
            else:
                consecutive_nans = 0

            batch_losses.append(batch_loss)
            num_batches += 1

    return batch_losses, num_batches


def _compute_batch_loss(language_model: AutoModelForCausalLM, batch: dict, device: torch.device) -> float:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = language_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    return loss.item()


def _calculate_evaluation_metrics(
    total_losses: list[float],
    num_batches: int,
    evaluation_config: DictDefault,
) -> dict[str, float]:
    valid_losses = [loss for loss in total_losses if not torch.isnan(torch.tensor(loss))]
    nan_count = len(total_losses) - len(valid_losses)
    nan_percentage = (nan_count / num_batches) * 100 if num_batches > 0 else 0

    if not valid_losses:
        logger.error("No valid losses were found during evaluation.")
        return {
            "eval_loss": float("inf"),
            "perplexity": float("inf"),
        }

    if nan_percentage > evaluation_config.get('max_nan_percentage'):
        logger.error(f"Too many nan values ({nan_percentage:.2f}% of batches)")
        return {
            "eval_loss": float("inf"),
            "perplexity": float("inf"),
        }

    average_loss = sum(valid_losses) / len(valid_losses)
    logger.info(f"Average loss: {average_loss} (calculated from {len(valid_losses)} valid batches)")

    if nan_count > 0:
        logger.warning(f"Skipped {nan_count} batches with nan values ({nan_percentage:.2f}% of total)")

    return {
        "eval_loss": average_loss,
        "perplexity": torch.exp(torch.tensor(average_loss)).item(),
    }


def evaluate_language_model_loss(
    evaluation_config: DictDefault,
    language_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> dict[str, float]:
    evaluation_config.tokenizer_config = tokenizer.name_or_path
    logger.info(f"Config: {evaluation_config}")

    eval_dataset = _load_evaluation_dataset(evaluation_config, tokenizer)
    _log_dataset_and_model_info(eval_dataset, language_model, tokenizer)
    eval_dataloader = _create_evaluation_dataloader(eval_dataset, evaluation_config, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    language_model.to(device)
    losses, num_batches = _process_evaluation_batches(language_model, eval_dataloader, device, evaluation_config)
    evaluation_results = _calculate_evaluation_metrics(losses, num_batches, evaluation_config)
    logger.info(f"Final evaluation results: {evaluation_results}")

    return evaluation_results


def evaluate_finetuned_model(
    dataset_name: str,
    finetuned_model: AutoModelForCausalLM,
    dataset_type: Union[DatasetType, CustomDatasetType],
    file_format: FileFormat,
    tokenizer: AutoTokenizer,
) -> dict[str, float]:
    evaluation_config = _load_and_update_evaluation_config(dataset_name, dataset_type, file_format, cst.VALI_CONFIG_PATH)
    return evaluate_language_model_loss(evaluation_config, finetuned_model, tokenizer)


def main():
    dataset = os.environ.get("DATASET")
    original_model = os.environ.get("ORIGINAL_MODEL")
    dataset_type_str = os.environ.get("DATASET_TYPE", "")
    file_format_str = os.environ.get("FILE_FORMAT")
    models_str = os.environ.get("MODELS", "")  # Comma-separated list of LoRA repos
    if not all([dataset, original_model, file_format_str, models_str]):
        logger.error("Missing required environment variables.")
        exit(1)

    file_format = FileFormat(file_format_str)

    try:
        dataset_type = DatasetType(dataset_type_str)
    except ValueError:
        dataset_type = CustomDatasetType.model_validate_json(dataset_type_str)

    base_model = AutoModelForCausalLM.from_pretrained(original_model, token=os.environ.get("HUGGINGFACE_TOKEN"))
    tokenizer = AutoTokenizer.from_pretrained(original_model, token=os.environ.get("HUGGINGFACE_TOKEN"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_repos = [m.strip() for m in models_str.split(",") if m.strip()]

    results_dict = {}
    for repo in lora_repos:
        try:
            try:
                finetuned_model = PeftModel.from_pretrained(base_model, repo, is_trainable=False)
                is_finetune = True
            except Exception as lora_error:
                logger.info(f"Loading full model... failed to load as LoRA: {lora_error}")
                finetuned_model = AutoModelForCausalLM.from_pretrained(repo, token=os.environ.get("HUGGINGFACE_TOKEN"))
                try:
                    is_finetune = model_is_a_finetune(original_model, finetuned_model)
                except Exception as e:
                    logger.info(f"Problem with detection of finetune for {repo}: {e}")
                    logger.info("Assuming False")
                    is_finetune = False

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
            logger.error(f"Error evaluating {repo}: {e}")
            results_dict[repo] = e

    output_file = "/aplp/evaluation_results.json"
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    serializable_results = {
        repo: (str(result) if isinstance(result, Exception) else result) for repo, result in results_dict.items()
    }

    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Evaluation results saved to {output_file}")
    logger.info(json.dumps(serializable_results, indent=2))


if __name__ == "__main__":
    main()
