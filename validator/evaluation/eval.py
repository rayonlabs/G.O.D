import json
import os
import time
from math import ceil
from pathlib import Path
from typing import Union
import requests

import re
import torch
import yaml
from accelerate.utils import find_executable_batch_size
from axolotl.utils.data import load_tokenized_prepared_datasets
from axolotl.utils.dict import DictDefault
from peft import PeftModel
from requests.exceptions import HTTPError
from tenacity import retry
from tenacity import retry_if_exception
from tenacity import stop_after_attempt
from tenacity import wait_exponential
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainerCallback
from transformers import TrainingArguments

from core.config.config_handler import create_dataset_entry
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import DatasetType
from core.models.utility_models import FileFormat
from validator.core import constants as cst
from validator.evaluation.utils import model_is_a_finetune
from validator.utils.logging import get_logger


logger = get_logger(__name__)


class ProgressLoggerCallback(TrainerCallback):
    """
    A callback that logs the progress of the evaluation every log_interval_seconds seconds.
    """
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


def _load_and_update_evaluation_config(
    dataset_name: str,
    dataset_type: Union[DatasetType, CustomDatasetType],
    file_format: FileFormat,
    finetuned_model: AutoModelForCausalLM,
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

    max_embeddings = getattr(finetuned_model.config, "max_position_embeddings", None)

    if max_embeddings and max_embeddings < 2 * config_dict["sequence_len"]:
        config_dict["sequence_len"] = ceil(max_embeddings / 2)

    return DictDefault(config_dict)


def _load_evaluation_dataset(evaluation_config: DictDefault, tokenizer: AutoTokenizer) -> Dataset:
    prepared_path = Path(evaluation_config.output_dir) / "prepared"
    eval_dataset, _ = load_tokenized_prepared_datasets(tokenizer, evaluation_config, prepared_path)
    eval_dataset = sorted(eval_dataset, key=lambda x: len(x["input_ids"]))
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
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = Trainer(
            model=language_model,
            args=training_args,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            data_collator=custom_data_collator,
            callbacks=[ProgressLoggerCallback(log_interval_seconds=evaluation_config.log_interval_seconds)]
        )

        eval_results = trainer.evaluate()
        return eval_results

    eval_results = evaluate_with_batch_size()
    logger.info(f"Final evaluation results: {eval_results}")
    evaluation_results = {
        "eval_loss": eval_results["eval_loss"],
        "perplexity": torch.exp(torch.tensor(eval_results["eval_loss"])).item(),
    }
    return evaluation_results


def evaluate_finetuned_model(
    dataset_name: str,
    finetuned_model: AutoModelForCausalLM,
    dataset_type: Union[DatasetType, CustomDatasetType],
    file_format: FileFormat,
    tokenizer: AutoTokenizer,
) -> dict[str, float]:
    evaluation_config = _load_and_update_evaluation_config(
        dataset_name, dataset_type, file_format, finetuned_model, cst.VALI_CONFIG_PATH
    )
    return evaluate_language_model_loss(evaluation_config, finetuned_model, tokenizer)


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

@retry_on_5xx()
def load_model(model_name_or_path: str) -> AutoModelForCausalLM:
    try:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, token=os.environ.get("HUGGINGFACE_TOKEN"))
    except RuntimeError as e:
        error_msg = str(e)
        if "size mismatch for" in error_msg and ("lm_head.weight" in error_msg or "model.embed_tokens.weight" in error_msg):
            pattern = re.search(r'shape torch\.Size\(\[(\d+), (\d+)\]\).*shape.*torch\.Size\(\[(\d+), \2\]\)', error_msg)
            if pattern and abs(int(pattern.group(1)) - int(pattern.group(3))) == 1:
                logger.info("Detected vocabulary size off-by-one error, attempting to load with ignore_mismatched_sizes=True")
                return AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    token=os.environ.get("HUGGINGFACE_TOKEN"),
                    ignore_mismatched_sizes=True
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


def load_finetuned_model(base_model, repo: str) -> PeftModel:
    try:

        precision_info = {}

        adapter_config_path = os.path.join(repo, "adapter_config.json")

        if not os.path.exists(adapter_config_path):
            try:
                config_url = f"https://huggingface.co/{repo}/raw/main/adapter_config.json"
                response = requests.get(config_url)
                if response.status_code == 200:
                    adapter_config = response.json()
                else:
                    adapter_config = {}
            except Exception:
                adapter_config = {}
        else:
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)

        if adapter_config and 'dtype' in adapter_config:
            if 'bfloat16' in adapter_config['dtype']:
                precision_info['torch_dtype'] = torch.bfloat16
                logger.info("Using bfloat16 precision from adapter config")
            elif 'float16' in adapter_config['dtype']:
                precision_info['torch_dtype'] = torch.float16
                logger.info("Using float16 precision from adapter config")
        else:
            precision_info['torch_dtype'] = torch.bfloat16
            logger.info("No precision info in adapter config, defaulting to bfloat16")

        return PeftModel.from_pretrained(
            base_model,
            repo,
            is_trainable=False,
            **precision_info
        )
    except Exception as e:
        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise


def _count_model_parameters(model: AutoModelForCausalLM) -> int:
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception as e:
        logger.error(f"Failed to count model parameters: {e}")
        return 0


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

    tokenizer = load_tokenizer(original_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_repos = [m.strip() for m in models_str.split(",") if m.strip()]

    results_dict = {}
    for repo in lora_repos:
        try:
            try:
                base_model = load_model(original_model)
                if "model_params_count" not in results_dict:
                    results_dict["model_params_count"] = _count_model_parameters(base_model)
                finetuned_model = load_finetuned_model(base_model, repo)
                is_finetune = True
            except Exception as lora_error:
                logger.info(f"Loading full model... failed to load as LoRA: {lora_error}")
                finetuned_model = load_model(repo)
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
