import os
import subprocess
import time

from accelerate.utils import find_executable_batch_size
from axolotl.utils.dict import DictDefault
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from trl import GRPOConfig
from trl import GRPOTrainer

from core.models.utility_models import GrpoDatasetType
from validator.core import constants as cst
from validator.core.models import EvaluationArgs
from validator.evaluation.common import ProgressLoggerCallback
from validator.evaluation.common import _load_and_update_evaluation_config
from validator.evaluation.common import _log_dataset_and_model_info
from validator.evaluation.common import check_and_log_base_model_size
from validator.evaluation.common import load_finetuned_model
from validator.evaluation.common import load_model
from validator.evaluation.common import load_results_dict
from validator.evaluation.common import load_tokenizer
from validator.evaluation.common import log_memory_stats
from validator.evaluation.common import save_results_dict
from validator.evaluation.utils import check_for_lora
from validator.evaluation.utils import model_is_a_finetune
from validator.utils.logging import get_logger
from validator.utils.reward_functions import validate_reward_function


logger = get_logger(__name__)


def _adapt_grpo_columns_to_trl(dataset: Dataset, dataset_type: GrpoDatasetType) -> Dataset:
    """
    Transform a GRPO dataset to match trl's expected column names.

    Args:
        dataset: Hugging Face dataset object
        dataset_type: GrpoDatasetType with field mappings
    """
    logger.info("Adapting GRPO columns to standard format")

    column_mapping = {
        dataset_type.field_prompt: cst.TRL_GRPO_FIELD_PROMPT,
    }
    for src_col, dst_col in column_mapping.items():
        if src_col in dataset.column_names and src_col != dst_col:
            dataset = dataset.rename_column(src_col, dst_col)

    return dataset


def evaluate_grpo_model(
    evaluation_config: DictDefault,
    finetuned_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    evaluation_args: EvaluationArgs
) -> dict[str, float]:
    evaluation_config.tokenizer_config = tokenizer.name_or_path
    logger.info(f"Config: {evaluation_config}")

    dataset_path = evaluation_config.datasets[0]["path"]
    eval_dataset = load_dataset("json", data_files=dataset_path, split="train")


    eval_dataset = _adapt_grpo_columns_to_trl(eval_dataset, evaluation_args.dataset_type)

    _log_dataset_and_model_info(eval_dataset, finetuned_model, tokenizer)

    reward_funcs_callable = []
    reward_func_names = []
    reward_weights = []

    logger.info(f"Processing {len(evaluation_args.dataset_type.reward_functions)} reward functions")

    reward_weights_list = [rf.reward_weight for rf in evaluation_args.dataset_type.reward_functions]
    logger.info(f"Using weights directly: {reward_weights_list}")

    for i, reward_function in enumerate(evaluation_args.dataset_type.reward_functions):
        reward_func_str = reward_function.reward_func
        is_valid, error_msg, reward_func_callable = validate_reward_function(reward_func_str)
        if not is_valid:
            logger.error(f"Invalid reward function:\n{reward_func_str}")
            raise ValueError(f"Invalid reward function: {error_msg}")

        reward_weight = reward_weights_list[i]
        reward_funcs_callable.append(reward_func_callable)

        func_name = getattr(reward_function, 'name', f"reward_func_{i}")
        weighted_name = f"{func_name}_weight_{reward_weight:.2f}"
        reward_func_names.append(weighted_name)
        reward_weights.append(reward_weight)

        logger.info(f"Using reward function {i}: {func_name} with weight {reward_weight:.4f}")

    captured_rewards = {name: [] for name in reward_func_names}
    raw_rewards = {name: [] for name in reward_func_names}
    wrapped_reward_funcs = []

    for i, (original_func, func_name, weight) in enumerate(zip(reward_funcs_callable, reward_func_names, reward_weights)):
        def create_wrapper(original_func, func_name, weight):
            def wrapper(completions, **kwargs):
                raw_results = original_func(completions, **kwargs)
                raw_rewards[func_name].extend(raw_results)
                weighted_results = [r * weight for r in raw_results]
                captured_rewards[func_name].extend(weighted_results)
                return weighted_results
            return wrapper

        wrapped_reward_funcs.append(create_wrapper(original_func, func_name, weight))

    @find_executable_batch_size(starting_batch_size=cst.GRPO_INITIAL_BATCH_SIZE)
    def evaluate_grpo_with_batch_size(batch_size):
        num_generations = max(2, cst.GRPO_DEFAULT_NUM_GENERATIONS)  # Ensure minimum of 2
        # Reduce generations if needed, but never below 2
        while batch_size < num_generations and num_generations > 2:
            num_generations = num_generations // 2
        logger.info(f"Using {num_generations} generations per prompt")
        training_args = GRPOConfig(
            output_dir=evaluation_config.output_dir,
            per_device_eval_batch_size=batch_size,
            report_to="none",
            bf16=True,
            beta=cst.BETA_GRPO,
            num_generations=num_generations,
        )
        grpo_trainer = GRPOTrainer(
            model=finetuned_model,
            reward_funcs=wrapped_reward_funcs,
            args=training_args,
            train_dataset=Dataset.from_dict({col: [] for col in eval_dataset.column_names}),
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=[ProgressLoggerCallback(log_interval_seconds=evaluation_config.log_interval_seconds)],
        )

        results = grpo_trainer.evaluate()
        return results

    eval_results = evaluate_grpo_with_batch_size()
    logger.info(f"Final GRPO evaluation results: {eval_results}")

    final_weighted_rewards = {}
    for name, captured_reward_list in captured_rewards.items():
        if captured_reward_list:
            final_weighted_rewards[name] = sum(captured_reward_list) / len(captured_reward_list)

    evaluation_results = {
        "eval_loss": sum(final_weighted_rewards.values()) - eval_results.get("eval_loss", 0.0),
    }
    return evaluation_results


def evaluate_finetuned_grpo_model(
    evaluation_args: EvaluationArgs,
    finetuned_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> dict[str, float]:
    evaluation_config = _load_and_update_evaluation_config(
        evaluation_args=evaluation_args,
        finetuned_model=finetuned_model,
        config_path=cst.VALI_CONFIG_PATH
    )
    return evaluate_grpo_model(evaluation_config, finetuned_model, tokenizer, evaluation_args)


def evaluate_grpo_repo(evaluation_args: EvaluationArgs) -> None:
    """Evaluate a single model repository and save results directly to file."""
    results_dict = load_results_dict()
    repo = evaluation_args.repo

    # Skip if duplicate
    if repo in results_dict:
        logger.info(f"Skipping {repo} as it's already evaluated")
        return

    tokenizer = load_tokenizer(evaluation_args.original_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        if check_for_lora(repo):
            logger.info("LoRA adapter detected. Loading as with Peft")
            finetuned_model = load_finetuned_model(repo)
            is_finetune = True
        else:
            logger.info("No LoRA adapter detected. Loading full model")
            finetuned_model = load_model(repo, is_base_model=False)
            try:
                is_finetune = model_is_a_finetune(evaluation_args.original_model, finetuned_model)
            except Exception as e:
                logger.info(f"Problem with detection of finetune for {repo}: {e}")
                logger.info("Assuming False")
                is_finetune = False
        log_memory_stats()
        finetuned_model.eval()

        results = evaluate_finetuned_grpo_model(
            evaluation_args=evaluation_args,
            finetuned_model=finetuned_model,
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


def main():
    logger.info("=== GRPO EVALUATION SCRIPT STARTING ===")
    dataset = os.environ.get("DATASET")
    original_model = os.environ.get("ORIGINAL_MODEL")
    dataset_type_str = os.environ.get("DATASET_TYPE", "")
    file_format_str = os.environ.get("FILE_FORMAT")
    models_str = os.environ.get("MODELS", "")
    if not all([dataset, original_model, file_format_str, models_str]):
        logger.error("Missing required environment variables.")
        exit(1)

    repos = [m.strip() for m in models_str.split(",") if m.strip()]

    timeout_seconds = 18_000
    timeout_adjusted = False

    for repo in repos:
        try:
            evaluation_args = EvaluationArgs(
                dataset=dataset,
                original_model=original_model,
                dataset_type=dataset_type_str,
                file_format=file_format_str,
                repo=repo
            )

            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    start_time = time.monotonic()
                    # Launching subprocess to purge memory
                    subprocess.run([
                        "python",
                        "-m",
                        "validator.evaluation.single_eval_grpo",
                        evaluation_args.model_dump_json()
                    ], check=True, timeout=timeout_seconds)
                    elapsed = time.monotonic() - start_time
                    logger.info(f"Subprocess completed for {repo} in {elapsed:.2f} seconds")
                    break
                except subprocess.TimeoutExpired as e:
                    retry_count += 1
                    logger.warning(f"Subprocess timed out for {repo} (attempt {retry_count}/{max_retries})")
                    if retry_count == max_retries:
                        logger.error(f"Max retries reached for {repo}")
                        raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running subprocess for {repo}: {e}")
    try:
        check_and_log_base_model_size(original_model)
    except Exception as e:
        logger.error(f"Error checking and logging base model size: {e}")

    logger.info("=== GRPO EVALUATION SCRIPT COMPLETED ===")


if __name__ == "__main__":
    main()
