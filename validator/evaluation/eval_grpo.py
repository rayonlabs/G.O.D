import os
import subprocess
import math
import numpy as np

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


def normalize_and_score_models(model_evaluations: list[dict]) -> list[dict]:
    """Calculate aggregate scores across all reward functions for each model."""

    evaluations_by_reward = {}
    for eval_result in model_evaluations:
        reward_func = eval_result['reward_function']
        if reward_func not in evaluations_by_reward:
            evaluations_by_reward[reward_func] = []
        evaluations_by_reward[reward_func].append(eval_result)

    all_normalized_evaluations = []
    for reward_func, evals in evaluations_by_reward.items():
        for eval_result in evals:
            reward = eval_result['results'].get('eval_reward', 0.0)
            loss = max(0.0001, eval_result['results'].get('eval_loss', 0.0))
            eval_result['raw_metrics'] = {'reward': reward, 'loss': loss}

        raw_rewards = [e['raw_metrics']['reward'] for e in evals]
        raw_losses = [e['raw_metrics']['loss'] for e in evals]

        sum_rewards = sum(raw_rewards)
        sum_losses = sum(raw_losses)

        for i, eval_result in enumerate(evals):
            norm_reward = raw_rewards[i] / sum_rewards if sum_rewards > 0 else 1.0 / len(evals)

            if sum_losses > 0:
                inverse_losses = [1.0/max(0.0001, loss) for loss in raw_losses]
                sum_inverse_losses = sum(inverse_losses)
                norm_loss = inverse_losses[i] / sum_inverse_losses if sum_inverse_losses > 0 else 1.0 / len(evals)
            else:
                norm_loss = 1.0 / len(evals)

            eval_result['normalized_metrics'] = {'reward': norm_reward, 'loss': norm_loss}
            # Use reward minus loss (KL penalty already beta-weighted)
            eval_result['grpo_score'] = norm_reward - norm_loss
        evals.sort(key=lambda x: x['grpo_score'], reverse=True)
        all_normalized_evaluations.extend(evals)

    return all_normalized_evaluations


def calculate_aggregate_scores(normalized_evaluations: list[dict]) -> list[dict]:
    """Calculate aggregate scores across all reward functions for each model."""
    scores_by_model = {}
    for eval_result in normalized_evaluations:
        model_name = eval_result['model_name']
        if model_name not in scores_by_model:
            scores_by_model[model_name] = {
                'scores': [], 'raw_rewards': [], 'raw_losses': [],
                'norm_rewards': [], 'norm_losses': []
            }

        data = scores_by_model[model_name]
        data['scores'].append(eval_result['grpo_score'])
        data['raw_rewards'].append(eval_result['raw_metrics']['reward'])
        data['raw_losses'].append(eval_result['raw_metrics']['loss'])
        data['norm_rewards'].append(eval_result['normalized_metrics']['reward'])
        data['norm_losses'].append(eval_result['normalized_metrics']['loss'])

    aggregate_results = []
    for model_name, data in scores_by_model.items():
        aggregate_results.append({
            'model_name': model_name,
            'aggregate_score': sum(data['scores']) / len(data['scores']),
            'avg_raw_reward': sum(data['raw_rewards']) / len(data['raw_rewards']),
            'avg_raw_loss': sum(data['raw_losses']) / len(data['raw_losses']),
            'avg_norm_reward': sum(data['norm_rewards']) / len(data['norm_rewards']),
            'avg_norm_loss': sum(data['norm_losses']) / len(data['norm_losses'])
        })

    aggregate_results.sort(key=lambda x: x['aggregate_score'], reverse=True)
    return aggregate_results


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

    total_weight = sum(rf.reward_weight for rf in evaluation_args.dataset_type.reward_functions)
    if total_weight <= 0:
        equal_weight = 1.0 / len(evaluation_args.dataset_type.reward_functions)
        normalized_weights = [equal_weight] * len(evaluation_args.dataset_type.reward_functions)
        logger.warning(f"Invalid total weight, using equal weights: {equal_weight}")
    else:
        normalized_weights = [rf.reward_weight / total_weight for rf in evaluation_args.dataset_type.reward_functions]

    for i, reward_function in enumerate(evaluation_args.dataset_type.reward_functions):
        reward_func_str = reward_function.reward_func
        is_valid, error_msg, reward_func_callable = validate_reward_function(reward_func_str)
        if not is_valid:
            logger.error(f"Invalid reward function:\n{reward_func_str}")
            raise ValueError(f"Invalid reward function: {error_msg}")

        reward_weight = normalized_weights[i]
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
        num_generations = cst.GRPO_DEFAULT_NUM_GENERATIONS
        while batch_size < num_generations:
            num_generations = num_generations // 2
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
        return results, grpo_trainer.reward_func_names

    eval_results, actual_reward_func_names = evaluate_grpo_with_batch_size()
    logger.info(f"Final GRPO evaluation results: {eval_results}")

    individual_rewards = {}
    for i, name in enumerate(actual_reward_func_names):
        reward_key = f"eval_rewards/{name}/mean"
        if reward_key in eval_results:
            individual_rewards[name] = eval_results[reward_key]
        elif i < len(reward_func_names) and reward_func_names[i] in captured_rewards and captured_rewards[reward_func_names[i]]:
            individual_rewards[name] = sum(captured_rewards[reward_func_names[i]]) / len(captured_rewards[reward_func_names[i]])
        else:
            individual_rewards[name] = 0.0

    # Also add raw (unweighted) rewards if available
    raw_individual_rewards = {}
    for name, rewards_list in raw_rewards.items():
        if rewards_list:
            raw_individual_rewards[name] = sum(rewards_list) / len(rewards_list)

    evaluation_results = {
        "eval_loss": eval_results.get("eval_loss", 0.0),
        "eval_reward": eval_results.get("eval_reward", 0.0),
        "individual_rewards": individual_rewards,
        "raw_rewards": raw_individual_rewards,
        "reward_weights": dict(zip(reward_func_names, reward_weights))
    }

    return evaluation_results


def evaluate_finetuned_grpo_model(
    evaluation_args: EvaluationArgs,
    finetuned_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_name: str = None,
) -> dict[str, float]:
    evaluation_config = _load_and_update_evaluation_config(
        evaluation_args=evaluation_args,
        finetuned_model=finetuned_model,
        config_path=cst.VALI_CONFIG_PATH
    )
    results = evaluate_grpo_model(
        evaluation_config, finetuned_model, tokenizer, evaluation_args
    )

    # Add model name if provided (used for multi-model comparison)
    if model_name:
        results["model_name"] = model_name

    return results


def evaluate_grpo_repo(evaluation_args: EvaluationArgs) -> None:
    """Evaluate a single model repository and save results directly to file."""
    results_dict = load_results_dict()
    repo = evaluation_args.repo

    if "model_evaluations" not in results_dict:
        results_dict["model_evaluations"] = []

    if "normalized_evaluations" not in results_dict:
        results_dict["normalized_evaluations"] = []

    if "aggregate_scores" not in results_dict:
        results_dict["aggregate_scores"] = []

    if repo in results_dict and "individual_rewards" in results_dict[repo]:
        logger.info(f"Skipping {repo} as it's already evaluated")
        return

    tokenizer = load_tokenizer(evaluation_args.original_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        if check_for_lora(repo):
            logger.info("LoRA adapter detected. Loading as with Peft")
            base_model = load_model(evaluation_args.original_model, is_base_model=True)
            if "model_params_count" not in results_dict:
                results_dict["model_params_count"] = count_model_parameters(base_model)
            finetuned_model = load_finetuned_model(base_model, repo)
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
            model_name=repo,
        )
        results["is_finetune"] = is_finetune

        results_dict[repo] = results

        if is_finetune and "individual_rewards" in results:
            for reward_name, reward_value in results["individual_rewards"].items():
                model_evaluation = {
                    "model_name": repo,
                    "reward_function": reward_name,
                    "results": {
                        "eval_reward": reward_value,
                        "eval_loss": results.get("eval_loss", 0.0),
                    }
                }
                results_dict["model_evaluations"].append(model_evaluation)

            if len(set(eval_result["model_name"] for eval_result in results_dict["model_evaluations"])) > 1:
                normalized_evals = normalize_and_score_models(results_dict["model_evaluations"])
                results_dict["normalized_evaluations"] = normalized_evals

                aggregate_scores = calculate_aggregate_scores(normalized_evals)
                results_dict["aggregate_scores"] = aggregate_scores

                for i, score in enumerate(aggregate_scores, 1):
                    logger.info(f"Rank {i}: {score['model_name']} - Score: {score['aggregate_score']:.4f}")

                for score in aggregate_scores:
                    model_repo = score["model_name"]
                    if model_repo in results_dict:
                        results_dict[model_repo]["eval_loss"] = score["aggregate_score"]

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

    for repo in repos:
        try:
            evaluation_args = EvaluationArgs(
                dataset=dataset,
                original_model=original_model,
                dataset_type=dataset_type_str,
                file_format=file_format_str,
                repo=repo
            )

            # Launching subprocess to purge memory
            subprocess.run([
                "python",
                "-m",
                "validator.evaluation.single_eval_grpo",
                evaluation_args.model_dump_json()
            ], check=True)
            logger.info(f"Subprocess completed for {repo}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running subprocess for {repo}: {e}")
    try:
        check_and_log_base_model_size(original_model)
    except Exception as e:
        logger.error(f"Error checking and logging base model size: {e}")

    logger.info("=== GRPO EVALUATION SCRIPT COMPLETED ===")


if __name__ == "__main__":
    main()
