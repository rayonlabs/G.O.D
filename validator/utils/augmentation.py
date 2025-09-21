import random
import re
import uuid

import validator.core.constants as cst
from validator.augmentation.word_honeypots import apply_word_honeypot_to_text
from validator.augmentation.word_honeypots import generate_text_transform_config
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def _rearrange_sentences(text: str) -> str:
    """Split text by sentences and rearrange with some at front, some at end."""
    # Split by sentence endings while keeping the punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 2:
        return text

    # Randomly split sentences into front and back portions
    split_point = random.randint(1, len(sentences) - 1)
    front_sentences = sentences[:split_point]
    back_sentences = sentences[split_point:]

    # Rearrange: back + front
    rearranged = back_sentences + front_sentences
    return " ".join(rearranged)


def _insert_uid_randomly(text: str, uid: str) -> str:
    """Insert UID at a random position in the text."""
    if not text or len(text) < 20:
        return text + f" {uid}"

    words = text.split()
    if len(words) > 2:
        insert_pos = random.randint(1, len(words) - 1)
        words.insert(insert_pos, uid)
        return " ".join(words)
    return text + f" {uid}"


def _generate_dpo_augmentation_config(dataset_size: int) -> dict:
    if random.random() < cst.DPO_AUGMENTATION_PROB:
        config = {
            "rearrange_sentences": random.random() < cst.DPO_AUGMENTATION_PROB,
            "add_prompt_honeypot": random.random() < cst.DPO_AUGMENTATION_PROB,
            "add_response_honeypot": random.random() < cst.DPO_AUGMENTATION_PROB,
            "swap_chosen_rejected": random.random() < cst.DPO_AUGMENTATION_PROB,
        }

        # Configure prompt honeypot (applies to ALL prompts if enabled)
        if config["add_prompt_honeypot"]:
            config["prompt_uid"] = uuid.uuid4().hex[:8]

        # Configure response honeypot (applies to a percentage of chosen OR rejected)
        if config["add_response_honeypot"]:
            config["response_uid"] = uuid.uuid4().hex[:8]
            config["honeypot_in_chosen"] = random.random() < 0.5
            config["honeypot_at_start"] = random.random() < 0.5
            # Select percentage of rows for response honeypot
            num_response_honeypot = int(dataset_size * cst.DPO_RESPONSE_HONEYPOT_PERCENTAGE)
            config["response_honeypot_indices"] = set(
                random.sample(range(dataset_size), min(num_response_honeypot, dataset_size))
            )

        return config
    else:
        return {}


def _generate_instruct_augmentation_config(dataset_size: int, train_dataset=None) -> dict:
    """Generate augmentation configuration for instruct tasks.

    Args:
        dataset_size: Size of the dataset for configuring augmentations
        train_dataset: Optional training dataset to extract instructions for conditional rules
    """
    if random.random() < cst.INSTRUCT_AUGMENTATION_PROB:
        config = {
            "rearrange_input": random.random() < cst.INSTRUCT_AUGMENTATION_PROB,
            "rearrange_output": random.random() < cst.INSTRUCT_AUGMENTATION_PROB,
            "add_input_honeypot": random.random() < cst.INSTRUCT_AUGMENTATION_PROB,
            "add_output_honeypot": random.random() < cst.INSTRUCT_AUGMENTATION_PROB,
        }

        # Configure input honeypot (applies to ALL instructions if enabled)
        if config["add_input_honeypot"]:
            config["input_uid"] = uuid.uuid4().hex[:8]
            config["input_honeypot_at_start"] = random.random() < 0.5

        # Configure output honeypot (applies to a percentage of outputs)
        if config["add_output_honeypot"]:
            config["output_uid"] = uuid.uuid4().hex[:8]
            config["output_honeypot_at_start"] = random.random() < 0.5
            # Select percentage of rows for output honeypot
            num_output_honeypot = int(dataset_size * cst.INSTRUCT_RESPONSE_HONEYPOT_PERCENTAGE)
            config["output_honeypot_indices"] = set(
                random.sample(range(dataset_size), min(num_output_honeypot, dataset_size))
            )

        # Generate text transform configuration
        if random.random() < cst.TEXT_TRANSFORM_PROB:
            # Extract instructions from dataset if provided for conditional rules
            instructions = None
            if train_dataset is not None:
                try:
                    # Extract instruction column data for conditional rule analysis
                    if hasattr(train_dataset, 'column_names') and cst.STANDARD_INSTRUCT_COLUMN in train_dataset.column_names:
                        instructions = [row[cst.STANDARD_INSTRUCT_COLUMN] for row in train_dataset]
                        logger.info(f"Extracted {len(instructions)} instructions for conditional rule analysis")
                except Exception as e:
                    logger.warning(f"Could not extract instructions for conditional rules: {e}")
                    instructions = None

            text_transform_config = generate_text_transform_config(dataset_size, instructions)
            config.update(text_transform_config)

        return config
    else:
        return {}


def _generate_grpo_augmentation_config(dataset_size: int) -> dict:
    """Generate augmentation configuration for GRPO tasks - only apply to prompts."""
    if random.random() < cst.GRPO_AUGMENTATION_PROB:
        config = {
            "add_prompt_honeypot": random.random() < cst.GRPO_PROMPT_HONEYPOT_PROB,
        }

        # Configure prompt honeypot (applies to ALL prompts if enabled)
        if config["add_prompt_honeypot"]:
            config["prompt_uid"] = uuid.uuid4().hex[:8]

        # Generate text transform configuration with GRPO-specific probability
        if random.random() < cst.GRPO_WORD_TRANSFORM_PROB:
            text_transform_config = generate_text_transform_config(dataset_size)

            # For GRPO, filter out output-specific configurations since GRPO only has prompts
            for key, value in text_transform_config.items():
                if key.startswith('output_') or 'OUTPUT' in key.upper():
                    continue
                config[key] = value

        return config
    else:
        return {}


def _apply_dpo_augmentations(row_dict: dict, augmentations: dict, idx: int) -> dict:
    """Apply DPO-specific augmentations to a row."""
    try:
        # 1. Rearrange prompt sentences (applies to ALL rows if enabled)
        if augmentations.get("rearrange_sentences") and cst.STANDARD_DPO_PROMPT_COLUMN in row_dict:
            row_dict[cst.STANDARD_DPO_PROMPT_COLUMN] = _rearrange_sentences(row_dict[cst.STANDARD_DPO_PROMPT_COLUMN])
    except Exception as e:
        logger.error(f"Error in rearrange_sentences: {e}")

    try:
        # 2. Add prompt honeypot (applies to ALL rows if enabled)
        if augmentations.get("add_prompt_honeypot") and cst.STANDARD_DPO_PROMPT_COLUMN in row_dict:
            row_dict[cst.STANDARD_DPO_PROMPT_COLUMN] = _insert_uid_randomly(
                row_dict[cst.STANDARD_DPO_PROMPT_COLUMN], augmentations["prompt_uid"]
            )
    except Exception as e:
        logger.error(f"Error in prompt honeypot: {e}")

    try:
        # 3. Add response honeypot (applies to percentage of rows if enabled)
        response_honeypot_indices = augmentations.get("response_honeypot_indices", set())
        if augmentations.get("add_response_honeypot") and idx in response_honeypot_indices:
            response_uid = augmentations["response_uid"]

            if augmentations["honeypot_in_chosen"] and cst.STANDARD_DPO_CHOSEN_COLUMN in row_dict:
                if augmentations["honeypot_at_start"]:
                    row_dict[cst.STANDARD_DPO_CHOSEN_COLUMN] = (
                        f"{response_uid} {row_dict[cst.STANDARD_DPO_CHOSEN_COLUMN]}"
                    )
                else:
                    row_dict[cst.STANDARD_DPO_CHOSEN_COLUMN] = (
                        f"{row_dict[cst.STANDARD_DPO_CHOSEN_COLUMN]} {response_uid}"
                    )
            elif not augmentations["honeypot_in_chosen"] and cst.STANDARD_DPO_REJECTED_COLUMN in row_dict:
                if augmentations["honeypot_at_start"]:
                    row_dict[cst.STANDARD_DPO_REJECTED_COLUMN] = (
                        f"{response_uid} {row_dict[cst.STANDARD_DPO_REJECTED_COLUMN]}"
                    )
                else:
                    row_dict[cst.STANDARD_DPO_REJECTED_COLUMN] = (
                        f"{row_dict[cst.STANDARD_DPO_REJECTED_COLUMN]} {response_uid}"
                    )
    except Exception as e:
        logger.error(f"Error in response honeypot: {e}")

    try:
        # 4. Swap chosen and rejected (applies to ALL rows if enabled)
        if augmentations.get("swap_chosen_rejected"):
            if cst.STANDARD_DPO_CHOSEN_COLUMN in row_dict and cst.STANDARD_DPO_REJECTED_COLUMN in row_dict:
                row_dict[cst.STANDARD_DPO_CHOSEN_COLUMN], row_dict[cst.STANDARD_DPO_REJECTED_COLUMN] = (
                    row_dict[cst.STANDARD_DPO_REJECTED_COLUMN],
                    row_dict[cst.STANDARD_DPO_CHOSEN_COLUMN],
                )
    except Exception as e:
        logger.error(f"Error in swap chosen/rejected: {e}")

    return row_dict


def _apply_instruct_augmentations(row_dict: dict, augmentations: dict, idx: int) -> dict:
    """Apply Instruct-specific augmentations to a row."""
    try:
        # 1. Rearrange input/instruction (applies to ALL rows if enabled)
        if augmentations.get("rearrange_input") and cst.STANDARD_INSTRUCT_COLUMN in row_dict:
            row_dict[cst.STANDARD_INSTRUCT_COLUMN] = _rearrange_sentences(row_dict[cst.STANDARD_INSTRUCT_COLUMN])
    except Exception as e:
        logger.error(f"Error in rearrange_input: {e}")

    try:
        # 2. Rearrange output (applies to ALL rows if enabled)
        if augmentations.get("rearrange_output") and cst.STANDARD_OUTPUT_COLUMN in row_dict:
            row_dict[cst.STANDARD_OUTPUT_COLUMN] = _rearrange_sentences(row_dict[cst.STANDARD_OUTPUT_COLUMN])
    except Exception as e:
        logger.error(f"Error in rearrange_output: {e}")

    try:
        # 3. Add input honeypot (applies to ALL rows if enabled)
        if augmentations.get("add_input_honeypot") and cst.STANDARD_INSTRUCT_COLUMN in row_dict:
            input_uid = augmentations["input_uid"]
            if augmentations.get("input_honeypot_at_start"):
                row_dict[cst.STANDARD_INSTRUCT_COLUMN] = f"{input_uid} {row_dict[cst.STANDARD_INSTRUCT_COLUMN]}"
            else:
                row_dict[cst.STANDARD_INSTRUCT_COLUMN] = _insert_uid_randomly(
                    row_dict[cst.STANDARD_INSTRUCT_COLUMN], input_uid
                )
    except Exception as e:
        logger.error(f"Error in input honeypot: {e}")

    try:
        # 4. Add output honeypot (applies to percentage of rows if enabled)
        output_honeypot_indices = augmentations.get("output_honeypot_indices", set())
        if augmentations.get("add_output_honeypot") and idx in output_honeypot_indices:
            output_uid = augmentations["output_uid"]
            if augmentations.get("output_honeypot_at_start"):
                row_dict[cst.STANDARD_OUTPUT_COLUMN] = f"{output_uid} {row_dict[cst.STANDARD_OUTPUT_COLUMN]}"
            else:
                row_dict[cst.STANDARD_OUTPUT_COLUMN] = f"{row_dict[cst.STANDARD_OUTPUT_COLUMN]} {output_uid}"
    except Exception as e:
        logger.error(f"Error in output honeypot: {e}")

    try:
        # 5. Apply word transformations to input (applies to ALL rows if enabled)
        if cst.STANDARD_INSTRUCT_COLUMN in row_dict:
            transformed_input = apply_word_honeypot_to_text(
                text=row_dict[cst.STANDARD_INSTRUCT_COLUMN],
                config=augmentations,
                is_input=True
            )
            if transformed_input != row_dict[cst.STANDARD_INSTRUCT_COLUMN]:
                row_dict[cst.STANDARD_INSTRUCT_COLUMN] = transformed_input
    except Exception as e:
        logger.error(f"Error in word transformations (input): {e}")

    try:
        # 6. Apply word transformations to output (applies conditionally if enabled)
        if cst.STANDARD_OUTPUT_COLUMN in row_dict:
            transformed_output = apply_word_honeypot_to_text(
                text=row_dict[cst.STANDARD_OUTPUT_COLUMN],
                config=augmentations,
                is_input=False
            )
            if transformed_output != row_dict[cst.STANDARD_OUTPUT_COLUMN]:
                row_dict[cst.STANDARD_OUTPUT_COLUMN] = transformed_output
    except Exception as e:
        logger.error(f"Error in word transformations (output): {e}")

    return row_dict


def _apply_grpo_augmentations(row_dict: dict, augmentations: dict, idx: int) -> dict:
    """Apply GRPO-specific augmentations to a row."""
    try:
        # 1. Add prompt honeypot (applies to ALL rows if enabled)
        if augmentations.get("add_prompt_honeypot") and cst.STANDARD_GRPO_PROMPT_COLUMN in row_dict:
            row_dict[cst.STANDARD_GRPO_PROMPT_COLUMN] = _insert_uid_randomly(
                row_dict[cst.STANDARD_GRPO_PROMPT_COLUMN], augmentations["prompt_uid"]
            )
    except Exception as e:
        logger.error(f"Error in GRPO prompt honeypot: {e}")

    try:
        # 2. Apply word transformations to GRPO prompts (treat prompts as input)
        if cst.STANDARD_GRPO_PROMPT_COLUMN in row_dict:
            transformed_prompt = apply_word_honeypot_to_text(
                text=row_dict[cst.STANDARD_GRPO_PROMPT_COLUMN],
                config=augmentations,
                is_input=True  # GRPO prompts are treated as input
            )
            if transformed_prompt != row_dict[cst.STANDARD_GRPO_PROMPT_COLUMN]:
                row_dict[cst.STANDARD_GRPO_PROMPT_COLUMN] = transformed_prompt
    except Exception as e:
        logger.error(f"Error in GRPO word transformations: {e}")

    return row_dict
