"""
Utility functions for DPO dataset processing and formatting
"""

import json
from typing import Dict, Any

import core.constants as cst
from core.models.utility_models import DpoDatasetType


def _dpo_format_prompt(item: dict, format_str: str) -> str:
    """
    Format a prompt for DPO task.

    Args:
        item: Dictionary with dataset fields
        format_str: Format string template

    Returns:
        Formatted prompt string
    """
    result = format_str

    # Replace placeholder with actual value if it exists
    if "{prompt}" in format_str and "prompt" in item:
        result = result.replace("{prompt}", str(item["prompt"]))

    if "{system}" in format_str and "system" in item:
        result = result.replace("{system}", str(item["system"]))

    return result


def _dpo_format_chosen(item: dict, format_str: str) -> str:
    """
    Format a chosen response for DPO task.

    Args:
        item: Dictionary with dataset fields
        format_str: Format string template

    Returns:
        Formatted chosen response string
    """
    result = format_str

    # Replace placeholders with actual values if they exist
    if "{chosen}" in format_str and "chosen" in item:
        result = result.replace("{chosen}", str(item["chosen"]))

    if "{prompt}" in format_str and "prompt" in item:
        result = result.replace("{prompt}", str(item["prompt"]))

    if "{system}" in format_str and "system" in item:
        result = result.replace("{system}", str(item["system"]))

    return result


def _dpo_format_rejected(item: dict, format_str: str) -> str:
    """
    Format a rejected response for DPO task.

    Args:
        item: Dictionary with dataset fields
        format_str: Format string template

    Returns:
        Formatted rejected response string
    """
    result = format_str

    # Replace placeholders with actual values if they exist
    if "{rejected}" in format_str and "rejected" in item:
        result = result.replace("{rejected}", str(item["rejected"]))

    if "{prompt}" in format_str and "prompt" in item:
        result = result.replace("{prompt}", str(item["prompt"]))

    if "{system}" in format_str and "system" in item:
        result = result.replace("{system}", str(item["system"]))

    return result


def adapt_columns_for_dpo_dataset(dataset_path: str, dataset_type: DpoDatasetType, apply_formatting: bool = True):
    """
    Transform a DPO JSON dataset file to match axolotl's expected column names.

    Args:
        dataset_path: Path to the JSON dataset file
        dataset_type: DpoDatasetType object with field mappings
        apply_formatting: If True, apply formatting templates to the content
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Use a direct approach instead of pandas to avoid any potential issues
    print(f"Original dataset keys: {list(data[0].keys()) if data else []}")

    # Create a new dataset with both chatml.intel field names AND the standard DPO field names
    transformed_data = []
    for item in data:
        new_item = {}

        # Get the prompt value from the input dataset
        prompt_value = item.get(dataset_type.field_prompt, "")

        # Add BOTH field names - "question" for chatml.intel and "prompt" for DPO processing
        new_item["question"] = prompt_value
        new_item["prompt"] = prompt_value  # Axolotl's DPO code explicitly looks for this

        # Add the chosen and rejected fields
        if dataset_type.field_chosen and dataset_type.field_chosen in item:
            new_item["chosen"] = item[dataset_type.field_chosen]

        if dataset_type.field_rejected and dataset_type.field_rejected in item:
            new_item["rejected"] = item[dataset_type.field_rejected]

        # Add system field if it exists
        if dataset_type.field_system and dataset_type.field_system in item:
            new_item["system"] = item[dataset_type.field_system]

        # Apply formatting if requested
        if apply_formatting:
            # Format prompt if a formatting template is provided
            if dataset_type.prompt_format and dataset_type.prompt_format != "{prompt}":
                # Create a formatting dictionary with the right keys
                format_dict = {
                    "prompt": new_item["prompt"],
                    "chosen": new_item.get("chosen", ""),
                    "rejected": new_item.get("rejected", "")
                }
                if "system" in new_item:
                    format_dict["system"] = new_item["system"]

                formatted_value = _dpo_format_prompt(format_dict, dataset_type.prompt_format)
                # Update both field names with the formatted value
                new_item["question"] = formatted_value
                new_item["prompt"] = formatted_value

            # Format chosen response if a formatting template is provided
            if dataset_type.chosen_format and dataset_type.chosen_format != "{chosen}":
                format_dict = {
                    "prompt": new_item["prompt"],
                    "chosen": new_item.get("chosen", ""),
                    "rejected": new_item.get("rejected", "")
                }
                if "system" in new_item:
                    format_dict["system"] = new_item["system"]

                new_item["chosen"] = _dpo_format_chosen(format_dict, dataset_type.chosen_format)

            # Format rejected response if a formatting template is provided
            if dataset_type.rejected_format and dataset_type.rejected_format != "{rejected}":
                format_dict = {
                    "prompt": new_item["prompt"],
                    "chosen": new_item.get("chosen", ""),
                    "rejected": new_item.get("rejected", "")
                }
                if "system" in new_item:
                    format_dict["system"] = new_item["system"]

                new_item["rejected"] = _dpo_format_rejected(format_dict, dataset_type.rejected_format)

        transformed_data.append(new_item)

    # Write the transformed data back to the file
    with open(dataset_path, 'w') as f:
        json.dump(transformed_data, f, indent=2)

    print(f"Transformed dataset to include both chatml.intel and DPO field names:")
    print(f"Final fields: {list(transformed_data[0].keys()) if transformed_data else []}")
    print(f"Mapped {dataset_type.field_prompt} to both 'question' and 'prompt' to satisfy all requirements")
    print(f"Saved dataset to {dataset_path}")


