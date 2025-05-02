"""
Utility functions for DPO dataset processing and formatting
"""

import json
from typing import Dict, Any

import core.constants as cst
from core.models.utility_models import DPODatasetType


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
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in item:
        result = result.replace("{prompt}", str(item[cst.DPO_DEFAULT_FIELD_PROMPT]))
        
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in item:
        result = result.replace("{system}", str(item[cst.DPO_DEFAULT_FIELD_SYSTEM]))
        
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
    if "{chosen}" in format_str and cst.DPO_DEFAULT_FIELD_CHOSEN in item:
        result = result.replace("{chosen}", str(item[cst.DPO_DEFAULT_FIELD_CHOSEN]))
        
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in item:
        result = result.replace("{prompt}", str(item[cst.DPO_DEFAULT_FIELD_PROMPT]))
        
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in item:
        result = result.replace("{system}", str(item[cst.DPO_DEFAULT_FIELD_SYSTEM]))
        
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
    if "{rejected}" in format_str and cst.DPO_DEFAULT_FIELD_REJECTED in item:
        result = result.replace("{rejected}", str(item[cst.DPO_DEFAULT_FIELD_REJECTED]))
        
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in item:
        result = result.replace("{prompt}", str(item[cst.DPO_DEFAULT_FIELD_PROMPT]))
        
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in item:
        result = result.replace("{system}", str(item[cst.DPO_DEFAULT_FIELD_SYSTEM]))
        
    return result


def adapt_columns_for_dpo_dataset(dataset_path: str, dataset_type: DPODatasetType, apply_formatting: bool = True):
    """
    Transform a DPO JSON dataset file to match axolotl's expected column names.

    Args:
        dataset_path: Path to the JSON dataset file
        dataset_type: DPODatasetType object with field mappings
        apply_formatting: If True, apply formatting templates to the content
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Use a direct approach instead of pandas to avoid any potential issues
    print(f"Original dataset keys: {list(data[0].keys()) if data else []}")
    
    # Create a new dataset with the correct field names for Axolotl
    transformed_data = []
    for item in data:
        new_item = {}
        
        # Map fields using constants from settings
        if dataset_type.field_prompt and dataset_type.field_prompt in item:
            new_item[cst.DPO_DEFAULT_FIELD_PROMPT] = item[dataset_type.field_prompt]
            print(f"Mapped '{dataset_type.field_prompt}' to '{cst.DPO_DEFAULT_FIELD_PROMPT}'")
            
        if dataset_type.field_chosen and dataset_type.field_chosen in item:
            new_item[cst.DPO_DEFAULT_FIELD_CHOSEN] = item[dataset_type.field_chosen]
            
        if dataset_type.field_rejected and dataset_type.field_rejected in item:
            new_item[cst.DPO_DEFAULT_FIELD_REJECTED] = item[dataset_type.field_rejected]
            
        if dataset_type.field_system and dataset_type.field_system in item:
            new_item[cst.DPO_DEFAULT_FIELD_SYSTEM] = item[dataset_type.field_system]
        
        # Apply formatting if requested
        if apply_formatting:
            # Format prompt if a formatting template is provided
            if dataset_type.prompt_format and dataset_type.prompt_format != "{prompt}":
                new_item[cst.DPO_DEFAULT_FIELD_PROMPT] = _dpo_format_prompt(new_item, dataset_type.prompt_format)
                
            # Format chosen response if a formatting template is provided
            if dataset_type.chosen_format and dataset_type.chosen_format != "{chosen}":
                new_item[cst.DPO_DEFAULT_FIELD_CHOSEN] = _dpo_format_chosen(new_item, dataset_type.chosen_format)
                
            # Format rejected response if a formatting template is provided
            if dataset_type.rejected_format and dataset_type.rejected_format != "{rejected}":
                new_item[cst.DPO_DEFAULT_FIELD_REJECTED] = _dpo_format_rejected(new_item, dataset_type.rejected_format)
        
        transformed_data.append(new_item)
    
    # Write the transformed data back to the file
    with open(dataset_path, 'w') as f:
        json.dump(transformed_data, f, indent=2)
    
    print(f"Transformed dataset keys: {list(transformed_data[0].keys()) if transformed_data else []}")
    print(f"Dataset has been adapted for {cst.DPO_DEFAULT_DATASET_TYPE} format with following mappings:")
    print(f"  - {dataset_type.field_prompt} → {cst.DPO_DEFAULT_FIELD_PROMPT}")
    print(f"  - {dataset_type.field_chosen} → {cst.DPO_DEFAULT_FIELD_CHOSEN}")
    print(f"  - {dataset_type.field_rejected} → {cst.DPO_DEFAULT_FIELD_REJECTED}")