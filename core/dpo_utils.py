"""
Utility functions for DPO dataset processing and formatting
"""

import json
import pandas as pd
from typing import Dict, Any

import core.constants as cst


def _dpo_format_prompt(row, format_str):
    """
    Format a prompt for DPO task.
    
    Args:
        row: DataFrame row with dataset fields
        format_str: Format string template
        
    Returns:
        Formatted prompt string
    """
    result = format_str
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _dpo_format_chosen(row, format_str):
    """
    Format a chosen response for DPO task.
    
    Args:
        row: DataFrame row with dataset fields
        format_str: Format string template
        
    Returns:
        Formatted chosen response string
    """
    result = format_str
    if "{chosen}" in format_str and cst.DPO_DEFAULT_FIELD_CHOSEN in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_CHOSEN]):
        result = result.replace("{chosen}", str(row[cst.DPO_DEFAULT_FIELD_CHOSEN]))
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _dpo_format_rejected(row, format_str):
    """
    Format a rejected response for DPO task.
    
    Args:
        row: DataFrame row with dataset fields
        format_str: Format string template
        
    Returns:
        Formatted rejected response string
    """
    result = format_str
    if "{rejected}" in format_str and cst.DPO_DEFAULT_FIELD_REJECTED in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_REJECTED]):
        result = result.replace("{rejected}", str(row[cst.DPO_DEFAULT_FIELD_REJECTED]))
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def adapt_columns_for_dpo_dataset(dataset_path: str, dataset_type, apply_formatting: bool = True):
    """
    Transform a DPO JSON dataset file to match axolotl's expected column names.

    Args:
        dataset_path: Path to the JSON dataset file
        dataset_type: DPODatasetType object with field mappings
        apply_formatting: If True, apply formatting templates to the content
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Build column mapping
    column_mapping = {}
    
    # Add required fields to column mapping
    if dataset_type.field_prompt:
        column_mapping[dataset_type.field_prompt] = cst.DPO_DEFAULT_FIELD_PROMPT
    
    if dataset_type.field_system:
        column_mapping[dataset_type.field_system] = cst.DPO_DEFAULT_FIELD_SYSTEM
    
    if dataset_type.field_chosen:
        column_mapping[dataset_type.field_chosen] = cst.DPO_DEFAULT_FIELD_CHOSEN
    
    if dataset_type.field_rejected:
        column_mapping[dataset_type.field_rejected] = cst.DPO_DEFAULT_FIELD_REJECTED
    
    # Rename columns
    df = df.rename(columns=column_mapping)

    if apply_formatting:
        # Apply formatting for the prompt
        if dataset_type.prompt_format and dataset_type.prompt_format != "{prompt}":
            df[cst.DPO_DEFAULT_FIELD_PROMPT] = df.apply(
                lambda row: _dpo_format_prompt(row, dataset_type.prompt_format), axis=1
            )
        
        # Apply formatting for the chosen response
        if dataset_type.chosen_format and dataset_type.chosen_format != "{chosen}":
            df[cst.DPO_DEFAULT_FIELD_CHOSEN] = df.apply(
                lambda row: _dpo_format_chosen(row, dataset_type.chosen_format), axis=1
            )
        
        # Apply formatting for the rejected response
        if dataset_type.rejected_format and dataset_type.rejected_format != "{rejected}":
            df[cst.DPO_DEFAULT_FIELD_REJECTED] = df.apply(
                lambda row: _dpo_format_rejected(row, dataset_type.rejected_format), axis=1
            )
    
    output_data = df.to_dict(orient='records')
    with open(dataset_path, 'w') as f:
        json.dump(output_data, f, indent=2)