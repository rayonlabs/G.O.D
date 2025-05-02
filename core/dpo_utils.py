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
    
    print(f"Original dataset columns: {df.columns.tolist()}")
    print(f"DPO_DEFAULT_FIELD_PROMPT: {cst.DPO_DEFAULT_FIELD_PROMPT}")
    
    # Build column mapping
    column_mapping = {}
    
    # Add required fields to column mapping
    if dataset_type.field_prompt:
        column_mapping[dataset_type.field_prompt] = cst.DPO_DEFAULT_FIELD_PROMPT
        print(f"Mapping {dataset_type.field_prompt} → {cst.DPO_DEFAULT_FIELD_PROMPT}")
    
    if dataset_type.field_system:
        column_mapping[dataset_type.field_system] = cst.DPO_DEFAULT_FIELD_SYSTEM
        print(f"Mapping {dataset_type.field_system} → {cst.DPO_DEFAULT_FIELD_SYSTEM}")
    
    if dataset_type.field_chosen:
        column_mapping[dataset_type.field_chosen] = cst.DPO_DEFAULT_FIELD_CHOSEN
        print(f"Mapping {dataset_type.field_chosen} → {cst.DPO_DEFAULT_FIELD_CHOSEN}")
    
    if dataset_type.field_rejected:
        column_mapping[dataset_type.field_rejected] = cst.DPO_DEFAULT_FIELD_REJECTED
        print(f"Mapping {dataset_type.field_rejected} → {cst.DPO_DEFAULT_FIELD_REJECTED}")
    
    print(f"Column mapping: {column_mapping}")
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    print(f"Columns after mapping: {df.columns.tolist()}")

    # Let's force create 'question' field from 'prompt' if only one exists and not the other
    if 'prompt' in df.columns and 'question' not in df.columns:
        print("Creating 'question' field from 'prompt'")
        df['question'] = df['prompt'].copy()
    elif 'question' in df.columns and 'prompt' not in df.columns:
        print("Creating 'prompt' field from 'question'")
        df['prompt'] = df['question'].copy()

    print(f"Final columns: {df.columns.tolist()}")

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
    
    # Make sure we have the 'question' field explicitly for chatml.intel
    if 'question' not in df.columns:
        print("WARNING: 'question' field is missing, which is required by chatml.intel format")
        print("Trying to create 'question' field from other available fields")
        if cst.DPO_DEFAULT_FIELD_PROMPT in df.columns:
            df['question'] = df[cst.DPO_DEFAULT_FIELD_PROMPT].copy()
            print(f"Created 'question' from {cst.DPO_DEFAULT_FIELD_PROMPT}")
    
    output_data = df.to_dict(orient='records')
    print(f"First sample keys: {list(output_data[0].keys())}")
    with open(dataset_path, 'w') as f:
        json.dump(output_data, f, indent=2)