#!/usr/bin/env python3

import asyncio
import os
import asyncpg
from datasets import load_dataset
from validator.core import constants as cst

async def debug_dataset_columns():
    """Debug dataset columns and extra_column configuration"""
    print("=== Debugging Dataset Columns ===")
    
    # Load env
    try:
        with open('.vali.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    except FileNotFoundError:
        print("⚠️ .vali.env file not found")
    
    connection_string = os.getenv("DATABASE_URL")
    pool = await asyncpg.create_pool(connection_string)
    
    try:
        async with pool.acquire() as conn:
            # Get task configuration
            task_query = """
                SELECT t.test_data, t.training_data, gt.extra_column
                FROM tasks t
                JOIN grpo_tasks gt ON t.task_id = gt.task_id
                WHERE t.task_id = '438b7328-859f-4d87-b99a-6d85acdbcc15'
            """
            task_row = await conn.fetchrow(task_query)
            
            if not task_row:
                print("❌ Task not found")
                return
                
            dataset_path = task_row['test_data'] or task_row['training_data']
            configured_extra_column = task_row['extra_column']
            
            print(f"📊 Dataset path: {dataset_path}")
            print(f"📋 Configured extra_column: {configured_extra_column}")
            print(f"📋 Expected column (STANDARD_GRPO_EXTRA_COLUMN): {cst.STANDARD_GRPO_EXTRA_COLUMN}")
            
            # Load dataset and check columns
            try:
                eval_dataset = load_dataset("json", data_files=dataset_path, split="train")
                print(f"📋 Dataset columns: {eval_dataset.column_names}")
                
                # Check the evaluation logic
                has_extra_column = configured_extra_column and cst.STANDARD_GRPO_EXTRA_COLUMN in eval_dataset.column_names
                print(f"\n🔍 Evaluation logic check:")
                print(f"  configured_extra_column: {configured_extra_column}")
                print(f"  STANDARD_GRPO_EXTRA_COLUMN in dataset: {cst.STANDARD_GRPO_EXTRA_COLUMN in eval_dataset.column_names}")
                print(f"  has_extra_column = {has_extra_column}")
                
                if has_extra_column:
                    print(f"✅ Should pass extra_data to reward functions")
                    # Show a sample
                    sample = eval_dataset[0]
                    extra_data_sample = sample.get(cst.STANDARD_GRPO_EXTRA_COLUMN, "NOT_FOUND")
                    print(f"📋 Sample extra_data: {str(extra_data_sample)[:200]}...")
                else:
                    print(f"❌ Will NOT pass extra_data to reward functions")
                    print(f"❌ This is why extra_data=None in the functions!")
                    
            except Exception as e:
                print(f"❌ Error loading dataset: {e}")
                
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(debug_dataset_columns())