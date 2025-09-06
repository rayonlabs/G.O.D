#!/usr/bin/env python3

import argparse
import json
import os
import asyncio

from core.models.utility_models import GrpoDatasetType
from validator.core.models import EvaluationArgs
from validator.utils.affine_reward_functions import sat_reward_function, abd_reward_function, ded_reward_function
from validator.utils.reward_functions import validate_reward_function
from validator.db.database import PSQLDB
from validator.db.models import Task, GrpoTask, RewardFunction

async def get_task_data(task_id: str):
    """Fetch task data and reward functions from database"""
    print(f"🔍 Fetching data for task_id: {task_id}")
    
    import asyncpg
    from uuid import UUID
    
    connection_string = os.getenv("DATABASE_URL")
    if not connection_string:
        raise ValueError("DATABASE_URL not found")
    
    pool = await asyncpg.create_pool(connection_string)
    
    try:
        async with pool.acquire() as conn:
            # Get task info including GRPO specific data
            task_query = """
                SELECT t.task_id, t.model_id, t.ds as dataset_url, t.test_data, t.training_data, 
                       t.task_type, gt.field_prompt, gt.file_format, gt.synthetic_data, gt.extra_column
                FROM tasks t
                JOIN grpo_tasks gt ON t.task_id = gt.task_id
                WHERE t.task_id = $1
            """
            task_data = await conn.fetchrow(task_query, UUID(task_id))
            
            if not task_data:
                print(f"❌ Task {task_id} not found")
                return None, None
            
            # Get reward functions
            reward_query = """
                SELECT rf.reward_id, rf.reward_func, gtf.reward_weight, rf.func_hash, rf.is_generic
                FROM grpo_task_functions gtf
                JOIN reward_functions rf ON gtf.reward_id = rf.reward_id
                WHERE gtf.task_id = $1
                ORDER BY gtf.reward_weight DESC
            """
            reward_functions_data = await conn.fetchall(reward_query, UUID(task_id))
            
            print(f"✅ Found task with {len(reward_functions_data)} reward functions")
            
            # Convert to RewardFunction objects
            reward_functions = []
            for rf_data in reward_functions_data:
                reward_functions.append(RewardFunction(
                    reward_function_id=str(rf_data['reward_id']),
                    name=f"reward_func_{len(reward_functions)}",
                    reward_func=rf_data['reward_func'],
                    reward_weight=rf_data['reward_weight'],
                    func_hash=rf_data['func_hash'],
                    is_generic=rf_data['is_generic']
                ))
            
            # Create dataset type
            dataset_type = GrpoDatasetType(
                task_name=f"task_{task_id[:8]}",
                field_prompt=task_data['field_prompt'],
                extra_column=task_data['extra_column'],
                reward_functions=reward_functions
            )
            
            return dict(task_data), dataset_type
    finally:
        await pool.close()

async def test_reward_functions_with_task(task_id: str):
    """Test reward functions with actual task data"""
    result = await get_task_data(task_id)
    
    if not result:
        return
    
    task, dataset_type = result
    
    print(f"\n=== Task Info ===")
    print(f"Task ID: {task['task_id']}")
    print(f"Model ID: {task['model_id']}")
    print(f"Dataset URL: {task['dataset_url']}")
    print(f"Field Prompt: {task['field_prompt']}")
    print(f"Extra Column: {task['extra_column']}")
    print(f"File Format: {task['file_format']}")
    
    # Load the dataset to get some sample data
    from datasets import load_dataset
    
    try:
        # Use the correct dataset path from the task
        dataset_path = task['test_data'] or task['training_data'] or task['dataset_url']
        print(f"Loading dataset from: {dataset_path}")
        
        eval_dataset = load_dataset("json", data_files=dataset_path, split="train")
        print(f"Dataset loaded with {len(eval_dataset)} samples")
        print(f"Dataset columns: {eval_dataset.column_names}")
        
        # Get first few samples
        sample_data = eval_dataset.to_list()[:3]
        print(f"\nSample data preview:")
        for i, sample in enumerate(sample_data):
            print(f"Sample {i}: {sample}")
        
        # Test completions (dummy for now)
        test_completions = [
            "I think x1=True, x2=False, x3=True would work for this SAT problem.",
            "Let me try x1=False, x2=True, x3=False for the SAT instance.",
        ]
        
        # Test each reward function
        print(f"\n=== Testing Reward Functions ===")
        for i, rf in enumerate(dataset_type.reward_functions):
            print(f"\n--- Reward Function {i}: {rf.name} ---")
            print(f"Weight: {rf.reward_weight}")
            print(f"Function preview: {rf.reward_func[:200]}...")
            
            # Validate the function
            is_valid, error_msg, reward_func_callable = validate_reward_function(rf.reward_func, sample_data)
            if not is_valid:
                print(f"❌ Invalid reward function: {error_msg}")
                continue
                
            print(f"✅ Reward function is valid")
            
            # Test with sample extra_data
            if task['extra_column'] and task['extra_column'] in eval_dataset.column_names:
                extra_data = eval_dataset[task['extra_column']][:2]
                print(f"Using extra_data from dataset column '{task['extra_column']}': {extra_data}")
                
                try:
                    result = reward_func_callable(test_completions, extra_data=extra_data)
                    print(f"✅ Result: {result}")
                except Exception as e:
                    print(f"❌ Error calling reward function: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"No extra_data column found (looking for '{task['extra_column']}'), testing without extra_data")
                try:
                    result = reward_func_callable(test_completions)
                    print(f"✅ Result: {result}")
                except Exception as e:
                    print(f"❌ Error calling reward function: {e}")
                    import traceback
                    traceback.print_exc()
                    
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test reward functions with actual task data")
    parser.add_argument("task_id", help="Task ID to test")
    args = parser.parse_args()
    
    asyncio.run(test_reward_functions_with_task(args.task_id))