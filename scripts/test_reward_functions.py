#!/usr/bin/env python3

import argparse
import json
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.models.utility_models import GrpoDatasetType
from validator.core.models import EvaluationArgs
from validator.utils.affine_reward_functions import sat_reward_function, abd_reward_function, ded_reward_function
from validator.utils.reward_functions import validate_reward_function
from core.database import Database
from validator.db.models import Task, GrpoTask, RewardFunction

def get_task_data(task_id: str):
    """Fetch task data and reward functions from database"""
    print(f"🔍 Fetching data for task_id: {task_id}")
    
    db = Database()
    
    with db.session() as session:
        # Get the task and grpo_task data
        task = session.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            print(f"❌ Task {task_id} not found")
            return None, None, None
        
        grpo_task = session.query(GrpoTask).filter(GrpoTask.task_id == task_id).first()
        if not grpo_task:
            print(f"❌ GRPO task {task_id} not found")
            return None, None, None
            
        # Get reward functions
        reward_functions = session.query(RewardFunction).filter(
            RewardFunction.reward_function_id.in_(grpo_task.reward_function_ids)
        ).all()
        
        print(f"✅ Found task with {len(reward_functions)} reward functions")
        
        # Create dataset type
        dataset_type = GrpoDatasetType(
            task_name=task.task_name,
            field_prompt=grpo_task.field_prompt,
            extra_column=grpo_task.extra_column,
            reward_functions=reward_functions
        )
        
        return task, grpo_task, dataset_type

def test_reward_functions_with_task(task_id: str):
    """Test reward functions with actual task data"""
    task, grpo_task, dataset_type = get_task_data(task_id)
    
    if not task:
        return
    
    print(f"\n=== Task Info ===")
    print(f"Task ID: {task.task_id}")
    print(f"Task Name: {task.task_name}")
    print(f"Dataset: {task.dataset}")
    print(f"Field Prompt: {grpo_task.field_prompt}")
    print(f"Extra Column: {grpo_task.extra_column}")
    
    # Load the dataset to get some sample data
    from datasets import load_dataset
    
    try:
        eval_dataset = load_dataset("json", data_files=task.dataset, split="train")
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
            
            # Validate the function
            is_valid, error_msg, reward_func_callable = validate_reward_function(rf.reward_func, sample_data)
            if not is_valid:
                print(f"❌ Invalid reward function: {error_msg}")
                continue
                
            print(f"✅ Reward function is valid")
            
            # Test with sample extra_data
            if grpo_task.extra_column and 'extra_data' in eval_dataset.column_names:
                extra_data = eval_dataset['extra_data'][:2]
                print(f"Using extra_data from dataset: {extra_data}")
                
                try:
                    result = reward_func_callable(test_completions, extra_data=extra_data)
                    print(f"✅ Result: {result}")
                except Exception as e:
                    print(f"❌ Error calling reward function: {e}")
            else:
                print(f"No extra_data column found, testing without extra_data")
                try:
                    result = reward_func_callable(test_completions)
                    print(f"✅ Result: {result}")
                except Exception as e:
                    print(f"❌ Error calling reward function: {e}")
                    
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test reward functions with actual task data")
    parser.add_argument("task_id", help="Task ID to test")
    args = parser.parse_args()
    
    test_reward_functions_with_task(args.task_id)