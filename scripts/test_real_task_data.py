#!/usr/bin/env python3

import asyncio
import json
import os
import asyncpg
from validator.core import constants as cst

async def test_real_task_data():
    """Test the actual reward functions with real task data"""
    print("=== Testing Real Task Data ===")
    
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
            # Get the actual task data
            task_query = """
                SELECT test_data, training_data
                FROM tasks 
                WHERE task_id = '438b7328-859f-4d87-b99a-6d85acdbcc15'
            """
            task_row = await conn.fetchrow(task_query)
            
            if not task_row:
                print("❌ Task not found")
                return
            
            # Load the dataset (use test_data first, fallback to training_data)
            dataset_path = task_row['test_data'] or task_row['training_data']
            print(f"📊 Loading dataset from: {dataset_path}")
            
            # Get reward functions
            reward_query = """
                SELECT gtf.reward_id, rf.reward_func
                FROM grpo_task_functions gtf
                JOIN reward_functions rf ON gtf.reward_id = rf.reward_id
                WHERE gtf.task_id = '438b7328-859f-4d87-b99a-6d85acdbcc15'
                ORDER BY gtf.created_at
            """
            reward_rows = await conn.fetch(reward_query)
            
            print(f"🎯 Found {len(reward_rows)} reward functions")
            
            # Load reward functions
            reward_functions = []
            for row in reward_rows:
                namespace = {}
                exec(row['reward_func'], namespace)
                # Find the reward function
                func_name = None
                for name, obj in namespace.items():
                    if callable(obj) and name.endswith('_reward_function'):
                        func_name = name
                        break
                if func_name:
                    reward_functions.append((row['reward_id'], func_name, namespace[func_name]))
                    print(f"✅ Loaded {func_name}: {row['reward_id'][:8]}...")
            
            # Load some sample data from the dataset
            from datasets import load_dataset
            try:
                eval_dataset = load_dataset("json", data_files=dataset_path, split="train")
                print(f"📋 Dataset has {len(eval_dataset)} samples")
                
                # Test with a few samples of each task type
                sample_data = eval_dataset.to_list()[:50]  # First 50 samples
                
                # Group by task type
                task_types = {}
                for item in sample_data:
                    extra_data_str = item.get('extra_data', '{}')
                    try:
                        extra_data = json.loads(extra_data_str)
                        task_type = extra_data.get('task_type', 'Unknown')
                        if task_type not in task_types:
                            task_types[task_type] = []
                        task_types[task_type].append((item, extra_data))
                    except json.JSONDecodeError:
                        print(f"⚠️ Failed to parse extra_data: {extra_data_str[:100]}...")
                
                print(f"\n📈 Task type distribution in first 50 samples:")
                for task_type, samples in task_types.items():
                    print(f"  {task_type}: {len(samples)} samples")
                
                # Test each reward function with appropriate data
                for reward_id, func_name, reward_func in reward_functions:
                    print(f"\n{'='*60}")
                    print(f"Testing {func_name} ({reward_id[:8]}...)")
                    
                    # Test with a few samples
                    tested = 0
                    for task_type, samples in task_types.items():
                        if tested >= 3:  # Test max 3 per function
                            break
                            
                        for item, extra_data in samples[:1]:  # Test 1 sample per task type
                            print(f"\n--- Testing with {task_type} data ---")
                            print(f"Extra data keys: {list(extra_data.keys())}")
                            
                            # Use a dummy completion for now
                            dummy_completion = "This is a test completion with code:\n```python\nprint('test')\n```\n<INPUT>\n1\n2\n</INPUT>\n<ASSIGNMENT>\nx1=True\nx2=False\n</ASSIGNMENT>"
                            
                            try:
                                result = reward_func([dummy_completion], extra_data=[extra_data])
                                print(f"🎯 Result: {result[0]}")
                                tested += 1
                            except Exception as e:
                                print(f"❌ Error: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            if tested >= 1:  # Only test 1 per function for now
                                break
                        
                        if tested >= 1:
                            break
                        
            except Exception as e:
                print(f"❌ Error loading dataset: {e}")
                import traceback
                traceback.print_exc()
                
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(test_real_task_data())