#!/usr/bin/env python3

import asyncio
import json
import os
import asyncpg
from validator.utils.reward_functions import validate_reward_function
from validator.core import constants as cst

async def debug_evaluation_environment():
    """Debug exactly how the evaluation environment calls reward functions"""
    print("=== Debugging Evaluation Environment ===")
    
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
            # Get the reward functions exactly as the evaluation does
            reward_query = """
                SELECT gtf.reward_id, rf.reward_func, gtf.reward_weight
                FROM grpo_task_functions gtf
                JOIN reward_functions rf ON gtf.reward_id = rf.reward_id
                WHERE gtf.task_id = '438b7328-859f-4d87-b99a-6d85acdbcc15'
                ORDER BY gtf.created_at
            """
            reward_rows = await conn.fetch(reward_query)
            
            print(f"🎯 Found {len(reward_rows)} reward functions from task")
            
            # Process each function exactly like the evaluation does
            for i, row in enumerate(reward_rows):
                print(f"\n{'='*50}")
                print(f"Processing reward function {i}")
                print(f"ID: {row['reward_id']}")
                print(f"Weight: {row['reward_weight']}")
                
                # Validate and process the function (same as evaluation)
                is_valid, error_msg, processed_func = validate_reward_function(
                    row['reward_func'], 
                    []  # No sample data like in evaluation
                )
                
                if not is_valid:
                    print(f"❌ Function invalid: {error_msg}")
                    continue
                    
                print(f"✅ Function processed successfully")
                
                # Test with the same data format as our individual test
                test_completion = """```python
from collections import defaultdict
import sys
def main():
    print("test")
if __name__ == "__main__":
    main()
```"""
                
                test_extra_data = {
                    "task_type": "DED",
                    "problem": "Write a test program",
                    "solution": "```python\nprint('test')\n```",
                    "premises": [""]
                }
                
                print(f"\n🧪 Testing processed function:")
                print(f"Completion preview: {test_completion[:100]}...")
                print(f"Extra data: {test_extra_data}")
                
                try:
                    # Call exactly as evaluation does
                    result = processed_func([test_completion], extra_data=[test_extra_data])
                    print(f"🎯 Processed function result: {result}")
                    
                    # Compare with direct call
                    print(f"\n🧪 Testing direct function call:")
                    namespace = {}
                    exec(row['reward_func'], namespace)
                    
                    # Find the function
                    func_name = None
                    for name, obj in namespace.items():
                        if callable(obj) and name.endswith('_reward_function'):
                            func_name = name
                            break
                    
                    if func_name:
                        direct_func = namespace[func_name]
                        direct_result = direct_func([test_completion], extra_data=[test_extra_data])
                        print(f"🎯 Direct function result: {direct_result}")
                        
                        # Compare results
                        if result == direct_result:
                            print("✅ Results match!")
                        else:
                            print(f"❌ Results differ! Processed: {result}, Direct: {direct_result}")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(debug_evaluation_environment())