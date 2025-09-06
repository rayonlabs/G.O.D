#!/usr/bin/env python3

import asyncio
import json
import os
import asyncpg
from validator.utils.reward_functions import validate_reward_function

async def debug_trl_batching():
    """Debug how TRL might be calling our reward functions"""
    print("=== Debugging TRL-style Batching ===")
    
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
            # Get DED function (the one that should work)
            reward_query = """
                SELECT reward_func
                FROM reward_functions 
                WHERE reward_id = '2c94fc44-686f-4667-a5e0-17bcbbcff85c'
            """
            row = await conn.fetchrow(reward_query)
            
            # Process the function
            is_valid, error_msg, processed_func = validate_reward_function(row['reward_func'], [])
            
            print("✅ DED function loaded and processed")
            
            # Test different ways TRL might call the function
            test_cases = [
                {
                    "name": "Single completion, single extra_data",
                    "completions": ["```python\nprint('test')\n```"],
                    "extra_data": [{"task_type": "DED", "solution": "```python\nprint('test')\n```", "premises": [""]}]
                },
                {
                    "name": "Single completion, single extra_data as dict",
                    "completions": ["```python\nprint('test')\n```"],
                    "extra_data": {"task_type": "DED", "solution": "```python\nprint('test')\n```", "premises": [""]}
                },
                {
                    "name": "Batch of completions",
                    "completions": [
                        "```python\nprint('test')\n```",
                        "```python\nprint('hello')\n```"
                    ],
                    "extra_data": [
                        {"task_type": "DED", "solution": "```python\nprint('test')\n```", "premises": [""]},
                        {"task_type": "DED", "solution": "```python\nprint('hello')\n```", "premises": [""]}
                    ]
                },
                {
                    "name": "Empty strings test",
                    "completions": [""],
                    "extra_data": [{"task_type": "DED", "solution": "```python\nprint('test')\n```", "premises": [""]}]
                },
                {
                    "name": "String vs JSON extra_data",
                    "completions": ["```python\nprint('test')\n```"],
                    "extra_data": ['{"task_type": "DED", "solution": "```python\\nprint(\'test\')\\n```", "premises": [""]}']
                },
                {
                    "name": "Large batch (like TRL might use)",
                    "completions": ["```python\nprint('test')\n```"] * 16,
                    "extra_data": [{"task_type": "DED", "solution": "```python\nprint('test')\n```", "premises": [""]}] * 16
                }
            ]
            
            for test_case in test_cases:
                print(f"\n{'='*60}")
                print(f"Testing: {test_case['name']}")
                print(f"Completions count: {len(test_case['completions'])}")
                print(f"Extra data count: {len(test_case['extra_data'])}")
                
                try:
                    result = processed_func(test_case['completions'], extra_data=test_case['extra_data'])
                    print(f"🎯 Result: {result}")
                    
                    # Check if any non-zero results
                    non_zero = [r for r in result if r > 0]
                    if non_zero:
                        print(f"✅ Found {len(non_zero)} non-zero scores: {non_zero}")
                    else:
                        print(f"❌ All scores are zero")
                        
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(debug_trl_batching())