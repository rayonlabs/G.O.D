#!/usr/bin/env python3

import asyncio
import os
import asyncpg
from validator.core import constants as cst

async def test_all_affine_functions():
    """Test all three affine reward functions from constants with known good data"""
    print("=== Testing All Affine Reward Functions from Constants ===")
    
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
    
    # Test data for each type with known correct results - using real data formats
    test_cases = [
        # SAT test cases - using cls and sol format
        {
            "name": "SAT",
            "reward_id": cst.AFFINE_REWARD_FN_IDS[0],
            "test_data": {
                "task_type": "SAT",
                "cls": [[1, -2, 3], [-1, 2], [2, -3]],  # Simple 3-SAT problem
                "sol": {"1": True, "2": True, "3": True}  # Known solution
            },
            "completions": [
                ("Good assignment", "Looking at this 3-SAT problem:\n<ASSIGNMENT>\nx1 = True\nx2 = True\nx3 = True\n</ASSIGNMENT>"),
                ("Wrong assignment", "Here's my solution:\n<ASSIGNMENT>\nx1 = False\nx2 = False\nx3 = False\n</ASSIGNMENT>"),
                ("No assignment", "This looks difficult to solve.")
            ]
        },
        # ABD test cases - same format (program, expected_output)
        {
            "name": "ABD", 
            "reward_id": cst.AFFINE_REWARD_FN_IDS[1],
            "test_data": {
                "task_type": "ABD",
                "program": "a = int(input())\nb = int(input())\nprint(a + b)",
                "expected_output": "15"
            },
            "completions": [
                ("Perfect input", "I need to find input that makes this program output 15. The program adds two numbers.\n<INPUT>\n5\n10\n</INPUT>"),
                ("Alternative input", "This adds two inputs. For output 15:\n<INPUT>\n7\n8\n</INPUT>"),
                ("Wrong input", "Let me try:\n<INPUT>\n1\n2\n</INPUT>"),
                ("No INPUT tags", "The answer is 5 and 10")
            ]
        },
        # DED test cases - using problem field like real data
        {
            "name": "DED",
            "reward_id": cst.AFFINE_REWARD_FN_IDS[2], 
            "test_data": {
                "task_type": "DED",
                "problem": "Write a program that doubles the input number.",
                "solution": "```python\nn = int(input())\nprint(n * 2)\n```",
                "premises": ["5"]
            },
            "completions": [
                ("Perfect solution", "I need to write code that doubles the input:\n```python\nn = int(input())\nprint(n * 2)\n```"),
                ("Alternative solution", "Here's my solution:\n```python\nn = int(input())\nprint(n + n)\n```"), 
                ("Wrong solution", "My code:\n```python\nn = int(input())\nprint(n + 1)\n```"),
                ("No code", "Just multiply by 2")
            ]
        }
    ]
    
    try:
        async with pool.acquire() as conn:
            for test_case in test_cases:
                print(f"\n{'='*60}")
                print(f"Testing {test_case['name']} Function")
                print(f"Reward ID: {test_case['reward_id']}")
                print(f"{'='*60}")
                
                # Get function from database
                query = """
                    SELECT reward_func, func_hash 
                    FROM reward_functions 
                    WHERE reward_id = $1
                """
                row = await conn.fetchrow(query, test_case['reward_id'])
                
                if not row:
                    print(f"❌ Function {test_case['reward_id']} not found in database")
                    continue
                    
                print(f"✅ Found function, hash: {row['func_hash'][:16]}...")
                
                # Execute the function code
                namespace = {}
                exec(row['reward_func'], namespace)
                
                # Get the reward function
                func_name = None
                for name, obj in namespace.items():
                    if callable(obj) and name.endswith('_reward_function'):
                        func_name = name
                        break
                        
                if not func_name:
                    print(f"❌ No reward function found in namespace")
                    continue
                    
                reward_func = namespace[func_name]
                print(f"✅ Found function: {func_name}")
                
                # Test each completion
                for completion_name, completion_text in test_case['completions']:
                    print(f"\n--- {completion_name} ---")
                    print(f"Completion: {completion_text[:80]}...")
                    
                    try:
                        result = reward_func([completion_text], extra_data=[test_case['test_data']])
                        print(f"🎯 Result: {result[0]}")
                        
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        import traceback
                        traceback.print_exc()
                        
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(test_all_affine_functions())