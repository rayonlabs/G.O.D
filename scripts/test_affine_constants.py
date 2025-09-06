#!/usr/bin/env python3

import asyncio
import os
import asyncpg
from core import constants as cst

async def test_affine_constants():
    """Test the affine reward functions from constants"""
    print("=== Testing Affine Reward Functions from Constants ===")
    
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
            print(f"📋 Testing {len(cst.AFFINE_REWARD_FN_IDS)} reward functions from constants")
            
            # Test data for each type
            test_completions = [
                "I think x1=True, x2=False, x3=True would work for this SAT problem.",
                "I need to find input:\n<INPUT>\n5\n10\n</INPUT>", 
                "Here's my solution:\n```python\na = int(input())\nb = int(input())\nprint(a + b)\n```"
            ]
            
            test_data = [
                # SAT test data
                {
                    "task_type": "SAT",
                    "cnf_formula": "p cnf 3 3\n1 -2 3 0\n-1 2 0\n2 -3 0",
                    "expected_satisfiable": True
                },
                # ABD test data  
                {
                    "task_type": "ABD",
                    "program": "a = int(input())\nb = int(input())\nprint(a + b)",
                    "expected_output": "15"
                },
                # DED test data
                {
                    "task_type": "DED",
                    "solution": "a = int(input())\nb = int(input())\nprint(a + b)",
                    "premises": ["5\n10"]
                }
            ]
            
            for i, reward_id in enumerate(cst.AFFINE_REWARD_FN_IDS):
                print(f"\n--- Testing Reward Function {i}: {reward_id} ---")
                
                # Get function from database
                query = """
                    SELECT reward_func, func_hash 
                    FROM reward_functions 
                    WHERE reward_id = $1
                """
                row = await conn.fetchrow(query, reward_id)
                
                if not row:
                    print(f"❌ Function {reward_id} not found in database")
                    continue
                    
                print(f"✅ Found function, hash: {row['func_hash'][:16]}...")
                
                # Execute the function code
                namespace = {}
                exec(row['reward_func'], namespace)
                
                # Get the function (should be the first function defined)
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
                
                # Test with appropriate data
                try:
                    result = reward_func([test_completions[i]], extra_data=[test_data[i]])
                    print(f"🎯 Result: {result}")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(test_affine_constants())