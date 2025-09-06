#!/usr/bin/env python3

import asyncio
import os
import asyncpg

async def create_file_logging_function():
    """Create a reward function that logs to file instead of stdout"""
    print("=== Creating File-Logging Reward Function ===")
    
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
    
    # Create a simple test function that bypasses print completely
    file_logging_function = '''
def test_reward_function(completions, extra_data=None, **kwargs):
    """Test function that logs to file instead of print"""
    import os
    import json
    
    # Log to file instead of stdout
    with open('/tmp/reward_debug.log', 'a') as f:
        f.write(f"\\n=== REWARD FUNCTION CALLED ===\\n")
        f.write(f"Completions count: {len(completions)}\\n")
        f.write(f"Extra data type: {type(extra_data)}\\n")
        f.write(f"Extra data: {str(extra_data)[:500]}\\n")
        if completions:
            f.write(f"First completion: {str(completions[0])[:500]}\\n")
        f.write("==================\\n")
    
    # Return non-zero scores to test if logging is the issue
    return [0.5] * len(completions)
'''
    
    try:
        async with pool.acquire() as conn:
            # Insert the test function
            test_id = "test0000-0000-0000-0000-000000000001"
            
            insert_query = """
                INSERT INTO reward_functions (reward_id, reward_func, func_hash, is_generic)
                VALUES ($1, $2, $3, false)
                ON CONFLICT (reward_id) 
                DO UPDATE SET reward_func = $2, func_hash = $3
            """
            
            await conn.execute(insert_query, test_id, file_logging_function, "test_file_logging")
            print(f"✅ Created file-logging test function: {test_id}")
            
            # Replace one of the task functions with this test
            print(f"\nRun this SQL to test:")
            print(f"UPDATE grpo_task_functions SET reward_id = '{test_id}' WHERE task_id = '438b7328-859f-4d87-b99a-6d85acdbcc15' LIMIT 1;")
            print(f"\nAfter evaluation, check: cat /tmp/reward_debug.log")
                
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(create_file_logging_function())