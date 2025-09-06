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
            # Use the first existing constant ID to replace it temporarily
            from validator.core import constants as cst
            test_id = cst.AFFINE_REWARD_FN_IDS[0]  # Replace the SAT function temporarily
            
            insert_query = """
                INSERT INTO reward_functions (reward_id, reward_func, func_hash, is_generic)
                VALUES ($1, $2, $3, false)
                ON CONFLICT (reward_id) 
                DO UPDATE SET reward_func = $2, func_hash = $3
            """
            
            await conn.execute(insert_query, test_id, file_logging_function, "test_file_logging")
            print(f"✅ Created file-logging test function: {test_id}")
            
            # Replace one of the task functions with this test
            print(f"\n✅ Replaced SAT function with file-logging test function")
            print(f"Now run evaluation and check: cat /tmp/reward_debug.log")
            print(f"The test function always returns 0.5 scores.")
                
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(create_file_logging_function())