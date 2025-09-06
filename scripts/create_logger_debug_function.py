#!/usr/bin/env python3

import asyncio
import os
import asyncpg
from validator.core import constants as cst

async def create_logger_debug_function():
    """Create a function that uses the evaluation's logger to debug"""
    print("=== Creating Logger-Debug Function ===")
    
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
    
    # Create a function that uses the evaluation logger
    logger_debug_function = '''
def debug_reward_function(completions, extra_data=None, **kwargs):
    """Debug function using evaluation logger"""
    import logging
    
    # Try to get the same logger the evaluation uses
    logger = logging.getLogger("eval_grpo")
    
    # Log what we received
    logger.error(f"🔍 REWARD DEBUG: Called with {len(completions)} completions")
    logger.error(f"🔍 REWARD DEBUG: extra_data type: {type(extra_data)}")
    
    if extra_data:
        logger.error(f"🔍 REWARD DEBUG: extra_data length: {len(extra_data) if hasattr(extra_data, '__len__') else 'no len'}")
        logger.error(f"🔍 REWARD DEBUG: first extra_data: {str(extra_data)[:300] if extra_data else 'None'}")
    
    if completions:
        logger.error(f"🔍 REWARD DEBUG: first completion preview: {str(completions[0])[:300] if completions else 'None'}")
    
    # Also try the root logger
    root_logger = logging.getLogger()
    root_logger.error(f"🔍 ROOT DEBUG: Function called with {len(completions)} completions")
    
    # Return some scores based on what we see
    scores = []
    for i in range(len(completions)):
        # Return different scores to help identify patterns
        scores.append(0.1 + (i * 0.1))  # 0.1, 0.2, 0.3, etc.
    
    logger.error(f"🔍 REWARD DEBUG: Returning scores: {scores}")
    return scores
'''
    
    try:
        async with pool.acquire() as conn:
            # Replace the DED function (the one we know should work best)
            test_id = cst.AFFINE_REWARD_FN_IDS[2]  # DED function
            
            insert_query = """
                INSERT INTO reward_functions (reward_id, reward_func, func_hash, is_generic)
                VALUES ($1, $2, $3, false)
                ON CONFLICT (reward_id) 
                DO UPDATE SET reward_func = $2, func_hash = $3
            """
            
            await conn.execute(insert_query, test_id, logger_debug_function, "logger_debug")
            print(f"✅ Created logger-debug function: {test_id}")
            print(f"\nNow run evaluation and look for '🔍 REWARD DEBUG:' messages in the logs!")
            print(f"This function returns incremental scores (0.1, 0.2, 0.3, etc.) to help identify patterns")
                
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(create_logger_debug_function())