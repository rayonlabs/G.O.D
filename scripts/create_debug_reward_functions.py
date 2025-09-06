#!/usr/bin/env python3

import asyncio
import os
import asyncpg
from validator.core import constants as cst

async def create_debug_reward_functions():
    """Create debug versions of reward functions that log everything"""
    print("=== Creating Debug Reward Functions ===")
    
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
            # Get each reward function and add debug logging
            for i, reward_id in enumerate(cst.AFFINE_REWARD_FN_IDS):
                print(f"Processing reward function {i}: {reward_id}")
                
                # Get original function
                query = "SELECT reward_func FROM reward_functions WHERE reward_id = $1"
                row = await conn.fetchrow(query, reward_id)
                
                if not row:
                    print(f"❌ Function {reward_id} not found")
                    continue
                
                original_code = row['reward_func']
                
                # Add debug logging at the start of each function
                debug_code = original_code.replace(
                    'scores = []',
                    '''print(f"🔍 EVAL DEBUG: Function called with {len(completions)} completions")
    print(f"🔍 EVAL DEBUG: extra_data type: {type(extra_data)}")
    if extra_data:
        print(f"🔍 EVAL DEBUG: extra_data length: {len(extra_data) if hasattr(extra_data, '__len__') else 'no len'}")
        print(f"🔍 EVAL DEBUG: first extra_data: {str(extra_data)[:200] if extra_data else 'None'}...")
    if completions:
        print(f"🔍 EVAL DEBUG: first completion: {str(completions[0])[:200] if completions else 'None'}...")
    scores = []'''
                )
                
                # Also add debug at the end
                debug_code = debug_code.replace(
                    'return scores',
                    '''print(f"🔍 EVAL DEBUG: Returning scores: {scores}")
    return scores'''
                )
                
                # Create new debug function ID
                debug_reward_id = f"{reward_id[:-3]}deb"  # Replace last 3 chars with 'deb'
                
                # Insert debug function
                insert_query = """
                    INSERT INTO reward_functions (reward_id, reward_func, func_hash, is_generic)
                    VALUES ($1, $2, $3, false)
                    ON CONFLICT (reward_id) 
                    DO UPDATE SET reward_func = $2, func_hash = $3
                """
                
                func_hash = f"debug_{reward_id[:8]}"
                await conn.execute(insert_query, debug_reward_id, debug_code, func_hash)
                print(f"✅ Created debug function: {debug_reward_id}")
            
            print(f"\n🔄 Now update the task to use debug functions:")
            debug_ids = [f"{rid[:-3]}deb" for rid in cst.AFFINE_REWARD_FN_IDS]
            
            for i, debug_id in enumerate(debug_ids):
                print(f"UPDATE grpo_task_functions SET reward_id = '{debug_id}' WHERE task_id = '438b7328-859f-4d87-b99a-6d85acdbcc15' AND reward_id = '{cst.AFFINE_REWARD_FN_IDS[i]}';")
                
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(create_debug_reward_functions())