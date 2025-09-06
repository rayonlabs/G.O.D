#!/usr/bin/env python3

import asyncio
import os
import asyncpg
from validator.core import constants as cst
from validator.utils.affine_reward_functions import sat_reward_function
import inspect

async def restore_sat_function():
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
    
    # Get the original function source
    sat_source = inspect.getsource(sat_reward_function)
    
    try:
        async with pool.acquire() as conn:
            # Restore the SAT function (first function)
            sat_id = cst.AFFINE_REWARD_FN_IDS[0]
            
            insert_query = """
                INSERT INTO reward_functions (reward_id, reward_func, func_hash, is_generic)
                VALUES ($1, $2, $3, false)
                ON CONFLICT (reward_id) 
                DO UPDATE SET reward_func = $2, func_hash = $3
            """
            
            await conn.execute(insert_query, sat_id, sat_source, "original_sat")
            print(f"✅ Restored original SAT function: {sat_id}")
                
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(restore_sat_function())