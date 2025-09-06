#!/usr/bin/env python3

import asyncio
import os
import asyncpg
from validator.core import constants as cst

async def check_current_functions():
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
            for i, reward_id in enumerate(cst.AFFINE_REWARD_FN_IDS):
                print(f"\n=== Function {i}: {reward_id} ===")
                
                query = "SELECT reward_func FROM reward_functions WHERE reward_id = $1"
                row = await conn.fetchrow(query, reward_id)
                
                if row:
                    func_code = row['reward_func']
                    if "debug_reward_function" in func_code:
                        print("❌ This is still the debug function")
                    elif "REWARD DEBUG" in func_code or "print(" in func_code:
                        print("❌ This contains debug code")
                    else:
                        print("✅ This looks like a clean production function")
                        
                    # Show first few lines
                    lines = func_code.split('\n')[:5]
                    for line in lines:
                        print(f"  {line}")
                else:
                    print("❌ Function not found")
                
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(check_current_functions())