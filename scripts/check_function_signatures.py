#!/usr/bin/env python3

import asyncio
import os
import asyncpg
import inspect
from validator.core import constants as cst
from validator.utils.reward_functions import validate_reward_function, supports_extra_data

async def check_function_signatures():
    """Check the signatures of our reward functions"""
    print("=== Checking Function Signatures ===")
    
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
            for i, reward_id in enumerate(cst.AFFINE_REWARD_FN_IDS):
                print(f"\n{'='*50}")
                print(f"Function {i}: {reward_id}")
                
                # Get function from database
                query = "SELECT reward_func FROM reward_functions WHERE reward_id = $1"
                row = await conn.fetchrow(query, reward_id)
                
                if not row:
                    print(f"❌ Function not found")
                    continue
                
                # Check original function signature
                print(f"\n--- Original Function ---")
                namespace = {}
                exec(row['reward_func'], namespace)
                
                func_name = None
                for name, obj in namespace.items():
                    if callable(obj) and name.endswith('_reward_function'):
                        func_name = name
                        break
                
                if func_name:
                    original_func = namespace[func_name]
                    original_sig = inspect.signature(original_func)
                    original_supports = supports_extra_data(original_func)
                    
                    print(f"Function name: {func_name}")
                    print(f"Original signature: {original_sig}")
                    print(f"Supports extra_data: {original_supports}")
                
                # Check processed function signature
                print(f"\n--- Processed Function ---")
                is_valid, error_msg, processed_func = validate_reward_function(row['reward_func'], [])
                
                if is_valid and processed_func:
                    processed_sig = inspect.signature(processed_func)
                    processed_supports = supports_extra_data(processed_func)
                    
                    print(f"Processed signature: {processed_sig}")
                    print(f"Supports extra_data: {processed_supports}")
                    
                    if original_supports != processed_supports:
                        print(f"🚨 SIGNATURE MISMATCH! Original: {original_supports}, Processed: {processed_supports}")
                    else:
                        print(f"✅ Signatures match")
                else:
                    print(f"❌ Processing failed: {error_msg}")
                    
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(check_function_signatures())