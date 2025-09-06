#!/usr/bin/env python3
"""
Script to add the new affine reward functions to the database.
"""
import asyncio
import os
from uuid import UUID

from validator.db.sql.manually_add_grpo_rewards import manually_store_reward_functions
from validator.db.sql.grpo import get_reward_function_by_id
from validator.utils.affine_reward_functions import (
    sat_reward_function,
    abd_reward_function, 
    ded_reward_function
)
from validator.core import constants as cst
import asyncpg


async def main():
    # Database connection string
    connection_string = os.getenv("DATABASE_URL")
    if not connection_string:
        print("❌ ERROR: DATABASE_URL environment variable not set")
        return
    
    print("🚀 Adding affine reward functions to database...")
    print(f"Database URL: {connection_string.split('@')[1] if '@' in connection_string else 'localhost'}")
    
    # List of functions to add
    reward_functions = [
        sat_reward_function,
        abd_reward_function,
        ded_reward_function
    ]
    
    print(f"\n📋 Functions to add:")
    for i, func in enumerate(reward_functions, 1):
        print(f"  {i}. {func.__name__}")
    
    # Add functions to database
    await manually_store_reward_functions(
        connection_string=connection_string,
        reward_functions=reward_functions,
        is_generic=True  # Set to True so they can be used across tasks
    )
    
    print("\n🔍 Checking what IDs were assigned...")
    
    # Connect to database to check the added functions
    pool = await asyncpg.create_pool(connection_string)
    
    try:
        async with pool.acquire() as conn:
            # Query for our functions by searching function names in the code
            for func in reward_functions:
                query = f"""
                    SELECT reward_id, func_hash 
                    FROM reward_functions 
                    WHERE reward_func LIKE $1
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                pattern = f"%def {func.__name__}%"
                result = await conn.fetchrow(query, pattern)
                
                if result:
                    reward_id = result['reward_id']
                    func_hash = result['func_hash'][:12]  # First 12 chars of hash
                    print(f"✅ {func.__name__}:")
                    print(f"   ID: {reward_id}")
                    print(f"   Hash: {func_hash}...")
                else:
                    print(f"❌ {func.__name__}: Not found in database")
                    
    finally:
        await pool.close()
    
    print(f"\n📝 Current constants in validator/core/constants.py:")
    print(f"AFFINE_REWARD_FN_IDS = {cst.AFFINE_REWARD_FN_IDS}")
    
    print(f"\n⚠️  NEXT STEPS:")
    print(f"1. Copy the actual reward_ids from above")
    print(f"2. Update AFFINE_REWARD_FN_IDS in validator/core/constants.py")
    print(f"3. Restart your validator to pick up the new constants")
    print(f"4. Check validator logs for 'Looking for affine reward functions' messages")


if __name__ == "__main__":
    asyncio.run(main())