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


def load_env_file():
    """Load DATABASE_URL from .vali.env file"""
    try:
        with open('.vali.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('DATABASE_URL=') and not line.startswith('#'):
                    return line.split('=', 1)[1]
        return None
    except FileNotFoundError:
        return None

async def delete_existing_functions(connection_string):
    """Delete existing affine reward functions"""
    pool = await asyncpg.create_pool(connection_string)
    
    try:
        async with pool.acquire() as conn:
            function_names = ['sat_reward_function', 'abd_reward_function', 'ded_reward_function']
            for func_name in function_names:
                query = """
                    DELETE FROM reward_functions 
                    WHERE reward_func LIKE $1
                """
                pattern = f"%def {func_name}%"
                result = await conn.execute(query, pattern)
                print(f"🗑️  Deleted existing {func_name} (if it existed)")
    finally:
        await pool.close()

async def main():
    # Database connection string
    connection_string = os.getenv("DATABASE_URL")
    
    # If not in env, try to load from .vali.env file
    if not connection_string:
        connection_string = load_env_file()
    
    if not connection_string:
        print("❌ ERROR: DATABASE_URL not found in environment or .vali.env file")
        return
    
    print("🚀 Setting up affine reward functions...")
    print(f"Database URL: {connection_string.split('@')[1] if '@' in connection_string else 'localhost'}")
    
    # Delete existing functions first to ensure clean state
    print("\n🗑️  Cleaning up existing functions...")
    await delete_existing_functions(connection_string)
    
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
    
    # Collect the actual IDs for auto-update
    actual_ids = []
    pool = await asyncpg.create_pool(connection_string)
    try:
        async with pool.acquire() as conn:
            for func in reward_functions:
                query = f"""
                    SELECT reward_id
                    FROM reward_functions 
                    WHERE reward_func LIKE $1
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                pattern = f"%def {func.__name__}%"
                result = await conn.fetchrow(query, pattern)
                if result:
                    actual_ids.append(str(result['reward_id']))
    finally:
        await pool.close()
    
    if len(actual_ids) == 3:
        print(f"\n🔄 Auto-updating constants file...")
        
        # Read current constants file
        constants_path = "validator/core/constants.py"
        with open(constants_path, 'r') as f:
            content = f.read()
        
        # Update the AFFINE_REWARD_FN_IDS
        old_pattern = r'AFFINE_REWARD_FN_IDS = \[[\s\S]*?\]'
        new_ids_str = f'''AFFINE_REWARD_FN_IDS = [
    "{actual_ids[0]}",  # sat_reward_function
    "{actual_ids[1]}",  # abd_reward_function  
    "{actual_ids[2]}",  # ded_reward_function
]'''
        
        import re
        updated_content = re.sub(old_pattern, new_ids_str, content)
        
        # Write back to file
        with open(constants_path, 'w') as f:
            f.write(updated_content)
            
        print(f"✅ Updated {constants_path} with actual reward function IDs")
        print(f"\n📝 New constants:")
        print(f"AFFINE_REWARD_FN_IDS = {actual_ids}")
        
        print(f"\n⚠️  NEXT STEPS:")
        print(f"1. Restart your validator to pick up the new constants")
        print(f"2. Check validator logs for 'Looking for affine reward functions' messages")
        print(f"3. Verify all 3 functions are found and loaded")
    else:
        print(f"\n❌ ERROR: Expected 3 function IDs, got {len(actual_ids)}")
        print(f"📝 Current constants in validator/core/constants.py:")
        print(f"AFFINE_REWARD_FN_IDS = {cst.AFFINE_REWARD_FN_IDS}")
        print(f"\n⚠️  MANUAL STEPS REQUIRED:")
        print(f"1. Copy the actual reward_ids from above")
        print(f"2. Update AFFINE_REWARD_FN_IDS in validator/core/constants.py")
        print(f"3. Restart your validator to pick up the new constants")


if __name__ == "__main__":
    asyncio.run(main())