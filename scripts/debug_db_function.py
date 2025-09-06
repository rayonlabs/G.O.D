#!/usr/bin/env python3

import asyncio
import os
import asyncpg

async def debug_db_function():
    """Debug what the database function code looks like"""
    print("=== Debugging Database Function Code ===")
    
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
            # Get ABD reward function
            query = """
                SELECT reward_func 
                FROM reward_functions 
                WHERE reward_id = 'd7000869-e268-4f84-b72f-838ca4da43f1'
            """
            row = await conn.fetchrow(query)
            
            if not row:
                print("❌ ABD function not found")
                return
                
            reward_func_code = row['reward_func']
            print(f"📋 Database function code ({len(reward_func_code)} chars):")
            print("=" * 80)
            
            # Show the part around restricted_execution call
            lines = reward_func_code.split('\n')
            for i, line in enumerate(lines):
                if 'restricted_execution(' in line:
                    print(f">>> Found restricted_execution call at line {i+1}:")
                    # Show context around this line
                    start = max(0, i-5)
                    end = min(len(lines), i+10)
                    for j in range(start, end):
                        marker = ">>> " if j == i else "    "
                        print(f"{marker}{j+1:3}: {lines[j]}")
                    print("=" * 40)
            
            # Check if restricted_execution function is defined in the code
            if "def restricted_execution(" in reward_func_code:
                print("✅ restricted_execution function is defined in the database code")
            else:
                print("❌ restricted_execution function is NOT defined in the database code")
                
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(debug_db_function())