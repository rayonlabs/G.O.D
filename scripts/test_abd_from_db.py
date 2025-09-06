#!/usr/bin/env python3

import asyncio
import os
import asyncpg
from uuid import UUID

async def test_abd_from_db():
    """Test ABD reward function loaded from database"""
    print("=== Testing ABD from Database ===")
    
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
    if not connection_string:
        raise ValueError("DATABASE_URL not found")
    
    pool = await asyncpg.create_pool(connection_string)
    
    try:
        async with pool.acquire() as conn:
            # Get ABD reward function
            query = """
                SELECT reward_func, func_hash 
                FROM reward_functions 
                WHERE reward_id = '912fa425-ac2f-4a9f-8767-c419fd4b9269'
            """
            row = await conn.fetchrow(query)
            
            if not row:
                print("❌ ABD function not found in database")
                return
                
            print(f"✅ Found ABD function, hash: {row['func_hash'][:16]}...")
            
            # Execute the function code
            reward_func_code = row['reward_func']
            print(f"📋 Function code length: {len(reward_func_code)} chars")
            
            # Create function from code
            namespace = {}
            exec(reward_func_code, namespace)
            abd_func = namespace['abd_reward_function']
            
            # Test data
            test_data = [{
                "task_type": "ABD",
                "program": "a = int(input())\nb = int(input())\nprint(a + b)",
                "expected_output": "15"
            }]
            completion = "I need to find input that makes this program output 15. The program adds two numbers.\n<INPUT>\n5\n10\n</INPUT>"
            
            print(f"🧪 Testing with completion: {completion[:50]}...")
            
            # Test the function
            result = abd_func([completion], extra_data=test_data)
            print(f"🎯 Database function result: {result}")
            
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(test_abd_from_db())