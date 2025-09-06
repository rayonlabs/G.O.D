#!/usr/bin/env python3

import asyncio
import os
import asyncpg

async def test_sat_from_db():
    """Test SAT reward function loaded from database"""
    print("=== Testing SAT from Database ===")
    
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
            # Get SAT reward function (ID from the upload output)
            query = """
                SELECT reward_func, func_hash 
                FROM reward_functions 
                WHERE reward_id = 'dec09e62-1e44-4baa-881f-5fba4bdfa225'
            """
            row = await conn.fetchrow(query)
            
            if not row:
                print("❌ SAT function not found in database")
                return
                
            print(f"✅ Found SAT function, hash: {row['func_hash'][:16]}...")
            
            # Execute the function code
            reward_func_code = row['reward_func']
            print(f"📋 Function code length: {len(reward_func_code)} chars")
            
            # Create function from code
            namespace = {}
            exec(reward_func_code, namespace)
            sat_func = namespace['sat_reward_function']
            
            # Test data - SAT problem (3-SAT with solution)
            test_data = [{
                "task_type": "SAT",
                "cnf_formula": "p cnf 3 3\n1 -2 3 0\n-1 2 0\n2 -3 0",
                "expected_satisfiable": True
            }]
            completion = "Looking at this 3-SAT problem, I need to find values that satisfy all clauses.\n\n<ASSIGNMENT>\nx1 = True\nx2 = True\nx3 = True\n</ASSIGNMENT>"
            
            print(f"🧪 Testing with SAT completion...")
            print(f"CNF: {test_data[0]['cnf_formula'].replace(chr(10), ' | ')}")
            
            # Test the function
            result = sat_func([completion], extra_data=test_data)
            print(f"🎯 Database SAT function result: {result}")
            
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(test_sat_from_db())