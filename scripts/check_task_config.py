#!/usr/bin/env python3

import asyncio
import os
import asyncpg
from validator.core import constants as cst

async def check_task_config():
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
            task_query = """
                SELECT t.test_data, t.training_data, gt.extra_column
                FROM tasks t
                JOIN grpo_tasks gt ON t.task_id = gt.task_id
                WHERE t.task_id = '438b7328-859f-4d87-b99a-6d85acdbcc15'
            """
            task_row = await conn.fetchrow(task_query)
            
            if task_row:
                print(f"Task extra_column config: {task_row['extra_column']}")
                print(f"This should be the actual column name in dataset, not the target name")
                
                print(f"\nUpdate needed:")
                print(f"UPDATE grpo_tasks SET extra_column = 'extra' WHERE task_id = '438b7328-859f-4d87-b99a-6d85acdbcc15';")
            else:
                print("Task not found")
                
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(check_task_config())