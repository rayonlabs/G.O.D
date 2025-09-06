#!/usr/bin/env python3
"""
Manual GRPO evaluation script using the actual evaluation container.
Usage: python -m scripts.manual_grpo_eval <task_id> <model_repo>
"""
import asyncio
import subprocess
import sys
import os
from uuid import UUID

import asyncpg

from validator.core import constants as cst
from validator.utils.logging import get_logger
from core.models.utility_models import GrpoDatasetType, RewardFunction
from core.models.data_models import FileFormat

logger = get_logger(__name__)


def load_env_file():
    """Load environment variables from .vali.env file"""
    try:
        with open('.vali.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
                    print(f"🔧 Loaded {key} from .vali.env")
    except FileNotFoundError:
        print("⚠️  .vali.env file not found")


async def get_task_info(task_id: str):
    """Get task information from database"""
    connection_string = os.getenv("DATABASE_URL")
    if not connection_string:
        raise ValueError("DATABASE_URL not found")
    
    pool = await asyncpg.create_pool(connection_string)
    
    try:
        async with pool.acquire() as conn:
            # Get basic task info
            query = """
                SELECT task_id, model_id, ds as dataset_url, test_data, training_data, 
                       synthetic_data, task_type, field_prompt
                FROM tasks 
                WHERE task_id = $1
            """
            task_row = await conn.fetchrow(query, UUID(task_id))
            
            if not task_row:
                raise ValueError(f"Task {task_id} not found")
            
            # Get GRPO specific info including reward functions
            grpo_query = """
                SELECT gt.file_format, gt.synthetic_data, gt.extra_column,
                       rf.reward_id, rf.reward_func, rf.reward_weight, rf.func_hash, rf.is_generic
                FROM grpo_tasks gt
                LEFT JOIN grpo_task_functions gtf ON gt.task_id = gtf.task_id
                LEFT JOIN reward_functions rf ON gtf.reward_id = rf.reward_id
                WHERE gt.task_id = $1
            """
            grpo_rows = await conn.fetch(grpo_query, UUID(task_id))
            
            return task_row, grpo_rows
            
    finally:
        await pool.close()


def run_grpo_evaluation(task_info, grpo_info, model_repo: str):
    """Run GRPO evaluation using Docker container"""
    
    task_row = task_info
    
    # Extract info from task
    dataset_url = task_row['dataset_url'] or task_row['synthetic_data']
    original_model = task_row['model_id']
    file_format = FileFormat(grpo_info[0]['file_format']) if grpo_info else FileFormat.S3
    
    # Create RewardFunction objects (matching validator pattern)
    reward_functions = []
    for row in grpo_info:
        if row['reward_func']:
            reward_function = RewardFunction(
                reward_id=row['reward_id'],
                reward_func=row['reward_func'],
                reward_weight=float(row['reward_weight']),
                func_hash=row['func_hash'],
                is_generic=row['is_generic']
            )
            reward_functions.append(reward_function)
    
    # Create GrpoDatasetType (matching validator pattern)
    dataset_type = GrpoDatasetType(
        field_prompt=task_row['field_prompt'],
        reward_functions=reward_functions,
        extra_column=grpo_info[0]['extra_column'] if grpo_info else 'extra_data'
    )
    
    print(f"🚀 Starting GRPO evaluation for task {task_row['task_id']}")
    print(f"📊 Dataset: {dataset_url}")
    print(f"🤖 Original Model: {original_model}")
    print(f"🎯 Model to evaluate: {model_repo}")
    print(f"🏆 Reward functions: {len(reward_functions)}")
    print(f"📋 Field prompt: {task_row['field_prompt']}")
    print(f"📝 Extra column: {dataset_type.extra_column}")
    
    # Get HF cache directory (platform independent)
    hf_cache = os.path.expanduser("~/.cache/huggingface")
    
    # Docker command to run evaluation (matching validator pattern)
    docker_cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "-v", f"{hf_cache}:/root/.cache/huggingface",
        "-e", f"DATASET={dataset_url}",
        "-e", f"ORIGINAL_MODEL={original_model}",
        "-e", f"MODELS={model_repo}",
        "-e", f"FILE_FORMAT={file_format.value}",
        "-e", f"DATASET_TYPE={dataset_type.model_dump_json()}",  # Use pydantic serialization
        "-e", "TRANSFORMERS_ALLOW_TORCH_LOAD=true",
        "-e", "HF_HOME=/root/.cache/huggingface",
        "-e", "TRANSFORMERS_CACHE=/root/.cache/huggingface/hub",
        "-e", "HF_DATASETS_CACHE=/root/.cache/huggingface/datasets",
        cst.VALIDATOR_DOCKER_IMAGE
    ]
    
    print(f"\n🐳 Docker image: {cst.VALIDATOR_DOCKER_IMAGE}")
    print(f"🐳 Running Docker command:")
    print(f"   docker run --rm --gpus all ... {cst.VALIDATOR_DOCKER_IMAGE}")
    
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        print(f"\n📋 STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print(f"\n❌ STDERR:")
            print(result.stderr)
        
        print(f"\n✅ Exit code: {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Evaluation timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return False


async def main():
    if len(sys.argv) != 3:
        print("Usage: python -m scripts.manual_grpo_eval <task_id> <model_repo>")
        print("Example: python -m scripts.manual_grpo_eval ba444f8a-dd60-40e2-bdad-524e1e754457 microsoft/Phi-3-mini-128k-instruct")
        sys.exit(1)
    
    task_id = sys.argv[1]
    model_repo = sys.argv[2]
    
    # Load environment variables
    load_env_file()
    
    try:
        print(f"🔍 Fetching task information for {task_id}...")
        task_info, grpo_info = await get_task_info(task_id)
        
        print(f"✅ Found task: {task_info['task_type']}")
        print(f"📊 Dataset: {task_info['dataset_url'] or 'synthetic_data'}")
        print(f"🤖 Original model: {task_info['model_id']}")
        
        success = run_grpo_evaluation(task_info, grpo_info, model_repo)
        
        if success:
            print("🎉 Evaluation completed successfully!")
        else:
            print("💥 Evaluation failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())