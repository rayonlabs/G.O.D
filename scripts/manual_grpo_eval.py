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

from core import constants as cst
from validator.utils.logging import get_logger
from core.models.utility_models import GrpoDatasetType, RewardFunction, FileFormat

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
            # Get task info including GRPO specific data
            query = """
                SELECT t.task_id, t.model_id, t.ds as dataset_url, t.test_data, t.training_data, 
                       t.task_type, gt.field_prompt, gt.file_format, gt.synthetic_data, gt.extra_column
                FROM tasks t
                JOIN grpo_tasks gt ON t.task_id = gt.task_id
                WHERE t.task_id = $1
            """
            task_row = await conn.fetchrow(query, UUID(task_id))
            
            if not task_row:
                raise ValueError(f"Task {task_id} not found")
            
            # Get reward functions
            reward_query = """
                SELECT rf.reward_id, rf.reward_func, gtf.reward_weight, rf.func_hash, rf.is_generic
                FROM grpo_task_functions gtf
                JOIN reward_functions rf ON gtf.reward_id = rf.reward_id
                WHERE gtf.task_id = $1
            """
            reward_rows = await conn.fetch(reward_query, UUID(task_id))
            
            return task_row, reward_rows
            
    finally:
        await pool.close()


def run_grpo_evaluation(task_info, reward_info, model_repo: str):
    """Run GRPO evaluation using Docker container"""
    
    task_row = task_info
    
    # Extract info from task
    dataset_url = task_row['dataset_url'] or (task_row['synthetic_data'] if task_row['synthetic_data'] else None)
    original_model = task_row['model_id']
    file_format = FileFormat(task_row['file_format']) if task_row['file_format'] else FileFormat.S3
    
    # Create RewardFunction objects (matching validator pattern)
    reward_functions = []
    for row in reward_info:
        if row['reward_func']:
            reward_function = RewardFunction(
                reward_id=str(row['reward_id']),
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
        extra_column=task_row['extra_column'] if task_row['extra_column'] else 'extra_data'
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
        task_info, reward_info = await get_task_info(task_id)
        
        print(f"✅ Found task: {task_info['task_type']}")
        print(f"📊 Dataset: {task_info['dataset_url'] or 'synthetic_data'}")
        print(f"🤖 Original model: {task_info['model_id']}")
        
        success = run_grpo_evaluation(task_info, reward_info, model_repo)
        
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