#!/usr/bin/env python3

import asyncio
import json
import os
import asyncpg
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

async def test_qwen_completions():
    """Test what Qwen actually generates on task data"""
    print("=== Testing Qwen Completions ===")
    
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
            # Get the task data
            task_query = """
                SELECT test_data, training_data, gt.field_prompt
                FROM tasks t
                JOIN grpo_tasks gt ON t.task_id = gt.task_id
                WHERE t.task_id = '438b7328-859f-4d87-b99a-6d85acdbcc15'
            """
            task_row = await conn.fetchrow(task_query)
            
            dataset_path = task_row['test_data'] or task_row['training_data']
            field_prompt = task_row['field_prompt']
            
            print(f"📊 Dataset: {dataset_path}")
            print(f"📋 Field prompt: {field_prompt}")
            
            # Load dataset
            eval_dataset = load_dataset("json", data_files=dataset_path, split="train")
            sample_data = eval_dataset.to_list()[:3]  # Test first 3 samples
            
            print(f"\n=== Loading Qwen Model ===")
            model_name = "Qwen/Qwen2.5-7B-Instruct"
            
            try:
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                print(f"✅ Loaded {model_name}")
                
                # Test each sample
                for i, sample in enumerate(sample_data):
                    print(f"\n{'='*60}")
                    print(f"Testing Sample {i}")
                    
                    # Get the prompt and extra data
                    prompt = sample.get(field_prompt, "")
                    extra_data_str = sample.get('extra_data', '{}')
                    
                    try:
                        extra_data = json.loads(extra_data_str)
                        task_type = extra_data.get('task_type', 'Unknown')
                        
                        print(f"Task type: {task_type}")
                        print(f"Prompt preview: {prompt[:200]}...")
                        
                        # Show the reference solution and premises for DED tasks
                        if task_type == 'DED':
                            solution = extra_data.get('solution', '')[:300]
                            premises = extra_data.get('premises', [])
                            print(f"Reference solution: {solution}...")
                            print(f"Premises: {premises}")
                            
                    except Exception as e:
                        task_type = 'Unknown'
                        print(f"❌ Error parsing extra_data: {e}")
                    
                    # Generate completion
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                    
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        generated_ids = model.generate(
                            model_inputs.input_ids,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]
                    
                    completion = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    print(f"\n🤖 Qwen completion:")
                    print(f"{completion}")
                    
                    # Now test this completion with our reward functions
                    print(f"\n🏆 Testing with reward functions:")
                    
                    # Get our reward functions from constants
                    from validator.core import constants as cst
                    
                    reward_query = """
                        SELECT reward_func
                        FROM reward_functions 
                        WHERE reward_id = ANY($1)
                        ORDER BY 
                            CASE 
                                WHEN reward_id = $2 THEN 1
                                WHEN reward_id = $3 THEN 2  
                                WHEN reward_id = $4 THEN 3
                            END
                    """
                    
                    reward_rows = await conn.fetch(reward_query, 
                                                   cst.AFFINE_REWARD_FN_IDS,
                                                   cst.AFFINE_REWARD_FN_IDS[0], 
                                                   cst.AFFINE_REWARD_FN_IDS[1], 
                                                   cst.AFFINE_REWARD_FN_IDS[2])
                    
                    for j, row in enumerate(reward_rows):
                        namespace = {}
                        exec(row['reward_func'], namespace)
                        
                        # Find the function
                        func_name = None
                        for name, obj in namespace.items():
                            if callable(obj) and name.endswith('_reward_function'):
                                func_name = name
                                break
                        
                        if func_name:
                            reward_func = namespace[func_name]
                            try:
                                result = reward_func([completion], extra_data=[extra_data])
                                print(f"  {func_name}: {result[0]}")
                            except Exception as e:
                                print(f"  {func_name}: ERROR - {e}")
                    
                    if i >= 0:  # Test only first sample for now
                        break
                        
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                import traceback
                traceback.print_exc()
                
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(test_qwen_completions())