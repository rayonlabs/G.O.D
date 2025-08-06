#!/usr/bin/env python3
"""
Test script for instruct evaluation of GPT-OSS-20B model.
This tests the base model directly as if it were a fine-tuned submission.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from uuid import uuid4

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from validator.core.models import EvaluationArgs
from core.models.utility_models import InstructTextDatasetType, FileFormat


def create_test_dataset(output_path: Path) -> str:
    """Create a small test dataset for instruct evaluation."""
    test_data = [
        {
            "instruction": "What is the capital of France?",
            "output": "The capital of France is Paris."
        },
        {
            "instruction": "Explain photosynthesis in simple terms.",
            "output": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to make their own food (glucose) and release oxygen as a byproduct."
        },
        {
            "instruction": "Write a Python function to calculate factorial.",
            "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)"
        },
        {
            "instruction": "What are the three states of matter?",
            "output": "The three states of matter are solid, liquid, and gas."
        },
        {
            "instruction": "Translate 'Hello, how are you?' to Spanish.",
            "output": "Hola, ¿cómo estás?"
        },
        {
            "instruction": "List three benefits of regular exercise.",
            "output": "Three benefits of regular exercise are: 1) Improved cardiovascular health, 2) Better mood and mental health, 3) Increased strength and endurance."
        },
        {
            "instruction": "What is machine learning?",
            "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
        },
        {
            "instruction": "Calculate 15% of 200.",
            "output": "15% of 200 is 30."
        },
        {
            "instruction": "Name the planets in our solar system.",
            "output": "The planets in our solar system are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."
        },
        {
            "instruction": "What is the difference between a compiler and an interpreter?",
            "output": "A compiler translates the entire source code into machine code before execution, while an interpreter translates and executes code line by line during runtime."
        }
    ]
    
    # Save as JSON file
    dataset_file = output_path / "test_instruct_dataset.json"
    with open(dataset_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Created test dataset at: {dataset_file}")
    return str(dataset_file)


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test GPT-OSS-20B instruct evaluation')
    parser.add_argument('--model', type=str, default='openai/gpt-oss-20b', help='Model ID to evaluate')
    parser.add_argument('--dataset', type=str, default=None, help='Path to dataset JSON file')
    parser.add_argument('--output-dir', type=str, default='./test_eval_output', help='Output directory')
    parser.add_argument('--field-instruction', type=str, default='instruction', help='Field name for instructions')
    parser.add_argument('--field-output', type=str, default='output', help='Field name for outputs')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GPT-OSS-20B Instruct Evaluation Test Script")
    print("="*60)
    
    # Configuration
    MODEL_ID = args.model
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Create or use provided dataset
    if args.dataset:
        dataset_path = args.dataset
        if not Path(dataset_path).exists():
            print(f"Error: Dataset file {dataset_path} does not exist")
            sys.exit(1)
        print(f"Using provided dataset: {dataset_path}")
    else:
        dataset_path = create_test_dataset(OUTPUT_DIR)
    
    # Define dataset type for instruct evaluation
    dataset_type = InstructTextDatasetType(
        field_instruction=args.field_instruction,
        field_output=args.field_output
    )
    
    # Set environment variables that the evaluation script expects
    os.environ["DATASET"] = dataset_path
    os.environ["ORIGINAL_MODEL"] = MODEL_ID
    os.environ["DATASET_TYPE"] = json.dumps(dataset_type.model_dump())
    os.environ["FILE_FORMAT"] = "json"
    os.environ["MODELS"] = MODEL_ID  # Testing the base model itself
    
    print(f"\nConfiguration:")
    print(f"  Model to evaluate: {MODEL_ID}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Dataset Type: {dataset_type.model_dump()}")
    print(f"  File Format: json")
    print(f"  Output Dir: {OUTPUT_DIR}")
    
    # Create evaluation arguments
    eval_args = EvaluationArgs(
        dataset=dataset_path,
        original_model=MODEL_ID,
        dataset_type=dataset_type,
        file_format=FileFormat.JSON,
        repo=MODEL_ID  # Evaluate the base model directly
    )
    
    print("\n" + "="*60)
    print("Running Instruct Evaluation")
    print("="*60)
    print("\nNote: This will evaluate the base model directly.")
    print("The expected_repo_name is only used when evaluating fine-tuned models from miners.")
    
    # Run the evaluation
    try:
        # Run evaluation subprocess (same pattern as the actual evaluation)
        result = subprocess.run([
            sys.executable,
            "-m",
            "validator.evaluation.single_eval_instruct_text",
            eval_args.model_dump_json()
        ], capture_output=True, text=True, check=False)
        
        print("\nEvaluation Output:")
        print(result.stdout)
        
        if result.stderr:
            print("\nErrors/Warnings:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✓ Evaluation completed successfully")
        else:
            print(f"\n⚠ Evaluation exited with code {result.returncode}")
            
        # Check for results file
        results_file = Path("results.json")
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
                print("\n" + "="*60)
                print("Evaluation Results:")
                print("="*60)
                print(json.dumps(results, indent=2))
                
                if MODEL_ID in results:
                    print(f"\nSummary:")
                    print(f"  Model: {MODEL_ID}")
                    print(f"  Eval Loss: {results[MODEL_ID].get('eval_loss', 'N/A')}")
                    print(f"  Is Finetune: {results[MODEL_ID].get('is_finetune', 'N/A')}")
        else:
            print(f"\nNote: Results file not found. Check current directory or logs for details.")
            
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Script Complete")
    print("="*60)
    print("\nUsage examples:")
    print("  python test_gpt_instruct_eval.py")
    print("  python test_gpt_instruct_eval.py --model meta-llama/Llama-2-7b-hf")
    print("  python test_gpt_instruct_eval.py --dataset /path/to/custom/dataset.json")


if __name__ == "__main__":
    main()