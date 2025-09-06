#!/usr/bin/env python3

from validator.utils.reward_functions import validate_reward_function
from validator.utils.affine_reward_functions import abd_reward_function
import inspect

def debug_processed_vs_manual():
    """Compare processed function vs manual execution"""
    print("=== Processed vs Manual Comparison ===")
    
    # Test data
    test_data = [{
        "task_type": "ABD",
        "program": "a = int(input())\nb = int(input())\nprint(a + b)",
        "expected_output": "15"
    }]
    completion = "I need to find input that makes this program output 15. The program adds two numbers.\n<INPUT>\n5\n10\n</INPUT>"
    
    print(f"Test data: {test_data[0]}")
    print(f"Completion: {completion}")
    
    # Get processed function
    original_code = inspect.getsource(abd_reward_function)
    is_valid, error_msg, processed_func = validate_reward_function(original_code, [])
    
    if not is_valid:
        print(f"❌ Processing failed: {error_msg}")
        return
    
    print("✅ Function processed successfully")
    
    # Test processed function
    print("\n=== Testing Processed Function ===")
    try:
        result = processed_func([completion], extra_data=test_data)
        print(f"🔍 Processed function result: {result}")
        
        # Also test the original function directly for comparison
        print("\n=== Testing Original Function ===")
        original_result = abd_reward_function([completion], extra_data=test_data)
        print(f"🔍 Original function result: {original_result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_processed_vs_manual()