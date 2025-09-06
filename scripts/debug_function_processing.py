#!/usr/bin/env python3

from validator.utils.reward_functions import validate_reward_function, process_reward_function_code
from validator.utils.affine_reward_functions import abd_reward_function
import inspect

def debug_function_processing():
    """Debug what happens during function processing"""
    print("=== Debugging Function Processing ===")
    
    # Get the original function code
    original_code = inspect.getsource(abd_reward_function)
    print(f"Original code length: {len(original_code)} chars")
    print(f"Original code preview:\n{original_code[:300]}...")
    
    # Process the code
    print("\n=== Processing Code ===")
    try:
        processed_code = process_reward_function_code(original_code)
        print(f"Processed code length: {len(processed_code)} chars")
        print(f"Processed code preview:\n{processed_code[:500]}...")
        
        # Check if restricted_execution was injected
        if "def restricted_execution" in processed_code:
            print("✅ restricted_execution was injected")
        else:
            print("❌ restricted_execution was NOT injected")
            
    except Exception as e:
        print(f"❌ Error processing code: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Now validate the processed function
    print("\n=== Validating Processed Function ===")
    try:
        is_valid, error_msg, processed_func = validate_reward_function(original_code, [])
        print(f"Valid: {is_valid}")
        if not is_valid:
            print(f"Error: {error_msg}")
            return
        print("✅ Function validation successful")
        
        # Test the processed function
        print("\n=== Testing Processed Function ===")
        test_data = [{
            "task_type": "ABD",
            "program": "a = int(input())\nb = int(input())\nprint(a + b)",
            "expected_output": "15"
        }]
        completion = "I need to find input:\n<INPUT>\n5\n10\n</INPUT>"
        
        result = processed_func([completion], extra_data=test_data)
        print(f"Processed function result: {result}")
        
    except Exception as e:
        print(f"❌ Error in validation/testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_function_processing()