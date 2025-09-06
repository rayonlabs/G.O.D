#!/usr/bin/env python3

from validator.utils.reward_functions import validate_reward_function
from validator.utils.affine_reward_functions import abd_reward_function
import inspect

def test_direct_comparison():
    """Direct comparison between processed function and expected behavior"""
    print("=== Direct Comparison Test ===")
    
    # Test data - exact same as our working manual test
    test_data = [{
        "task_type": "ABD",
        "program": "a = int(input())\nb = int(input())\nprint(a + b)",
        "expected_output": "15"
    }]
    completion = "I need to find input that makes this program output 15. The program adds two numbers.\n<INPUT>\n5\n10\n</INPUT>"
    
    print(f"Test completion: {completion[:50]}...")
    print(f"Test data: {test_data[0]}")
    
    # Get processed function
    original_code = inspect.getsource(abd_reward_function)
    is_valid, error_msg, processed_func = validate_reward_function(original_code, [])
    
    if not is_valid:
        print(f"❌ Processing failed: {error_msg}")
        return
    
    print("✅ Function processed successfully")
    
    # Test the processed function with debug
    print("\n=== Testing Processed Function ===")
    
    try:
        # Test with single completion and single extra_data item
        result = processed_func([completion], extra_data=test_data)
        print(f"Processed function result: {result}")
        
        # Test with different parameter formats
        print("\n=== Testing different parameter formats ===")
        
        # Test 1: extra_data as list
        result1 = processed_func([completion], extra_data=test_data)
        print(f"List format result: {result1}")
        
        # Test 2: extra_data as single dict
        result2 = processed_func([completion], extra_data=test_data[0])
        print(f"Single dict format result: {result2}")
        
        # Test 3: extra_data as kwarg
        result3 = processed_func([completion], extra_data=test_data)
        print(f"Kwarg format result: {result3}")
        
    except Exception as e:
        print(f"❌ Error testing processed function: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_direct_comparison()