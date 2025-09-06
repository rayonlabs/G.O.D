#!/usr/bin/env python3

from validator.utils.reward_functions import validate_reward_function, restricted_execution
from validator.utils.affine_reward_functions import sat_reward_function, abd_reward_function, ded_reward_function
import inspect

def test_direct_restricted_execution():
    """Test restricted_execution directly"""
    print("=== Testing restricted_execution directly ===")
    
    code = """
input_lines = ['5']
input_index = 0

def input(prompt=""):
    global input_index
    if input_index < len(input_lines):
        result = input_lines[input_index]
        input_index += 1
        return result
    return ""

n = int(input())
print(n * 2)
"""
    
    output, error = restricted_execution(code, "5")
    print(f"Output: '{output}'")
    print(f"Error: '{error}'")
    
    if error:
        print("❌ Direct test failed")
        return False
    else:
        print("✅ Direct test passed")
        return True

def test_processed_functions():
    """Test processed versions of reward functions"""
    print("\n=== Testing processed reward functions ===")
    
    functions = [
        ("SAT", sat_reward_function),
        ("ABD", abd_reward_function), 
        ("DED", ded_reward_function)
    ]
    
    for name, func in functions:
        print(f"\n--- Testing {name} ---")
        
        # Get and process the function
        code = inspect.getsource(func)
        is_valid, error, processed_func = validate_reward_function(code, [])
        
        if not is_valid:
            print(f"❌ {name} processing failed: {error}")
            continue
            
        print(f"✅ {name} processed successfully")
        
        # Test with simple data
        if name == "SAT":
            test_data = [{"task_type": "SAT", "cls": [[1, -2], [2, -1]]}]
            test_completions = ["x1=True, x2=False"]
        elif name == "ABD":
            test_data = [{"task_type": "ABD", "program": "print('hello')", "expected_output": "hello"}]
            test_completions = ["<INPUT>\ntest\n</INPUT>"]
        else:  # DED
            test_data = [{"task_type": "DED", "solution": "print('hello')", "premises": ["test"]}]
            test_completions = ["```python\nprint('hello')\n```"]
        
        try:
            result = processed_func(test_completions, extra_data=test_data)
            print(f"✅ {name} test result: {result}")
        except Exception as e:
            print(f"❌ {name} test failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Testing Processed Reward Functions")
    print("=" * 60)
    
    # Test restricted_execution directly first
    if test_direct_restricted_execution():
        # If that works, test the processed functions
        test_processed_functions()
    else:
        print("\n❌ Skipping processed function tests due to restricted_execution failure")
        
    print("\n✅ All tests completed!")