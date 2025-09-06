#!/usr/bin/env python3

from validator.utils.reward_functions import validate_reward_function
from validator.utils.affine_reward_functions import abd_reward_function
import inspect

print("=== Checking reward function processing ===")

code = inspect.getsource(abd_reward_function)
print('=== Raw Function Code (first 300 chars) ===')
print(code[:300] + '...')

print('\n=== Processing Function ===')  
is_valid, error, processed = validate_reward_function(code, [])
print(f'Valid: {is_valid}')
if not is_valid:
    print(f'Error: {error}')
else:
    print('✅ Processed function created successfully')
    
    # Test the processed function
    print('\n=== Testing processed function ===')
    try:
        test_data = [{"task_type": "ABD", "program": "print('test')", "expected_output": "test"}]
        result = processed(["test completion"], extra_data=test_data)
        print(f"Test result: {result}")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()