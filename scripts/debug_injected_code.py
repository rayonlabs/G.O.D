#!/usr/bin/env python3

from validator.utils.reward_functions import process_reward_function_code
from validator.utils.affine_reward_functions import abd_reward_function
import inspect

def debug_injected_code():
    """Debug the injected restricted_execution code"""
    print("=== Debugging Injected Code ===")
    
    # Get the processed code
    original_code = inspect.getsource(abd_reward_function)
    processed_code = process_reward_function_code(original_code)
    
    # Find the restricted_execution function in the processed code
    lines = processed_code.split('\n')
    in_restricted_execution = False
    restricted_lines = []
    
    for line in lines:
        if 'def restricted_execution(' in line:
            in_restricted_execution = True
            restricted_lines.append(line)
        elif in_restricted_execution:
            if line.startswith('def ') and 'restricted_execution' not in line:
                # Found another function, stop
                break
            restricted_lines.append(line)
    
    injected_restricted_execution = '\n'.join(restricted_lines)
    print("=== Injected restricted_execution function ===")
    print(injected_restricted_execution[:1000] + "..." if len(injected_restricted_execution) > 1000 else injected_restricted_execution)
    
    # Check key differences
    if 'create_input_func' in injected_restricted_execution:
        print("✅ create_input_func found in injected version")
    else:
        print("❌ create_input_func NOT found in injected version")
    
    if "'input': create_input_func(input_lines)" in injected_restricted_execution:
        print("✅ input function setup found in injected version")
    else:
        print("❌ input function setup NOT found in injected version")


if __name__ == "__main__":
    debug_injected_code()