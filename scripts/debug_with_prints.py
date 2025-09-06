#!/usr/bin/env python3

from validator.utils.reward_functions import process_reward_function_code
from validator.utils.affine_reward_functions import abd_reward_function
import inspect

def create_debug_abd():
    """Create ABD function with debug prints"""
    
    # Get the processed code
    original_code = inspect.getsource(abd_reward_function)
    processed_code = process_reward_function_code(original_code)
    
    # Add debug prints to key locations
    debug_code = processed_code.replace(
        'for completion, extra_data_item in zip(completions, extra_data_list):',
        '''for completion, extra_data_item in zip(completions, extra_data_list):
        print(f"🔍 PROCESSED DEBUG: Processing completion: {str(completion)[:50]}...")
        print(f"🔍 PROCESSED DEBUG: extra_data_item: {extra_data_item}")'''
    )
    
    debug_code = debug_code.replace(
        'if extra_data_item.get("task_type", "").upper() != "ABD":',
        '''task_type = extra_data_item.get("task_type", "")
        print(f"🔍 PROCESSED DEBUG: task_type: {task_type}")
        if task_type.upper() != "ABD":
            print(f"🔍 PROCESSED DEBUG: Wrong task type, expected ABD, got {task_type}")'''
    )
    
    debug_code = debug_code.replace(
        'output, error = restricted_execution(program, generated_input)',
        '''print(f"🔍 PROCESSED DEBUG: About to call restricted_execution")
        print(f"🔍 PROCESSED DEBUG: program: {program}")
        print(f"🔍 PROCESSED DEBUG: generated_input: {generated_input}")
        output, error = restricted_execution(program, generated_input)
        print(f"🔍 PROCESSED DEBUG: output: {output}")
        print(f"🔍 PROCESSED DEBUG: error: {error}")'''
    )
    
    debug_code = debug_code.replace(
        'scores.append(1.0)',
        '''print(f"🔍 PROCESSED DEBUG: Perfect match! Appending 1.0")
        scores.append(1.0)'''
    )
    
    debug_code = debug_code.replace(
        'scores.append(0.0)',
        '''print(f"🔍 PROCESSED DEBUG: Appending 0.0")
        scores.append(0.0)'''
    )
    
    # Create the function
    exec(debug_code, globals())
    return abd_reward_function


def test_debug_abd():
    """Test the debug version"""
    print("=== Testing Debug ABD Function ===")
    
    debug_abd_func = create_debug_abd()
    
    test_data = [{
        "task_type": "ABD",
        "program": "a = int(input())\nb = int(input())\nprint(a + b)",
        "expected_output": "15"
    }]
    completion = "I need to find input that makes this program output 15. The program adds two numbers.\n<INPUT>\n5\n10\n</INPUT>"
    
    print("Calling debug ABD function...")
    result = debug_abd_func([completion], extra_data=test_data)
    print(f"Final result: {result}")


if __name__ == "__main__":
    test_debug_abd()