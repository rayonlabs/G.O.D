#!/usr/bin/env python3

from validator.utils.reward_functions import process_reward_function_code
from validator.utils.affine_reward_functions import abd_reward_function
import inspect

def create_debug_abd():
    """Create ABD function with debug prints"""
    
    # Get the processed code
    original_code = inspect.getsource(abd_reward_function)
    processed_code = process_reward_function_code(original_code)
    
    # Add simple debug print at function start
    debug_code = processed_code.replace(
        'scores = []',
        '''print("🔍 PROCESSED DEBUG: Function started, about to process completions")
    scores = []'''
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