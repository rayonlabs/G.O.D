#!/usr/bin/env python3

from validator.utils.affine_reward_functions import abd_reward_function
import re

def debug_original_function():
    """Debug what happens in the original function"""
    print("=== Debugging Original ABD Function ===")
    
    # Test data - same as working manual test
    test_data = [{
        "task_type": "ABD",
        "program": "a = int(input())\nb = int(input())\nprint(a + b)",
        "expected_output": "15"
    }]
    completion = "I need to find input that makes this program output 15. The program adds two numbers.\n<INPUT>\n5\n10\n</INPUT>"
    
    print(f"Completion: {completion}")
    print(f"Test data: {test_data[0]}")
    
    # Test the function with detailed error catching
    try:
        result = abd_reward_function([completion], extra_data=test_data)
        print(f"Result: {result}")
        
        # Also test what restricted_execution from the original function produces
        print("\n=== Testing Original restricted_execution directly ===")
        from validator.utils.reward_functions import restricted_execution
        
        program = "a = int(input())\nb = int(input())\nprint(a + b)"
        generated_input = "5\n10"
        
        print(f"Program: {program}")
        print(f"Input: '{generated_input}'")
        
        output, error = restricted_execution(program, generated_input)
        print(f"Output: '{output}'")
        print(f"Error: '{error}'")
        
        if error:
            print("❌ Original restricted_execution has error")
        else:
            print("✅ Original restricted_execution works")
            
            # Compare output
            expected = "15"
            output_clean = "\n".join(line.rstrip() for line in output.strip().splitlines())
            expected_clean = "\n".join(line.rstrip() for line in str(expected).strip().splitlines())
            
            print(f"Output clean: '{output_clean}'")
            print(f"Expected clean: '{expected_clean}'")
            print(f"Match: {output_clean == expected_clean}")
        
    except Exception as e:
        print(f"❌ Error in original function: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_original_function()