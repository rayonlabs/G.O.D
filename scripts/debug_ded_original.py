#!/usr/bin/env python3

from validator.utils.affine_reward_functions import ded_reward_function

def debug_ded_original():
    """Debug what happens in the original DED function"""
    print("=== Debugging Original DED Function ===")
    
    # Test data - DED problem with perfect solution
    test_data = [{
        "task_type": "DED",
        "solution": "a = int(input())\nb = int(input())\nprint(a + b)",
        "premises": ["5\n10"]
    }]
    completion = "I need to write code that adds two numbers:\n\n```python\na = int(input())\nb = int(input())\nprint(a + b)\n```"
    
    print(f"Completion: {completion}")
    print(f"Solution: {test_data[0]['solution']}")
    print(f"Premises: {test_data[0]['premises']}")
    
    # Test the function
    try:
        result = ded_reward_function([completion], extra_data=test_data)
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"❌ Error in original DED function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ded_original()