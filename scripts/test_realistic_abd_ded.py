#!/usr/bin/env python3

from validator.utils.reward_functions import validate_reward_function
from validator.utils.affine_reward_functions import abd_reward_function, ded_reward_function
import inspect

def test_abd_realistic():
    """Test ABD with realistic data and completion"""
    print("=== Testing ABD with realistic data ===")
    
    # Get processed function
    abd_code = inspect.getsource(abd_reward_function)
    is_valid, error, processed_abd = validate_reward_function(abd_code, [])
    
    if not is_valid:
        print(f"❌ ABD processing failed: {error}")
        return
        
    print("✅ ABD processed successfully")
    
    # Realistic ABD data - simple addition program
    abd_data = [{
        "task_type": "ABD",
        "program": "a = int(input())\nb = int(input())\nprint(a + b)",
        "expected_output": "15"
    }]
    
    # Test cases
    test_cases = [
        # Perfect solution
        ("Perfect", "I need to find input that makes this program output 15. The program adds two numbers.\n<INPUT>\n5\n10\n</INPUT>"),
        
        # Alternative perfect solution  
        ("Alternative", "This adds two inputs. For output 15:\n<INPUT>\n7\n8\n</INPUT>"),
        
        # Wrong input
        ("Wrong", "Let me try:\n<INPUT>\n1\n2\n</INPUT>"),
        
        # No INPUT tags
        ("No tags", "The answer is 5 and 10")
    ]
    
    for name, completion in test_cases:
        print(f"\n--- {name} ---")
        print(f"Completion: {completion[:50]}...")
        result = processed_abd([completion], extra_data=abd_data)
        print(f"Result: {result[0]}")


def test_ded_realistic():
    """Test DED with realistic data and completion"""
    print("\n=== Testing DED with realistic data ===")
    
    # Get processed function
    ded_code = inspect.getsource(ded_reward_function)
    is_valid, error, processed_ded = validate_reward_function(ded_code, [])
    
    if not is_valid:
        print(f"❌ DED processing failed: {error}")
        return
        
    print("✅ DED processed successfully")
    
    # Realistic DED data - simple doubling program
    ded_data = [{
        "task_type": "DED",
        "solution": "n = int(input())\nprint(n * 2)",
        "premises": ["5"]
    }]
    
    # Test cases
    test_cases = [
        # Perfect solution
        ("Perfect", "I need to write code that doubles the input:\n```python\nn = int(input())\nprint(n * 2)\n```"),
        
        # Alternative correct solution
        ("Alternative", "Here's my solution:\n```python\nn = int(input())\nprint(n + n)\n```"),
        
        # Wrong logic
        ("Wrong", "My code:\n```python\nn = int(input())\nprint(n + 1)\n```"),
        
        # No code block
        ("No code", "Just multiply by 2")
    ]
    
    for name, completion in test_cases:
        print(f"\n--- {name} ---")
        print(f"Completion: {completion[:50]}...")
        result = processed_ded([completion], extra_data=ded_data)
        print(f"Result: {result[0]}")


if __name__ == "__main__":
    test_abd_realistic()
    test_ded_realistic()