#!/usr/bin/env python3

from validator.utils.affine_reward_functions import sat_reward_function, abd_reward_function, ded_reward_function

def test_sat_function():
    """Test SAT function with manually solved example"""
    print("=== Testing SAT Reward Function ===")
    
    # From the actual data - SAT problem with known solution
    sat_extra_data = [{
        "task_type": "SAT", 
        "cls": [[10, 12, -1, 5, 7, 8, 15, -2, 13, 9], [12, -1, 7, 15, 9, -14, 8, -11, -5, 10]],  # Just first 2 clauses for simplicity
        "sol": {"1": True, "2": False, "3": True, "4": True, "5": True, "6": False, "7": True, "8": True, "9": False, "10": False, "11": False, "12": True, "13": True, "14": True, "15": True}
    }]
    
    # Test 1: Perfect solution (should get 1.0)
    perfect_completion = "x1=True, x2=False, x3=True, x4=True, x5=True, x6=False, x7=True, x8=True, x9=False, x10=False, x11=False, x12=True, x13=True, x14=True, x15=True"
    
    # Test 2: Partial solution (should get partial credit)
    partial_completion = "x1=True, x5=True, x8=True, x12=True"
    
    # Test 3: Wrong format (should get 0.0)
    wrong_completion = "This is not a valid assignment"
    
    completions = [perfect_completion, partial_completion, wrong_completion]
    result = sat_reward_function(completions, extra_data=sat_extra_data)
    
    print(f"Perfect solution result: {result[0]}")
    print(f"Partial solution result: {result[1]}")
    print(f"Wrong format result: {result[2]}")
    print()


def test_abd_function():
    """Test ABD function with manually solved example"""
    print("=== Testing ABD Reward Function ===")
    
    # Simple ABD example
    abd_extra_data = [{
        "task_type": "ABD",
        "program": "a = int(input())\nb = int(input())\nprint(a + b)",
        "expected_output": "15"
    }]
    
    # Test 1: Perfect solution (should get 1.0)
    perfect_completion = "Looking at this program, it reads two integers and adds them. To get output 15, I need:\n<INPUT>\n5\n10\n</INPUT>"
    
    # Test 2: Alternative perfect solution
    alternative_completion = "The program adds two numbers. For output 15:\n<INPUT>\n7\n8\n</INPUT>"
    
    # Test 3: Wrong input (should get partial credit for format but wrong result)
    wrong_input_completion = "The program adds two numbers:\n<INPUT>\n1\n2\n</INPUT>"
    
    # Test 4: No INPUT tags (should get 0.0)
    no_input_completion = "The program adds two numbers. The answer is 5 and 10."
    
    completions = [perfect_completion, alternative_completion, wrong_input_completion, no_input_completion]
    result = abd_reward_function(completions, extra_data=abd_extra_data)
    
    print(f"Perfect solution result: {result[0]}")
    print(f"Alternative perfect result: {result[1]}")
    print(f"Wrong input result: {result[2]}")
    print(f"No INPUT tags result: {result[3]}")
    print()


def test_ded_function():
    """Test DED function with manually solved example"""
    print("=== Testing DED Reward Function ===")
    
    # Simple DED example
    # The DED function will run the solution with premises[0] as input to get expected output
    # If premises[0] = "5", and solution reads input() and doubles it, expected output = "10"
    ded_extra_data = [{
        "task_type": "DED",
        "solution": "n = int(input())\nprint(n * 2)",
        "premises": ["5"]  # This input will be fed to both solution and submitted code
    }]
    
    # Test 1: Perfect solution (should get 1.0)
    perfect_completion = "Looking at this problem, I need to double the input:\n```python\nn = int(input())\nprint(n * 2)\n```"
    
    # Test 2: Alternative correct solution (should get 1.0)
    alternative_completion = "The solution is:\n```python\nn = int(input())\nprint(n + n)\n```"
    
    # Test 3: Wrong logic but valid syntax (should get partial credit)
    wrong_logic_completion = "Here's my solution:\n```python\nn = int(input())\nprint(n + 1)\n```"
    
    # Test 4: Syntax error (should get lower credit)
    syntax_error_completion = "My solution:\n```python\nn = int(input(\nprint(n * 2)\n```"
    
    # Test 5: No code block (should get 0.0)
    no_code_completion = "The answer is to multiply by 2"
    
    completions = [perfect_completion, alternative_completion, wrong_logic_completion, syntax_error_completion, no_code_completion]
    result = ded_reward_function(completions, extra_data=ded_extra_data)
    
    print(f"Perfect solution result: {result[0]}")
    print(f"Alternative correct result: {result[1]}")
    print(f"Wrong logic result: {result[2]}")
    print(f"Syntax error result: {result[3]}")
    print(f"No code block result: {result[4]}")
    print()


def test_complex_ded_example():
    """Test DED with a more complex example to really verify execution"""
    print("=== Testing Complex DED Example ===")
    
    # More complex example with multiple inputs
    # premises[0] = "3\n2\n3\n4" means: first input()="3", second="2", third="3", fourth="4"
    # solution: read n=3, then loop 3 times reading 2,3,4 and print their squares: 4,9,16
    complex_ded_extra_data = [{
        "task_type": "DED", 
        "solution": "n = int(input())\nfor i in range(n):\n    x = int(input())\n    print(x * x)",
        "premises": ["3\n2\n3\n4"]  # Expected output will be "4\n9\n16"
    }]
    
    # Perfect solution
    perfect_completion = """Looking at this, it reads n numbers and prints their squares:
```python
n = int(input())
for i in range(n):
    x = int(input())
    print(x * x)
```"""
    
    # Alternative solution (different style but same logic)
    alternative_completion = """Here's the solution:
```python
n = int(input())
for _ in range(n):
    x = int(input())
    print(x**2)
```"""
    
    # Wrong solution
    wrong_completion = """My solution:
```python
n = int(input())
for i in range(n):
    x = int(input())
    print(x)
```"""
    
    completions = [perfect_completion, alternative_completion, wrong_completion]
    result = ded_reward_function(completions, extra_data=complex_ded_extra_data)
    
    print(f"Perfect complex solution result: {result[0]}")
    print(f"Alternative complex solution result: {result[1]}")
    print(f"Wrong complex solution result: {result[2]}")
    print()


def debug_restricted_execution():
    """Debug what restricted_execution actually produces"""
    print("=== Debugging Restricted Execution ===")
    
    from validator.utils.reward_functions import restricted_execution
    
    # Test simple example
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
    print(f"Simple example - Output: '{output}', Error: '{error}'")
    
    # Test complex example  
    complex_code = """
input_lines = ['3', '2', '3', '4']
input_index = 0

def input(prompt=""):
    global input_index
    if input_index < len(input_lines):
        result = input_lines[input_index]
        input_index += 1
        return result
    return ""

n = int(input())
for i in range(n):
    x = int(input())
    print(x * x)
"""
    
    output, error = restricted_execution(complex_code, "3\n2\n3\n4")
    print(f"Complex example - Output: '{output}', Error: '{error}'")
    print()


if __name__ == "__main__":
    print("🧪 Testing Affine Reward Functions with Manual Solutions")
    print("=" * 60)
    
    debug_restricted_execution()
    test_sat_function()
    test_abd_function() 
    test_ded_function()
    test_complex_ded_example()
    
    print("✅ All tests completed!")
    print("\nIf you see non-zero scores for correct solutions and zero scores for wrong ones,")
    print("then the reward functions are working correctly and the issue is with model completions")
    print("not being in the expected format during GRPO evaluation.")