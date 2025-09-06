#!/usr/bin/env python3

def simple_restricted_execution(code: str, input_data: str) -> tuple[str, str]:
    """Simple restricted execution that works with RestrictedPython"""
    import contextlib
    import io
    from RestrictedPython import compile_restricted
    from RestrictedPython.Guards import safe_builtins, safe_globals
    from RestrictedPython.PrintCollector import PrintCollector
    
    stderr_capture = io.StringIO()
    
    try:
        compiled_code = compile_restricted(code, '<string>', 'exec')
        if compiled_code is None:
            return "", "Failed to compile restricted code"
        
        # Parse input into lines
        input_lines = input_data.split('\n') if input_data else []
        
        # Create input function
        def create_input_func(lines):
            lines_iter = iter(lines)
            def input_func(prompt=""):
                try:
                    return next(lines_iter)
                except StopIteration:
                    return ""
            return input_func
        
        # Set up environment
        restricted_globals = {
            '__builtins__': safe_builtins,
            '_print_': PrintCollector,
            '_getattr_': getattr,
            '_getitem_': lambda obj, key: obj[key],
            '_getiter_': iter,
            'input': create_input_func(input_lines),
            'int': int,
            'str': str,
            'len': len,
            'range': range,
            'print': lambda *args: None,  # Capture print calls
        }
        restricted_globals.update(safe_globals)
        
        local_vars = {}
        
        with contextlib.redirect_stderr(stderr_capture):
            exec(compiled_code, restricted_globals, local_vars)
        
        # Get output
        print_collector = local_vars.get('_print')
        if print_collector and hasattr(print_collector, 'txt'):
            output = '\n'.join(str(item) for item in print_collector.txt)
        else:
            output = ""
            
        error = stderr_capture.getvalue()
        return output, error
        
    except Exception as e:
        return "", str(e)


def ded_reward_function_standalone(completions, extra_data=None, **kwargs):
    """Standalone DED reward function that works with RestrictedPython"""
    import re
    import json
    
    # Handle extra_data parameter  
    extra_data_list = extra_data if extra_data is not None else kwargs.get('extra_data', [])

    if not extra_data_list:
        return [0.0] * len(completions)

    # Handle both single dict and list of dicts
    if isinstance(extra_data_list, dict):
        extra_data_list = [extra_data_list] * len(completions)
    elif len(extra_data_list) == 1 and len(completions) > 1:
        extra_data_list = extra_data_list * len(completions)

    scores = []

    for i, (completion, extra_data_item) in enumerate(zip(completions, extra_data_list)):
        print(f"🔍 DED DEBUG: Processing completion {i}")
        print(f"🔍 DED DEBUG: extra_data_item: {extra_data_item}")
        print(f"🔍 DED DEBUG: completion preview: {str(completion)[:100]}...")
        
        # Handle JSON string extra_data
        if isinstance(extra_data_item, str):
            try:
                extra_data_item = json.loads(extra_data_item)
                print(f"🔍 DED DEBUG: Parsed JSON: {extra_data_item}")
            except json.JSONDecodeError as e:
                print(f"🔍 DED DEBUG: JSON decode error: {e}")
                scores.append(0.0)
                continue
        
        # Validate extra_data
        if not isinstance(extra_data_item, dict):
            print("🔍 DED DEBUG: extra_data_item is not a dict")
            scores.append(0.0)
            continue

        # Check task type
        task_type = extra_data_item.get("task_type", "")
        print(f"🔍 DED DEBUG: task_type: {task_type}")
        if task_type.upper() != "DED":
            print(f"🔍 DED DEBUG: Wrong task type, expected DED, got {task_type}")
            scores.append(0.0)
            continue

        solution = extra_data_item.get("solution", "")
        premises = extra_data_item.get("premises", [])
        print(f"🔍 DED DEBUG: solution: {solution[:100]}...")
        print(f"🔍 DED DEBUG: premises: {premises}")

        if not solution:
            print("🔍 DED DEBUG: No solution provided")
            scores.append(0.0)
            continue

        # Evaluate DED
        try:
            # Extract code from fence in completion
            fence_pattern = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
            match = fence_pattern.search(str(completion))

            if not match:
                print("🔍 DED DEBUG: No code block found in completion")
                # Small credit for code-like content
                if any(keyword in str(completion) for keyword in ["def ", "print", "input", "return"]):
                    print("🔍 DED DEBUG: Found code-like content, giving 0.1")
                    scores.append(0.1)
                else:
                    scores.append(0.0)
                continue

            submitted_code = match.group(1).strip()
            print(f"🔍 DED DEBUG: Extracted submitted code: {submitted_code}")

            # Check syntax validity
            try:
                compile(submitted_code, '<string>', 'exec')
                print("🔍 DED DEBUG: Submitted code has valid syntax")
            except Exception as syntax_error:
                print(f"🔍 DED DEBUG: Invalid syntax: {syntax_error}")
                scores.append(0.2)  # Invalid syntax
                continue

            # Extract solution code if fenced
            sol_match = fence_pattern.search(solution)
            if sol_match:
                solution = sol_match.group(1).strip()
            print(f"🔍 DED DEBUG: Solution code: {solution}")

            if not premises or not isinstance(premises, list):
                print("🔍 DED DEBUG: No valid premises")
                scores.append(0.3)  # Valid syntax but can't test
                continue

            # Get test input
            test_input = str(premises[0]) if premises else ""
            print(f"🔍 DED DEBUG: test_input: {test_input}")

            # Execute expected solution
            print("🔍 DED DEBUG: Executing expected solution...")
            expected_output, expected_error = simple_restricted_execution(solution, test_input)
            print(f"🔍 DED DEBUG: Expected output: '{expected_output}'")
            print(f"🔍 DED DEBUG: Expected error: '{expected_error}'")

            if expected_error:
                print("🔍 DED DEBUG: Error in expected solution")
                scores.append(0.35)  # Can't get expected output
                continue

            # Execute submitted code
            print("🔍 DED DEBUG: Executing submitted solution...")
            actual_output, actual_error = simple_restricted_execution(submitted_code, test_input)
            print(f"🔍 DED DEBUG: Actual output: '{actual_output}'")
            print(f"🔍 DED DEBUG: Actual error: '{actual_error}'")

            if actual_error:
                print("🔍 DED DEBUG: Error in submitted solution")
                scores.append(0.4)  # Valid syntax but runtime error
                continue

            # Compare outputs
            expected_clean = "\n".join(line.rstrip() for line in expected_output.strip().splitlines())
            actual_clean = "\n".join(line.rstrip() for line in actual_output.strip().splitlines())
            
            print(f"🔍 DED DEBUG: Expected clean: '{expected_clean}'")
            print(f"🔍 DED DEBUG: Actual clean: '{actual_clean}'")

            if expected_clean == actual_clean:
                print("🔍 DED DEBUG: Perfect match! Score = 1.0")
                scores.append(1.0)
            elif expected_clean in actual_clean or actual_clean in expected_clean:
                print("🔍 DED DEBUG: Partial match, Score = 0.8")
                scores.append(0.8)
            else:
                print("🔍 DED DEBUG: Different outputs, Score = 0.5")
                scores.append(0.5)  # Successful execution but wrong output

        except Exception as e:
            print(f"🔍 DED DEBUG: Exception in evaluation: {e}")
            import traceback
            traceback.print_exc()
            scores.append(0.0)

    return scores


def test_ded_function():
    """Test the standalone DED function"""
    print("=== Testing Standalone DED Reward Function ===")
    
    # Simple test case
    ded_extra_data = [{
        "task_type": "DED",
        "solution": "n = int(input())\nprint(n * 2)",
        "premises": ["5"]
    }]
    
    # Test cases
    test_cases = [
        # Perfect solution
        ("Perfect", "Looking at this problem, I need to double the input:\n```python\nn = int(input())\nprint(n * 2)\n```"),
        
        # Alternative correct solution
        ("Alternative", "The solution is:\n```python\nn = int(input())\nprint(n + n)\n```"),
        
        # Wrong logic
        ("Wrong logic", "Here's my solution:\n```python\nn = int(input())\nprint(n + 1)\n```"),
        
        # No code block
        ("No code", "The answer is to multiply by 2")
    ]
    
    for name, completion in test_cases:
        print(f"\n--- Testing {name} ---")
        result = ded_reward_function_standalone([completion], extra_data=ded_extra_data)
        print(f"Result: {result[0]}")


if __name__ == "__main__":
    test_ded_function()