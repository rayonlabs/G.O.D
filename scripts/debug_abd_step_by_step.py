#!/usr/bin/env python3

from validator.utils.reward_functions import restricted_execution
import re
import json

def debug_abd_step_by_step():
    """Debug ABD function step by step"""
    print("=== Debugging ABD Step by Step ===")
    
    # Test data
    completion = "I need to find input that makes this program output 15. The program adds two numbers.\n<INPUT>\n5\n10\n</INPUT>"
    extra_data_item = {
        "task_type": "ABD",
        "program": "a = int(input())\nb = int(input())\nprint(a + b)",
        "expected_output": "15"
    }
    
    print(f"Completion: {completion}")
    print(f"Extra data: {extra_data_item}")
    
    # Step 1: Check task type
    task_type = extra_data_item.get("task_type", "")
    print(f"✅ Task type: {task_type}")
    
    if task_type.upper() != "ABD":
        print("❌ Wrong task type")
        return
    
    # Step 2: Get program and expected output
    program = extra_data_item.get("program", "")
    expected_output = extra_data_item.get("expected_output", "")
    print(f"✅ Program: {program}")
    print(f"✅ Expected output: {expected_output}")
    
    if not program:
        print("❌ No program")
        return
    
    # Step 3: Extract program from fence if present
    fence_pattern = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
    match = fence_pattern.search(program)
    if match:
        program = match.group(1).strip()
        print(f"✅ Extracted program from fence: {program}")
    else:
        print("✅ No fence in program, using as-is")
    
    # Step 4: Remove thinking tags from completion
    response = str(completion)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL)
    print(f"✅ Response after cleaning: {response}")
    
    # Step 5: Extract INPUT
    input_matches = re.findall(r"<INPUT>(.*?)</INPUT>", response, re.IGNORECASE | re.DOTALL)
    print(f"✅ Input matches found: {input_matches}")
    
    if not input_matches:
        if "<INPUT" in response.upper():
            print("❌ Found <INPUT but no matches - score 0.1")
            return 0.1
        else:
            print("❌ No <INPUT found - score 0.0")
            return 0.0
    
    # Step 6: Get the input
    generated_input = input_matches[-1].strip()
    lines = [ln.rstrip() for ln in generated_input.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    generated_input = "\n".join(lines)
    print(f"✅ Generated input: '{generated_input}'")
    
    # Step 7: Use restricted_execution
    print(f"✅ Running program: {program}")
    print(f"✅ With input: '{generated_input}'")
    
    output, error = restricted_execution(program, generated_input)
    print(f"✅ Execution output: '{output}'")
    print(f"✅ Execution error: '{error}'")
    
    if error:
        print("❌ Execution error - score 0.2")
        return 0.2
    
    # Step 8: Compare outputs
    output_clean = "\n".join(line.rstrip() for line in output.strip().splitlines())
    expected_clean = "\n".join(line.rstrip() for line in str(expected_output).strip().splitlines())
    
    print(f"✅ Output clean: '{output_clean}'")
    print(f"✅ Expected clean: '{expected_clean}'")
    
    if output_clean == expected_clean:
        print("🎉 Perfect match! Score = 1.0")
        return 1.0
    else:
        # Calculate similarity
        if not output_clean or not expected_clean:
            print("❌ Empty output - score 0.3")
            return 0.3
        else:
            matches = sum(c1 == c2 for c1, c2 in zip(output_clean, expected_clean))
            similarity = matches / max(len(output_clean), len(expected_clean))
            score = min(0.3 + (0.6 * similarity), 0.95)
            print(f"✅ Similarity: {similarity}, Score: {score}")
            return score


if __name__ == "__main__":
    result = debug_abd_step_by_step()
    print(f"\n🎯 Final result: {result}")