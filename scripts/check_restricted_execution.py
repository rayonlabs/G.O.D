#!/usr/bin/env python3

from validator.utils.reward_functions import restricted_execution
import inspect

print("=== Checking restricted_execution function ===")
source = inspect.getsource(restricted_execution)

# Check if our fix is present
if "local_vars = restricted_globals.copy()" in source:
    print("✅ Fixed version is loaded")
else:
    print("❌ Old version still loaded")

# Show the relevant part
lines = source.split('\n')
for i, line in enumerate(lines):
    if 'local_vars' in line:
        print(f"\nLine {i}: {line.strip()}")
        if i+1 < len(lines):
            print(f"Line {i+1}: {lines[i+1].strip()}")