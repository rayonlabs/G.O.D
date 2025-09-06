#!/usr/bin/env python3

import sys

modules_to_clear = [
    'validator.utils.reward_functions',
    'validator.utils.affine_reward_functions'
]

print("=== Clearing module cache ===")
for module in modules_to_clear:
    if module in sys.modules:
        del sys.modules[module]
        print(f"✅ Cleared {module}")
    else:
        print(f"ℹ️  {module} not in cache")

print("\n✅ Module cache cleared. Next import will reload fresh code.")