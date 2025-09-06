#!/usr/bin/env python3

from validator.utils.affine_reward_functions import sat_reward_function

def debug_sat_original():
    """Debug what happens in the original SAT function"""
    print("=== Debugging Original SAT Function ===")
    
    # Test data - SAT problem (3-SAT with solution)
    test_data = [{
        "task_type": "SAT",
        "cnf_formula": "p cnf 3 3\n1 -2 3 0\n-1 2 0\n2 -3 0",
        "expected_satisfiable": True
    }]
    completion = "Looking at this 3-SAT problem, I need to find values that satisfy all clauses.\n\n<ASSIGNMENT>\nx1 = True\nx2 = True\nx3 = True\n</ASSIGNMENT>"
    
    print(f"Completion: {completion}")
    print(f"CNF: {test_data[0]['cnf_formula'].replace(chr(10), ' | ')}")
    print(f"Expected satisfiable: {test_data[0]['expected_satisfiable']}")
    
    # Test the function
    try:
        result = sat_reward_function([completion], extra_data=test_data)
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"❌ Error in original SAT function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_sat_original()