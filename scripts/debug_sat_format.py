#!/usr/bin/env python3

from validator.utils.affine_reward_functions import sat_reward_function

def debug_sat_format():
    """Debug SAT function format requirements"""
    print("=== Debugging SAT Format Requirements ===")
    
    # Test what the SAT function expects
    # Looking at the code, it expects "cls" (clauses) not "cnf_formula"
    
    # Convert CNF formula to clause list format
    cnf_formula = "p cnf 3 3\n1 -2 3 0\n-1 2 0\n2 -3 0"
    
    # Parse CNF to clause list
    clauses = []
    for line in cnf_formula.split('\n'):
        if line.startswith('p cnf'):
            continue
        if line.strip() and not line.startswith('c'):
            # Parse clause: "1 -2 3 0" -> [1, -2, 3]
            literals = [int(x) for x in line.split() if x != '0']
            if literals:
                clauses.append(literals)
    
    print(f"Original CNF: {cnf_formula}")
    print(f"Parsed clauses: {clauses}")
    
    # Test data with correct format
    correct_test_data = [{
        "task_type": "SAT",
        "cls": clauses,  # Use "cls" not "cnf_formula"
        "expected_satisfiable": True
    }]
    
    completion = "Looking at this 3-SAT problem:\n<ASSIGNMENT>\nx1 = True\nx2 = True\nx3 = True\n</ASSIGNMENT>"
    
    print(f"\nTesting with correct format...")
    print(f"Test data: {correct_test_data[0]}")
    print(f"Completion: {completion}")
    
    result = sat_reward_function([completion], extra_data=correct_test_data)
    print(f"Result: {result}")

if __name__ == "__main__":
    debug_sat_format()