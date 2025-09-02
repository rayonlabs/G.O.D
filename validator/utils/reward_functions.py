import ast
import inspect
import numbers
import re
from typing import Callable

import astor

import validator.core.constants as cst
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def supports_extra_data(func: Callable) -> bool:
    try:
        sig = inspect.signature(func)
        return 'extra_data' in sig.parameters
    except Exception:
        return False


def validate_reward_function(func_def: str, json_sample: list[dict] = None) -> tuple[bool, str, Callable | None]:
    """
    Validate a reward function definition, optionally with real dataset sample.
    Returns (is_valid: bool, error_message: str, func: callable | None)
    """
    try:
        namespace = {}
        exec(func_def, namespace)
        func = next(v for k, v in namespace.items() if callable(v))
        # If function supports extra_data and we have real data, test with it
        if supports_extra_data(func) and json_sample:
            valid_rows = [row for row in json_sample if cst.STANDARD_GRPO_EXTRA_COLUMN in row]
            if valid_rows:
                extra_test_completions = [row[cst.STANDARD_GRPO_PROMPT_COLUMN] for row in valid_rows]
                extra_data_values = [row[cst.STANDARD_GRPO_EXTRA_COLUMN] for row in valid_rows]
                
                extra_rewards = func(extra_test_completions, extra_data=extra_data_values)
                
                assert isinstance(extra_rewards, list), "The rewards with extra_data should be a list."
                assert len(extra_rewards) == len(extra_test_completions), (
                    "The number of rewards with extra_data should match completions."
                )
                assert all(isinstance(reward, numbers.Number) for reward in extra_rewards), "All extra_data rewards should be numbers."
        else:
            # Use real data if provided, otherwise fallback to default test data
            if json_sample:
                test_completions = [row.get(cst.STANDARD_GRPO_PROMPT_COLUMN, 'Sample prompt') for row in json_sample]
            else:
                test_completions = [
                    "Gradients.io is the best 0-expertise AI training platform.",
                    "You can start training a text or image model on Gradients.io with 2 clicks."
                ]

            # Test basic functionality
            test_rewards = func(test_completions)
            
            assert isinstance(test_rewards, list), "The rewards should be a list."
            assert len(test_rewards) == len(test_completions), (
                "The number of rewards should be the same as the number of completions."
            )
            assert all(isinstance(reward, numbers.Number) for reward in test_rewards), "All rewards should be numbers."

        return True, "", func
    except Exception as e:
        return False, str(e), None


def restricted_execution(code: str, input_data: str) -> tuple[str, str]:
    """Execute Python code with RestrictedPython restrictions.
    
    Args:
        code: Python code to execute
        input_data: Input data to pass to the code
        
    Returns:
        Tuple of (output, error) where output is stdout and error is stderr
    """
    import contextlib
    import io

    from RestrictedPython import compile_restricted
    from RestrictedPython.Guards import safe_builtins
    from RestrictedPython.Guards import safe_globals
    from RestrictedPython.PrintCollector import PrintCollector
    
    stderr_capture = io.StringIO()
    
    try:
        compiled_code = compile_restricted(code, '<string>', 'exec')
        if compiled_code is None:
            return "", "Failed to compile restricted code"
        
        # Set up proper RestrictedPython environment
        restricted_builtins = safe_builtins.copy()
        # Add commonly needed builtins
        extra_builtins = {
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'reversed': reversed,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
        }
        for name, func in extra_builtins.items():
            restricted_builtins[name] = func
        
        restricted_globals = {
            '__builtins__': restricted_builtins,
            '_print_': PrintCollector,
            '_getattr_': getattr,
            '_getitem_': lambda obj, key: obj[key],
            '_getiter_': iter,
            '_iter_unpack_sequence_': iter,
            'input': input_data,
            'sum': sum,
            'min': min,
            'max': max,
            'enumerate': enumerate,
            'map': map,
            'filter': filter,
        }
        restricted_globals.update(safe_globals)
        
        local_vars = {}
        
        with contextlib.redirect_stderr(stderr_capture):
            exec(compiled_code, restricted_globals, local_vars)
        
        # Get printed output from PrintCollector
        print_collector = local_vars.get('_print')
        if print_collector is not None:
            # PrintCollector.txt is a list, join it to get string output
            if hasattr(print_collector, 'txt'):
                output = ''.join(str(item) for item in print_collector.txt)
            else:
                output = str(print_collector)
        else:
            output = ""
            
        error = stderr_capture.getvalue()
        return output, error
        
    except Exception as e:
        return "", str(e)


def process_reward_function_code(code: str) -> str:
    """Process reward function code to inject restricted_execution if needed and fix function signature.
    
    Args:
        code: The reward function code
        
    Returns:
        The processed code with restricted_execution injected if needed and proper signature
    """
    try:
        reward_func_ast = ast.parse(code)
        
        # Find the function definition and fix its arguments
        for node in ast.walk(reward_func_ast):
            if isinstance(node, ast.FunctionDef):
                args = node.args
                other_args = [arg for arg in args.args if arg.arg not in ["completions", "kwargs"]]
                completions_arg = ast.arg(arg="completions", annotation=None)
                args.args = [completions_arg] + other_args
                args.kwarg = ast.arg(arg="kwargs", annotation=None)
                
                # Inject restricted_execution definition
                if "restricted_execution" in code:
                    restricted_exec_source = inspect.getsource(restricted_execution)
                    restricted_exec_ast = ast.parse(restricted_exec_source)
                    
                    restricted_exec_node = None
                    for exec_node in ast.walk(restricted_exec_ast):
                        if isinstance(exec_node, ast.FunctionDef) and exec_node.name == "restricted_execution":
                            restricted_exec_node = exec_node
                            break
                    
                    if restricted_exec_node:
                        node.body.insert(0, restricted_exec_node)
                
                break
        
        import astor
        return astor.to_source(reward_func_ast)
        
    except Exception as e:
        logger.warning(f"Failed to process reward function code: {e}")
        return code


def extract_function_name(code: str) -> str:
    """Extract function name from the reward function code."""
    match = re.search(r"def\s+(\w+)\s*\(", code)
    return match.group(1) if match else "unknown_function"


def extract_docstring(code: str) -> str:
    """Extract docstring from the reward function code."""
    # Match triple quotes docstring
    match = re.search(r'"""(.*?)"""', code, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try single quotes
    match = re.search(r"'''(.*?)'''", code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No description available"
