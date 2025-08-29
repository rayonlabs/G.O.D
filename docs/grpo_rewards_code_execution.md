# GRPO Reward Functions & Safe Code Execution

## Using `restricted_execution` for Code-Based Rewards

When developing GRPO reward functions that need to execute user-provided code, **you must use the `restricted_execution` utility function**. This is a security requirement - reward functions that execute code without this wrapper will be rejected.

### Function Signature
```python
def restricted_execution(code: str, input_data: str) -> tuple[str, str]:
    """
    Returns:
        tuple[str, str]: (output, error) where:
        - output: All printed content with newlines preserved
        - error: Error messages if execution fails, empty string if successful
    """
```

### Key Features
- **Security**: Uses RestrictedPython to prevent dangerous operations (file I/O, imports, system calls)
- **Output Capture**: Captures all `print()` statements as the primary output mechanism
- **Error Handling**: Returns errors as strings rather than raising exceptions
- **Built-in Functions**: Provides common functions like `sum`, `min`, `max`, `enumerate`, etc.

### Output Format
- **Success**: `output` contains all printed content, `error` is empty
- **Failure**: `output` is empty, `error` contains the error message
- **Newlines**: Each `print()` call automatically adds a newline (`\n`)

### Usage Example
```python
def my_reward_function(completions, extra_data=None, **kwargs):
    scores = []
    
    for response in completions:
        # Extract code from response
        user_code = extract_code_from_response(response)
        expected_output = extra_data.get('expected_output', '')
        
        # Execute safely
        output, error = restricted_execution(user_code, input_data='')
        
        # Score based on correctness
        if not error and output.strip() == expected_output.strip():
            scores.append(1.0)
        else:
            scores.append(0.0)
    
    return scores
```

### Security Restrictions
The following operations are **blocked** for security:
- File system access (`open`, file operations)
- Network requests (`urllib`, `requests`, etc.), additionnally reward functions are ran in a containerized environment with no network access
- System commands (`os`, `subprocess`, etc.)  
- Module imports (`import`, `__import__`)
- Dangerous built-ins (`eval`, `exec`, `globals`, `locals`)

### Available Built-ins
Common functions are available: `sum`, `min`, `max`, `abs`, `round`, `sorted`, `len`, `str`, `int`, `float`, `list`, `dict`, `range`, `enumerate`, `zip`, `map`, `filter`