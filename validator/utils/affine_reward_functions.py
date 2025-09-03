"""
Affine-style reward functions for GRPO tasks.
"""

from .reward_functions import restricted_execution


def affine_reward_function(completions, extra_data=None, **kwargs):
    """
    Single reward function for affine tasks that routes to appropriate handler based on task_type.

    Args:
        completions: List of model responses to evaluate
        extra_data: Dictionary containing task-specific data and task_type field
        **kwargs: Additional keyword arguments

    Returns:
        List of scores (0.0 or 1.0) for each completion
    """

    def handle_abd(completions, extra_data, **kwargs):
        """Handle Algorithmic Backward Design (ABD) problems."""
        import re

        scores = []

        for completion in completions:
            try:
                program = extra_data["program"]
                expected_output = extra_data["expected_output"]

                fence_pattern = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
                match = fence_pattern.search(program)
                if match:
                    program = match.group(1).strip()
                else:
                    program = program.strip()

                response = re.sub(r"<think>.*?</thinking>", "", completion, flags=re.DOTALL)
                response = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL)

                matches = re.findall(r"<INPUT>(.*?)</INPUT>", response, re.IGNORECASE | re.DOTALL)

                if not matches:
                    scores.append(0.0)
                    continue

                lines = [ln.rstrip() for ln in matches[-1].strip().splitlines()]
                while lines and not lines[-1].strip():
                    lines.pop()

                generated_input = "\n".join(lines)

                if not generated_input:
                    scores.append(0.0)
                    continue

                execution_code = f'''
# Define input() function to simulate stdin
input_lines = "{generated_input}".split("\\n")

def input():
    if input_lines:
        return input_lines.pop(0)
    return ""

# Execute the program
{program}
'''

                output, error = restricted_execution(execution_code, input_data=generated_input)

                if error:
                    scores.append(0.0)
                    continue

                if expected_output == output:
                    outputs_match = True
                else:
                    exp = expected_output.strip().replace("\r\n", "\n")
                    act = output.strip().replace("\r\n", "\n")

                    if exp == act:
                        outputs_match = True
                    else:
                        exp_lines = [l.rstrip() for l in exp.splitlines()]
                        act_lines = [l.rstrip() for l in act.splitlines()]
                        outputs_match = exp_lines == act_lines

                scores.append(1.0 if outputs_match else 0.0)

            except Exception:
                scores.append(0.0)

        return scores

    def handle_sat(completions, extra_data, **kwargs):
        """Handle SAT (Boolean Satisfiability) problems."""
        import re

        scores = []

        for completion in completions:
            try:
                sol = extra_data["sol"]
                cls = extra_data["cls"]

                got = {int(v): val.lower() in ("true", "1") for v, val in re.findall(r"x(\d+)=(True|False|1|0)", completion)}

                ok = all(any((lit > 0) == got.get(abs(lit), None) for lit in c) for c in cls)

                scores.append(float(ok))

            except Exception:
                scores.append(0.0)

        return scores

    def handle_ded(completions, extra_data, **kwargs):
        """Handle DED (Deduction) problems."""
        import ast
        import json
        import re

        scores = []

        for completion in completions:
            try:
                fence_pattern = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
                match = fence_pattern.search(completion)
                if not match:
                    scores.append(0.0)
                    continue

                program = match.group(1).strip()

                sample = extra_data
                ver_raw = sample.get("verification_info") or sample.get("test_cases")

                if not ver_raw:
                    scores.append(0.0)
                    continue

                try:
                    if isinstance(ver_raw, str):
                        try:
                            ver_json = json.loads(ver_raw)
                        except json.JSONDecodeError:
                            ver_json = ast.literal_eval(ver_raw)
                    else:
                        ver_json = ver_raw
                except Exception:
                    scores.append(0.0)
                    continue

                cases = ver_json.get("test_cases")
                if not cases:
                    scores.append(0.0)
                    continue

                def _to_str(x):
                    if isinstance(x, str):
                        return x
                    if isinstance(x, (bytes, bytearray)):
                        return x.decode()
                    if isinstance(x, list):
                        return "\n".join(_to_str(e) for e in x)
                    return json.dumps(x, ensure_ascii=False)

                def _normalize(text):
                    return "\n".join(line.rstrip() for line in text.rstrip().splitlines())

                passed, total = 0, len(cases)

                for case in cases:
                    ctype = case.get("type")
                    raw_inp = case.get("input")
                    raw_exp = case.get("output")

                    if ctype == "stdin_stdout":
                        inp = _to_str(raw_inp)
                        if not inp.endswith("\n"):
                            inp += "\n"
                        exec_prog = program
                        exp = _to_str(raw_exp)
                    elif ctype == "function_call":
                        fn = case.get("fn_name")
                        args = case.get("input", [])
                        exec_prog = (
                            program
                            + "\n"
                            + "if __name__ == '__main__':\n"
                            + f"    result = {fn}(*{args!r})\n"
                            + "    print(result)"
                        )
                        inp = ""
                        exp = _to_str(raw_exp[0]) if isinstance(raw_exp, list) and raw_exp else _to_str(raw_exp)
                    else:
                        total -= 1
                        continue

                    execution_code = f'''
# Define input() function to simulate stdin
input_lines = "{inp}".split("\\n")

def input():
    if input_lines:
        return input_lines.pop(0)
    return ""

# Execute the program
{exec_prog}
'''

                    output, error = restricted_execution(execution_code, input_data=inp)

                    ok_run = not error.strip()
                    out_norm = _normalize(output)
                    exp_norm = _normalize(exp) if exp is not None else None
                    correct = ok_run and (exp_norm is None or out_norm == exp_norm)

                    if correct:
                        passed += 1

                score = 1.0 if passed == total else 0.0
                scores.append(score)

            except Exception:
                scores.append(0.0)

        return scores

    if not extra_data or "task_type" not in extra_data:
        raise ValueError("task_type field is required in extra_data")

    task_type = extra_data["task_type"].upper()

    if task_type == "ABD":
        return handle_abd(completions, extra_data, **kwargs)
    elif task_type == "SAT":
        return handle_sat(completions, extra_data, **kwargs)
    elif task_type == "DED":
        return handle_ded(completions, extra_data, **kwargs)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}. Must be one of: ABD, SAT, DED")
