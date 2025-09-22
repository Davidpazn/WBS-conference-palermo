"""E2B code execution functions."""

from typing import Dict, Any, Optional
from e2b_code_interpreter import Sandbox


def run_in_e2b(code: str) -> dict:
    """Create a temporary sandbox, run the code, return logs."""
    # Sandbox is auto-terminated after leaving the context
    with Sandbox.create() as sbx:
        exec_result = sbx.run_code(code)
        return summarize_execution(exec_result)


def summarize_execution(execution) -> dict:
    """Best-effort extraction of stdout/stderr/text depending on SDK version."""
    # Handle None result
    if execution is None:
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "results": [],
            "error": None
        }

    out = {
        "success": True,
        "stdout": "",
        "stderr": "",
        "results": [],
        "error": None
    }

    # Handle E2B Code Interpreter SDK Execution objects
    if hasattr(execution, 'logs'):
        logs = execution.logs
        if hasattr(logs, 'stdout'):
            stdout = logs.stdout
            if isinstance(stdout, list):
                out["stdout"] = '\n'.join(str(line) for line in stdout)
            else:
                out["stdout"] = str(stdout) if stdout else ""

        if hasattr(logs, 'stderr'):
            stderr = logs.stderr
            if isinstance(stderr, list):
                out["stderr"] = '\n'.join(str(line) for line in stderr)
            else:
                out["stderr"] = str(stderr) if stderr else ""

    # Check for results
    if hasattr(execution, 'results'):
        results = execution.results
        if isinstance(results, list):
            out["results"] = [getattr(r, 'text', str(r)) for r in results]
        else:
            out["results"] = [str(results)] if results else []

    # Check for error
    if hasattr(execution, 'error') and execution.error:
        out["error"] = execution.error
        out["success"] = False

    # Check if there's an error in stderr
    if out["stderr"] and out["stderr"].strip():
        out["success"] = False

    return out


def extract_output_from_execution(execution) -> str:
    """Extract text output from E2B execution result."""
    if execution is None:
        return ""

    # The E2B SDK returns Execution objects with logs.stdout as a list
    if hasattr(execution, 'logs') and hasattr(execution.logs, 'stdout'):
        stdout_list = execution.logs.stdout
        if isinstance(stdout_list, list):
            return '\n'.join(stdout_list)
        else:
            return str(stdout_list) if stdout_list else ""

    # Fallback checks
    if hasattr(execution, 'text') and execution.text:
        return execution.text

    if hasattr(execution, 'stdout'):
        stdout = execution.stdout
        if isinstance(stdout, list):
            return '\n'.join(stdout)
        else:
            return str(stdout) if stdout else ""

    return str(execution) if execution else ""


def failed(exe_summary: dict) -> bool:
    """Check if execution failed based on stderr content."""
    stderr = exe_summary.get("stderr")
    if stderr and str(stderr).strip():
        return True
    # fallbacks for older SDKs
    text = exe_summary.get("text")
    if isinstance(text, str) and any(tok in text.lower() for tok in ("traceback", "error", "exception")):
        return True
    return False