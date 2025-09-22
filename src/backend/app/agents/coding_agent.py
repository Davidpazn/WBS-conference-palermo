"""Main coding agent implementation."""

from typing import Dict, Any
from ..llm.openai_client import llm_generate_code
from ..e2b.execution import run_in_e2b, failed


def coding_agent(task: str) -> str:
    """
    Simple coding agent that generates code for a given task.

    Args:
        task: Description of what code to generate

    Returns:
        Generated code as a string
    """
    return llm_generate_code(task)