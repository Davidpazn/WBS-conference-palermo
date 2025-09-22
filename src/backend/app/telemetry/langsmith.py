"""LangSmith integration utilities."""

import os
from typing import Optional, Dict, Any
from langsmith import Client as LangSmithClient


def get_langsmith_client() -> Optional[LangSmithClient]:
    """Get LangSmith client if API key is available."""
    api_key = os.getenv("LANGSMITH_API_KEY")
    if api_key:
        return LangSmithClient(api_key=api_key)
    return None


def create_langsmith_run(
    name: str,
    project: str,
    inputs: Optional[Dict[str, Any]] = None,
    run_type: str = "chain"
) -> Optional[str]:
    """Create a LangSmith run and return run ID."""
    client = get_langsmith_client()
    if not client:
        return None

    try:
        run = client.create_run(
            name=name,
            inputs=inputs or {},
            run_type=run_type,
            project_name=project
        )
        return str(run.id)
    except Exception as e:
        print(f"⚠️  Failed to create LangSmith run: {e}")
        return None


def update_langsmith_run(
    run_id: str,
    outputs: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    end_time: Optional[str] = None
) -> bool:
    """Update a LangSmith run with outputs or error."""
    client = get_langsmith_client()
    if not client or not run_id:
        return False

    try:
        update_data = {}
        if outputs is not None:
            update_data["outputs"] = outputs
        if error is not None:
            update_data["error"] = error
        if end_time is not None:
            update_data["end_time"] = end_time

        client.update_run(run_id=run_id, **update_data)
        return True
    except Exception as e:
        print(f"⚠️  Failed to update LangSmith run: {e}")
        return False