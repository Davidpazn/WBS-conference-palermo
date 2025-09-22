"""E2B Sandbox lifecycle management functions."""

import os
from typing import List, Iterable, Optional
from e2b_code_interpreter import Sandbox


# Constants
PERSIST_TIMEOUT_SECONDS = int(os.getenv("NB1_PERSIST_TIMEOUT", "600"))  # 10 min


def new_persistent_sandbox(timeout_seconds: int = PERSIST_TIMEOUT_SECONDS) -> Sandbox:
    """Create a long-lived E2B sandbox."""
    sbx = Sandbox.create(timeout=timeout_seconds)
    print({"sandboxId": sbx.sandbox_id, "timeout_s": timeout_seconds})
    return sbx


def list_running_or_paused(limit: int = 100) -> list:
    """List running/paused sandboxes using E2B SDK."""
    try:
        # Use E2B's list method with proper pagination
        paginator = Sandbox.list()
        print(f"Paginator type: {type(paginator)}")

        items = []

        # Check if it has nextItems method (proper E2B pagination)
        if hasattr(paginator, 'nextItems'):
            try:
                # Get first page
                first_page = paginator.nextItems()
                items.extend(first_page)
                print(f"Got {len(first_page)} items from first page")

                # Get remaining pages if hasNext is True
                while hasattr(paginator, 'hasNext') and paginator.hasNext:
                    next_page = paginator.nextItems()
                    items.extend(next_page)
                    print(f"Got {len(next_page)} items from next page")
            except Exception as e:
                print(f"Pagination failed: {e}")

        # Fallback: try direct iteration
        elif hasattr(paginator, '__iter__'):
            try:
                items = list(paginator)
                print(f"Got {len(items)} items via iteration")
            except Exception as e:
                print(f"Iteration failed: {e}")

        return items
    except Exception as e:
        print(f"[list_running_or_paused] error: {e}")
        return []


def pretty_sbx_info(items: Iterable) -> None:
    """Display formatted sandbox information."""
    try:
        items_list = list(items) if hasattr(items, '__iter__') else [items]
    except TypeError:
        print(f"Cannot iterate over items: {type(items)}")
        return

    for it in items_list:
        if it is None:
            continue

        try:
            # Extract sandbox information using E2B SDK methods
            if hasattr(it, 'get_info'):
                # Use get_info() method if available
                info = it.get_info()
                print(f"Sandbox info: {info}")
            else:
                # Try direct attribute access
                print(f"Item type: {type(it)}")
                sid = getattr(it, 'sandbox_id', getattr(it, 'id', 'unknown'))
                state = getattr(it, 'state', 'unknown')
                metadata = getattr(it, 'metadata', {})
                print({'sandboxId': sid, 'state': state, 'metadata': metadata})
        except Exception as e:
            print(f"Error processing sandbox info: {e}, item: {it}")


def kill_by_id(sandbox_id: str) -> bool:
    """Kill a sandbox by its ID using E2B's static kill method."""
    try:
        return Sandbox.kill(sandbox_id)
    except Exception as e:
        print(f"[kill_by_id] error: {e}")
        return False


def kill_all_running() -> None:
    """Kill all running sandboxes."""
    items = list_running_or_paused()
    for it in items:
        if it is None:
            continue

        try:
            # Try to get sandbox ID
            sid = getattr(it, 'sandbox_id', getattr(it, 'id', None))
            if not sid:
                print(f"No sandbox ID found for item: {it}")
                continue

            ok = Sandbox.kill(sid)
            print({'killed': sid, 'ok': ok})
        except Exception as e:
            print({'killed': getattr(it, 'sandbox_id', 'unknown'), 'ok': False, 'error': str(e)})