from datetime import datetime, timezone
from typing import Any

from .state import AppState
from . import memory as letta

def recall_memory_node(state: AppState) -> AppState:
    """
    Long-term memory recall (Letta). Adds top-k items into state.recalled and
    appends a light system hint into messages for the next node.
    """
    agent_id = state.meta.get("letta_agent_id") or state.user_id
    topk = int(state.meta.get("memory_topk", 3))
    items = letta.recall(agent_id, query=state.user_query, k=topk)
    state.recalled = items or []
    if items:
        preview = [
            {"summary": it.get("summary") or it.get("text", "")[:200], "tags": it.get("tags", [])}
            for it in items[:2]
        ]
        state.messages.append(
            {
                "role": "system",
                "content": f"Previously recalled context (top {min(topk, len(items))}): {preview}",
            }
        )
    return state

def save_memory_node(state: AppState) -> AppState:
    """
    Long-term memory write-back (Letta). Saves a compact memory item if present.
    Gracefully no-ops when disabled.
    """
    agent_id = state.meta.get("letta_agent_id") or state.user_id
    # Prepare a compact item if caller didn't set one
    item = state.memory_item or {
        "user_query": state.user_query,
        "summary": (state.result or "")[:500],
        "tags": list({*state.meta.get("tags", []), "demo", "agents"}),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    letta.save(agent_id, item)
    return state
