from __future__ import annotations

import os
import logging
import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING, TypeAlias
from collections.abc import Mapping

from .utils import to_dictish, safe_get

log = logging.getLogger(__name__)

# ---- Letta imports: separate runtime vs type-only ---------------------------------
try:
    # Runtime symbol used to actually construct the client
    from letta_client import Letta as _LettaRuntime
    LETTA_AVAILABLE = True
except ImportError:
    _LettaRuntime = None  # type: ignore[assignment]
    LETTA_AVAILABLE = False

# Type-only alias used in annotations; falls back to Any when package missing
if TYPE_CHECKING:
    from letta_client import Letta as LettaClient  # real type for static checkers
else:
    LettaClient: TypeAlias = Any  # safe fallback at runtime
# -----------------------------------------------------------------------------------

def _is_true(v: str | None) -> bool:
    return str(v).lower() in {"1", "true", "yes", "on"}

# Feature flag: enable if explicitly requested OR if client available + key present
USE_LETTA_EXPLICIT = os.getenv("USE_LETTA")
if USE_LETTA_EXPLICIT is not None:
    USE_LETTA = _is_true(USE_LETTA_EXPLICIT) and LETTA_AVAILABLE and bool(os.getenv("LETTA_API_KEY"))
else:
    USE_LETTA = LETTA_AVAILABLE and bool(os.getenv("LETTA_API_KEY"))

LETTA_BASE = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
LETTA_KEY = os.getenv("LETTA_API_KEY", "")
LETTA_PROJECT = os.getenv("LETTA_PROJECT", None)  # optional, for cloud projects
LETTA_TIMEOUT = float(os.getenv("LETTA_TIMEOUT", "30.0"))  # currently unused (SDK default timeouts)

# Global client instance
_letta_client: Optional[LettaClient] = None

def _get_client() -> Optional[LettaClient]:
    """Get or create Letta client instance with proper error handling."""
    global _letta_client

    if not USE_LETTA or not LETTA_AVAILABLE or _LettaRuntime is None:
        return None

    if _letta_client is None:
        try:
            if LETTA_KEY:
                # Cloud/authenticated connection (token + optional project)
                if LETTA_PROJECT:
                    _letta_client = _LettaRuntime(token=LETTA_KEY, project=LETTA_PROJECT)  # type: ignore[call-arg]
                else:
                    _letta_client = _LettaRuntime(token=LETTA_KEY)  # type: ignore[call-arg]
                log.info("[letta] Initialized cloud client with API key")
            else:
                # Local server connection
                _letta_client = _LettaRuntime(base_url=LETTA_BASE)  # type: ignore[call-arg]
                log.info(f"[letta] Initialized local client at {LETTA_BASE}")
        except Exception as e:
            log.error(f"[letta] Client initialization failed: {e}")
            _letta_client = None

    return _letta_client

def normalize_letta_search(res):
    """Handles tuple vs dict returns across letta/letta-client versions"""
    # common object shapes
    for attr in ("results", "data", "items"):
        if hasattr(res, attr):
            return {"results": getattr(res, attr) or [], "meta": getattr(res, "meta", {})}
    # dict shape
    if isinstance(res, Mapping):
        results = res.get("results") or res.get("data") or res.get("items") or []
        return {"results": results, "meta": res.get("meta", {})}
    # tuple shape (results, meta)
    if isinstance(res, tuple) and len(res) == 2:
        results, meta = res
        return {"results": results or [], "meta": meta or {}}
    # fallback
    return {"results": [], "meta": {}}

def recall(agent_id: str, query: str, k: int = 3) -> List[Dict]:
    """
    Recall relevant memory items for an agent using the official Letta client.
    Gracefully returns empty list if Letta is unavailable.
    """
    if not USE_LETTA:
        log.debug("[letta] recall skipped (USE_LETTA disabled or client unavailable)")
        return []

    client = _get_client()
    if not client:
        log.debug("[letta] recall skipped (client unavailable)")
        return []

    try:
        # Get or create agent first
        agent = _ensure_agent(client, agent_id)
        if not agent:
            log.warning(f"[letta] Agent {agent_id} not found and could not be created")
            return []

        # Search archival memory
        raw_response = client.agents.passages.search(
            agent_id=agent["id"],
            query=query,
            top_k=k
        )

        # Normalize response across different Letta client versions
        normalized = normalize_letta_search(raw_response)
        passages_list = normalized["results"]

        # Format results for compatibility
        results: List[Dict[str, Any]] = []
        for passage in passages_list or []:
            p = to_dictish(passage)
            results.append({
                "id": safe_get(p, "id"),
                "text": safe_get(p, "text", ""),
                "metadata": safe_get(p, "metadata", {}),
                "score": safe_get(p, "score", 0.0),
                "timestamp": safe_get(p, "created_at")
            })

        log.debug(f"[letta] recalled {len(results)} memories for agent {agent_id}")
        return results

    except Exception as e:
        log.warning("[letta] recall failed: %s", e)
        return []

def save(agent_id: str, item: Dict) -> bool:
    """
    Save a memory item to agent's archival memory using official Letta client.
    Returns True on success, False otherwise.
    """
    if not USE_LETTA:
        log.debug("[letta] save skipped (USE_LETTA disabled or client unavailable)")
        return False

    client = _get_client()
    if not client:
        log.debug("[letta] save skipped (client unavailable)")
        return False

    try:
        # Get or create agent first
        agent = _ensure_agent(client, agent_id)
        if not agent:
            log.warning(f"[letta] Agent {agent_id} not found and could not be created")
            return False

        # Convert item to text for archival storage
        if isinstance(item, dict):
            text_content = json.dumps(item, indent=2)
        else:
            text_content = str(item)

        # Extract tags from metadata for passage creation
        tags = ["user_save"]
        if isinstance(item, dict):
            tags.append("dict_format")
            if "metadata" in item and isinstance(item["metadata"], dict):
                for key, value in item["metadata"].items():
                    if isinstance(value, str) and len(value) < 50:
                        tags.append(f"{key}:{value}")
        else:
            tags.append("string_format")

        # Create passage with tags only (no metadata parameter in create())
        passage_result = client.agents.passages.create(
            agent_id=agent["id"],
            text=text_content,
            tags=tags
        )

        # Log the successful creation
        passage_id = None
        if isinstance(passage_result, list) and passage_result:
            pr = to_dictish(passage_result[0])
            passage_id = safe_get(pr, "id")
        else:
            pr = to_dictish(passage_result)
            passage_id = safe_get(pr, "id")

        if passage_id:
            log.debug(f"[letta] created passage {passage_id} for agent {agent_id}")

        log.debug(f"[letta] saved memory item for agent {agent_id}")
        return True

    except Exception as e:
        log.warning("[letta] save failed: %s", e)
        return False

def health() -> dict:
    """
    Check Letta service health using official client. Returns comprehensive diagnostic info.
    Never raises; safe to call from debug routes or health checks.
    """
    info = {
        "use_letta": bool(USE_LETTA),
        "letta_available": LETTA_AVAILABLE,
        "base_url": LETTA_BASE,
        "has_api_key": bool(LETTA_KEY),
        "ok": False,
        "version": None,
        "error": None
    }

    if not USE_LETTA:
        info["error"] = "USE_LETTA disabled or requirements not met"
        return info

    if not LETTA_AVAILABLE or _LettaRuntime is None:
        info["error"] = "Letta client library not available"
        return info

    try:
        client = _get_client()
        if not client:
            info["error"] = "Failed to initialize Letta client"
            return info

        # Try health check endpoint
        health_result = client.health.check()
        if health_result:
            info["ok"] = True
            info["version"] = getattr(health_result, "version", "unknown")

    except Exception as e:
        info["error"] = str(e)
        log.debug(f"[letta] health check failed: {e}")

    return info

def _ensure_agent(client: LettaClient, agent_id: str) -> Optional[Dict[str, Any]]:
    """
    Ensure an agent exists, creating it if necessary.
    Returns agent info dict or None on failure.
    """
    try:
        # Try to get existing agent
        agents_response = client.agents.list()
        agents_list = agents_response.data if hasattr(agents_response, 'data') else agents_response

        for agent in agents_list or []:
            a = to_dictish(agent)
            if safe_get(a, "name") == agent_id or safe_get(a, "id") == agent_id:
                return {
                    "id": safe_get(a, "id"),
                    "name": safe_get(a, "name"),
                    "created_at": safe_get(a, "created_at")
                }

        # Agent doesn't exist, create it
        log.info(f"[letta] Creating new agent: {agent_id}")
        new_agent = client.agents.create(
            name=agent_id,
            system="You are a helpful assistant for portfolio and memory management.",
            model=os.getenv("LETTA_MODEL", "openai/gpt-5-nano"),
            embedding=os.getenv("LETTA_EMBEDDING", "openai/text-embedding-3-small"),
            memory_blocks=[
                {
                    "label": "persona",
                    "description": "Agent persona and role",
                    "value": f"Agent for user {agent_id}",
                    "limit": 6000
                },
                {
                    "label": "human",
                    "description": "Information about the user",
                    "value": "User working with portfolio rebalancing and financial analysis",
                    "limit": 6000
                }
            ]
        )

        agent_dict = to_dictish(new_agent)
        return {
            "id": safe_get(agent_dict, "id"),
            "name": safe_get(agent_dict, "name"),
            "created_at": safe_get(agent_dict, "created_at")
        }

    except Exception as e:
        log.error(f"[letta] Failed to ensure agent {agent_id}: {e}")
        return None

def list_agents() -> List[Dict[str, Any]]:
    """
    List all available agents. Returns empty list if Letta is unavailable.
    """
    if not USE_LETTA:
        return []

    client = _get_client()
    if not client:
        return []

    try:
        agents_response = client.agents.list()
        agents_list = agents_response.data if hasattr(agents_response, 'data') else agents_response

        result: List[Dict[str, Any]] = []
        for agent in agents_list or []:
            a = to_dictish(agent)
            result.append({
                "id": safe_get(a, "id"),
                "name": safe_get(a, "name"),
                "created_at": safe_get(a, "created_at"),
                "last_updated": safe_get(a, "last_updated")
            })
        return result
    except Exception as e:
        log.warning(f"[letta] Failed to list agents: {e}")
        return []

def reset_client():
    """
    Reset the global client instance. Useful for testing or configuration changes.
    """
    global _letta_client
    _letta_client = None
    log.info("[letta] Client instance reset")
