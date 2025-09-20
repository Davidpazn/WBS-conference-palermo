import os
import logging
from typing import Dict, List

import httpx

log = logging.getLogger(__name__)

def _is_true(v: str | None) -> bool:
    return str(v).lower() in {"1", "true", "yes", "on"}

# Feature flag (explicit) + key presence (implicit)
USE_LETTA = _is_true(os.getenv("USE_LETTA")) and bool(os.getenv("LETTA_API_KEY"))

LETTA_BASE = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
LETTA_KEY = os.getenv("LETTA_API_KEY", "")
AGENTS_URL = f"{LETTA_BASE.rstrip('/')}/v1/agents"

def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {LETTA_KEY}"} if LETTA_KEY else {}

def recall(agent_id: str, query: str, k: int = 3) -> List[Dict]:
    """
    Recall relevant memory items for an agent. Gracefully no-ops if USE_LETTA is false
    or the API key is missing. Adjust endpoint paths to your Letta deployment.
    """
    if not USE_LETTA:
        log.debug("[letta] recall skipped (USE_LETTA disabled or no API key)")
        return []
    url = f"{AGENTS_URL}/{agent_id}/memory/recall"
    try:
        with httpx.Client(timeout=10) as client:
            r = client.get(url, params={"q": query, "k": k}, headers=_headers())
            r.raise_for_status()
            payload = r.json() or {}
            return payload.get("items", [])
    except Exception as e:
        log.warning("[letta] recall failed: %s", e)
        return []

def save(agent_id: str, item: Dict) -> bool:
    """
    Save a compact memory item. Returns True on success, False otherwise.
    """
    if not USE_LETTA:
        log.debug("[letta] save skipped (USE_LETTA disabled or no API key)")
        return False
    url = f"{AGENTS_URL}/{agent_id}/memory/archival"
    try:
        with httpx.Client(timeout=10) as client:
            r = client.post(url, json=item, headers=_headers())
            r.raise_for_status()
            return True
    except Exception as e:
        log.warning("[letta] save failed: %s", e)
        return False

def health() -> dict:
    """
    Try a couple of common Letta health endpoints; return a diagnostic dict.
    Never raises; safe to call from a debug route.
    """
    info = {"use_letta": bool(USE_LETTA), "base_url": LETTA_BASE, "ok": False, "endpoint": None}
    if not USE_LETTA:
        return info
    for path in ("/health", "/v1/health", "/"):
        try:
            with httpx.Client(timeout=5) as client:
                r = client.get(f"{LETTA_BASE.rstrip('/')}{path}", headers=_headers())
                if r.status_code < 500:
                    info["ok"] = True
                    info["endpoint"] = path
                    return info
        except Exception:
            continue
    return info
