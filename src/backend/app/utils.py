from __future__ import annotations

from typing import Any, Dict
from dataclasses import is_dataclass, asdict
from collections.abc import Mapping


def to_dictish(x: Any) -> Dict[str, Any]:
    """
    Coerces anything (dict, Pydantic model, dataclass, namedtuple, object, list of pairs)
    into a plain dict. Safe fallback for handling various Letta SDK return types.
    """
    if x is None:
        return {}
    if isinstance(x, Mapping):
        return dict(x)
    if is_dataclass(x):
        try:
            return asdict(x)
        except Exception:
            pass
    # Pydantic v2 / v1
    if hasattr(x, "model_dump"):
        try:
            return x.model_dump()  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(x, "dict"):
        try:
            return x.dict()  # type: ignore[attr-defined]
        except Exception:
            pass
    # namedtuple
    if hasattr(x, "_asdict"):
        try:
            return x._asdict()  # type: ignore[attr-defined]
        except Exception:
            pass
    # generic python object
    if hasattr(x, "__dict__"):
        try:
            return dict(vars(x))
        except Exception:
            pass
    # list/tuple of (key, value) pairs
    if isinstance(x, (list, tuple)):
        try:
            if all(isinstance(i, tuple) and len(i) == 2 and isinstance(i[0], str) for i in x):
                return {k: v for k, v in x}
        except Exception:
            pass
    return {}


def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """
    Safely reads a key from either a dict or an object (via attribute),
    falling back to to_dictish if needed. Never raises on .get() calls.
    """
    # dict-like
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    # object attribute
    if hasattr(obj, key):
        return getattr(obj, key, default)
    # mapping-like with .get
    if hasattr(obj, "get"):
        try:
            return obj.get(key, default)  # type: ignore[attr-defined]
        except Exception:
            pass
    # last resort: coerce then get
    d = to_dictish(obj)
    return d.get(key, default)