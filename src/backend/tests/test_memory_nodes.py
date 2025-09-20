import types
from typing import List, Dict
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.backend.app.state import AppState
from src.backend.app.nodes_memory import recall_memory_node, save_memory_node
from src.backend.app import memory as letta

class DummyResp:
    def __init__(self, json_obj): self._json = json_obj
    def json(self): return self._json
    def raise_for_status(self): return None

@pytest.fixture(autouse=True)
def enable_letta_env(monkeypatch):
    # Enable feature flag but don't require real network
    monkeypatch.setenv("USE_LETTA", "true")
    monkeypatch.setenv("LETTA_API_KEY", "test_key")
    monkeypatch.setenv("LETTA_BASE_URL", "http://letta.local")
    importlib.reload(letta)
    yield
    # teardown not strictly necessary here

def test_recall_memory_node(monkeypatch):
    # Stub httpx.Client.get
    class DummyClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def get(self, url, params=None, headers=None):
            return DummyResp({"items": [
                {"summary": "Past summary 1", "tags": ["t1"]},
                {"summary": "Past summary 2", "tags": ["t2"]},
            ]})
    monkeypatch.setattr(letta.httpx, "Client", DummyClient)

    st = AppState(user_id="u1", user_query="rebalance tech tilt")
    st = recall_memory_node(st)
    assert st.recalled and len(st.recalled) == 2
    assert any(m.get("role") == "system" for m in st.messages)

def test_save_memory_node(monkeypatch):
    # Stub httpx.Client.post
    class DummyClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def post(self, url, json=None, headers=None):
            # validate minimal shape
            assert "summary" in json or "user_query" in json
            return DummyResp({"ok": True})
    monkeypatch.setattr(letta.httpx, "Client", DummyClient)

    st = AppState(user_id="u1", user_query="hello")
    st.result = "final answer"
    st = save_memory_node(st)
    # no exception == pass

def test_graceful_noop_when_disabled(monkeypatch):
    # Disable feature flag to ensure graceful behavior
    monkeypatch.delenv("LETTA_API_KEY", raising=False)
    monkeypatch.setenv("USE_LETTA", "false")
    importlib.reload(letta)

    st = AppState(user_id="u1", user_query="x")
    st2 = recall_memory_node(st)
    assert st2.recalled == []
    st3 = save_memory_node(st2)  # should not raise
