"""Shared test fixtures and configuration."""

import pytest
from unittest.mock import Mock, MagicMock
import os
from typing import Generator

@pytest.fixture
def mock_e2b_sandbox():
    """Mock E2B sandbox for testing."""
    mock_sandbox = Mock()
    mock_sandbox.sandbox_id = "test-sandbox-123"
    mock_sandbox.run_code.return_value = Mock(
        logs=Mock(stdout="print('Hello, World!')", stderr=""),
        results=[Mock(text="Hello, World!", is_main_result=True)],
        error=None
    )
    mock_sandbox.filesystem = Mock()
    mock_sandbox.filesystem.list.return_value = [
        Mock(name="test.py", path="/home/user/test.py", is_dir=False)
    ]
    mock_sandbox.filesystem.read.return_value = "print('test')"
    mock_sandbox.filesystem.write.return_value = None
    return mock_sandbox

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [
        Mock(message=Mock(content="```python\nprint('Generated code')\n```"))
    ]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client

@pytest.fixture
def mock_langsmith_client():
    """Mock LangSmith client for testing."""
    mock_client = Mock()
    mock_client.create_run.return_value = Mock(id="test-run-123")
    mock_client.update_run.return_value = None
    return mock_client

@pytest.fixture
def sample_hitl_state():
    """Sample HITL state for testing."""
    from app.models import HITLState, HITLStage
    return HITLState(
        user_query="Write a function to calculate fibonacci",
        stage=HITLStage.GENERATE_CODE,
        session_id="test-session-123",
        generated_code="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        review_comments=[],
        execution_result=None,
        artifacts={},
        approval_needed=False,
        user_decision=None
    )

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "OPENAI_API_KEY": "test-key",
        "E2B_API_KEY": "test-e2b-key",
        "LANGSMITH_API_KEY": "test-langsmith-key",
        "LANGSMITH_PROJECT": "test-project",
        "OTEL_SERVICE_NAME": "test-service",
    }

    original_values = {}
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value

    yield

    # Restore original values
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

@pytest.fixture
def temp_tar_file():
    """Temporary tar file for testing."""
    import tempfile
    import os

    fd, path = tempfile.mkstemp(suffix='.tar.gz')
    os.close(fd)
    yield path
    os.unlink(path)