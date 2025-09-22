"""Tests for model definitions and validation."""

import pytest
from datetime import datetime
from typing import Dict, Any

from app.models import (
    HITLDecision,
    HITLStage,
    ApprovalPayload,
    DecisionPayload,
    HITLState,
    create_hitl_session,
    FileEntry
)


class TestHITLDecision:
    """Test HITL decision enum."""

    def test_hitl_decision_values(self):
        """Test that all expected decision values exist."""
        assert HITLDecision.APPROVE == "approve"
        assert HITLDecision.REJECT == "reject"
        assert HITLDecision.MODIFY == "modify"

    def test_hitl_decision_from_string(self):
        """Test creating decision from string values."""
        assert HITLDecision("approve") == HITLDecision.APPROVE
        assert HITLDecision("reject") == HITLDecision.REJECT
        assert HITLDecision("modify") == HITLDecision.MODIFY


class TestHITLStage:
    """Test HITL stage enum."""

    def test_hitl_stage_values(self):
        """Test that all expected stage values exist."""
        assert HITLStage.GENERATE_CODE == "generate_code"
        assert HITLStage.REVIEW_CODE == "review_code"
        assert HITLStage.EXECUTE_CODE == "execute_code"
        assert HITLStage.SAVE_ARTIFACTS == "save_artifacts"
        assert HITLStage.COMPLETE == "complete"

    def test_hitl_stage_from_string(self):
        """Test creating stage from string values."""
        assert HITLStage("generate_code") == HITLStage.GENERATE_CODE
        assert HITLStage("review_code") == HITLStage.REVIEW_CODE
        assert HITLStage("execute_code") == HITLStage.EXECUTE_CODE


class TestApprovalPayload:
    """Test approval payload model."""

    def test_approval_payload_creation(self):
        """Test creating approval payload with valid data."""
        payload = ApprovalPayload(
            session_id="test-session-123",
            stage=HITLStage.REVIEW_CODE,
            content="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            message="Please review this factorial implementation"
        )

        assert payload.session_id == "test-session-123"
        assert payload.stage == HITLStage.REVIEW_CODE
        assert "factorial" in payload.content
        assert "review" in payload.message.lower()

    def test_approval_payload_validation(self):
        """Test approval payload validation."""
        with pytest.raises(ValueError):
            ApprovalPayload(
                session_id="",  # Empty session_id should fail
                stage=HITLStage.REVIEW_CODE,
                content="test content",
                message="test message"
            )

    def test_approval_payload_serialization(self):
        """Test approval payload can be serialized to dict."""
        payload = ApprovalPayload(
            session_id="test-session-123",
            stage=HITLStage.REVIEW_CODE,
            content="test content",
            message="test message"
        )

        data = payload.model_dump()
        assert data["session_id"] == "test-session-123"
        assert data["stage"] == "review_code"
        assert data["content"] == "test content"
        assert data["message"] == "test message"


class TestDecisionPayload:
    """Test decision payload model."""

    def test_decision_payload_creation(self):
        """Test creating decision payload with valid data."""
        payload = DecisionPayload(
            session_id="test-session-123",
            decision=HITLDecision.APPROVE,
            feedback="Looks good to me!"
        )

        assert payload.session_id == "test-session-123"
        assert payload.decision == HITLDecision.APPROVE
        assert payload.feedback == "Looks good to me!"

    def test_decision_payload_optional_feedback(self):
        """Test decision payload with optional feedback."""
        payload = DecisionPayload(
            session_id="test-session-123",
            decision=HITLDecision.REJECT
        )

        assert payload.session_id == "test-session-123"
        assert payload.decision == HITLDecision.REJECT
        assert payload.feedback is None

    def test_decision_payload_serialization(self):
        """Test decision payload serialization."""
        payload = DecisionPayload(
            session_id="test-session-123",
            decision=HITLDecision.MODIFY,
            feedback="Please add error handling"
        )

        data = payload.model_dump()
        assert data["session_id"] == "test-session-123"
        assert data["decision"] == "modify"
        assert data["feedback"] == "Please add error handling"


class TestHITLState:
    """Test HITL state model."""

    def test_hitl_state_creation(self):
        """Test creating HITL state with required fields."""
        state = HITLState(
            user_query="Write a function to calculate fibonacci",
            stage=HITLStage.GENERATE_CODE,
            session_id="test-session-123"
        )

        assert state.user_query == "Write a function to calculate fibonacci"
        assert state.stage == HITLStage.GENERATE_CODE
        assert state.session_id == "test-session-123"
        assert state.generated_code is None
        assert state.review_comments == []
        assert state.execution_result is None
        assert state.artifacts == {}
        assert state.approval_needed is False
        assert state.user_decision is None

    def test_hitl_state_with_all_fields(self):
        """Test creating HITL state with all fields populated."""
        state = HITLState(
            user_query="Write a function to calculate fibonacci",
            stage=HITLStage.EXECUTE_CODE,
            session_id="test-session-123",
            generated_code="def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
            review_comments=["Add input validation", "Consider iterative approach"],
            execution_result={"success": True, "output": "0, 1, 1, 2, 3, 5, 8"},
            artifacts={"output_file": "fibonacci_results.txt"},
            approval_needed=True,
            user_decision=HITLDecision.APPROVE
        )

        assert state.user_query == "Write a function to calculate fibonacci"
        assert state.stage == HITLStage.EXECUTE_CODE
        assert state.generated_code is not None
        assert len(state.review_comments) == 2
        assert state.execution_result["success"] is True
        assert "output_file" in state.artifacts
        assert state.approval_needed is True
        assert state.user_decision == HITLDecision.APPROVE

    def test_hitl_state_serialization(self):
        """Test HITL state serialization."""
        state = HITLState(
            user_query="Test query",
            stage=HITLStage.GENERATE_CODE,
            session_id="test-session-123",
            generated_code="print('hello')",
            artifacts={"test": "value"}
        )

        data = state.model_dump()
        assert data["user_query"] == "Test query"
        assert data["stage"] == "generate_code"
        assert data["session_id"] == "test-session-123"
        assert data["generated_code"] == "print('hello')"
        assert data["artifacts"] == {"test": "value"}

    def test_hitl_state_validation(self):
        """Test HITL state validation."""
        with pytest.raises(ValueError):
            HITLState(
                user_query="",  # Empty query should fail
                stage=HITLStage.GENERATE_CODE,
                session_id="test-session-123"
            )

        with pytest.raises(ValueError):
            HITLState(
                user_query="Valid query",
                stage=HITLStage.GENERATE_CODE,
                session_id=""  # Empty session_id should fail
            )


class TestFileEntry:
    """Test file entry model."""

    def test_file_entry_creation(self):
        """Test creating file entry with valid data."""
        entry = FileEntry(
            name="test.py",
            path="/home/user/test.py",
            content="print('hello world')",
            size=19
        )

        assert entry.name == "test.py"
        assert entry.path == "/home/user/test.py"
        assert entry.content == "print('hello world')"
        assert entry.size == 19

    def test_file_entry_optional_fields(self):
        """Test file entry with optional fields."""
        entry = FileEntry(
            name="test.py",
            path="/home/user/test.py"
        )

        assert entry.name == "test.py"
        assert entry.path == "/home/user/test.py"
        assert entry.content is None
        assert entry.size is None

    def test_file_entry_validation(self):
        """Test file entry validation."""
        with pytest.raises(ValueError):
            FileEntry(
                name="",  # Empty name should fail
                path="/home/user/test.py"
            )

        with pytest.raises(ValueError):
            FileEntry(
                name="test.py",
                path=""  # Empty path should fail
            )

    def test_file_entry_serialization(self):
        """Test file entry serialization."""
        entry = FileEntry(
            name="test.py",
            path="/home/user/test.py",
            content="print('test')",
            size=12
        )

        data = entry.model_dump()
        assert data["name"] == "test.py"
        assert data["path"] == "/home/user/test.py"
        assert data["content"] == "print('test')"
        assert data["size"] == 12


class TestCreateHITLSession:
    """Test HITL session creation utility."""

    def test_create_hitl_session(self):
        """Test creating new HITL session."""
        query = "Write a function to sort a list"
        session = create_hitl_session(query)

        assert isinstance(session, HITLState)
        assert session.user_query == query
        assert session.stage == HITLStage.GENERATE_CODE
        assert session.session_id is not None
        assert len(session.session_id) > 0
        assert session.generated_code is None
        assert session.review_comments == []
        assert session.execution_result is None
        assert session.artifacts == {}
        assert session.approval_needed is False
        assert session.user_decision is None

    def test_create_hitl_session_unique_ids(self):
        """Test that each session gets a unique ID."""
        session1 = create_hitl_session("Query 1")
        session2 = create_hitl_session("Query 2")

        assert session1.session_id != session2.session_id


class TestModuleImports:
    """Test that all model modules can be imported correctly."""

    def test_import_hitl_models(self):
        """Test importing HITL models."""
        from app.models import HITLDecision, HITLStage, HITLState, ApprovalPayload, DecisionPayload
        assert HITLDecision is not None
        assert HITLStage is not None
        assert HITLState is not None
        assert ApprovalPayload is not None
        assert DecisionPayload is not None

    def test_import_execution_models(self):
        """Test importing execution models."""
        from app.models import FileEntry
        assert FileEntry is not None

    def test_import_utilities(self):
        """Test importing utility functions."""
        from app.models import create_hitl_session
        assert callable(create_hitl_session)

    def test_models_module_all_exports(self):
        """Test that __all__ exports are correct."""
        import app.models as models_module

        expected_exports = [
            "HITLDecision",
            "HITLStage",
            "ApprovalPayload",
            "DecisionPayload",
            "HITLState",
            "create_hitl_session",
            "FileEntry"
        ]

        for export in expected_exports:
            assert hasattr(models_module, export), f"Missing export: {export}"