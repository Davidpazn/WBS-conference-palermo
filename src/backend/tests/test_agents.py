"""Integration tests for agent workflows."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
from typing import Dict, Any

from app.agents import (
    coding_agent,
    generate_code_node,
    code_review_node,
    execute_code_node,
    save_artifacts_node,
    create_hitl_graph,
    start_hitl_workflow,
    resume_hitl_workflow,
    display_approval_request,
    SelfContainedHITLWorkflow
)
from app.models import HITLState, HITLStage, HITLDecision


class TestCodingAgent:
    """Test basic coding agent functionality."""

    @patch('app.agents.coding_agent.llm_generate_code')
    def test_coding_agent_simple_task(self, mock_generate_code):
        """Test coding agent with simple task."""
        mock_generate_code.return_value = "def hello(): return 'Hello, World!'"

        result = coding_agent("Write a hello world function")

        assert result == "def hello(): return 'Hello, World!'"
        mock_generate_code.assert_called_once_with("Write a hello world function")

    @patch('app.agents.coding_agent.llm_generate_code')
    def test_coding_agent_error_handling(self, mock_generate_code):
        """Test coding agent error handling."""
        mock_generate_code.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            coding_agent("Write a function")


class TestHITLWorkflowNodes:
    """Test individual HITL workflow nodes."""

    @patch('app.agents.hitl_workflow.llm_generate_code')
    @patch('app.agents.hitl_workflow.tracer')
    def test_generate_code_node(self, mock_tracer, mock_generate_code, sample_hitl_state):
        """Test code generation node."""
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        mock_generate_code.return_value = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"

        result = generate_code_node(sample_hitl_state)

        assert isinstance(result, HITLState)
        assert result.generated_code is not None
        assert "factorial" in result.generated_code
        assert result.stage == HITLStage.REVIEW_CODE
        mock_generate_code.assert_called_once()

    @patch('app.agents.hitl_workflow.tracer')
    def test_code_review_node(self, mock_tracer, sample_hitl_state):
        """Test code review node."""
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        # Set up state with generated code
        sample_hitl_state.generated_code = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
        sample_hitl_state.stage = HITLStage.REVIEW_CODE

        result = code_review_node(sample_hitl_state)

        assert isinstance(result, HITLState)
        assert len(result.review_comments) > 0
        assert result.approval_needed is True

    @patch('app.agents.hitl_workflow.run_in_e2b')
    @patch('app.agents.hitl_workflow.tracer')
    def test_execute_code_node(self, mock_tracer, mock_run_in_e2b, sample_hitl_state):
        """Test code execution node."""
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        mock_run_in_e2b.return_value = {
            "success": True,
            "stdout": "Function executed successfully",
            "stderr": "",
            "results": ["5"]
        }

        # Set up state with generated code
        sample_hitl_state.generated_code = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
        sample_hitl_state.stage = HITLStage.EXECUTE_CODE

        result = execute_code_node(sample_hitl_state)

        assert isinstance(result, HITLState)
        assert result.execution_result is not None
        assert result.execution_result["success"] is True
        assert result.stage == HITLStage.SAVE_ARTIFACTS
        mock_run_in_e2b.assert_called_once()

    @patch('app.agents.hitl_workflow.new_persistent_sandbox')
    @patch('app.agents.hitl_workflow.download_all_as_tar')
    @patch('app.agents.hitl_workflow.tracer')
    def test_save_artifacts_node(self, mock_tracer, mock_download_tar, mock_new_sandbox, sample_hitl_state):
        """Test save artifacts node."""
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        mock_sandbox = Mock(sandbox_id="sandbox-123")
        mock_new_sandbox.return_value = mock_sandbox
        mock_download_tar.return_value = "/tmp/artifacts.tar.gz"

        # Set up state with execution result
        sample_hitl_state.execution_result = {"success": True, "stdout": "Output"}
        sample_hitl_state.stage = HITLStage.SAVE_ARTIFACTS

        result = save_artifacts_node(sample_hitl_state)

        assert isinstance(result, HITLState)
        assert "artifacts_path" in result.artifacts
        assert result.stage == HITLStage.COMPLETE
        mock_new_sandbox.assert_called_once()
        mock_download_tar.assert_called_once()


class TestHITLWorkflowManagement:
    """Test HITL workflow management functions."""

    @patch('app.agents.hitl_workflow.StateGraph')
    @patch('app.agents.hitl_workflow.MemorySaver')
    def test_create_hitl_graph(self, mock_memory_saver, mock_state_graph):
        """Test creating HITL graph."""
        mock_graph_instance = Mock()
        mock_state_graph.return_value = mock_graph_instance
        mock_memory_instance = Mock()
        mock_memory_saver.return_value = mock_memory_instance

        graph = create_hitl_graph()

        assert graph == mock_graph_instance.compile.return_value
        mock_state_graph.assert_called_once()
        mock_graph_instance.add_node.assert_called()
        mock_graph_instance.add_edge.assert_called()

    @patch('app.agents.hitl_workflow.create_hitl_graph')
    def test_start_hitl_workflow(self, mock_create_graph):
        """Test starting HITL workflow."""
        mock_graph = Mock()
        mock_create_graph.return_value = mock_graph
        mock_config = {"configurable": {"thread_id": "test-session-123"}}

        initial_state = HITLState(
            user_query="Test query",
            stage=HITLStage.GENERATE_CODE,
            session_id="test-session-123"
        )

        start_hitl_workflow(initial_state)

        mock_graph.invoke.assert_called_once()

    @patch('app.agents.hitl_workflow.create_hitl_graph')
    def test_resume_hitl_workflow(self, mock_create_graph):
        """Test resuming HITL workflow."""
        mock_graph = Mock()
        mock_create_graph.return_value = mock_graph

        session_id = "test-session-123"
        user_decision = HITLDecision.APPROVE

        resume_hitl_workflow(session_id, user_decision)

        mock_graph.invoke.assert_called_once()

    def test_display_approval_request(self, sample_hitl_state):
        """Test displaying approval request."""
        sample_hitl_state.generated_code = "def test(): pass"
        sample_hitl_state.review_comments = ["Add docstring", "Add error handling"]

        # This should not raise an exception
        display_approval_request(sample_hitl_state)


class TestSelfContainedWorkflow:
    """Test self-contained HITL workflow."""

    @patch('app.agents.self_contained_workflow.create_hitl_graph')
    def test_self_contained_workflow_creation(self, mock_create_graph):
        """Test creating self-contained workflow."""
        mock_graph = Mock()
        mock_create_graph.return_value = mock_graph

        workflow = SelfContainedHITLWorkflow()

        assert workflow.graph == mock_graph
        assert workflow.sessions == {}

    @patch('app.agents.self_contained_workflow.create_hitl_graph')
    def test_self_contained_workflow_start_session(self, mock_create_graph):
        """Test starting session in self-contained workflow."""
        mock_graph = Mock()
        mock_create_graph.return_value = mock_graph

        workflow = SelfContainedHITLWorkflow()
        session_id = workflow.start_session("Write a test function")

        assert session_id in workflow.sessions
        assert workflow.sessions[session_id].user_query == "Write a test function"
        assert workflow.sessions[session_id].stage == HITLStage.GENERATE_CODE

    @patch('app.agents.self_contained_workflow.create_hitl_graph')
    def test_self_contained_workflow_get_session(self, mock_create_graph):
        """Test getting session from self-contained workflow."""
        mock_graph = Mock()
        mock_create_graph.return_value = mock_graph

        workflow = SelfContainedHITLWorkflow()
        session_id = workflow.start_session("Test query")

        retrieved_session = workflow.get_session(session_id)

        assert retrieved_session is not None
        assert retrieved_session.session_id == session_id

    @patch('app.agents.self_contained_workflow.create_hitl_graph')
    def test_self_contained_workflow_get_nonexistent_session(self, mock_create_graph):
        """Test getting non-existent session."""
        mock_graph = Mock()
        mock_create_graph.return_value = mock_graph

        workflow = SelfContainedHITLWorkflow()
        retrieved_session = workflow.get_session("non-existent-id")

        assert retrieved_session is None

    @patch('app.agents.self_contained_workflow.create_hitl_graph')
    def test_self_contained_workflow_run_step(self, mock_create_graph):
        """Test running workflow step."""
        mock_graph = Mock()
        mock_graph.invoke.return_value = HITLState(
            user_query="Test query",
            stage=HITLStage.REVIEW_CODE,
            session_id="test-session-123",
            generated_code="def test(): pass"
        )
        mock_create_graph.return_value = mock_graph

        workflow = SelfContainedHITLWorkflow()
        session_id = workflow.start_session("Test query")

        updated_state = workflow.run_step(session_id)

        assert updated_state is not None
        assert updated_state.stage == HITLStage.REVIEW_CODE
        mock_graph.invoke.assert_called_once()

    @patch('app.agents.self_contained_workflow.create_hitl_graph')
    def test_self_contained_workflow_submit_decision(self, mock_create_graph):
        """Test submitting user decision."""
        mock_graph = Mock()
        mock_create_graph.return_value = mock_graph

        workflow = SelfContainedHITLWorkflow()
        session_id = workflow.start_session("Test query")

        # Set up session to need approval
        workflow.sessions[session_id].approval_needed = True

        result = workflow.submit_decision(session_id, HITLDecision.APPROVE, "Looks good!")

        assert result is True
        assert workflow.sessions[session_id].user_decision == HITLDecision.APPROVE


class TestWorkflowIntegration:
    """Test end-to-end workflow integration."""

    @patch('app.agents.hitl_workflow.llm_generate_code')
    @patch('app.agents.hitl_workflow.run_in_e2b')
    @patch('app.agents.hitl_workflow.new_persistent_sandbox')
    @patch('app.agents.hitl_workflow.download_all_as_tar')
    @patch('app.agents.hitl_workflow.tracer')
    def test_full_workflow_simulation(self, mock_tracer, mock_download_tar, mock_new_sandbox,
                                    mock_run_in_e2b, mock_generate_code):
        """Test simulating a full workflow."""
        # Mock all dependencies
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        mock_generate_code.return_value = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
        mock_run_in_e2b.return_value = {"success": True, "stdout": "Test output", "stderr": "", "results": ["120"]}
        mock_sandbox = Mock(sandbox_id="sandbox-123")
        mock_new_sandbox.return_value = mock_sandbox
        mock_download_tar.return_value = "/tmp/artifacts.tar.gz"

        # Create initial state
        initial_state = HITLState(
            user_query="Write a factorial function",
            stage=HITLStage.GENERATE_CODE,
            session_id="test-session-123"
        )

        # Run through workflow steps
        state_after_generate = generate_code_node(initial_state)
        assert state_after_generate.generated_code is not None
        assert state_after_generate.stage == HITLStage.REVIEW_CODE

        state_after_review = code_review_node(state_after_generate)
        assert state_after_review.approval_needed is True
        assert len(state_after_review.review_comments) > 0

        # Simulate approval
        state_after_review.user_decision = HITLDecision.APPROVE
        state_after_review.stage = HITLStage.EXECUTE_CODE

        state_after_execute = execute_code_node(state_after_review)
        assert state_after_execute.execution_result is not None
        assert state_after_execute.stage == HITLStage.SAVE_ARTIFACTS

        final_state = save_artifacts_node(state_after_execute)
        assert final_state.stage == HITLStage.COMPLETE
        assert "artifacts_path" in final_state.artifacts


class TestModuleImports:
    """Test that all agent modules can be imported correctly."""

    def test_import_coding_agent(self):
        """Test importing coding agent."""
        from app.agents import coding_agent
        assert callable(coding_agent)

    def test_import_hitl_workflow_nodes(self):
        """Test importing HITL workflow nodes."""
        from app.agents import (
            generate_code_node,
            code_review_node,
            execute_code_node,
            save_artifacts_node
        )
        assert callable(generate_code_node)
        assert callable(code_review_node)
        assert callable(execute_code_node)
        assert callable(save_artifacts_node)

    def test_import_hitl_workflow_management(self):
        """Test importing HITL workflow management."""
        from app.agents import (
            create_hitl_graph,
            start_hitl_workflow,
            resume_hitl_workflow,
            display_approval_request
        )
        assert callable(create_hitl_graph)
        assert callable(start_hitl_workflow)
        assert callable(resume_hitl_workflow)
        assert callable(display_approval_request)

    def test_import_self_contained_workflow(self):
        """Test importing self-contained workflow."""
        from app.agents import SelfContainedHITLWorkflow
        assert SelfContainedHITLWorkflow is not None

    def test_agents_module_all_exports(self):
        """Test that __all__ exports are correct."""
        import app.agents as agents_module

        expected_exports = [
            "coding_agent",
            "generate_code_node",
            "code_review_node",
            "execute_code_node",
            "save_artifacts_node",
            "create_hitl_graph",
            "start_hitl_workflow",
            "resume_hitl_workflow",
            "display_approval_request",
            "hitl_graph",
            "SelfContainedHITLWorkflow"
        ]

        for export in expected_exports:
            assert hasattr(agents_module, export), f"Missing export: {export}"