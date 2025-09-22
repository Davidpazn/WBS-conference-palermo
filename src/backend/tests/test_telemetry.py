"""Tests for telemetry and tracing functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from typing import Optional

from app.telemetry import (
    setup_telemetry,
    tracer,
    set_genai_span_attributes,
    set_hitl_span_attributes,
    set_sandbox_span_attributes,
    handle_span_error,
    handle_span_success,
    flush_telemetry_spans,
    get_langsmith_client,
    create_langsmith_run,
    update_langsmith_run,
    TELEMETRY_CONFIG,
    SPAN_ATTRIBUTES
)


class TestTelemetrySetup:
    """Test telemetry setup and configuration."""

    @patch('app.telemetry.tracing.TracerProvider')
    @patch('app.telemetry.tracing.Resource')
    @patch('app.telemetry.tracing.OTLPSpanExporter')
    @patch('app.telemetry.tracing.BatchSpanProcessor')
    @patch('app.telemetry.tracing.trace.set_tracer_provider')
    @patch('app.telemetry.tracing.trace.get_tracer')
    def test_setup_telemetry_with_otlp(self, mock_get_tracer, mock_set_provider, mock_batch_processor,
                                      mock_otlp_exporter, mock_resource, mock_tracer_provider):
        """Test telemetry setup with OTLP exporter."""
        # Mock environment variables for OTLP
        with patch.dict(os.environ, {
            'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://localhost:4317',
            'OTEL_SERVICE_NAME': 'test-service'
        }):
            mock_provider_instance = Mock()
            mock_tracer_provider.return_value = mock_provider_instance
            mock_tracer_instance = Mock()
            mock_get_tracer.return_value = mock_tracer_instance

            tracer_result, langsmith_client = setup_telemetry()

            assert tracer_result == mock_tracer_instance
            mock_tracer_provider.assert_called_once()
            mock_otlp_exporter.assert_called_once()
            mock_batch_processor.assert_called_once()

    @patch('app.telemetry.tracing.TracerProvider')
    @patch('app.telemetry.tracing.Resource')
    @patch('app.telemetry.tracing.ConsoleSpanExporter')
    @patch('app.telemetry.tracing.BatchSpanProcessor')
    @patch('app.telemetry.tracing.trace.get_tracer')
    def test_setup_telemetry_console_fallback(self, mock_get_tracer, mock_batch_processor, mock_console_exporter,
                                             mock_resource, mock_tracer_provider):
        """Test telemetry setup with console exporter fallback."""
        # Clear OTLP environment variables
        with patch.dict(os.environ, {}, clear=True):
            mock_provider_instance = Mock()
            mock_tracer_provider.return_value = mock_provider_instance
            mock_tracer_instance = Mock()
            mock_get_tracer.return_value = mock_tracer_instance

            tracer_result, langsmith_client = setup_telemetry()

            assert tracer_result == mock_tracer_instance
            mock_console_exporter.assert_called_once()

    def test_telemetry_config_constants(self):
        """Test that telemetry configuration constants are defined."""
        assert isinstance(TELEMETRY_CONFIG, dict)
        assert "service_name" in TELEMETRY_CONFIG
        assert "version" in TELEMETRY_CONFIG

    def test_span_attributes_constants(self):
        """Test that span attributes constants are defined."""
        assert isinstance(SPAN_ATTRIBUTES, dict)
        assert "GENAI" in SPAN_ATTRIBUTES
        assert "HITL" in SPAN_ATTRIBUTES
        assert "SANDBOX" in SPAN_ATTRIBUTES


class TestSpanAttributeHelpers:
    """Test span attribute helper functions."""

    def test_set_genai_span_attributes(self):
        """Test setting GenAI span attributes."""
        mock_span = Mock()

        set_genai_span_attributes(
            span=mock_span,
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost=0.005
        )

        # Verify all expected attributes were set
        expected_calls = [
            ('gen_ai.system', 'openai'),
            ('gen_ai.request.model', 'gpt-4o'),
            ('gen_ai.usage.prompt_tokens', 100),
            ('gen_ai.usage.completion_tokens', 50),
            ('gen_ai.usage.total_tokens', 150),
            ('gen_ai.usage.cost', 0.005)
        ]

        for attr_name, attr_value in expected_calls:
            mock_span.set_attribute.assert_any_call(attr_name, attr_value)

    def test_set_hitl_span_attributes(self):
        """Test setting HITL span attributes."""
        mock_span = Mock()

        set_hitl_span_attributes(
            span=mock_span,
            session_id="test-session-123",
            stage="generate_code",
            approval_needed=True,
            user_decision="approve"
        )

        expected_calls = [
            ('hitl.session_id', 'test-session-123'),
            ('hitl.stage', 'generate_code'),
            ('hitl.approval_needed', True),
            ('hitl.user_decision', 'approve')
        ]

        for attr_name, attr_value in expected_calls:
            mock_span.set_attribute.assert_any_call(attr_name, attr_value)

    def test_set_sandbox_span_attributes(self):
        """Test setting sandbox span attributes."""
        mock_span = Mock()

        set_sandbox_span_attributes(
            span=mock_span,
            sandbox_id="sandbox-123",
            timeout_seconds=600,
            execution_time_ms=1500,
            exit_code=0
        )

        expected_calls = [
            ('sandbox.id', 'sandbox-123'),
            ('sandbox.timeout_seconds', 600),
            ('sandbox.execution_time_ms', 1500),
            ('sandbox.exit_code', 0)
        ]

        for attr_name, attr_value in expected_calls:
            mock_span.set_attribute.assert_any_call(attr_name, attr_value)

    def test_handle_span_success(self):
        """Test handling successful span completion."""
        mock_span = Mock()

        handle_span_success(mock_span, "Operation completed successfully")

        mock_span.set_status.assert_called_once_with(
            mock_span.set_status.call_args[0][0]  # StatusCode.OK
        )
        mock_span.add_event.assert_called_once_with(
            "success", {"message": "Operation completed successfully"}
        )

    def test_handle_span_error(self):
        """Test handling span errors."""
        mock_span = Mock()
        test_exception = Exception("Test error")

        handle_span_error(mock_span, test_exception)

        mock_span.set_status.assert_called_once()
        mock_span.record_exception.assert_called_once_with(test_exception)

    @patch('app.telemetry.tracing.trace.get_tracer_provider')
    def test_flush_telemetry_spans(self, mock_get_provider):
        """Test flushing telemetry spans."""
        mock_provider = Mock()
        mock_provider.force_flush = Mock()
        mock_get_provider.return_value = mock_provider

        flush_telemetry_spans()

        mock_provider.force_flush.assert_called_once()


class TestLangSmithIntegration:
    """Test LangSmith integration functionality."""

    @patch('app.telemetry.langsmith.LangSmithClient')
    def test_get_langsmith_client_with_api_key(self, mock_langsmith_class):
        """Test getting LangSmith client with API key."""
        mock_client = Mock()
        mock_langsmith_class.return_value = mock_client

        with patch.dict(os.environ, {'LANGSMITH_API_KEY': 'test-key'}):
            client = get_langsmith_client()

        assert client == mock_client
        mock_langsmith_class.assert_called_once_with(api_key='test-key')

    def test_get_langsmith_client_no_api_key(self):
        """Test getting LangSmith client without API key."""
        with patch.dict(os.environ, {}, clear=True):
            client = get_langsmith_client()

        assert client is None

    @patch('app.telemetry.langsmith.get_langsmith_client')
    def test_create_langsmith_run(self, mock_get_client):
        """Test creating LangSmith run."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_run = Mock(id="run-123")
        mock_client.create_run.return_value = mock_run

        run_id = create_langsmith_run(
            name="test-run",
            project="test-project",
            run_type="llm",
            inputs={"query": "test"}
        )

        assert run_id == "run-123"
        mock_client.create_run.assert_called_once()

    @patch('app.telemetry.langsmith.get_langsmith_client')
    def test_create_langsmith_run_no_client(self, mock_get_client):
        """Test creating LangSmith run when client is not available."""
        mock_get_client.return_value = None

        run_id = create_langsmith_run(
            name="test-run",
            project="test-project",
            run_type="llm",
            inputs={"query": "test"}
        )

        assert run_id is None

    @patch('app.telemetry.langsmith.get_langsmith_client')
    def test_update_langsmith_run(self, mock_get_client):
        """Test updating LangSmith run."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        update_langsmith_run(
            run_id="run-123",
            outputs={"result": "success"},
            end_time="2023-12-01T10:00:00Z"
        )

        mock_client.update_run.assert_called_once_with(
            run_id="run-123",
            outputs={"result": "success"},
            end_time="2023-12-01T10:00:00Z"
        )

    @patch('app.telemetry.langsmith.get_langsmith_client')
    def test_update_langsmith_run_no_client(self, mock_get_client):
        """Test updating LangSmith run when client is not available."""
        mock_get_client.return_value = None

        # Should not raise an exception
        update_langsmith_run(
            run_id="run-123",
            outputs={"result": "success"}
        )


class TestTracerAccess:
    """Test tracer access and usage."""

    def test_tracer_available(self):
        """Test that tracer is available for import."""
        assert tracer is not None

    @patch('app.telemetry.tracing.tracer')
    def test_tracer_start_span(self, mock_tracer):
        """Test starting a span with tracer."""
        mock_span = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracer.start_span.return_value = mock_context_manager

        with mock_tracer.start_span("test-operation") as span:
            assert span == mock_span

        mock_tracer.start_span.assert_called_once_with("test-operation")


class TestModuleImports:
    """Test that all telemetry modules can be imported correctly."""

    def test_import_telemetry_tracing(self):
        """Test importing tracing module."""
        from app.telemetry import setup_telemetry, tracer, flush_telemetry_spans
        assert callable(setup_telemetry)
        assert tracer is not None
        assert callable(flush_telemetry_spans)

    def test_import_telemetry_langsmith(self):
        """Test importing LangSmith module."""
        from app.telemetry import get_langsmith_client, create_langsmith_run, update_langsmith_run
        assert callable(get_langsmith_client)
        assert callable(create_langsmith_run)
        assert callable(update_langsmith_run)

    def test_import_span_helpers(self):
        """Test importing span helper functions."""
        from app.telemetry import (
            set_genai_span_attributes,
            set_hitl_span_attributes,
            set_sandbox_span_attributes,
            handle_span_error,
            handle_span_success
        )
        assert callable(set_genai_span_attributes)
        assert callable(set_hitl_span_attributes)
        assert callable(set_sandbox_span_attributes)
        assert callable(handle_span_error)
        assert callable(handle_span_success)

    def test_telemetry_module_all_exports(self):
        """Test that __all__ exports are correct."""
        import app.telemetry as telemetry_module

        expected_exports = [
            "setup_telemetry",
            "tracer",
            "langsmith_client",
            "flush_telemetry_spans",
            "set_genai_span_attributes",
            "set_hitl_span_attributes",
            "set_sandbox_span_attributes",
            "handle_span_error",
            "handle_span_success",
            "TELEMETRY_CONFIG",
            "SPAN_ATTRIBUTES",
            "get_langsmith_client",
            "create_langsmith_run",
            "update_langsmith_run"
        ]

        for export in expected_exports:
            assert hasattr(telemetry_module, export), f"Missing export: {export}"


class TestErrorHandling:
    """Test error handling in telemetry functions."""

    @patch('app.telemetry.tracing.TracerProvider')
    def test_setup_telemetry_handles_exceptions(self, mock_tracer_provider):
        """Test that telemetry setup handles exceptions gracefully."""
        mock_tracer_provider.side_effect = Exception("Provider setup failed")

        # Should not raise an exception
        try:
            tracer_result, langsmith_client = setup_telemetry()
            # Depending on implementation, might return None or fallback
        except Exception as e:
            pytest.fail(f"setup_telemetry should handle exceptions gracefully: {e}")

    def test_span_attribute_helpers_handle_none_span(self):
        """Test that span attribute helpers handle None span gracefully."""
        # Should not raise an exception
        set_genai_span_attributes(None, "gpt-4o", 100, 50, 150, 0.005)
        set_hitl_span_attributes(None, "session-123", "generate_code", True, "approve")
        set_sandbox_span_attributes(None, "sandbox-123", 600, 1500, 0)
        handle_span_success(None, "Success message")
        handle_span_error(None, Exception("Test error"))