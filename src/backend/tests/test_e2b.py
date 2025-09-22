"""Tests for E2B module functionality."""

import pytest
from unittest.mock import Mock, patch, mock_open
import tempfile
import os

from app.e2b import (
    new_persistent_sandbox,
    run_in_e2b,
    summarize_execution,
    download_all_as_tar
)


class TestSandboxManagement:
    """Test sandbox lifecycle management."""

    @patch('app.e2b.sandbox.Sandbox')
    def test_new_persistent_sandbox(self, mock_sandbox_class, mock_e2b_sandbox):
        """Test creating a new persistent sandbox."""
        mock_sandbox_class.create.return_value = mock_e2b_sandbox

        result = new_persistent_sandbox(timeout_seconds=600)

        assert result == mock_e2b_sandbox
        mock_sandbox_class.create.assert_called_once_with(timeout=600)

    @patch('app.e2b.sandbox.Sandbox')
    def test_new_persistent_sandbox_default_timeout(self, mock_sandbox_class, mock_e2b_sandbox):
        """Test creating sandbox with default timeout."""
        mock_sandbox_class.create.return_value = mock_e2b_sandbox

        result = new_persistent_sandbox()

        assert result == mock_e2b_sandbox
        mock_sandbox_class.create.assert_called_once_with(timeout=600)  # Default from PERSIST_TIMEOUT_SECONDS


class TestCodeExecution:
    """Test code execution functionality."""

    @patch('app.e2b.execution.Sandbox')
    def test_run_in_e2b_success(self, mock_sandbox_class, mock_e2b_sandbox):
        """Test successful code execution in E2B."""
        mock_sandbox_class.create.return_value.__enter__.return_value = mock_e2b_sandbox

        code = "print('Hello, World!')"
        result = run_in_e2b(code)

        assert result is not None
        mock_e2b_sandbox.run_code.assert_called_once_with(code)

    def test_summarize_execution_success(self):
        """Test summarizing successful execution results."""
        mock_result = Mock()
        mock_result.logs.stdout = "Hello, World!"
        mock_result.logs.stderr = ""
        mock_result.results = [Mock(text="Hello, World!", is_main_result=True)]
        mock_result.error = None

        summary = summarize_execution(mock_result)

        assert summary["success"] is True
        assert summary["stdout"] == "Hello, World!"
        assert summary["stderr"] == ""
        assert len(summary["results"]) == 1
        assert summary["error"] is None

    def test_summarize_execution_error(self):
        """Test summarizing execution with error."""
        mock_result = Mock()
        mock_result.logs.stdout = ""
        mock_result.logs.stderr = "SyntaxError: invalid syntax"
        mock_result.results = []
        mock_result.error = Mock(name="SyntaxError", value="invalid syntax")

        summary = summarize_execution(mock_result)

        assert summary["success"] is False
        assert summary["stderr"] == "SyntaxError: invalid syntax"
        assert summary["error"] is not None


class TestArtifactManagement:
    """Test artifact management functionality."""

    @patch('app.e2b.artifacts.Sandbox')
    def test_download_all_as_tar(self, mock_sandbox_class, mock_e2b_sandbox, temp_tar_file):
        """Test downloading all files as tar archive."""
        # Mock the two run_code calls: tar creation and base64 encoding
        import base64
        test_content = b"test tar content"
        encoded_content = base64.b64encode(test_content).decode()

        mock_e2b_sandbox.run_code.side_effect = [
            # First call: tar creation
            Mock(
                logs=Mock(stdout="tar created successfully", stderr=""),
                error=None
            ),
            # Second call: base64 encoding
            Mock(
                logs=Mock(stdout=f"BASE64_START\n{encoded_content}\nBASE64_END", stderr=""),
                error=None
            )
        ]

        result = download_all_as_tar(
            mock_e2b_sandbox,
            remote_root="/home/user",
            local_tar_path=temp_tar_file
        )

        assert result == temp_tar_file
        # Verify that run_code was called twice
        assert mock_e2b_sandbox.run_code.call_count == 2



class TestModuleImports:
    """Test that all E2B modules can be imported correctly."""

    def test_import_e2b_sandbox(self):
        """Test importing sandbox module."""
        from app.e2b import new_persistent_sandbox
        assert callable(new_persistent_sandbox)

    def test_import_e2b_execution(self):
        """Test importing execution module."""
        from app.e2b import run_in_e2b, summarize_execution
        assert callable(run_in_e2b)
        assert callable(summarize_execution)

    def test_import_e2b_artifacts(self):
        """Test importing artifacts module."""
        from app.e2b import download_all_as_tar
        assert callable(download_all_as_tar)

    def test_e2b_module_all_exports(self):
        """Test that __all__ exports are correct."""
        import app.e2b as e2b_module

        expected_exports = [
            "new_persistent_sandbox",
            "run_in_e2b",
            "summarize_execution",
            "download_all_as_tar"
        ]

        for export in expected_exports:
            assert hasattr(e2b_module, export), f"Missing export: {export}"