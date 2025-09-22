"""Tests for LLM module functionality."""

import pytest
from unittest.mock import Mock, patch

from app.llm import (
    llm_generate_code,
    extract_code_blocks,
    get_openai_client
)


class TestOpenAIClient:
    """Test OpenAI client functionality."""

    @patch('app.llm.openai_client.OpenAI')
    def test_get_openai_client(self, mock_openai_class, mock_openai_client):
        """Test getting OpenAI client instance."""
        mock_openai_class.return_value = mock_openai_client

        client = get_openai_client()

        assert client == mock_openai_client
        mock_openai_class.assert_called_once_with(api_key="test-key")

    @patch('app.llm.openai_client.get_openai_client')
    @patch('app.llm.openai_client.trace')
    def test_llm_generate_code_success(self, mock_trace, mock_get_client, mock_openai_client):
        """Test successful code generation."""
        mock_get_client.return_value = mock_openai_client
        mock_span = Mock()
        mock_trace.get_tracer.return_value.start_as_current_span.return_value.__enter__.return_value = mock_span

        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="```python\nprint('Generated code')\n```"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
        mock_openai_client.chat.completions.create.return_value = mock_response

        task = "Write a function to calculate factorial"
        result = llm_generate_code(task)

        assert result == "print('Generated code')"
        mock_openai_client.chat.completions.create.assert_called_once()

        # Verify the call arguments
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-5-nano"
        assert len(call_args[1]["messages"]) == 2
        assert call_args[1]["messages"][0]["role"] == "system"
        assert call_args[1]["messages"][1]["role"] == "user"
        assert call_args[1]["messages"][1]["content"] == task

    @patch('app.llm.openai_client.get_openai_client')
    @patch('app.llm.openai_client.trace')
    def test_llm_generate_code_with_retry(self, mock_trace, mock_get_client, mock_openai_client):
        """Test code generation with retry on failure."""
        mock_get_client.return_value = mock_openai_client
        mock_span = Mock()
        mock_trace.get_tracer.return_value.start_as_current_span.return_value.__enter__.return_value = mock_span

        # First call fails, second succeeds
        mock_success_response = Mock()
        mock_success_response.choices = [Mock(message=Mock(content="```python\nprint('retry success')\n```"))]
        mock_success_response.usage = Mock(prompt_tokens=10, completion_tokens=5)

        mock_openai_client.chat.completions.create.side_effect = [
            Exception("API Error"),
            mock_success_response
        ]

        with patch('app.llm.openai_client.sleep'):  # Mock sleep to speed up test
            result = llm_generate_code("test task")

        assert result == "print('retry success')"
        assert mock_openai_client.chat.completions.create.call_count == 2


class TestCodeExtraction:
    """Test code extraction functionality."""

    def test_extract_code_blocks_single_block(self):
        """Test extracting single code block."""
        content = """
        Here's the solution:

        ```python
        def factorial(n):
            return 1 if n <= 1 else n * factorial(n-1)
        ```

        This function calculates factorial recursively.
        """

        result = extract_code_blocks(content)

        expected = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"
        assert result == expected

    def test_extract_code_blocks_multiple_blocks(self):
        """Test extracting multiple code blocks."""
        content = """
        First function:

        ```python
        def add(a, b):
            return a + b
        ```

        Second function:

        ```python
        def multiply(a, b):
            return a * b
        ```
        """

        result = extract_code_blocks(content)

        expected = "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b"
        assert result == expected

    def test_extract_code_blocks_no_blocks(self):
        """Test extracting when no code blocks exist."""
        content = "This is just text with no code blocks."

        result = extract_code_blocks(content)

        assert result == ""

    def test_extract_code_blocks_different_languages(self):
        """Test extracting blocks with different language specifiers."""
        content = """
        ```javascript
        function hello() { console.log('hello'); }
        ```

        ```python
        print('hello')
        ```

        ```
        echo "hello"
        ```
        """

        result = extract_code_blocks(content)

        expected = "function hello() { console.log('hello'); }\n\nprint('hello')\n\necho \"hello\""
        assert result == expected

    def test_extract_code_blocks_with_whitespace(self):
        """Test extracting blocks with leading/trailing whitespace."""
        content = """
        ```python

        def test():
            pass

        ```
        """

        result = extract_code_blocks(content)

        expected = "def test():\n    pass"
        assert result == expected


class TestModuleImports:
    """Test that all LLM modules can be imported correctly."""

    def test_import_llm_openai_client(self):
        """Test importing OpenAI client module."""
        from app.llm import llm_generate_code, get_openai_client
        assert callable(llm_generate_code)
        assert callable(get_openai_client)

    def test_import_llm_code_extraction(self):
        """Test importing code extraction module."""
        from app.llm import extract_code_blocks
        assert callable(extract_code_blocks)

    def test_llm_module_all_exports(self):
        """Test that __all__ exports are correct."""
        import app.llm as llm_module

        expected_exports = [
            "llm_generate_code",
            "get_openai_client",
            "extract_code_blocks"
        ]

        for export in expected_exports:
            assert hasattr(llm_module, export), f"Missing export: {export}"


class TestErrorHandling:
    """Test error handling in LLM functions."""

    @patch('app.llm.openai_client.get_openai_client')
    @patch('app.llm.openai_client.trace')
    def test_llm_generate_code_max_retries_exceeded(self, mock_trace, mock_get_client, mock_openai_client):
        """Test behavior when max retries are exceeded."""
        mock_get_client.return_value = mock_openai_client
        mock_span = Mock()
        mock_trace.get_tracer.return_value.start_as_current_span.return_value.__enter__.return_value = mock_span
        mock_openai_client.chat.completions.create.side_effect = Exception("Persistent API Error")

        with patch('app.llm.openai_client.sleep'):  # Mock sleep to speed up test
            with pytest.raises(Exception, match="Persistent API Error"):
                llm_generate_code("test task")

        # Should retry 3 times (initial + 2 retries)
        assert mock_openai_client.chat.completions.create.call_count == 3

    def test_extract_code_blocks_malformed_markdown(self):
        """Test extraction with malformed markdown."""
        content = """
        ```python
        def incomplete_function(
        # Missing closing backticks

        ```
        def complete_function():
            pass
        ```
        """

        result = extract_code_blocks(content)

        # With malformed markdown, the first opening ``` consumes until the first closing ```
        # So we should only get the incomplete function
        assert "def incomplete_function(" in result
        assert "Missing closing backticks" in result
        # The complete function is not extracted due to malformed structure
        assert "def complete_function():" not in result