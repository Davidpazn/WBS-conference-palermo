"""Code extraction utilities for LLM responses."""

import re


def extract_code_blocks(text: str, language: str = None) -> str:
    """Extract all code blocks from text, returning concatenated code as a string."""
    if not text:
        return ""

    # Pattern to match fenced code blocks with any language or no language specifier
    # This pattern handles:
    # - ```python\ncode\n```
    # - ```javascript\ncode\n```
    # - ```\ncode\n```
    # - Whitespace around language specifiers
    pattern = r"```(?:[a-zA-Z0-9_+-]*\s*)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Strip each match and normalize indentation
        code_blocks = []
        for match in matches:
            stripped = match.strip()
            if stripped:  # Only include non-empty blocks
                # Normalize indentation properly
                import textwrap
                # First, restore consistent indentation by finding the pattern
                lines = match.split('\n')  # Use original match, not stripped
                if lines:
                    # Remove empty lines at start and end
                    while lines and not lines[0].strip():
                        lines.pop(0)
                    while lines and not lines[-1].strip():
                        lines.pop()

                    if lines:
                        # Use textwrap.dedent on the original content to normalize
                        restored = '\n'.join(lines)
                        normalized = textwrap.dedent(restored).strip()
                        code_blocks.append(normalized)

        # Join multiple blocks with double newlines
        return "\n\n".join(code_blocks)

    # If no fenced blocks found, return empty string
    return ""