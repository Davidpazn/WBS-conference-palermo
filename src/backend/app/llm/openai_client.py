"""OpenAI client configuration and LLM functions."""

import os
import time
from time import sleep
from typing import Optional
from openai import OpenAI
from opentelemetry import trace


# Model configuration
MODEL = os.getenv("NB1_OPENAI_MODEL", "gpt-5-nano")  # keep cost low; upgrade as you like
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")

# OpenAI client configuration will be created lazily

# System prompt for code generation
SYSTEM_PROMPT = (
    "You are a disciplined coding agent. Write a *single* runnable Python script.\n"
    "Constraints:\n"
    "- No external files; everything in one script.\n"
    "- Print clear results to stdout.\n"
    "- If tests are needed, use simple asserts in __main__ instead of pytest.\n"
    "Output strictly as a fenced code block:```python ...``` and nothing else."
)


def get_openai_client() -> OpenAI:
    """Get OpenAI client instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return OpenAI(api_key=api_key)


def llm_generate_code(task: str, max_retries: int = 3) -> str:
    """Ask the model to produce a single Python script as a fenced block."""
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("llm_generate_code") as span:
        span.set_attribute("task", task[:100])  # Truncate for telemetry
        span.set_attribute("model", MODEL)
        span.set_attribute("max_retries", max_retries)

        client = get_openai_client()

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": task},
                    ],
                )
                latency_ms = (time.time() - start_time) * 1000

                # Extract text from the response
                text = resp.choices[0].message.content

                # Add telemetry attributes
                span.set_attribute("latency_ms", latency_ms)
                span.set_attribute("attempt", attempt + 1)
                span.set_attribute("success", True)

                if hasattr(resp, 'usage') and resp.usage:
                    span.set_attribute("tokens_in", resp.usage.prompt_tokens)
                    span.set_attribute("tokens_out", resp.usage.completion_tokens)

                from .code_extraction import extract_code_blocks
                code = extract_code_blocks(text)
                span.set_attribute("code_extracted", bool(code))
                return code

            except Exception as e:
                span.set_attribute("error", str(e))
                span.set_attribute("attempt", attempt + 1)

                if attempt == max_retries - 1:
                    span.set_attribute("success", False)
                    raise e

                # Exponential backoff
                sleep_time = 2 ** attempt
                span.add_event(f"Retrying after {sleep_time}s due to: {str(e)}")
                sleep(sleep_time)

        return ""