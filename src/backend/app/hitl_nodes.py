"""Human-in-the-loop (HITL) nodes for LangGraph with dynamic interrupts."""

import os
from typing import Any, Dict
from opentelemetry import trace

from langgraph.types import interrupt, Command
from .state import AppState
from src.infra.telemetry import get_current_span_context


def code_generation_node(state: AppState) -> AppState:
    """Generate code using OpenAI Responses API with tracing."""
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("node.code_generation") as span:
        span.set_attribute("node.name", "code_generation")
        span.set_attribute("user.query", state.user_query[:200])

        try:
            from openai import OpenAI
            import re

            client = OpenAI()
            model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

            # Create GenAI span
            with tracer.start_as_current_span("gen_ai.responses") as gen_span:
                gen_span.set_attribute("gen_ai.system", "openai")
                gen_span.set_attribute("gen_ai.operation.name", "responses")
                gen_span.set_attribute("gen_ai.request.model", model)
                gen_span.set_attribute("gen_ai.prompt", state.user_query[:1000])

                system_prompt = (
                    "You are a disciplined coding agent. Write a *single* runnable Python script.\n"
                    "Constraints:\n"
                    "- No external files; everything in one script.\n"
                    "- Print clear results to stdout.\n"
                    "- If tests are needed, use simple asserts in __main__ instead of pytest.\n"
                    "Output strictly as a fenced code block:```python ...``` and nothing else."
                )

                resp = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": state.user_query},
                    ],
                )

                # Extract code from response
                try:
                    text = resp.output[0].content[0].text
                except Exception:
                    text = getattr(resp, "output_text", str(resp))

                # Parse code blocks
                fence = re.compile(r"```python\n(.*?)```", re.DOTALL | re.IGNORECASE)
                m = fence.search(text)
                code = m.group(1).strip() if m else text.strip()

                gen_span.set_attribute("gen_ai.completion", code[:1000])
                span.set_attribute("code.size_bytes", len(code.encode()))

            state.generated_code = code
            state.messages.append({
                "role": "assistant",
                "content": f"Generated code:\n```python\n{code}\n```"
            })

            # Update trace context
            span_context = get_current_span_context()
            if span_context:
                state.span_context = span_context

        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error", True)
            state.result = f"Code generation failed: {e}"

    return state


def hitl_code_review_node(state: AppState) -> AppState:
    """Human-in-the-loop code review with dynamic interrupt."""
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("node.hitl_code_review") as span:
        span.set_attribute("node.name", "hitl_code_review")
        span.set_attribute("code.size_bytes", len(state.generated_code.encode()) if state.generated_code else 0)

        # Prepare payload for human review
        review_payload = {
            "code": state.generated_code,
            "task": state.user_query,
            "suggestion": "Please review the code and choose: approve, edit, or reject",
            "options": ["approve", "edit", "reject"],
            "stage": "code_review"
        }

        state.approval_payload = review_payload
        state.hitl_stage = "code_review"

        span.set_attribute("hitl.stage", "code_review")
        span.set_attribute("hitl.payload_size", len(str(review_payload)))

        # Dynamic interrupt - pauses execution until human input
        try:
            decision = interrupt(review_payload)
            state.approval_decision = decision
            span.set_attribute("hitl.decision", decision.get("decision", "unknown"))

            # Handle edited code if provided
            if decision.get("code") and decision.get("decision") == "approve":
                state.generated_code = decision["code"]
                span.set_attribute("code.edited", True)

        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error", True)
            # Set default approval for error cases
            state.approval_decision = {"decision": "reject", "reason": f"HITL error: {e}"}

    return state


def route_after_code_review(state: AppState) -> str:
    """Route graph based on human code review decision."""
    if not state.approval_decision:
        return "end"

    decision = state.approval_decision.get("decision", "reject")

    if decision == "approve":
        return "execute_code"
    elif decision == "edit":
        return "code_generation"  # Loop back to regenerate
    else:  # reject
        return "end"


def sandbox_execution_node(state: AppState) -> AppState:
    """Execute code in E2B sandbox with tracing."""
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("node.sandbox_execution") as span:
        span.set_attribute("node.name", "sandbox_execution")
        span.set_attribute("code.size_bytes", len(state.generated_code.encode()) if state.generated_code else 0)
        span.set_attribute("sandbox.provider", "e2b")

        try:
            from e2b_code_interpreter import Sandbox

            with Sandbox.create() as sbx:
                span.set_attribute("sandbox.id", sbx.sandbox_id)

                exec_result = sbx.run_code(state.generated_code)

                # Extract execution summary
                execution_summary = {}
                for attr in ("text", "output", "result"):
                    if hasattr(exec_result, attr):
                        execution_summary[attr] = getattr(exec_result, attr)

                # Logs API (preferred on newer SDKs)
                logs = getattr(exec_result, "logs", None)
                if logs is not None:
                    execution_summary["stdout"] = getattr(logs, "stdout", None)
                    execution_summary["stderr"] = getattr(logs, "stderr", None)

                state.sandbox_execution = execution_summary

                # Check for errors
                stderr = execution_summary.get("stderr")
                failed = stderr and str(stderr).strip()

                span.set_attribute("sandbox.success", not failed)
                if stderr:
                    span.set_attribute("sandbox.stderr_len", len(str(stderr)))

                if failed and state.code_retries < 1:
                    # Prepare for retry with error feedback
                    state.code_retries += 1
                    state.user_query += f"\n\nThe previous script failed. Here are the logs:\nSTDERR:\n{str(stderr)[:2000]}\n\nPlease FIX the bug and output ONLY a single ```python``` block containing the full corrected script."
                    span.set_attribute("sandbox.retry", True)
                else:
                    # Success or max retries reached
                    stdout = execution_summary.get("stdout", execution_summary.get("text", ""))
                    state.result = f"Code executed successfully.\nOutput:\n{stdout}" if not failed else f"Code execution failed after retries.\nError:\n{stderr}"
                    span.set_attribute("sandbox.final", True)

        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error", True)
            state.result = f"Sandbox execution failed: {e}"

    return state


def route_after_execution(state: AppState) -> str:
    """Route based on execution results."""
    if state.result:
        return "save"  # Final result ready
    else:
        return "code_generation"  # Retry with error feedback


def hitl_execution_review_node(state: AppState) -> AppState:
    """Optional HITL node to review execution results before final output."""
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("node.hitl_execution_review") as span:
        span.set_attribute("node.name", "hitl_execution_review")

        # Only interrupt if execution had warnings or user wants review
        execution = state.sandbox_execution or {}
        stderr = execution.get("stderr", "")

        # Skip HITL if clean execution (configurable)
        skip_clean_review = os.getenv("SKIP_CLEAN_EXECUTION_REVIEW", "true").lower() == "true"
        if skip_clean_review and not stderr:
            span.set_attribute("hitl.skipped", True)
            return state

        review_payload = {
            "execution_result": state.sandbox_execution,
            "code": state.generated_code,
            "task": state.user_query,
            "suggestion": "Review execution results. Approve to continue or request changes.",
            "options": ["approve", "regenerate", "abort"],
            "stage": "execution_review"
        }

        state.approval_payload = review_payload
        state.hitl_stage = "execution_review"

        span.set_attribute("hitl.stage", "execution_review")

        try:
            decision = interrupt(review_payload)
            state.approval_decision = decision
            span.set_attribute("hitl.decision", decision.get("decision", "unknown"))

        except Exception as e:
            span.record_exception(e)
            # Default to approve for error cases
            state.approval_decision = {"decision": "approve"}

    return state


def route_after_execution_review(state: AppState) -> str:
    """Route based on execution review decision."""
    if not state.approval_decision:
        return "save"

    decision = state.approval_decision.get("decision", "approve")

    if decision == "regenerate":
        return "code_generation"
    elif decision == "abort":
        state.result = "Execution aborted by user"
        return "save"
    else:  # approve
        return "save"