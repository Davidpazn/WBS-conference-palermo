"""HITL (Human-in-the-Loop) workflow implementation."""

import os
import time
import uuid
from typing import Dict, Any

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# LangSmith imports
from langsmith import traceable

# Local imports
from ..models.hitl_models import HITLState, HITLStage, HITLDecision, ApprovalPayload, DecisionPayload, create_hitl_session
from ..llm.openai_client import llm_generate_code, MODEL
from ..e2b.execution import run_in_e2b, summarize_execution, failed
from ..e2b.artifacts import download_all_as_tar, extract_output_from_execution
from ..e2b.sandbox import new_persistent_sandbox
from ..telemetry.tracing import tracer


# Global variable for persistent sandbox (to be set externally)
PERSIST_SBX = None


@traceable(name="generate_code_node")
def generate_code_node(state: HITLState) -> HITLState:
    """Generate code using OpenAI with full tracing."""
    with tracer.start_as_current_span("generate_code_node") as span:
        span.set_attribute("node", "generate_code")

        # Access Pydantic model fields directly
        user_query = state.user_query
        span.set_attribute("user_query", user_query)

        try:
            # Use existing LLM generation
            start_time = time.time()
            code = llm_generate_code(user_query)
            latency_ms = (time.time() - start_time) * 1000

            # Set span attributes for GenAI
            span.set_attribute("gen_ai.system", "openai")
            span.set_attribute("gen_ai.operation.name", "chat.completions")
            span.set_attribute("gen_ai.request.model", MODEL)
            span.set_attribute("gen_ai.prompt", user_query[:500])  # Truncated
            span.set_attribute("gen_ai.completion", code[:500])  # Truncated
            span.set_attribute("latency_ms", latency_ms)

            # Create new state with updated fields
            updated_state = state.model_copy(update={
                "generated_code": code,
                "stage": HITLStage.REVIEW_CODE
            })

            span.set_status(Status(StatusCode.OK))
            print(f"📝 Code generated ({len(code)} chars)")

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            updated_state = state.model_copy(update={"error_message": str(e)})

    return updated_state


@traceable(name="code_review_node")
def code_review_node(state: HITLState) -> HITLState:
    """Generate automated code review comments."""
    with tracer.start_as_current_span("code_review_node") as span:
        span.set_attribute("node", "code_review")
        span.set_attribute("hitl.stage", HITLStage.REVIEW_CODE.value)

        try:
            # Get generated code
            generated_code = state.generated_code or ""

            # Generate basic review comments based on code analysis
            review_comments = []

            if generated_code:
                # Basic code quality checks
                if "def " in generated_code and '"""' not in generated_code and "'''" not in generated_code:
                    review_comments.append("Consider adding docstrings to functions")

                if "try:" not in generated_code and "except:" not in generated_code:
                    review_comments.append("Consider adding error handling")

                if len(generated_code.split('\n')) > 20:
                    review_comments.append("Consider breaking down large functions")

                if not any(line.strip().startswith('#') for line in generated_code.split('\n')):
                    review_comments.append("Consider adding comments for clarity")

                # If no specific issues found, add generic positive feedback
                if not review_comments:
                    review_comments.append("Code looks well-structured")
            else:
                review_comments.append("No code to review")

            # Create updated state
            updated_state = state.model_copy(update={
                "review_comments": review_comments,
                "approval_needed": True,
                "stage": HITLStage.REVIEW_CODE
            })

            span.set_attribute("review_comments_count", len(review_comments))
            span.set_status(Status(StatusCode.OK))
            print(f"🔍 Code reviewed - {len(review_comments)} comments generated")

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            updated_state = state.model_copy(update={
                "error_message": str(e),
                "review_comments": ["Error during code review"],
                "approval_needed": True
            })

    return updated_state


@traceable(name="execute_code_node")
def execute_code_node(state: HITLState) -> HITLState:
    """Execute code in E2B sandbox with tracing."""
    with tracer.start_as_current_span("execute_code_node") as span:
        span.set_attribute("node", "execute_code")

        try:
            start_time = time.time()

            # Get code to execute from Pydantic model
            generated_code = state.generated_code or "print('No code to execute')"

            # Use the persistent sandbox for execution to preserve files for artifact creation
            if PERSIST_SBX:
                try:
                    exec_result = PERSIST_SBX.run_code(generated_code)
                    execution = summarize_execution(exec_result)
                except Exception as e:
                    print(f"⚠️  Persistent sandbox failed: {e}, falling back to temporary")
                    execution = run_in_e2b(generated_code)
            else:
                # Fallback to temporary sandbox
                execution = run_in_e2b(generated_code)

            latency_ms = (time.time() - start_time) * 1000

            # Set sandbox attributes
            span.set_attribute("sandbox.success", not failed(execution))
            span.set_attribute("latency_ms", latency_ms)

            if failed(execution):
                error_message = str(execution.get("stderr", "Unknown error"))
                span.set_attribute("sandbox.error", error_message)
                updated_state = state.model_copy(update={
                    "execution_result": execution,
                    "error_message": error_message,
                    "stage": HITLStage.SAVE_ARTIFACTS
                })
            else:
                updated_state = state.model_copy(update={
                    "execution_result": execution,
                    "stage": HITLStage.SAVE_ARTIFACTS
                })
                span.set_status(Status(StatusCode.OK))

            print(f"🏃 Code executed in E2B ({latency_ms:.1f}ms)")

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            updated_state = state.model_copy(update={
                "error_message": str(e),
                "stage": HITLStage.SAVE_ARTIFACTS
            })

    return updated_state


@traceable(name="save_artifacts_node")
def save_artifacts_node(state: HITLState) -> HITLState:
    """Save project artifacts as tar for download."""
    with tracer.start_as_current_span("save_artifacts_node") as span:
        span.set_attribute("node", "save_artifacts")

        try:
            # Get required state values from Pydantic model
            thread_id = state.session_id or f"unknown_{uuid.uuid4().hex[:8]}"
            user_query = state.user_query or "Unknown task"
            generated_code = state.generated_code or "# No code generated"

            artifacts = {}

            # Create or use persistent sandbox for artifact creation
            sandbox_to_use = PERSIST_SBX
            # Create sandbox if one doesn't exist
            if not sandbox_to_use:
                sandbox_to_use = new_persistent_sandbox()

            try:
                # Write the generated code to a file in the persistent sandbox
                write_code = f'''
import os
project_dir = "/home/user/hitl_project_{thread_id}"
os.makedirs(project_dir, exist_ok=True)

# Write the main code
with open(os.path.join(project_dir, "main.py"), "w") as f:
    f.write({repr(generated_code)})

# Write a README
with open(os.path.join(project_dir, "README.md"), "w") as f:
    f.write(f"# HITL Generated Project\\n\\nTask: {repr(user_query)}\\n\\nGenerated on: {{__import__('datetime').datetime.now()}}\\n")

print(f"Project saved to {{project_dir}}")
'''
                result = sandbox_to_use.run_code(write_code)
                output = extract_output_from_execution(result)
                print(f"📁 Project structure created: {output}")

                # Create tar archive
                tar_path = download_all_as_tar(
                    sandbox_to_use,
                    remote_root=f"/home/user/hitl_project_{thread_id}",
                    local_tar_path=f"artifacts/hitl_project_{thread_id}.tar.gz"
                )

                artifacts["artifacts_path"] = tar_path
                span.set_attribute("artifact.path", tar_path)
                print(f"📦 Artifacts saved to: {tar_path}")

            except Exception as e:
                print(f"⚠️  Could not save artifacts: {e}")
                span.set_attribute("artifact.error", str(e))
                artifacts["error"] = str(e)

            # Create updated state
            updated_state = state.model_copy(update={
                "artifacts": artifacts,
                "stage": HITLStage.COMPLETE
            })

            span.set_status(Status(StatusCode.OK))

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            updated_state = state.model_copy(update={
                "artifacts": {"error": str(e)},
                "stage": HITLStage.COMPLETE
            })

    return updated_state


def create_hitl_graph() -> StateGraph:
    """Create the HITL coding workflow graph."""

    # Create the graph
    graph = StateGraph(HITLState)

    # Add nodes
    graph.add_node("generate_code", generate_code_node)
    graph.add_node("code_review", code_review_node)
    graph.add_node("execute_code", execute_code_node)
    graph.add_node("save_artifacts", save_artifacts_node)

    # Define routing logic
    def decide_next_node(state: HITLState) -> str:
        """Route to next node based on current state and human decisions."""

        # Check for human decision
        if state.user_decision:
            decision = state.user_decision

            if decision == HITLDecision.APPROVE:
                return "execute_code"
            elif decision == HITLDecision.EDIT:
                return "execute_code"
            elif decision == HITLDecision.REJECT:
                return END

        # Default routing based on stage
        current_stage = state.stage
        if current_stage == HITLStage.REVIEW_CODE:
            return "execute_code"
        else:
            return END

    # Add edges
    graph.add_edge(START, "generate_code")
    graph.add_edge("generate_code", "code_review")

    # Conditional routing after code review
    graph.add_conditional_edges(
        "code_review",
        decide_next_node,
        {
            "execute_code": "execute_code",
            END: END
        }
    )

    graph.add_edge("execute_code", "save_artifacts")
    graph.add_edge("save_artifacts", END)

    # Compile and return the graph
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Create the compiled graph
hitl_graph = create_hitl_graph()


def start_hitl_workflow(initial_state: HITLState) -> HITLState:
    """Start the HITL workflow with the given initial state."""
    with tracer.start_as_current_span("start_hitl_workflow") as span:
        span.set_attribute("user_query", initial_state.user_query)
        span.set_attribute("session_id", initial_state.session_id)

        graph = create_hitl_graph()
        config = {"configurable": {"thread_id": initial_state.session_id}}

        # Run the workflow
        result = graph.invoke(initial_state, config=config)

        span.set_attribute("stage", result.stage.value if result.stage else "unknown")

        return result


def resume_hitl_workflow(session_id: str, user_decision: HITLDecision) -> HITLState:
    """Resume the HITL workflow with human decision."""
    with tracer.start_as_current_span("resume_hitl_workflow") as span:
        span.set_attribute("session_id", session_id)
        span.set_attribute("hitl.decision", user_decision.value)

        graph = create_hitl_graph()
        config = {"configurable": {"thread_id": session_id}}

        try:
            # Get the current state from the checkpointer first
            current_state = graph.get_state(config)
            if current_state and current_state.values:
                # Update the state with user decision
                updated_state = current_state.values.model_copy(update={
                    "user_decision": user_decision
                })

                # Continue execution with updated state
                result = graph.invoke(updated_state, config=config)
            else:
                # Fallback: create minimal state if checkpoint is missing
                print("⚠️  No checkpoint found, creating minimal state")
                from ..models.hitl_models import HITLState, HITLStage
                minimal_state = HITLState(
                    user_query="Resume workflow",
                    session_id=session_id,
                    stage=HITLStage.REVIEW_CODE,
                    user_decision=user_decision
                )
                result = graph.invoke(minimal_state, config=config)

            return result

        except Exception as e:
            print(f"❌ Error resuming workflow: {e}")
            # Return error state
            from ..models.hitl_models import HITLState, HITLStage
            return HITLState(
                user_query="Error state",
                session_id=session_id,
                stage=HITLStage.COMPLETE,
                artifacts={"error": str(e)}
            )


def display_approval_request(state: dict):
    """Display the approval request for human review."""
    if "approval_payload" in state and state["approval_payload"]:
        payload = state["approval_payload"]

        print("\n" + "="*60)
        print("🤖 HUMAN REVIEW REQUIRED")
        print("="*60)
        print(f"Task: {payload.task}")
        print(f"Stage: {payload.stage.value}")
        print(f"\nGenerated Code:")
        print("-" * 40)
        print(payload.code)
        print("-" * 40)
        print(f"\n{payload.suggestion}")
        print(f"Options: {', '.join(payload.options)}")
        print("="*60)

        return True
    return False