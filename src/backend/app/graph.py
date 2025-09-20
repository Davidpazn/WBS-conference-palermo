import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AppState
from .nodes_memory import recall_memory_node, save_memory_node
from .nodes import plan_node, act_node, observe_node
from .hitl_nodes import (
    code_generation_node,
    hitl_code_review_node,
    route_after_code_review,
    sandbox_execution_node,
    route_after_execution,
    hitl_execution_review_node,
    route_after_execution_review
)

def build_graph():
    """Build LangGraph with HITL support for coding agent workflow."""
    g = StateGraph(AppState)

    # Memory integration
    g.add_node("recall", recall_memory_node)

    # HITL Coding Agent Flow
    g.add_node("code_generation", code_generation_node)
    g.add_node("code_review", hitl_code_review_node)
    g.add_node("execute_code", sandbox_execution_node)
    g.add_node("execution_review", hitl_execution_review_node)

    # Fallback legacy nodes (for non-coding workflows)
    g.add_node("plan", plan_node)
    g.add_node("act", act_node)
    g.add_node("observe", observe_node)

    # Memory write-back
    g.add_node("save", save_memory_node)

    # Define flow: memory recall -> code generation -> HITL review -> execution -> final save
    g.add_edge(START, "recall")

    # Route to coding workflow by default (configurable)
    use_coding_workflow = os.getenv("USE_CODING_WORKFLOW", "true").lower() == "true"
    if use_coding_workflow:
        g.add_edge("recall", "code_generation")
        g.add_edge("code_generation", "code_review")

        # Conditional routing after code review
        g.add_conditional_edges(
            "code_review",
            route_after_code_review,
            {
                "execute_code": "execute_code",
                "code_generation": "code_generation",  # Retry
                "end": "save"  # Reject
            }
        )

        # Conditional routing after execution
        g.add_conditional_edges(
            "execute_code",
            route_after_execution,
            {
                "code_generation": "code_generation",  # Retry with feedback
                "save": "save"  # Success
            }
        )

        # Optional execution review (can be disabled via env)
        enable_execution_review = os.getenv("ENABLE_EXECUTION_REVIEW", "false").lower() == "true"
        if enable_execution_review:
            g.add_conditional_edges(
                "execute_code",
                route_after_execution,
                {
                    "code_generation": "code_generation",
                    "execution_review": "execution_review",
                    "save": "save"
                }
            )
            g.add_conditional_edges(
                "execution_review",
                route_after_execution_review,
                {
                    "code_generation": "code_generation",
                    "save": "save"
                }
            )
    else:
        # Legacy flow for non-coding workflows
        g.add_edge("recall", "plan")
        g.add_edge("plan", "act")
        g.add_edge("act", "observe")
        g.add_edge("observe", "save")

    g.add_edge("save", END)

    # Use MemorySaver for HITL sessions (persistent within process)
    # Note: For production, implement proper SQLite persistence
    checkpointer = MemorySaver()
    print(f"[graph] Using MemorySaver for HITL checkpoints (in-memory persistence)")

    return g.compile(checkpointer=checkpointer)
