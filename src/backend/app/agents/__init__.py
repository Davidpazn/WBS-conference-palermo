"""Agent implementations and workflows."""

from .coding_agent import coding_agent

from .hitl_workflow import (
    generate_code_node,
    code_review_node,
    execute_code_node,
    save_artifacts_node,
    create_hitl_graph,
    start_hitl_workflow,
    resume_hitl_workflow,
    display_approval_request,
    hitl_graph
)

from .self_contained_workflow import SelfContainedHITLWorkflow

__all__ = [
    # Basic coding agent
    "coding_agent",

    # HITL workflow nodes
    "generate_code_node",
    "code_review_node",
    "execute_code_node",
    "save_artifacts_node",

    # HITL workflow management
    "create_hitl_graph",
    "start_hitl_workflow",
    "resume_hitl_workflow",
    "display_approval_request",
    "hitl_graph",

    # Self-contained workflow
    "SelfContainedHITLWorkflow"
]