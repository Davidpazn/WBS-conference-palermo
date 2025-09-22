"""
AI Agents Backend Application

This package contains all the extracted functionality from NB1_E2B_coding_agent_v2.ipynb,
organized into proper modules for reuse in the full-stack application.
"""

# Version info
__version__ = "0.1.0"
__title__ = "AI Agents Backend"
__description__ = "Backend application for AI agent workflows with E2B, LangGraph, and OpenTelemetry"

# Import main components for easy access
from .agents import coding_agent, SelfContainedHITLWorkflow
from .e2b import run_in_e2b, new_persistent_sandbox
from .llm import llm_generate_code
from .models import HITLState, HITLDecision, DecisionPayload
from .telemetry import tracer, setup_telemetry

# Export commonly used functions
__all__ = [
    # Core agent functionality
    "coding_agent",
    "SelfContainedHITLWorkflow",

    # E2B integration
    "run_in_e2b",
    "new_persistent_sandbox",

    # LLM integration
    "llm_generate_code",

    # Models
    "HITLState",
    "HITLDecision",
    "DecisionPayload",

    # Telemetry
    "tracer",
    "setup_telemetry",

    # Package info
    "__version__",
    "__title__",
    "__description__"
]