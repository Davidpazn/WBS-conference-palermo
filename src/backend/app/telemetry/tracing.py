"""OpenTelemetry and LangSmith tracing configuration."""

import os
import time
from typing import Optional, Tuple

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode

# LangSmith imports
from langsmith import Client as LangSmithClient, traceable
from langsmith.run_helpers import tracing_context


# Environment variables for telemetry
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "agents-demo")
SERVICE_NAME = os.getenv("SERVICE_NAME", "hitl-coding-agent")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")  # notebook, development, production


# Telemetry configuration constants
TELEMETRY_CONFIG = {
    "service_name": "hitl-coding-agent",
    "version": "0.1.0",
    "langsmith_endpoint": "https://api.smith.langchain.com/otel",
    "max_attribute_length": 500,  # For truncating long strings
    "default_project": "agents-demo"
}

# Span attribute keys
SPAN_ATTRIBUTES = {
    "GENAI": {
        "system": "gen_ai.system",
        "model": "gen_ai.request.model",
        "prompt_tokens": "gen_ai.usage.prompt_tokens",
        "completion_tokens": "gen_ai.usage.completion_tokens",
        "total_tokens": "gen_ai.usage.total_tokens",
        "cost": "gen_ai.usage.cost"
    },
    "HITL": {
        "session_id": "hitl.session_id",
        "stage": "hitl.stage",
        "approval_needed": "hitl.approval_needed",
        "user_decision": "hitl.user_decision"
    },
    "SANDBOX": {
        "id": "sandbox.id",
        "timeout_seconds": "sandbox.timeout_seconds",
        "execution_time_ms": "sandbox.execution_time_ms",
        "exit_code": "sandbox.exit_code"
    }
}


def setup_telemetry(
    service_name: str = SERVICE_NAME,
    service_version: str = SERVICE_VERSION,
    environment: str = ENVIRONMENT,
    langsmith_api_key: Optional[str] = LANGSMITH_API_KEY,
    langsmith_project: str = LANGSMITH_PROJECT
) -> Tuple[trace.Tracer, Optional[LangSmithClient]]:
    """
    Configure OpenTelemetry + LangSmith tracing.

    Returns:
        tuple: (tracer, langsmith_client)
    """
    try:
        # Initialize OpenTelemetry Resource
        resource = Resource.create({
            "service.name": service_name,
            "service.version": service_version,
            "environment": environment
        })

        # Set up tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Check for OTLP endpoint in environment
        otlp_endpoint = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')
        if otlp_endpoint:
            # Configure OTLP exporter
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            print(f"✅ OTLP exporter configured for endpoint: {otlp_endpoint}")
        elif langsmith_api_key:
            # Configure OTLP exporter for LangSmith
            otlp_exporter = OTLPSpanExporter(
                endpoint="https://api.smith.langchain.com/otel",
                headers={
                    "x-api-key": langsmith_api_key,
                    "Langsmith-Project": langsmith_project
                }
            )
            tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            print(f"✅ OTLP exporter configured for LangSmith project: {langsmith_project}")
        else:
            # Always add console exporter for local debugging/fallback
            console_exporter = ConsoleSpanExporter()
            tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))
            print("⚠️  No OTLP endpoint or LangSmith API key, using console exporter")

        # Get tracer
        tracer = trace.get_tracer(__name__)

        # Initialize LangSmith client
        langsmith_client = LangSmithClient(api_key=langsmith_api_key) if langsmith_api_key else None
        if langsmith_client:
            print(f"✅ LangSmith client initialized")

        print("✅ OpenTelemetry + LangSmith tracing configured")

        return tracer, langsmith_client
    except Exception as e:
        print(f"⚠️  Error setting up telemetry: {e}")
        # Return fallback tracer
        fallback_tracer = trace.get_tracer(__name__)
        return fallback_tracer, None


# Initialize telemetry components
tracer, langsmith_client = setup_telemetry()


def set_genai_span_attributes(
    span: Optional[trace.Span],
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cost: float
):
    """Set GenAI semantic convention attributes on a span."""
    if span is None:
        return

    span.set_attribute("gen_ai.system", "openai")
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
    span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
    span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
    span.set_attribute("gen_ai.usage.cost", cost)


def set_hitl_span_attributes(
    span: Optional[trace.Span],
    session_id: str,
    stage: str,
    approval_needed: bool,
    user_decision: str
):
    """Set HITL-specific attributes on a span."""
    if span is None:
        return

    span.set_attribute("hitl.session_id", session_id)
    span.set_attribute("hitl.stage", stage)
    span.set_attribute("hitl.approval_needed", approval_needed)
    span.set_attribute("hitl.user_decision", user_decision)


def set_sandbox_span_attributes(
    span: Optional[trace.Span],
    sandbox_id: str,
    timeout_seconds: int,
    execution_time_ms: int,
    exit_code: int
):
    """Set sandbox execution attributes on a span."""
    if span is None:
        return

    span.set_attribute("sandbox.id", sandbox_id)
    span.set_attribute("sandbox.timeout_seconds", timeout_seconds)
    span.set_attribute("sandbox.execution_time_ms", execution_time_ms)
    span.set_attribute("sandbox.exit_code", exit_code)


def handle_span_error(span: Optional[trace.Span], exception: Exception):
    """Handle errors in spans with proper exception recording."""
    if span is None:
        return

    span.record_exception(exception)
    span.set_status(Status(StatusCode.ERROR, str(exception)))


def handle_span_success(span: Optional[trace.Span], message_or_attributes=None):
    """Mark span as successful with optional message or attributes."""
    if span is None:
        return

    span.set_status(Status(StatusCode.OK))

    if message_or_attributes:
        if isinstance(message_or_attributes, str):
            # Simple string message
            span.add_event("success", {"message": message_or_attributes})
        elif isinstance(message_or_attributes, dict):
            # Dictionary of attributes - serialize complex values
            sanitized_attrs = {}
            for key, value in message_or_attributes.items():
                if isinstance(value, (str, int, float, bool)):
                    sanitized_attrs[key] = value
                elif value is None:
                    sanitized_attrs[key] = "null"
                else:
                    # Convert complex objects to string representation
                    sanitized_attrs[key] = str(value)
            span.add_event("success", sanitized_attrs)
        else:
            # Convert other types to string
            span.add_event("success", {"message": str(message_or_attributes)})


def safe_set_span_attribute(span: Optional[trace.Span], key: str, value) -> None:
    """Safely set span attribute, handling complex types by serialization."""
    if span is None:
        return

    # OpenTelemetry accepts: str, bool, int, float, and sequences of these types
    if isinstance(value, (str, bool, int, float)):
        span.set_attribute(key, value)
    elif value is None:
        span.set_attribute(key, "null")
    elif isinstance(value, (list, tuple)):
        # Convert sequence elements to strings if needed
        safe_list = []
        for item in value:
            if isinstance(item, (str, bool, int, float)):
                safe_list.append(item)
            else:
                safe_list.append(str(item))
        span.set_attribute(key, safe_list)
    elif isinstance(value, dict):
        # Serialize dict to JSON string
        import json
        try:
            span.set_attribute(key, json.dumps(value))
        except (TypeError, ValueError):
            span.set_attribute(key, str(value))
    else:
        # Convert other types to string
        span.set_attribute(key, str(value))


def safe_set_span_attributes(span: Optional[trace.Span], attributes: dict) -> None:
    """Safely set multiple span attributes at once."""
    if span is None or not attributes:
        return

    for key, value in attributes.items():
        safe_set_span_attribute(span, key, value)


def flush_telemetry_spans():
    """Force flush all pending spans to exporters."""
    try:
        # Get the active tracer provider
        current_tracer_provider = trace.get_tracer_provider()
        if hasattr(current_tracer_provider, 'force_flush'):
            current_tracer_provider.force_flush()
        print("✅ Spans flushed to exporters")
    except Exception as e:
        print(f"⚠️  Warning: Could not flush spans: {e}")