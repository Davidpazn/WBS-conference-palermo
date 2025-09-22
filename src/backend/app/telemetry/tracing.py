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


# Module-level flags to track initialization status
_TELEMETRY_INITIALIZED = False
_TRACER_INSTANCE = None
_LANGSMITH_CLIENT_INSTANCE = None

# Environment variables for telemetry
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "agents-demo")
SERVICE_NAME = os.getenv("SERVICE_NAME", "hitl-coding-agent")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")  # notebook, development, production

# Feature flags for telemetry components
ENABLE_OPENTELEMETRY = os.getenv("ENABLE_OPENTELEMETRY", "true").lower() == "true"
ENABLE_LANGSMITH = os.getenv("ENABLE_LANGSMITH", "true").lower() == "true" and LANGSMITH_API_KEY
ENABLE_CONSOLE_EXPORTER = os.getenv("ENABLE_CONSOLE_EXPORTER", "false").lower() == "true"


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
    langsmith_project: str = LANGSMITH_PROJECT,
    force_reinit: bool = False
) -> Tuple[trace.Tracer, Optional[LangSmithClient]]:
    """
    Configure OpenTelemetry + LangSmith tracing with proper initialization guards.

    Args:
        service_name: Name of the service
        service_version: Version of the service
        environment: Environment (production, development, notebook)
        langsmith_api_key: LangSmith API key
        langsmith_project: LangSmith project name
        force_reinit: Force re-initialization even if already initialized

    Returns:
        tuple: (tracer, langsmith_client)
    """
    global _TELEMETRY_INITIALIZED, _TRACER_INSTANCE, _LANGSMITH_CLIENT_INSTANCE

    # Return cached instances if already initialized (unless forced)
    if _TELEMETRY_INITIALIZED and not force_reinit:
        if _TRACER_INSTANCE is not None:
            return _TRACER_INSTANCE, _LANGSMITH_CLIENT_INSTANCE

    # Check if OpenTelemetry is disabled
    if not ENABLE_OPENTELEMETRY:
        print("⚠️  OpenTelemetry disabled via ENABLE_OPENTELEMETRY=false")
        fallback_tracer = trace.get_tracer(__name__)
        return fallback_tracer, None

    try:
        # Check if TracerProvider is already configured
        current_provider = trace.get_tracer_provider()
        if not isinstance(current_provider, TracerProvider) or force_reinit:
            # Initialize OpenTelemetry Resource
            resource = Resource.create({
                "service.name": service_name,
                "service.version": service_version,
                "environment": environment
            })

            # Set up new tracer provider
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)

            # Configure exporters based on environment and flags
            exporters_configured = False

            # Check for OTLP endpoint in environment
            otlp_endpoint = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')
            if otlp_endpoint:
                # Configure OTLP exporter
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                print(f"✅ OTLP exporter configured for endpoint: {otlp_endpoint}")
                exporters_configured = True

            # Configure LangSmith if enabled
            if ENABLE_LANGSMITH and langsmith_api_key:
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
                exporters_configured = True

            # Add console exporter if enabled or as fallback
            if ENABLE_CONSOLE_EXPORTER or not exporters_configured:
                console_exporter = ConsoleSpanExporter()
                tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))
                if not exporters_configured:
                    print("⚠️  No OTLP endpoint or LangSmith API key, using console exporter")
                else:
                    print("✅ Console exporter enabled")
        else:
            # Provider already configured, just get the tracer
            print("ℹ️  TracerProvider already configured, reusing existing provider")

        # Get tracer
        tracer = trace.get_tracer(__name__)

        # Initialize LangSmith client if enabled
        langsmith_client = None
        if ENABLE_LANGSMITH and langsmith_api_key:
            langsmith_client = LangSmithClient(api_key=langsmith_api_key)
            print(f"✅ LangSmith client initialized")
        elif not ENABLE_LANGSMITH:
            print("ℹ️  LangSmith disabled via ENABLE_LANGSMITH=false")

        print("✅ OpenTelemetry + LangSmith tracing configured")

        # Cache instances
        _TRACER_INSTANCE = tracer
        _LANGSMITH_CLIENT_INSTANCE = langsmith_client
        _TELEMETRY_INITIALIZED = True

        return tracer, langsmith_client

    except Exception as e:
        print(f"⚠️  Error setting up telemetry: {e}")
        # Return fallback tracer
        fallback_tracer = trace.get_tracer(__name__)
        _TRACER_INSTANCE = fallback_tracer
        _LANGSMITH_CLIENT_INSTANCE = None
        return fallback_tracer, None


# Initialize telemetry components
tracer, langsmith_client = setup_telemetry()


def get_telemetry_status() -> dict:
    """
    Get current telemetry configuration status.

    Returns:
        dict: Status of telemetry components
    """
    global _TELEMETRY_INITIALIZED, _TRACER_INSTANCE, _LANGSMITH_CLIENT_INSTANCE

    return {
        "initialized": _TELEMETRY_INITIALIZED,
        "opentelemetry_enabled": ENABLE_OPENTELEMETRY,
        "langsmith_enabled": ENABLE_LANGSMITH,
        "console_exporter_enabled": ENABLE_CONSOLE_EXPORTER,
        "tracer_configured": _TRACER_INSTANCE is not None,
        "langsmith_client_configured": _LANGSMITH_CLIENT_INSTANCE is not None,
        "service_name": SERVICE_NAME,
        "service_version": SERVICE_VERSION,
        "environment": ENVIRONMENT,
        "langsmith_project": LANGSMITH_PROJECT if ENABLE_LANGSMITH else None,
        "otlp_endpoint": os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT'),
        "provider_type": type(trace.get_tracer_provider()).__name__
    }


def get_tracer() -> trace.Tracer:
    """
    Get the cached tracer instance or create one if needed.

    Returns:
        trace.Tracer: The configured tracer
    """
    global _TRACER_INSTANCE

    if _TRACER_INSTANCE is None:
        _TRACER_INSTANCE, _ = setup_telemetry()

    return _TRACER_INSTANCE


def get_langsmith_client() -> Optional[LangSmithClient]:
    """
    Get the cached LangSmith client instance.

    Returns:
        Optional[LangSmithClient]: The configured client or None
    """
    global _LANGSMITH_CLIENT_INSTANCE

    if _LANGSMITH_CLIENT_INSTANCE is None and ENABLE_LANGSMITH and LANGSMITH_API_KEY:
        _, _LANGSMITH_CLIENT_INSTANCE = setup_telemetry()

    return _LANGSMITH_CLIENT_INSTANCE


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