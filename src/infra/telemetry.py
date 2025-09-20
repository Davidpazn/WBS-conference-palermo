"""OpenTelemetry setup for tracing and observability."""

import os
import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

log = logging.getLogger(__name__)

def setup_tracer(service_name: str = "agents-backend") -> trace.Tracer:
    """
    Set up OpenTelemetry tracing with multiple exporter options.

    Environment variables:
    - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (e.g., http://localhost:4317)
    - OTEL_SERVICE_NAME: Service name override
    - OTEL_SERVICE_VERSION: Service version
    - LANGSMITH_TRACING: Enable LangSmith integration
    """

    # Create resource with service info
    resource = Resource.create({
        SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", service_name),
        SERVICE_VERSION: os.getenv("OTEL_SERVICE_VERSION", "0.1.0"),
        "environment": os.getenv("ENVIRONMENT", "development"),
    })

    # Set up tracer provider
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # Configure exporters based on environment
    exporters_configured = 0

    # OTLP Exporter (for services like Jaeger, Zipkin, LangSmith, or cloud providers)
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            # Parse headers for LangSmith integration
            headers_env = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
            headers = {}
            if headers_env:
                for header_pair in headers_env.split(","):
                    if "=" in header_pair:
                        key, value = header_pair.strip().split("=", 1)
                        headers[key] = value

            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                headers=headers,
                timeout=10
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            log.info(f"[telemetry] OTLP exporter configured: {otlp_endpoint}")
            if headers:
                log.info(f"[telemetry] OTLP headers configured: {list(headers.keys())}")
            exporters_configured += 1
        except Exception as e:
            log.warning(f"[telemetry] OTLP exporter failed: {e}")

    # Console exporter (fallback or development)
    if exporters_configured == 0 or os.getenv("OTEL_CONSOLE_EXPORTER", "false").lower() == "true":
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))
        log.info("[telemetry] Console exporter configured")
        exporters_configured += 1

    if exporters_configured == 0:
        log.warning("[telemetry] No exporters configured")

    # Set up auto-instrumentation
    setup_auto_instrumentation()

    # Get tracer for this service
    tracer = trace.get_tracer(__name__)
    log.info(f"[telemetry] Tracer setup complete for service: {service_name}")

    return tracer

def setup_auto_instrumentation():
    """Set up automatic instrumentation for common libraries."""
    try:
        # Instrument FastAPI
        FastAPIInstrumentor().instrument()
        log.debug("[telemetry] FastAPI instrumentation enabled")
    except Exception as e:
        log.warning(f"[telemetry] FastAPI instrumentation failed: {e}")

    try:
        # Instrument HTTPX (for external API calls like OpenAI, Letta)
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
        log.debug("[telemetry] HTTPX instrumentation enabled")
    except ImportError:
        log.debug("[telemetry] HTTPX instrumentation not available (install: pip install opentelemetry-instrumentation-httpx)")
    except Exception as e:
        log.warning(f"[telemetry] HTTPX instrumentation failed: {e}")

def setup_langsmith_integration():
    """
    Set up LangSmith integration for LLM tracing.
    This should be called if LANGSMITH_TRACING=true.
    """
    langsmith_enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    if not langsmith_enabled:
        log.debug("[telemetry] LangSmith tracing disabled")
        return

    try:
        # Import LangSmith tracing
        from langsmith import traceable
        from langchain_core.tracers.langchain import LangChainTracer

        # Set up LangChain tracer
        tracer = LangChainTracer(
            project_name=os.getenv("LANGSMITH_PROJECT", "agents-demo"),
        )

        log.info("[telemetry] LangSmith integration enabled")
        return tracer

    except ImportError:
        log.warning("[telemetry] LangSmith not available (install with: pip install langsmith)")
    except Exception as e:
        log.warning(f"[telemetry] LangSmith setup failed: {e}")

    return None

def get_current_span_context() -> Optional[dict]:
    """Get current span context for correlation."""
    span = trace.get_current_span()
    if span.is_recording():
        span_context = span.get_span_context()
        return {
            "trace_id": format(span_context.trace_id, "032x"),
            "span_id": format(span_context.span_id, "016x"),
        }
    return None

def create_gen_ai_span(tracer: trace.Tracer, operation: str, model: str, system: str = "openai"):
    """
    Create a span with GenAI semantic conventions for LLM operations.

    Args:
        tracer: OpenTelemetry tracer
        operation: Operation name (e.g., "chat", "completion", "responses")
        model: Model name (e.g., "gpt-5-nano")
        system: AI system name (e.g., "openai")
    """
    span = tracer.start_span(f"gen_ai.{operation}")
    span.set_attribute("gen_ai.system", system)
    span.set_attribute("gen_ai.operation.name", operation)
    span.set_attribute("gen_ai.request.model", model)
    return span

def add_gen_ai_attributes(span: trace.Span,
                         prompt: str = None,
                         completion: str = None,
                         input_tokens: int = None,
                         output_tokens: int = None,
                         cost: float = None):
    """Add GenAI semantic convention attributes to a span."""
    if prompt:
        span.set_attribute("gen_ai.prompt", prompt[:1000])  # Truncate for safety
    if completion:
        span.set_attribute("gen_ai.completion", completion[:1000])
    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
    if cost is not None:
        span.set_attribute("gen_ai.usage.cost", cost)