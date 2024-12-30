from contextlib import contextmanager
from typing import Any
from uuid import UUID

from opentelemetry import context
from opentelemetry import trace
from opentelemetry.trace import SpanContext
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode
from opentelemetry.trace import TraceFlags


tracer = trace.get_tracer(__name__)


@contextmanager
def task_span(**attributes: Any):
    """Context manager to create a span with task context"""
    span_name = "task_operation"

    # Extract task_id if provided in attributes
    task_id = attributes.get("task_id")

    # Create a trace ID from task_id if available
    trace_id = int(UUID(task_id).hex, 16) if task_id else None

    # Create custom span context if we have a trace_id
    ctx = None
    if trace_id:
        # Generate a random span ID (16 random hex digits)
        span_id = int.from_bytes(UUID(int=trace_id).bytes[:8], byteorder="big")
        span_context = SpanContext(trace_id=trace_id, span_id=span_id, is_remote=False, trace_flags=TraceFlags.SAMPLED)
        ctx = context.Context()

    with tracer.start_as_current_span(span_name, context=ctx) as span:
        for key, value in attributes.items():
            span.set_attribute(key, str(value))
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(e)
            raise


def add_span_attribute(key: str, value: Any) -> None:
    """Add an attribute to the current span"""
    current_span = trace.get_current_span()
    if current_span.is_recording():
        current_span.set_attribute(key, str(value))


def get_span_attribute(key: str) -> Any:
    """Get an attribute from the current span"""
    current_span = trace.get_current_span()
    if current_span.is_recording():
        return current_span.attributes.get(key)
    return None
