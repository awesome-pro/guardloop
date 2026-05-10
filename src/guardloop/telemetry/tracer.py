"""OpenTelemetry span creation and optional exporter setup."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from guardloop.models import TelemetryConfig
from guardloop.telemetry.conventions import Attributes

_otel_configured = False


def configure_otel_export(config: TelemetryConfig) -> None:
    """Configure an SDK exporter only when the user asks for one."""

    global _otel_configured
    if _otel_configured or (not config.otlp_endpoint and not config.console_exporter):
        return

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError as exc:
        raise RuntimeError(
            "OpenTelemetry export requires the 'otel' extra: pip install guardloop[otel]"
        ) from exc

    resource = Resource.create({"service.name": config.service_name})
    provider = TracerProvider(resource=resource)

    if config.otlp_endpoint:
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=config.otlp_endpoint))
        )
    if config.console_exporter:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _otel_configured = True


class Telemetry:
    """Thin wrapper around OpenTelemetry spans."""

    def __init__(self, config: TelemetryConfig, *, tracer: Tracer | None = None) -> None:
        self.config = config
        if config.enabled:
            configure_otel_export(config)
        self.tracer = tracer or trace.get_tracer("guardloop")

    @contextmanager
    def start_span(self, name: str, attributes: Attributes | None = None) -> Generator[Span]:
        if not self.config.enabled:
            yield trace.get_current_span()
            return
        with self.tracer.start_as_current_span(name, attributes=attributes) as span:
            yield span

    def set_attributes(self, span: Span, attributes: Attributes) -> None:
        if not self.config.enabled:
            return
        for key, value in attributes.items():
            span.set_attribute(key, value)

    def record_exception(self, span: Span, exc: BaseException) -> None:
        if not self.config.enabled:
            return
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))

    def mark_ok(self, span: Span) -> None:
        if self.config.enabled:
            span.set_status(Status(StatusCode.OK))

    @staticmethod
    def trace_id(span: Span) -> str | None:
        context = span.get_span_context()
        if not context.is_valid:
            return None
        return f"{context.trace_id:032x}"
