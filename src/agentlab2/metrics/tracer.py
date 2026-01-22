import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from agentlab2.metrics.disk_exporter import DiskSpanExporter
from agentlab2.metrics.processor import AL2_EXPERIMENT, AL2_NAME, AL2_TYPE, TYPE_EPISODE, TYPE_EXPERIMENT, TYPE_STEP

ENV_TRACEPARENT = "TRACEPARENT"
ENV_TRACE_OUTPUT = "AGENTLAB_TRACE_OUTPUT"
ENV_OTLP_ENDPOINT = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
ENV_AUTH_CODE = "AGENTLAB_AUTH_CODE"
AL2_AUTH = "al2.auth"


class _AgentTracer:
    """Internal tracer. Use get_tracer() to create instances."""

    def __init__(
        self,
        service_name: str,
        output_dir: str | Path | None = None,
        otlp_endpoint: str | None = None,
        auth_code: str | None = None,
    ) -> None:
        assert output_dir or otlp_endpoint, "At least one collector (output_dir or otlp_endpoint) required"

        self.output_dir: Path | None = None
        resource_attrs = {SERVICE_NAME: service_name}
        if auth_code:
            resource_attrs[AL2_AUTH] = auth_code
            os.environ[ENV_AUTH_CODE] = auth_code
        self._provider = TracerProvider(resource=Resource.create(resource_attrs))

        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._provider.add_span_processor(BatchSpanProcessor(DiskSpanExporter(self.output_dir)))
            os.environ[ENV_TRACE_OUTPUT] = str(self.output_dir)

        if otlp_endpoint:
            os.environ[ENV_OTLP_ENDPOINT] = otlp_endpoint
            self._provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

        # Set as global provider so OTEL-instrumented libraries emit spans into this trace
        trace.set_tracer_provider(self._provider)
        self._tracer = self._provider.get_tracer(__name__)
        self._current_experiment: str | None = None


    @contextmanager
    def benchmark(self, name: str) -> Iterator[trace.Span]:
        with self._tracer.start_as_current_span(name) as span:
            span.set_attribute(AL2_TYPE, TYPE_EXPERIMENT)
            span.set_attribute(AL2_NAME, name)
            self._current_experiment = name
            _set_traceparent_env()
            try:
                yield span
            finally:
                self._current_experiment = None


    @contextmanager
    def episode(
            self,
            name: str,
            experiment: str | None = None,
    ) -> Iterator[trace.Span]:
        exp = experiment or self._current_experiment or "default"
        parent_ctx = _get_parent_ctx_env()

        with self._tracer.start_as_current_span(name, context=parent_ctx) as span:
            span.set_attribute(AL2_TYPE, TYPE_EPISODE)
            span.set_attribute(AL2_NAME, name)
            span.set_attribute(AL2_EXPERIMENT, exp)
            yield span


    @contextmanager
    def step(self, name: str) -> Iterator[trace.Span]:
        with self._tracer.start_as_current_span(name) as span:
            span.set_attribute(AL2_TYPE, TYPE_STEP)
            yield span


    def log(self, data: dict[str, Any], name: str = "step") -> None:
        with self.step(name) as span:
            for k, v in data.items():
                span.set_attribute(k, v)


    @contextmanager
    def span(self, name: str) -> Iterator[trace.Span]:
        with self._tracer.start_as_current_span(name) as span:
            yield span


    def shutdown(self) -> None:
        self._provider.shutdown()


def _set_traceparent_env() -> None:
    carrier: dict[str, str] = {}
    inject(carrier)
    if tp := carrier.get("traceparent"):
        os.environ[ENV_TRACEPARENT] = tp


def _get_parent_ctx_env() -> Context | None:
    if tp := os.environ.get(ENV_TRACEPARENT):
        return extract({"traceparent": tp})
    return None


def get_trace_env_vars() -> dict[str, str]:
    env_vars = {}
    if tp := os.environ.get(ENV_TRACEPARENT):
        env_vars[ENV_TRACEPARENT] = tp
    if output := os.environ.get(ENV_TRACE_OUTPUT):
        env_vars[ENV_TRACE_OUTPUT] = output
    if otlp := os.environ.get(ENV_OTLP_ENDPOINT):
        env_vars[ENV_OTLP_ENDPOINT] = otlp
    if auth := os.environ.get(ENV_AUTH_CODE):
        env_vars[ENV_AUTH_CODE] = auth
    return env_vars


class _NoOpSpan:
    """A no-op span that ignores all attribute calls."""


    def set_attribute(self, key: str, value: Any) -> None:
        pass


_NOOP_SPAN = _NoOpSpan()


class _NoOpTracer:
    @contextmanager
    def benchmark(self, name: str) -> Iterator[_NoOpSpan]:
        yield _NOOP_SPAN

    @contextmanager
    def episode(self, name: str, experiment: str | None = None) -> Iterator[_NoOpSpan]:
        yield _NOOP_SPAN

    @contextmanager
    def step(self, name: str) -> Iterator[_NoOpSpan]:
        yield _NOOP_SPAN

    @contextmanager
    def span(self, name: str) -> Iterator[_NoOpSpan]:
        yield _NOOP_SPAN

    def log(self, data: dict[str, Any], name: str = "step") -> None:
        pass

    def shutdown(self) -> None:
        pass


def get_tracer(
    service_name: str,
    output_dir: str | Path | None = None,
    otlp_endpoint: str | None = None,
    auth_code: str | None = None,
) -> _AgentTracer | _NoOpTracer:
    output_dir = output_dir or os.environ.get(ENV_TRACE_OUTPUT)
    otlp_endpoint = otlp_endpoint or os.environ.get(ENV_OTLP_ENDPOINT)
    auth_code = auth_code or os.environ.get(ENV_AUTH_CODE)

    if output_dir or otlp_endpoint:
        return _AgentTracer(
            service_name=service_name,
            output_dir=output_dir,
            otlp_endpoint=otlp_endpoint,
            auth_code=auth_code,
        )
    return _NoOpTracer()
