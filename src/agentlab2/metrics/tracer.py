import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import litellm
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind

from agentlab2.core import Action
from agentlab2.metrics.disk_exporter import DiskSpanExporter
from agentlab2.metrics.processor import AL2_EXPERIMENT, AL2_NAME, AL2_TYPE, TYPE_EPISODE, TYPE_EXPERIMENT, TYPE_STEP

_logger = logging.getLogger(__name__)

ENV_TRACEPARENT = "TRACEPARENT"
ENV_TRACE_OUTPUT = "AGENTLAB_TRACE_OUTPUT"
ENV_OTLP_ENDPOINT = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"

# https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#gen-ai-agent-attributes
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_AGENT_ID = "gen_ai.agent.id"
GEN_AI_AGENT_DESCRIPTION = "gen_ai.agent.description"

# https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"

_tool_tracer = trace.get_tracer(__name__)


@contextmanager
def tool_span(action: Action) -> Iterator[trace.Span]:
    """Create a span for tool execution with GenAI semantic attributes."""
    with _tool_tracer.start_as_current_span(f"execute_tool {action.name}", kind=SpanKind.INTERNAL) as span:
        span.set_attribute(GEN_AI_TOOL_NAME, action.name)
        span.set_attribute(GEN_AI_TOOL_CALL_ID, action.id)
        span.set_attribute(GEN_AI_TOOL_CALL_ARGUMENTS, json.dumps(action.arguments))
        yield span


class _AgentTracer:
    """Internal tracer. Use get_tracer() to create instances."""

    def __init__(
        self,
        service_name: str,
        output_dir: str | Path | None = None,
        otlp_endpoint: str | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
        agent_description: str | None = None,
    ) -> None:
        assert output_dir or otlp_endpoint, "At least one collector (output_dir or otlp_endpoint) required"
        _logger.info(
            f"Creating _AgentTracer: service={service_name}, output_dir={output_dir}, otlp_endpoint={otlp_endpoint}"
        )

        self.output_dir: Path | None = None
        resource_attrs = {SERVICE_NAME: service_name}

        default_agent_id = agent_id or agent_name or uuid4().hex
        resource_attrs[GEN_AI_AGENT_NAME] = agent_name or default_agent_id
        resource_attrs[GEN_AI_AGENT_ID] = agent_id or default_agent_id
        if agent_description is not None:
            resource_attrs[GEN_AI_AGENT_DESCRIPTION] = agent_description

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

        # Enable litellm OTEL callback now that a proper TracerProvider is configured.
        # This must happen after set_tracer_provider() to avoid ConsoleSpanExporter fallback.
        os.environ["USE_OTEL_LITELLM_REQUEST_SPAN"] = "true"
        litellm.callbacks = ["otel"]

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
        """Shutdown the tracer provider.

        Note: The provider is shared. Only call once per process. Calling
        shutdown() multiple times will cause errors on subsequent calls.
        """
        _logger.info("Shutting down tracer and flushing spans")
        self._provider.shutdown()
        _logger.info("Tracer shutdown complete")


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
    agent_name: str | None = None,
    agent_id: str | None = None,
    agent_description: str | None = None,
) -> _AgentTracer | _NoOpTracer:
    output_dir = output_dir or os.environ.get(ENV_TRACE_OUTPUT)
    otlp_endpoint = otlp_endpoint or os.environ.get(ENV_OTLP_ENDPOINT)

    if output_dir or otlp_endpoint:
        return _AgentTracer(
            service_name=service_name,
            output_dir=output_dir,
            otlp_endpoint=otlp_endpoint,
            agent_name=agent_name,
            agent_id=agent_id,
            agent_description=agent_description,
        )
    return _NoOpTracer()
