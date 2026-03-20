import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol, override
from uuid import uuid4

import litellm
from cube.core import Action
from opentelemetry import baggage, context, trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind

from cube_harness.metrics.disk_exporter import DiskSpanExporter
from cube_harness.metrics.processor import (
    CH_EPISODE,
    CH_EXPERIMENT,
    CH_NAME,
    CH_TYPE,
    TYPE_EPISODE,
    TYPE_EXPERIMENT,
    TYPE_STEP,
)

_logger = logging.getLogger(__name__)

RAY_ENV_TRACEPARENT = "TRACEPARENT"
RAY_ENV_TRACE_OUTPUT = "CUBE_HARNESS_TRACE_OUTPUT"
RAY_ENV_OTLP_ENDPOINT = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
RAY_ENV_MODEL = "CUBE_HARNESS_MODEL"
RAY_ENV_AGENT_NAME = "CUBE_HARNESS_AGENT_NAME"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"

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


class _EpisodeBaggageSpanProcessor(SpanProcessor):
    """Stamps the episode ID baggage value as a span attribute on every span."""

    @override
    def on_start(self, span: trace.Span, parent_context: Optional[Context] = None) -> None:
        episode_id = baggage.get_baggage(CH_EPISODE, context=parent_context)
        if episode_id is not None:
            span.set_attribute(CH_EPISODE, str(episode_id))


@contextmanager
def tool_span(action: Action) -> Iterator[trace.Span]:
    """Create a span for tool execution with GenAI semantic attributes."""
    with _tool_tracer.start_as_current_span(f"execute_tool {action.name}", kind=SpanKind.INTERNAL) as span:
        span.set_attribute(GEN_AI_TOOL_NAME, action.name)
        span.set_attribute(GEN_AI_TOOL_CALL_ID, action.id or "")
        span.set_attribute(GEN_AI_TOOL_CALL_ARGUMENTS, json.dumps(action.arguments))
        yield span


class Tracer(Protocol):
    def benchmark(self, name: str) -> Iterator[trace.Span]: ...
    def episode(self, name: str, experiment: str | None = None) -> Iterator[trace.Span]: ...
    def step(self, name: str) -> Iterator[trace.Span]: ...
    def span(self, name: str) -> Iterator[trace.Span]: ...
    def log(self, data: dict[str, Any], name: str = "step") -> None: ...
    def shutdown(self) -> None: ...


class _AgentTracer:
    """Internal tracer. Use get_tracer() to create instances."""

    def __init__(self, provider: TracerProvider, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir
        self._provider = provider
        self._tracer = provider.get_tracer(__name__)
        self._current_experiment: str | None = None

    @contextmanager
    def benchmark(self, name: str) -> Iterator[trace.Span]:
        with self._tracer.start_as_current_span(name) as span:
            span.set_attribute(CH_TYPE, TYPE_EXPERIMENT)
            span.set_attribute(CH_NAME, name)
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
        episode_id = str(uuid4())

        ctx = baggage.set_baggage(CH_EPISODE, episode_id, context=parent_ctx)
        token = context.attach(ctx)
        try:
            with self._tracer.start_as_current_span(name) as span:
                span.set_attribute(CH_TYPE, TYPE_EPISODE)
                span.set_attribute(CH_NAME, name)
                span.set_attribute(CH_EXPERIMENT, exp)
                yield span
        finally:
            context.detach(token)

    @contextmanager
    def step(self, name: str) -> Iterator[trace.Span]:
        with self._tracer.start_as_current_span(name) as span:
            span.set_attribute(CH_TYPE, TYPE_STEP)
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
        os.environ[RAY_ENV_TRACEPARENT] = tp


def _get_parent_ctx_env() -> Context | None:
    if tp := os.environ.get(RAY_ENV_TRACEPARENT):
        return extract({"traceparent": tp})
    return None


def get_trace_env_vars() -> dict[str, str]:
    env_vars = {}
    for key in (RAY_ENV_TRACEPARENT, RAY_ENV_TRACE_OUTPUT, RAY_ENV_OTLP_ENDPOINT, RAY_ENV_MODEL, RAY_ENV_AGENT_NAME):
        if val := os.environ.get(key):
            env_vars[key] = val
    return env_vars


class _NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any, description: str | None = None) -> None:
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


def make_tracer(provider: TracerProvider, output_dir: Path | None = None) -> Tracer:
    provider.add_span_processor(_EpisodeBaggageSpanProcessor())
    return _AgentTracer(provider, output_dir=output_dir)


def get_tracer(
    service_name: str,
    output_dir: str | Path | None = None,
    otlp_endpoint: str | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
    agent_description: str | None = None,
    model: str | None = None,
) -> Tracer:
    output_dir = output_dir or os.environ.get(RAY_ENV_TRACE_OUTPUT)
    otlp_endpoint = otlp_endpoint or os.environ.get(RAY_ENV_OTLP_ENDPOINT)
    model = model or os.environ.get(RAY_ENV_MODEL)
    agent_name = agent_name or os.environ.get(RAY_ENV_AGENT_NAME)

    if not (output_dir or otlp_endpoint):
        return _NoOpTracer()

    _logger.info(
        f"Creating _AgentTracer: service={service_name}, output_dir={output_dir}, otlp_endpoint={otlp_endpoint}"
    )

    resource_attrs: dict[str, str] = {SERVICE_NAME: service_name}
    default_agent_id = agent_id or agent_name or uuid4().hex
    resource_attrs[GEN_AI_AGENT_NAME] = agent_name or default_agent_id
    resource_attrs[GEN_AI_AGENT_ID] = agent_id or default_agent_id
    if agent_description is not None:
        resource_attrs[GEN_AI_AGENT_DESCRIPTION] = agent_description
    if model:
        resource_attrs[GEN_AI_REQUEST_MODEL] = model
        os.environ[RAY_ENV_MODEL] = model
    os.environ[RAY_ENV_AGENT_NAME] = resource_attrs[GEN_AI_AGENT_NAME]

    provider = TracerProvider(resource=Resource.create(resource_attrs))

    resolved_output_dir: Path | None = None
    if output_dir:
        resolved_output_dir = Path(output_dir)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        provider.add_span_processor(BatchSpanProcessor(DiskSpanExporter(resolved_output_dir)))
        os.environ[RAY_ENV_TRACE_OUTPUT] = str(resolved_output_dir)

    if otlp_endpoint:
        os.environ[RAY_ENV_OTLP_ENDPOINT] = otlp_endpoint
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    # Set as global provider so OTEL-instrumented libraries emit spans into this trace
    trace.set_tracer_provider(provider)

    # Enable litellm OTEL callback now that a proper TracerProvider is configured.
    # This must happen after set_tracer_provider() to avoid ConsoleSpanExporter fallback.
    os.environ["USE_OTEL_LITELLM_REQUEST_SPAN"] = "true"
    litellm.callbacks = ["otel"]

    return make_tracer(provider, output_dir=resolved_output_dir)
