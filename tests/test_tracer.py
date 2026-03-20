from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from cube_harness.metrics.processor import CH_EPISODE
from cube_harness.metrics.tracer import Tracer, make_tracer


class _CollectingExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS


def _make_test_tracer() -> tuple[Tracer, _CollectingExporter]:
    exporter = _CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = make_tracer(provider)
    return tracer, exporter


class TestEpisodeBaggage:
    def test_step_span_has_episode_id(self) -> None:
        tracer, exporter = _make_test_tracer()
        with tracer.episode("ep1"):
            with tracer.step("step1"):
                pass
        tracer.shutdown()

        step_span = next(s for s in exporter.spans if s.name == "step1")
        episode_span = next(s for s in exporter.spans if s.name == "ep1")

        assert step_span.attributes[CH_EPISODE] is not None
        assert step_span.attributes[CH_EPISODE] == episode_span.attributes[CH_EPISODE]

    def test_episode_span_has_episode_id(self) -> None:
        tracer, exporter = _make_test_tracer()
        with tracer.episode("ep1"):
            pass
        tracer.shutdown()

        episode_span = next(s for s in exporter.spans if s.name == "ep1")
        assert episode_span.attributes[CH_EPISODE] is not None

    def test_sequential_episodes_have_distinct_ids(self) -> None:
        tracer, exporter = _make_test_tracer()
        with tracer.episode("ep1"):
            pass
        with tracer.episode("ep2"):
            pass
        tracer.shutdown()

        ep1 = next(s for s in exporter.spans if s.name == "ep1")
        ep2 = next(s for s in exporter.spans if s.name == "ep2")
        assert ep1.attributes[CH_EPISODE] != ep2.attributes[CH_EPISODE]

    def test_baggage_does_not_leak_across_episodes(self) -> None:
        tracer, exporter = _make_test_tracer()
        with tracer.episode("ep1"):
            with tracer.step("step1"):
                pass
        with tracer.episode("ep2"):
            with tracer.step("step2"):
                pass
        tracer.shutdown()

        ep1 = next(s for s in exporter.spans if s.name == "ep1")
        step1 = next(s for s in exporter.spans if s.name == "step1")
        ep2 = next(s for s in exporter.spans if s.name == "ep2")
        step2 = next(s for s in exporter.spans if s.name == "step2")

        assert step1.attributes[CH_EPISODE] == ep1.attributes[CH_EPISODE]
        assert step2.attributes[CH_EPISODE] == ep2.attributes[CH_EPISODE]
        assert step1.attributes[CH_EPISODE] != step2.attributes[CH_EPISODE]

    def test_span_outside_episode_has_no_episode_id(self) -> None:
        tracer, exporter = _make_test_tracer()
        with tracer.step("orphan"):
            pass
        tracer.shutdown()

        orphan = next(s for s in exporter.spans if s.name == "orphan")
        assert CH_EPISODE not in (orphan.attributes or {})
