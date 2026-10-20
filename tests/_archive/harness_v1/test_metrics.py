"""Tests for mullet.harness.observability.metrics."""


from mullet.harness.observability.metrics import (
    MetricsRegistry,
    _format_name,
    inc,
    gauge,
    histogram,
    snapshot,
    reset,
)


class TestFormatName:
    def test_no_tags(self):
        assert _format_name("foo", None) == "foo"
        assert _format_name("foo", {}) == "foo"

    def test_with_tags(self):
        assert _format_name("foo", {"a": "1"}) == "foo.a=1"
        assert _format_name("foo", {"b": "2", "a": "1"}) == "foo.a=1,b=2"


class TestMetricsRegistry:
    def test_counter_increment(self):
        reg = MetricsRegistry()
        reg.inc("requests")
        reg.inc("requests", 5)
        assert reg.snapshot().counters["requests"] == 6

    def test_counter_with_tags(self):
        reg = MetricsRegistry()
        reg.inc("tool_calls", tags={"tool": "Read"})
        reg.inc("tool_calls", tags={"tool": "Write"})
        reg.inc("tool_calls", tags={"tool": "Read"})
        snap = reg.snapshot()
        assert snap.counters["tool_calls.tool=Read"] == 2
        assert snap.counters["tool_calls.tool=Write"] == 1

    def test_gauge(self):
        reg = MetricsRegistry()
        reg.gauge("cpu", 0.5)
        reg.gauge("cpu", 0.8)
        assert reg.snapshot().gauges["cpu"] == 0.8

    def test_histogram(self):
        reg = MetricsRegistry()
        reg.histogram("latency", 10.0)
        reg.histogram("latency", 20.0)
        reg.histogram("latency", 30.0)
        snap = reg.snapshot()
        assert snap.histograms["latency"] == [10.0, 20.0, 30.0]

    def test_histogram_max_samples_eviction(self):
        reg = MetricsRegistry()
        for i in range(5):
            reg.histogram("lat", float(i), max_samples=3)
        snap = reg.snapshot()
        assert snap.histograms["lat"] == [2.0, 3.0, 4.0]

    def test_snapshot_is_copy(self):
        reg = MetricsRegistry()
        reg.inc("x")
        snap1 = reg.snapshot()
        reg.inc("x")
        snap2 = reg.snapshot()
        assert snap1.counters["x"] == 1
        assert snap2.counters["x"] == 2

    def test_reset(self):
        reg = MetricsRegistry()
        reg.inc("x")
        reg.gauge("y", 1.0)
        reg.histogram("z", 1.0)
        reg.reset()
        snap = reg.snapshot()
        assert snap.counters == {}
        assert snap.gauges == {}
        assert snap.histograms == {}


class TestModuleLevelSingleton:
    def test_round_trip(self):
        reset()
        inc("a")
        gauge("b", 2.0)
        histogram("c", 3.0)
        snap = snapshot()
        assert snap.counters["a"] == 1
        assert snap.gauges["b"] == 2.0
        assert snap.histograms["c"] == [3.0]
        reset()
