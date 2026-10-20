"""Tests for Prometheus metrics."""

from mullet.server.metrics import Metrics, _label_key


def test_counter_increment():
    m = Metrics()
    m.inc("requests_total")
    m.inc("requests_total")
    assert m._counters["requests_total"] == 2


def test_counter_with_labels():
    m = Metrics()
    m.inc("requests_total", labels={"method": "GET"})
    m.inc("requests_total", labels={"method": "POST"})
    assert m._counters['requests_total{method="GET"}'] == 1
    assert m._counters['requests_total{method="POST"}'] == 1


def test_gauge():
    m = Metrics()
    m.set_gauge("active_tasks", 5)
    assert m._gauges["active_tasks"] == 5
    m.set_gauge("active_tasks", 3)
    assert m._gauges["active_tasks"] == 3


def test_histogram():
    m = Metrics()
    for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
        m.observe("latency_seconds", v)
    assert len(m._histograms["latency_seconds"]) == 5


def test_format_prometheus():
    m = Metrics()
    m.inc("http_requests_total", labels={"status": "200"})
    m.set_gauge("active_connections", 10)
    m.observe("response_time", 0.5)

    output = m.format_prometheus()
    assert "http_requests_total" in output
    assert "active_connections 10" in output
    assert "response_time_count 1" in output
    assert "mullet_uptime_seconds" in output


def test_label_key_no_labels():
    assert _label_key("metric", None) == "metric"


def test_label_key_with_labels():
    key = _label_key("metric", {"a": "1", "b": "2"})
    assert key == 'metric{a="1",b="2"}'


def test_histogram_quantiles():
    m = Metrics()
    for i in range(100):
        m.observe("latency", i / 100.0)
    output = m.format_prometheus()
    assert 'quantile="0.5"' in output
    assert 'quantile="0.95"' in output
    assert 'quantile="0.99"' in output


def test_thread_safety():
    """Metrics should be thread-safe."""
    import threading
    m = Metrics()
    errors = []

    def worker():
        try:
            for _ in range(100):
                m.inc("counter")
                m.observe("hist", 0.1)
                m.set_gauge("gauge", 1)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert m._counters["counter"] == 400


def test_histogram_trim_at_1000():
    """Line 41: histogram trimmed to last 1000 when it exceeds 1000 entries."""
    m = Metrics()
    for i in range(1001):
        m.observe("req_duration", float(i))
    assert len(m._histograms[_label_key("req_duration", None)]) == 1000


def test_histogram_empty_values_skipped():
    """Line 68: empty histogram list skipped in format_prometheus."""
    m = Metrics()
    m._histograms["empty_hist"] = []
    output = m.format_prometheus()
    assert "empty_hist" not in output


async def test_metrics_endpoint(aiohttp_client):
    """Metrics endpoint should return Prometheus format."""
    try:
        from mullet.server.app import create_app
        client = await aiohttp_client(create_app())
        resp = await client.get("/metrics")
        assert resp.status == 200
        text = await resp.text()
        assert "mullet_uptime_seconds" in text
    except ImportError:
        pass  # aiohttp not available
