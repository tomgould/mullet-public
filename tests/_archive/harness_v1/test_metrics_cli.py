"""Tests for mullet.cli.metrics_cli."""

import json

import pytest

from mullet.harness.observability import metrics as _metrics
from mullet.cli.metrics_cli import cmd_metrics


@pytest.fixture(autouse=True)
def _reset_metrics():
    _metrics.reset()
    yield
    _metrics.reset()


class TestMetricsCli:
    def test_list_empty(self, capsys):
        assert cmd_metrics(["list"]) == 0
        out = capsys.readouterr().out
        assert "No metrics recorded" in out

    def test_list_with_metrics(self, capsys):
        _metrics.inc("foo")
        _metrics.gauge("bar", 1.5)
        _metrics.histogram("baz", 10.0)
        assert cmd_metrics(["list"]) == 0
        out = capsys.readouterr().out
        assert "foo: 1" in out
        assert "bar: 1.5" in out
        assert "baz:" in out

    def test_reset(self, capsys):
        _metrics.inc("x")
        assert cmd_metrics(["reset"]) == 0
        out = capsys.readouterr().out
        assert "cleared" in out.lower()
        snap = _metrics.snapshot()
        assert snap.counters == {}

    def test_export_json(self, capsys):
        _metrics.inc("c")
        _metrics.gauge("g", 2.0)
        assert cmd_metrics(["export", "--format", "json"]) == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["counters"]["c"] == 1
        assert data["gauges"]["g"] == 2.0

    def test_export_prometheus(self, capsys):
        _metrics.inc("requests")
        _metrics.gauge("cpu", 0.5)
        _metrics.histogram("lat", 10.0)
        assert cmd_metrics(["export", "--format", "prometheus"]) == 0
        out = capsys.readouterr().out
        assert "requests_count 1" in out or "requests 1" in out
        assert "cpu 0.5" in out
        assert "lat_count 1" in out
        assert "lat_sum 10.0" in out

    def test_no_args_prints_help(self, capsys):
        assert cmd_metrics([]) == 1
        out = capsys.readouterr().out
        assert "usage" in out.lower()
