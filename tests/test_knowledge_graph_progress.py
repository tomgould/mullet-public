from unittest.mock import patch
from mullet.knowledge.graph.progress import GraphBuildProgress


class TestGraphBuildProgressPercent:
    def test_percent_zero_total(self):
        p = GraphBuildProgress(total_chunks=0)
        assert p.percent == 100.0

    def test_percent_zero_processed(self):
        p = GraphBuildProgress(total_chunks=100)
        assert p.percent == 0.0

    def test_percent_half(self):
        p = GraphBuildProgress(total_chunks=100, processed_chunks=50)
        assert p.percent == 50.0

    def test_percent_complete(self):
        p = GraphBuildProgress(total_chunks=10, processed_chunks=10)
        assert p.percent == 100.0


class TestGraphBuildProgressETA:
    def test_eta_none_before_any_processed(self):
        p = GraphBuildProgress(total_chunks=10)
        assert p.eta_seconds is None

    def test_format_eta_calculating(self):
        p = GraphBuildProgress(total_chunks=10)
        assert p.format_eta() == "calculating..."

    @patch("time.time", return_value=1000.0)
    def test_eta_computed(self, mock_time):
        p = GraphBuildProgress(total_chunks=10, processed_chunks=5, start_time=995.0)
        with patch("time.time", return_value=1000.0):
            eta = p.eta_seconds
        assert eta == 5.0

    @patch("time.time", return_value=1000.0)
    def test_format_eta_seconds(self, mock_time):
        p = GraphBuildProgress(total_chunks=10, processed_chunks=5, start_time=995.0)
        assert p.format_eta() == "5s"

    @patch("time.time", return_value=1000.0)
    def test_format_eta_minutes_seconds(self, mock_time):
        p = GraphBuildProgress(total_chunks=100, processed_chunks=10, start_time=990.0)
        assert p.format_eta() == "1m 30s"

    @patch("time.time", return_value=1000.0)
    def test_format_eta_exact_minute(self, mock_time):
        p = GraphBuildProgress(total_chunks=130, processed_chunks=10, start_time=990.0)
        assert p.format_eta() == "2m 00s"

    def test_eta_none_zero_rate(self):
        with patch("time.time", return_value=1000.0):
            p = GraphBuildProgress(total_chunks=10, processed_chunks=1, start_time=1000.0)
            assert p.eta_seconds is None
