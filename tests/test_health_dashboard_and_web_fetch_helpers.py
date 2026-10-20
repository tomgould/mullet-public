"""Tests for mullet.core.providers.health_dashboard and mullet.core.tools.web_fetch helpers."""

import pytest
from mullet.core.providers.health_dashboard import ProviderHealthDashboard
from mullet.core.tools.web_fetch import HTMLStripper


# ---------------------------------------------------------------------------
# ProviderHealthDashboard — record_call
# ---------------------------------------------------------------------------

class TestProviderHealthDashboardRecordCall:
    def setup_method(self):
        self.dash = ProviderHealthDashboard()

    def test_initial_empty(self):
        assert self.dash.providers_data == {}

    def test_record_creates_entry(self):
        self.dash.record_call("kimi", 1.0, True)
        assert "kimi" in self.dash.providers_data

    def test_record_total_calls_incremented(self):
        self.dash.record_call("kimi", 0.5, True)
        self.dash.record_call("kimi", 0.3, False)
        assert self.dash.providers_data["kimi"]["total_calls"] == 2

    def test_record_successful_calls_incremented_on_success(self):
        self.dash.record_call("gemini", 1.2, True)
        assert self.dash.providers_data["gemini"]["successful_calls"] == 1

    def test_record_failure_does_not_increment_successful(self):
        self.dash.record_call("gemini", 0.8, False)
        assert self.dash.providers_data["gemini"]["successful_calls"] == 0

    def test_total_latency_accumulated(self):
        self.dash.record_call("kimi", 1.0, True)
        self.dash.record_call("kimi", 2.0, True)
        assert self.dash.providers_data["kimi"]["total_latency"] == pytest.approx(3.0)

    def test_min_latency_tracked(self):
        self.dash.record_call("kimi", 2.0, True)
        self.dash.record_call("kimi", 0.5, True)
        self.dash.record_call("kimi", 1.5, True)
        assert self.dash.providers_data["kimi"]["min_latency"] == pytest.approx(0.5)

    def test_max_latency_tracked(self):
        self.dash.record_call("kimi", 2.0, True)
        self.dash.record_call("kimi", 0.5, True)
        self.dash.record_call("kimi", 5.0, True)
        assert self.dash.providers_data["kimi"]["max_latency"] == pytest.approx(5.0)

    def test_multiple_providers_independent(self):
        self.dash.record_call("kimi", 1.0, True)
        self.dash.record_call("gemini", 2.0, False)
        assert self.dash.providers_data["kimi"]["successful_calls"] == 1
        assert self.dash.providers_data["gemini"]["successful_calls"] == 0


# ---------------------------------------------------------------------------
# ProviderHealthDashboard — get_report
# ---------------------------------------------------------------------------

class TestProviderHealthDashboardGetReport:
    def setup_method(self):
        self.dash = ProviderHealthDashboard()

    def test_empty_dashboard_empty_report(self):
        assert self.dash.get_report() == {}

    def test_report_has_provider_key(self):
        self.dash.record_call("kimi", 1.0, True)
        report = self.dash.get_report()
        assert "kimi" in report

    def test_total_calls_in_report(self):
        self.dash.record_call("kimi", 1.0, True)
        self.dash.record_call("kimi", 2.0, False)
        report = self.dash.get_report()
        assert report["kimi"]["total_calls"] == 2

    def test_error_rate_all_success(self):
        self.dash.record_call("kimi", 1.0, True)
        self.dash.record_call("kimi", 1.0, True)
        report = self.dash.get_report()
        assert report["kimi"]["error_rate"] == pytest.approx(0.0)

    def test_error_rate_all_fail(self):
        self.dash.record_call("kimi", 1.0, False)
        self.dash.record_call("kimi", 1.0, False)
        report = self.dash.get_report()
        assert report["kimi"]["error_rate"] == pytest.approx(1.0)

    def test_error_rate_half(self):
        self.dash.record_call("kimi", 1.0, True)
        self.dash.record_call("kimi", 1.0, False)
        report = self.dash.get_report()
        assert report["kimi"]["error_rate"] == pytest.approx(0.5)

    def test_average_latency_calculated(self):
        self.dash.record_call("gemini", 1.0, True)
        self.dash.record_call("gemini", 3.0, True)
        report = self.dash.get_report()
        assert report["gemini"]["average_latency"] == pytest.approx(2.0)

    def test_min_latency_in_report(self):
        self.dash.record_call("kimi", 0.5, True)
        self.dash.record_call("kimi", 2.0, True)
        report = self.dash.get_report()
        assert report["kimi"]["min_latency"] == pytest.approx(0.5)

    def test_max_latency_in_report(self):
        self.dash.record_call("kimi", 0.5, True)
        self.dash.record_call("kimi", 5.0, True)
        report = self.dash.get_report()
        assert report["kimi"]["max_latency"] == pytest.approx(5.0)

    def test_single_call_report(self):
        self.dash.record_call("haiku", 0.7, True)
        report = self.dash.get_report()
        assert report["haiku"]["total_calls"] == 1
        assert report["haiku"]["error_rate"] == pytest.approx(0.0)
        assert report["haiku"]["average_latency"] == pytest.approx(0.7)

    def test_no_inf_in_report_for_single_call(self):
        self.dash.record_call("kimi", 1.5, True)
        report = self.dash.get_report()
        assert report["kimi"]["min_latency"] != float("inf")
        assert report["kimi"]["max_latency"] != float("-inf")


# ---------------------------------------------------------------------------
# HTMLStripper
# ---------------------------------------------------------------------------

class TestHTMLStripper:
    def setup_method(self):
        self.stripper = HTMLStripper()

    def test_plain_text_passthrough(self):
        self.stripper.feed("Hello world")
        assert self.stripper.get_data() == "Hello world"

    def test_strips_paragraph_tags(self):
        self.stripper.feed("<p>Hello</p>")
        assert "Hello" in self.stripper.get_data()
        assert "<p>" not in self.stripper.get_data()

    def test_strips_div_tags(self):
        self.stripper.feed("<div>Content</div>")
        assert "Content" in self.stripper.get_data()
        assert "<div>" not in self.stripper.get_data()

    def test_strips_script_content(self):
        self.stripper.feed("<p>Text</p><script>alert('xss')</script><p>More</p>")
        data = self.stripper.get_data()
        assert "alert" not in data
        assert "Text" in data
        assert "More" in data

    def test_strips_heading_tags(self):
        self.stripper.feed("<h1>Title</h1><p>Body</p>")
        data = self.stripper.get_data()
        assert "Title" in data
        assert "Body" in data
        assert "<h1>" not in data

    def test_empty_html_returns_empty(self):
        self.stripper.feed("")
        assert self.stripper.get_data() == ""

    def test_nested_tags_stripped(self):
        self.stripper.feed("<div><span><b>Bold</b></span></div>")
        assert "Bold" in self.stripper.get_data()
        assert "<" not in self.stripper.get_data()

    def test_get_data_returns_string(self):
        self.stripper.feed("<p>test</p>")
        assert isinstance(self.stripper.get_data(), str)
