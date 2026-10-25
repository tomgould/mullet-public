"""Tests for wrap, health_dashboard, and cost_tracker pure helpers."""

import pytest

from mullet.hud.render.wrap import wcwidth_truncate, split_wrap_parts, _visible_width
from mullet.core.providers.health_dashboard import ProviderHealthDashboard
from mullet.core.providers.cost_tracker import CostTracker


# ---------------------------------------------------------------------------
# _visible_width
# ---------------------------------------------------------------------------

class TestVisibleWidth:
    def test_plain_ascii(self):
        assert _visible_width("hello") == 5

    def test_empty_string(self):
        assert _visible_width("") == 0

    def test_ansi_not_counted(self):
        assert _visible_width("\x1b[31mhello\x1b[0m") == 5

    def test_multiple_ansi(self):
        text = "\x1b[1m\x1b[32mOK\x1b[0m"
        assert _visible_width(text) == 2


# ---------------------------------------------------------------------------
# wcwidth_truncate
# ---------------------------------------------------------------------------

class TestWcwidthTruncate:
    def test_short_line_unchanged(self):
        assert wcwidth_truncate("hi", 20) == "hi"

    def test_zero_max_width_returns_line(self):
        result = wcwidth_truncate("hello", 0)
        assert result == "hello"

    def test_negative_max_width_returns_line(self):
        result = wcwidth_truncate("hello", -1)
        assert result == "hello"

    def test_truncated_ends_with_ellipsis(self):
        result = wcwidth_truncate("hello world testing", 8)
        assert result.endswith("…")

    def test_truncated_visible_width_at_most_max(self):
        result = wcwidth_truncate("hello world testing", 8)
        assert _visible_width(result) <= 8

    def test_exact_fit_not_truncated(self):
        result = wcwidth_truncate("hello", 5)
        assert "hello" in result
        assert "…" not in result

    def test_ansi_preserved_in_output(self):
        result = wcwidth_truncate("\x1b[31mhello world test\x1b[0m", 8)
        assert "\x1b[31m" in result

    def test_returns_string(self):
        assert isinstance(wcwidth_truncate("test string here", 10), str)

    def test_single_char_max_width(self):
        result = wcwidth_truncate("hello", 1)
        # With ellipsis width of 1, budget=0 → just ellipsis
        assert "…" in result


# ---------------------------------------------------------------------------
# split_wrap_parts
# ---------------------------------------------------------------------------

class TestSplitWrapParts:
    def test_short_line_single_element(self):
        result = split_wrap_parts("Short", 80)
        assert result == ["Short"]

    def test_zero_max_width_returns_original(self):
        result = split_wrap_parts("A | B | C", 0)
        assert result == ["A | B | C"]

    def test_fits_on_one_line_not_split(self):
        result = split_wrap_parts("A | B", 80)
        assert len(result) == 1

    def test_overflow_splits_into_two(self):
        result = split_wrap_parts("A | B | C | D", 6)
        assert len(result) >= 2

    def test_returns_list(self):
        result = split_wrap_parts("Part1 | Part2", 80)
        assert isinstance(result, list)

    def test_all_parts_present(self):
        result = split_wrap_parts("A | B | C", 80)
        joined = " | ".join(result)
        assert "A" in joined
        assert "B" in joined
        assert "C" in joined

    def test_empty_string_returns_list(self):
        result = split_wrap_parts("", 80)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# ProviderHealthDashboard
# ---------------------------------------------------------------------------

class TestProviderHealthDashboardEmpty:
    def test_empty_report_is_empty_dict(self):
        d = ProviderHealthDashboard()
        assert d.get_report() == {}

    def test_providers_data_initially_empty(self):
        d = ProviderHealthDashboard()
        assert d.providers_data == {}


class TestProviderHealthDashboardRecordCall:
    def test_record_call_adds_provider(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        assert "kimi" in d.get_report()

    def test_total_calls_increments(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        d.record_call("kimi", 0.3, True)
        assert d.get_report()["kimi"]["total_calls"] == 2

    def test_success_reduces_error_rate(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        assert d.get_report()["kimi"]["error_rate"] == 0.0

    def test_failure_increases_error_rate(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, False)
        assert d.get_report()["kimi"]["error_rate"] == 1.0

    def test_mixed_error_rate(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        d.record_call("kimi", 0.5, False)
        assert d.get_report()["kimi"]["error_rate"] == pytest.approx(0.5)

    def test_average_latency_computed(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.2, True)
        d.record_call("kimi", 0.4, True)
        assert d.get_report()["kimi"]["average_latency"] == pytest.approx(0.3)

    def test_min_latency_tracked(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        d.record_call("kimi", 0.1, True)
        assert d.get_report()["kimi"]["min_latency"] == pytest.approx(0.1)

    def test_max_latency_tracked(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.1, True)
        d.record_call("kimi", 0.9, True)
        assert d.get_report()["kimi"]["max_latency"] == pytest.approx(0.9)

    def test_multiple_providers_independent(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.3, True)
        d.record_call("gemini", 0.5, False)
        report = d.get_report()
        assert report["kimi"]["error_rate"] == 0.0
        assert report["gemini"]["error_rate"] == 1.0

    def test_report_has_all_keys(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.3, True)
        entry = d.get_report()["kimi"]
        for key in ("total_calls", "error_rate", "average_latency", "min_latency", "max_latency"):
            assert key in entry


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class TestCostTrackerEmpty:
    def test_total_cost_zero_initially(self):
        ct = CostTracker()
        assert ct.total_cost() == 0.0

    def test_breakdown_empty_initially(self):
        ct = CostTracker()
        assert ct.breakdown_by_model() == {}


class TestCostTrackerRecord:
    def test_record_adds_cost(self):
        ct = CostTracker()
        ct.record("claude-3-opus", 1000, 500)
        assert ct.total_cost() > 0

    def test_total_cost_accumulates(self):
        ct = CostTracker()
        ct.record("claude-3-opus", 1000, 0)
        ct.record("claude-3-opus", 1000, 0)
        first = CostTracker()
        first.record("claude-3-opus", 1000, 0)
        assert ct.total_cost() == pytest.approx(first.total_cost() * 2)

    def test_breakdown_by_model_has_entry(self):
        ct = CostTracker()
        ct.record("claude-3-opus", 1000, 500)
        breakdown = ct.breakdown_by_model()
        assert "claude-3-opus" in breakdown

    def test_breakdown_cost_positive(self):
        ct = CostTracker()
        ct.record("claude-3-opus", 1000, 500)
        assert ct.breakdown_by_model()["claude-3-opus"] > 0

    def test_multiple_models_tracked_separately(self):
        ct = CostTracker()
        ct.record("claude-3-opus", 1000, 0)
        ct.record("claude-3-haiku", 1000, 0)
        breakdown = ct.breakdown_by_model()
        assert "claude-3-opus" in breakdown
        assert "claude-3-haiku" in breakdown

    def test_negative_tokens_in_raises(self):
        ct = CostTracker()
        with pytest.raises(ValueError):
            ct.record("claude-3-opus", -1, 0)

    def test_negative_tokens_out_raises(self):
        ct = CostTracker()
        with pytest.raises(ValueError):
            ct.record("claude-3-opus", 0, -1)

    def test_zero_tokens_zero_cost(self):
        ct = CostTracker()
        ct.record("claude-3-opus", 0, 0)
        assert ct.total_cost() == 0.0
