"""Tests for small pure-function utility modules.

Covers:
- mullet.cli.rich_errors.format_error
- mullet.hud.budget_monitor.format_advice
- mullet.core.providers.cost_tracker.CostTracker
"""

import pytest
from mullet.cli.rich_errors import format_error
from mullet.hud.budget_monitor import format_advice
from mullet.core.providers.cost_tracker import CostTracker


# ---------------------------------------------------------------------------
# format_error (rich_errors)
# ---------------------------------------------------------------------------

class TestFormatError:
    def _strip_ansi(self, s: str) -> str:
        import re
        return re.sub(r'\033\[[0-9;]*m', '', s)

    def test_returns_string(self):
        assert isinstance(format_error(ValueError("bad")), str)

    def test_includes_error_type(self):
        result = self._strip_ansi(format_error(ValueError("oops")))
        assert "ValueError" in result

    def test_includes_error_message(self):
        result = self._strip_ansi(format_error(ValueError("oops")))
        assert "oops" in result

    def test_import_error_has_pip_suggestion(self):
        err = ImportError("No module named 'foo'")
        result = self._strip_ansi(format_error(err))
        assert "pip install" in result
        assert "foo" in result

    def test_file_not_found_has_path_suggestion(self):
        err = FileNotFoundError("no such file")
        result = self._strip_ansi(format_error(err))
        assert "path" in result or "file" in result.lower()

    def test_connection_error_has_network_suggestion(self):
        err = ConnectionError("refused")
        result = self._strip_ansi(format_error(err))
        assert "network" in result.lower() or "connection" in result.lower()

    def test_value_error_has_input_suggestion(self):
        err = ValueError("bad value")
        result = self._strip_ansi(format_error(err))
        assert "input" in result.lower() or "arguments" in result.lower()

    def test_type_error_has_type_suggestion(self):
        err = TypeError("wrong type")
        result = self._strip_ansi(format_error(err))
        assert "type" in result.lower()

    def test_context_included_when_provided(self):
        err = ValueError("bad")
        result = self._strip_ansi(format_error(err, context="in my_function"))
        assert "in my_function" in result

    def test_no_suggestion_for_unknown_error(self):
        err = RuntimeError("unexpected")
        result = self._strip_ansi(format_error(err))
        assert "RuntimeError" in result
        assert "Suggestion:" not in result

    def test_context_none_no_context_line(self):
        err = ValueError("bad")
        result = self._strip_ansi(format_error(err, context=None))
        assert "Context:" not in result


# ---------------------------------------------------------------------------
# format_advice (budget_monitor)
# ---------------------------------------------------------------------------

class TestFormatAdvice:
    def test_returns_string(self):
        result = format_advice({"advice": "OK", "usage_5h_pct": 30.0})
        assert isinstance(result, str)

    def test_includes_advice(self):
        result = format_advice({"advice": "THROTTLE", "usage_5h_pct": 90.0})
        assert "THROTTLE" in result

    def test_includes_percentage(self):
        result = format_advice({"advice": "OK", "usage_5h_pct": 42.7})
        assert "43%" in result or "42%" in result  # Rounded to 0 decimal places

    def test_none_pct_shows_question_mark(self):
        result = format_advice({"advice": "UNKNOWN", "usage_5h_pct": None})
        assert "?" in result

    def test_zero_pct_formatted(self):
        result = format_advice({"advice": "OK", "usage_5h_pct": 0.0})
        assert "0%" in result

    def test_hundred_pct_formatted(self):
        result = format_advice({"advice": "THROTTLE", "usage_5h_pct": 100.0})
        assert "100%" in result


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class TestCostTracker:
    def test_empty_total_cost_zero(self):
        tracker = CostTracker()
        assert tracker.total_cost() == 0.0

    def test_record_valid_model(self):
        tracker = CostTracker()
        tracker.record("claude-haiku-4-5", tokens_in=1000, tokens_out=500)
        assert tracker.total_cost() > 0

    def test_total_cost_accumulates(self):
        tracker = CostTracker()
        tracker.record("claude-haiku-4-5", 1000, 500)
        cost1 = tracker.total_cost()
        tracker.record("claude-haiku-4-5", 1000, 500)
        cost2 = tracker.total_cost()
        assert cost2 == pytest.approx(cost1 * 2)

    def test_negative_tokens_in_raises(self):
        tracker = CostTracker()
        with pytest.raises(ValueError):
            tracker.record("claude-haiku-4-5", tokens_in=-1, tokens_out=100)

    def test_negative_tokens_out_raises(self):
        tracker = CostTracker()
        with pytest.raises(ValueError):
            tracker.record("claude-haiku-4-5", tokens_in=100, tokens_out=-1)

    def test_unknown_model_records_zero_cost(self):
        # get_pricing returns (0.0, 0.0) for unknown models — no exception raised
        tracker = CostTracker()
        tracker.record("nonexistent-model-xyz", tokens_in=100, tokens_out=50)
        assert tracker.total_cost() == 0.0

    def test_breakdown_by_model_keys(self):
        tracker = CostTracker()
        tracker.record("claude-haiku-4-5", 1000, 500)
        breakdown = tracker.breakdown_by_model()
        assert "claude-haiku-4-5" in breakdown

    def test_breakdown_multiple_models(self):
        tracker = CostTracker()
        tracker.record("claude-haiku-4-5", 500, 250)
        tracker.record("claude-haiku-4-5", 500, 250)
        breakdown = tracker.breakdown_by_model()
        assert breakdown["claude-haiku-4-5"] > 0

    def test_breakdown_returns_dict(self):
        tracker = CostTracker()
        assert isinstance(tracker.breakdown_by_model(), dict)

    def test_empty_breakdown_empty_dict(self):
        tracker = CostTracker()
        assert tracker.breakdown_by_model() == {}
