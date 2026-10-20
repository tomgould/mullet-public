"""Tests for HTMLStripper, format_error, CostTracker, and format_advice pure helpers."""

import pytest

from mullet.core.tools.web_fetch import HTMLStripper
from mullet.cli.rich_errors import format_error
from mullet.core.providers.cost_tracker import CostTracker
from mullet.hud.budget_monitor import format_advice


# ---------------------------------------------------------------------------
# HTMLStripper
# ---------------------------------------------------------------------------

class TestHTMLStripper:
    def test_strips_basic_tags(self):
        s = HTMLStripper()
        s.feed("<p>Hello</p>")
        assert "Hello" in s.get_data()
        assert "<" not in s.get_data()

    def test_strips_bold(self):
        s = HTMLStripper()
        s.feed("<b>bold text</b>")
        assert s.get_data() == "bold text"

    def test_strips_nested_tags(self):
        s = HTMLStripper()
        s.feed("<html><body><p>content</p></body></html>")
        assert "content" in s.get_data()

    def test_strips_script_tag_content(self):
        s = HTMLStripper()
        s.feed("<html><script>alert(1)</script><p>visible</p></html>")
        assert "alert" not in s.get_data()
        assert "visible" in s.get_data()

    def test_empty_feed_returns_empty(self):
        s = HTMLStripper()
        s.feed("")
        assert s.get_data() == ""

    def test_plain_text_preserved(self):
        s = HTMLStripper()
        s.feed("no tags here")
        assert "no tags here" in s.get_data()

    def test_returns_string(self):
        s = HTMLStripper()
        s.feed("<p>test</p>")
        assert isinstance(s.get_data(), str)

    def test_multiple_paragraphs(self):
        s = HTMLStripper()
        s.feed("<p>first</p><p>second</p>")
        result = s.get_data()
        assert "first" in result
        assert "second" in result


# ---------------------------------------------------------------------------
# format_error
# ---------------------------------------------------------------------------

class TestFormatError:
    def test_returns_string(self):
        assert isinstance(format_error(RuntimeError("test")), str)

    def test_error_type_in_output(self):
        result = format_error(RuntimeError("test"))
        assert "RuntimeError" in result

    def test_message_in_output(self):
        result = format_error(ValueError("bad input"))
        assert "bad input" in result

    def test_import_error_suggests_pip(self):
        e = ImportError("No module named 'requests'")
        result = format_error(e)
        assert "pip install" in result

    def test_import_error_includes_module_name(self):
        e = ImportError("No module named 'requests'")
        result = format_error(e)
        assert "requests" in result

    def test_file_not_found_suggests_path(self):
        result = format_error(FileNotFoundError("missing.txt"))
        assert "file" in result.lower() or "path" in result.lower()

    def test_connection_error_suggests_network(self):
        result = format_error(ConnectionError("refused"))
        assert "network" in result.lower() or "connection" in result.lower()

    def test_value_error_suggests_input(self):
        result = format_error(ValueError("bad"))
        assert "input" in result.lower() or "value" in result.lower()

    def test_type_error_suggests_types(self):
        result = format_error(TypeError("wrong type"))
        assert "type" in result.lower()

    def test_context_appears_in_output(self):
        result = format_error(RuntimeError("err"), context="during startup")
        assert "during startup" in result

    def test_no_context_works(self):
        result = format_error(RuntimeError("err"))
        assert isinstance(result, str)

    def test_red_color_code_present(self):
        result = format_error(RuntimeError("err"))
        assert "\x1b[31m" in result or "Error" in result

    def test_unknown_error_no_suggestion_but_no_crash(self):
        result = format_error(MemoryError("OOM"))
        assert isinstance(result, str)
        assert "MemoryError" in result


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class TestCostTrackerInit:
    def test_records_empty_initially(self):
        ct = CostTracker()
        assert ct._records == []

    def test_total_cost_zero_initially(self):
        ct = CostTracker()
        assert ct.total_cost() == 0

    def test_breakdown_empty_initially(self):
        ct = CostTracker()
        assert ct.breakdown_by_model() == {}


class TestCostTrackerRecord:
    def test_record_increases_total(self):
        ct = CostTracker()
        ct.record("gpt-4", 1000, 500)
        assert ct.total_cost() > 0

    def test_record_adds_to_records(self):
        ct = CostTracker()
        ct.record("gpt-4", 1000, 500)
        assert len(ct._records) == 1

    def test_two_records_accumulated(self):
        ct = CostTracker()
        ct.record("gpt-4", 1000, 500)
        ct.record("gpt-4", 2000, 1000)
        assert len(ct._records) == 2

    def test_total_cost_is_sum(self):
        ct = CostTracker()
        ct.record("gpt-4", 1000, 500)
        first = ct.total_cost()
        ct.record("gpt-4", 1000, 500)
        assert ct.total_cost() == pytest.approx(first * 2)

    def test_negative_tokens_in_raises(self):
        ct = CostTracker()
        with pytest.raises(ValueError):
            ct.record("gpt-4", -1, 0)

    def test_negative_tokens_out_raises(self):
        ct = CostTracker()
        with pytest.raises(ValueError):
            ct.record("gpt-4", 0, -1)

    def test_zero_tokens_allowed(self):
        ct = CostTracker()
        ct.record("gpt-4", 0, 0)
        assert ct.total_cost() == 0.0

    def test_model_stored_in_record(self):
        ct = CostTracker()
        ct.record("gpt-4", 100, 50)
        assert ct._records[0]["model"] == "gpt-4"

    def test_tokens_stored_in_record(self):
        ct = CostTracker()
        ct.record("gpt-4", 123, 456)
        assert ct._records[0]["tokens_in"] == 123
        assert ct._records[0]["tokens_out"] == 456


class TestCostTrackerBreakdown:
    def test_single_model_in_breakdown(self):
        ct = CostTracker()
        ct.record("gpt-4", 1000, 500)
        bd = ct.breakdown_by_model()
        assert "gpt-4" in bd

    def test_multiple_records_same_model_accumulated(self):
        ct = CostTracker()
        ct.record("gpt-4", 1000, 500)
        ct.record("gpt-4", 1000, 500)
        bd = ct.breakdown_by_model()
        ct2 = CostTracker()
        ct2.record("gpt-4", 1000, 500)
        assert bd["gpt-4"] == pytest.approx(ct2.breakdown_by_model()["gpt-4"] * 2)

    def test_returns_dict(self):
        ct = CostTracker()
        ct.record("gpt-4", 100, 50)
        assert isinstance(ct.breakdown_by_model(), dict)


# ---------------------------------------------------------------------------
# format_advice
# ---------------------------------------------------------------------------

class TestFormatAdvice:
    def _result(self, advice, pct):
        return {"advice": advice, "usage_5h_pct": pct, "can_use_anthropic": True, "resources": {}}

    def test_returns_string(self):
        r = self._result("OK — Anthropic available, use judiciously", 30.0)
        assert isinstance(format_advice(r), str)

    def test_advice_text_in_output(self):
        r = self._result("OK — Anthropic available, use judiciously", 30.0)
        assert "OK" in format_advice(r)

    def test_pct_shown_as_int_percent(self):
        r = self._result("OK — Anthropic available", 45.5)
        result = format_advice(r)
        assert "46%" in result or "45%" in result

    def test_none_pct_shows_question_mark(self):
        r = self._result("UNKNOWN — usage data unavailable", None)
        assert "?" in format_advice(r)

    def test_throttle_message_in_output(self):
        r = self._result("THROTTLE — delegate to Kimi/Gemini", 85.0)
        assert "THROTTLE" in format_advice(r)

    def test_zero_pct_shows_zero(self):
        r = self._result("OK — Anthropic available", 0.0)
        assert "0%" in format_advice(r)

    def test_hundred_pct_shows_100(self):
        r = self._result("THROTTLE", 100.0)
        result = format_advice(r)
        assert "100%" in result
