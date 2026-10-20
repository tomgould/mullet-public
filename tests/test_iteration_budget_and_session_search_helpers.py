"""Tests for IterationBudget and SessionIndex pure helpers."""

import json
import time

from mullet.core.agent.iteration_budget import IterationBudget
from mullet.core.tools.session_search import SessionIndex


# ---------------------------------------------------------------------------
# IterationBudget — construction and defaults
# ---------------------------------------------------------------------------

class TestIterationBudgetDefaults:
    def test_default_max_total(self):
        b = IterationBudget()
        assert b.max_total == 90

    def test_custom_max_total(self):
        b = IterationBudget(max_total=50)
        assert b.max_total == 50

    def test_used_starts_at_zero(self):
        b = IterationBudget()
        assert b.used == 0

    def test_remaining_equals_max_at_start(self):
        b = IterationBudget(max_total=10)
        assert b.remaining == 10

    def test_not_exhausted_initially(self):
        b = IterationBudget()
        assert b.exhausted is False

    def test_not_timed_out_initially(self):
        b = IterationBudget(timeout_seconds=60.0)
        assert b.timed_out is False

    def test_no_timeout_never_times_out(self):
        b = IterationBudget(timeout_seconds=None)
        assert b.timed_out is False

    def test_elapsed_seconds_returns_float(self):
        b = IterationBudget()
        assert isinstance(b.elapsed_seconds, float)

    def test_elapsed_seconds_non_negative(self):
        b = IterationBudget()
        assert b.elapsed_seconds >= 0


# ---------------------------------------------------------------------------
# IterationBudget — consume
# ---------------------------------------------------------------------------

class TestIterationBudgetConsume:
    def test_consume_returns_true(self):
        b = IterationBudget(max_total=5)
        assert b.consume() is True

    def test_consume_increments_used(self):
        b = IterationBudget(max_total=5)
        b.consume()
        assert b.used == 1

    def test_consume_decrements_remaining(self):
        b = IterationBudget(max_total=5)
        b.consume()
        assert b.remaining == 4

    def test_consume_at_limit_returns_false(self):
        b = IterationBudget(max_total=2)
        b.consume()
        b.consume()
        assert b.consume() is False

    def test_used_does_not_exceed_max(self):
        b = IterationBudget(max_total=2)
        b.consume()
        b.consume()
        b.consume()
        assert b.used == 2

    def test_remaining_never_negative(self):
        b = IterationBudget(max_total=1)
        b.consume()
        b.consume()
        assert b.remaining == 0

    def test_exhausted_when_at_max(self):
        b = IterationBudget(max_total=2)
        b.consume()
        b.consume()
        assert b.exhausted is True

    def test_consume_returns_false_when_timed_out(self):
        b = IterationBudget(max_total=100, timeout_seconds=0.001)
        time.sleep(0.01)
        assert b.consume() is False

    def test_used_unchanged_after_timeout_consume(self):
        b = IterationBudget(max_total=100, timeout_seconds=0.001)
        time.sleep(0.01)
        b.consume()
        assert b.used == 0


# ---------------------------------------------------------------------------
# IterationBudget — refund
# ---------------------------------------------------------------------------

class TestIterationBudgetRefund:
    def test_refund_decrements_used(self):
        b = IterationBudget(max_total=10)
        b.consume()
        b.consume()
        b.refund(1)
        assert b.used == 1

    def test_refund_multiple(self):
        b = IterationBudget(max_total=10)
        for _ in range(5):
            b.consume()
        b.refund(3)
        assert b.used == 2

    def test_refund_does_not_go_negative(self):
        b = IterationBudget(max_total=10)
        b.refund(100)
        assert b.used == 0

    def test_refund_restores_remaining(self):
        b = IterationBudget(max_total=10)
        b.consume()
        b.refund(1)
        assert b.remaining == 10


# ---------------------------------------------------------------------------
# IterationBudget — pressure
# ---------------------------------------------------------------------------

class TestIterationBudgetPressure:
    def test_none_pressure_initially(self):
        b = IterationBudget(max_total=10)
        assert b.pressure == "none"

    def test_caution_at_70_percent(self):
        b = IterationBudget(max_total=10)
        for _ in range(7):
            b.consume()
        assert b.pressure == "caution"

    def test_urgent_at_90_percent_with_remaining_gt_1(self):
        b = IterationBudget(max_total=20)
        for _ in range(18):
            b.consume()
        assert b.pressure == "urgent"

    def test_critical_when_remaining_le_1(self):
        b = IterationBudget(max_total=10)
        for _ in range(9):
            b.consume()
        assert b.pressure == "critical"

    def test_critical_when_exhausted(self):
        b = IterationBudget(max_total=3)
        for _ in range(3):
            b.consume()
        assert b.pressure == "critical"

    def test_pressure_returns_string(self):
        b = IterationBudget(max_total=10)
        assert isinstance(b.pressure, str)

    def test_pressure_values_are_known(self):
        b = IterationBudget(max_total=10)
        assert b.pressure in ("none", "caution", "urgent", "critical")


# ---------------------------------------------------------------------------
# IterationBudget — get_pressure_message
# ---------------------------------------------------------------------------

class TestIterationBudgetPressureMessage:
    def test_returns_none_when_no_pressure(self):
        b = IterationBudget(max_total=10)
        assert b.get_pressure_message() is None

    def test_returns_string_at_caution(self):
        b = IterationBudget(max_total=10)
        for _ in range(7):
            b.consume()
        msg = b.get_pressure_message()
        assert isinstance(msg, str)

    def test_caution_message_contains_note(self):
        b = IterationBudget(max_total=10)
        for _ in range(7):
            b.consume()
        assert "NOTE" in b.get_pressure_message()

    def test_urgent_message_contains_urgent(self):
        b = IterationBudget(max_total=20)
        for _ in range(18):
            b.consume()
        assert "URGENT" in b.get_pressure_message()

    def test_critical_message_contains_critical(self):
        b = IterationBudget(max_total=10)
        for _ in range(9):
            b.consume()
        assert "CRITICAL" in b.get_pressure_message()

    def test_critical_message_mentions_remaining(self):
        b = IterationBudget(max_total=10)
        for _ in range(9):
            b.consume()
        msg = b.get_pressure_message()
        assert "1" in msg

    def test_message_mentions_max_total_at_urgent(self):
        b = IterationBudget(max_total=20)
        for _ in range(18):
            b.consume()
        msg = b.get_pressure_message()
        assert "20" in msg


# ---------------------------------------------------------------------------
# IterationBudget — timed_out
# ---------------------------------------------------------------------------

class TestIterationBudgetTimeout:
    def test_not_timed_out_before_deadline(self):
        b = IterationBudget(timeout_seconds=60.0)
        assert b.timed_out is False

    def test_timed_out_after_deadline(self):
        b = IterationBudget(timeout_seconds=0.001)
        time.sleep(0.01)
        assert b.timed_out is True

    def test_no_timeout_never_times_out(self):
        b = IterationBudget(timeout_seconds=None)
        assert b.timed_out is False


# ---------------------------------------------------------------------------
# SessionIndex
# ---------------------------------------------------------------------------

def _make_session(tmp_path, name, entries):
    """Write a .jsonl session file with given entries."""
    filepath = tmp_path / name
    with open(filepath, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return filepath


class TestSessionIndexConstruction:
    def test_empty_dir_no_sessions(self, tmp_path):
        idx = SessionIndex(str(tmp_path))
        assert idx.sessions == []

    def test_ignores_non_jsonl_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("some text")
        idx = SessionIndex(str(tmp_path))
        assert idx.sessions == []

    def test_loads_jsonl_files(self, tmp_path):
        _make_session(tmp_path, "sess.jsonl", [{"user_message": "hello"}])
        idx = SessionIndex(str(tmp_path))
        assert len(idx.sessions) == 1

    def test_session_has_filename_key(self, tmp_path):
        _make_session(tmp_path, "sess.jsonl", [{"user_message": "hi"}])
        idx = SessionIndex(str(tmp_path))
        assert "filename" in idx.sessions[0]

    def test_session_filename_matches(self, tmp_path):
        _make_session(tmp_path, "mysess.jsonl", [{"user_message": "hello"}])
        idx = SessionIndex(str(tmp_path))
        assert idx.sessions[0]["filename"] == "mysess.jsonl"

    def test_session_has_messages_key(self, tmp_path):
        _make_session(tmp_path, "sess.jsonl", [{"user_message": "hi"}])
        idx = SessionIndex(str(tmp_path))
        assert "messages" in idx.sessions[0]

    def test_malformed_lines_skipped(self, tmp_path):
        filepath = tmp_path / "bad.jsonl"
        filepath.write_text('{"user_message": "good"}\nnot json\n{"assistant_message": "ok"}\n')
        idx = SessionIndex(str(tmp_path))
        messages = idx.sessions[0]["messages"]
        assert len(messages) == 2


class TestSessionIndexSearch:
    def test_search_returns_list(self, tmp_path):
        _make_session(tmp_path, "s.jsonl", [{"user_message": "hello world"}])
        idx = SessionIndex(str(tmp_path))
        result = idx.search("hello")
        assert isinstance(result, list)

    def test_search_finds_match(self, tmp_path):
        _make_session(tmp_path, "s.jsonl", [{"user_message": "deploy to production"}])
        idx = SessionIndex(str(tmp_path))
        result = idx.search("deploy")
        assert len(result) == 1

    def test_search_no_match_returns_empty(self, tmp_path):
        _make_session(tmp_path, "s.jsonl", [{"user_message": "hello world"}])
        idx = SessionIndex(str(tmp_path))
        result = idx.search("xyzzy")
        assert result == []

    def test_search_case_insensitive(self, tmp_path):
        _make_session(tmp_path, "s.jsonl", [{"user_message": "Deploy Application"}])
        idx = SessionIndex(str(tmp_path))
        result = idx.search("deploy")
        assert len(result) == 1

    def test_search_result_has_session_file_key(self, tmp_path):
        _make_session(tmp_path, "sess.jsonl", [{"user_message": "test query"}])
        idx = SessionIndex(str(tmp_path))
        result = idx.search("test")
        assert "session_file" in result[0]

    def test_search_result_has_matches_key(self, tmp_path):
        _make_session(tmp_path, "s.jsonl", [{"user_message": "test query"}])
        idx = SessionIndex(str(tmp_path))
        result = idx.search("test")
        assert "matches" in result[0]

    def test_search_max_results_respected(self, tmp_path):
        for i in range(10):
            _make_session(tmp_path, f"s{i}.jsonl", [{"user_message": f"keyword content {i}"}])
        idx = SessionIndex(str(tmp_path))
        result = idx.search("keyword", max_results=3)
        assert len(result) <= 3

    def test_search_assistant_messages_also_matched(self, tmp_path):
        _make_session(tmp_path, "s.jsonl", [{"assistant_message": "the answer is 42"}])
        idx = SessionIndex(str(tmp_path))
        result = idx.search("answer")
        assert len(result) == 1

    def test_search_multiple_sessions_returns_most_relevant_first(self, tmp_path):
        _make_session(tmp_path, "many.jsonl", [
            {"user_message": "deploy deploy deploy"},
            {"assistant_message": "deploy confirmed"},
        ])
        _make_session(tmp_path, "few.jsonl", [{"user_message": "deploy once"}])
        idx = SessionIndex(str(tmp_path))
        result = idx.search("deploy")
        assert result[0]["session_file"] == "many.jsonl"
