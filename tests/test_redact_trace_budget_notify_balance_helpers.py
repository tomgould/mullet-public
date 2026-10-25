"""Tests for redact, trace, and iteration_budget pure helpers."""

import time
import threading
import pytest

from mullet.core.agent.redact import _mask_token, redact_sensitive_text, RedactingFormatter
from mullet.core.agent.trace import TraceEvent, SpanContext, AgentTracer
from mullet.core.agent.iteration_budget import IterationBudget


# ---------------------------------------------------------------------------
# _mask_token
# ---------------------------------------------------------------------------

class TestMaskToken:
    def test_short_token_fully_masked(self):
        assert _mask_token("short") == "***"

    def test_exactly_17_chars_masked(self):
        assert _mask_token("a" * 17) == "***"

    def test_18_chars_not_fully_masked(self):
        result = _mask_token("a" * 18)
        assert result != "***"

    def test_long_token_preserves_first_six(self):
        token = "ABCDEF_middle_WXYZ"
        result = _mask_token(token)
        assert result.startswith("ABCDEF")

    def test_long_token_preserves_last_four(self):
        token = "ABCDEF_middle_WXYZ"
        result = _mask_token(token)
        assert result.endswith("WXYZ")

    def test_long_token_contains_ellipsis(self):
        result = _mask_token("A" * 25)
        assert "..." in result

    def test_returns_string(self):
        assert isinstance(_mask_token("token"), str)


# ---------------------------------------------------------------------------
# redact_sensitive_text
# ---------------------------------------------------------------------------

class TestRedactSensitiveText:
    def test_none_returns_none(self):
        assert redact_sensitive_text(None) is None

    def test_empty_returns_empty(self):
        assert redact_sensitive_text("") == ""

    def test_plain_text_unchanged(self):
        text = "Hello, this is a regular sentence."
        assert redact_sensitive_text(text) == text

    def test_openai_key_masked(self):
        text = "key is sk-abcdefghijklmnopqrstuvwxyz12345"
        result = redact_sensitive_text(text)
        assert "sk-abcdefghijklmnopqrstuvwxyz12345" not in result

    def test_github_pat_masked(self):
        text = "token=ghp_abcdefghijklmnopqrstuvwxyz"
        result = redact_sensitive_text(text)
        assert "ghp_abcdefghijklmnopqrstuvwxyz" not in result

    def test_env_assignment_masked(self):
        text = "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz"
        result = redact_sensitive_text(text)
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in result

    def test_json_apikey_masked(self):
        text = '{"apiKey": "some_secret_value_1234567890"}'
        result = redact_sensitive_text(text)
        assert "some_secret_value_1234567890" not in result

    def test_auth_bearer_masked(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc.def"
        result = redact_sensitive_text(text)
        assert "eyJhbGciOiJIUzI1NiJ9.abc.def" not in result

    def test_private_key_block_replaced(self):
        text = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBAD\n-----END PRIVATE KEY-----"
        result = redact_sensitive_text(text)
        assert "REDACTED PRIVATE KEY" in result

    def test_db_connstr_password_masked(self):
        text = "postgres://user:supersecret@localhost:5432/db"
        result = redact_sensitive_text(text)
        assert "supersecret" not in result

    def test_db_connstr_user_preserved(self):
        text = "postgres://user:supersecret@localhost:5432/db"
        result = redact_sensitive_text(text)
        assert "user" in result

    def test_returns_string(self):
        assert isinstance(redact_sensitive_text("any text"), str)

    def test_stripe_live_key_masked(self):
        text = "sk_live_abcdefghijklmnopqrstu"
        result = redact_sensitive_text(text)
        assert "sk_live_abcdefghijklmnopqrstu" not in result

    def test_huggingface_token_masked(self):
        text = "hf_abcdefghijklmnopqrstu"
        result = redact_sensitive_text(text)
        assert "hf_abcdefghijklmnopqrstu" not in result

    def test_json_token_field_masked(self):
        text = '{"token": "my_token_value_very_long_12345"}'
        result = redact_sensitive_text(text)
        assert "my_token_value_very_long_12345" not in result

    def test_json_password_field_masked(self):
        text = '{"password": "hunter2_supersecret_12345"}'
        result = redact_sensitive_text(text)
        assert "hunter2_supersecret_12345" not in result


# ---------------------------------------------------------------------------
# RedactingFormatter
# ---------------------------------------------------------------------------

class TestRedactingFormatter:
    def test_format_returns_string(self):
        import logging
        formatter = RedactingFormatter("%(message)s")
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello world", (), None)
        result = formatter.format(record)
        assert isinstance(result, str)

    def test_format_redacts_secret(self):
        import logging
        formatter = RedactingFormatter("%(message)s")
        record = logging.LogRecord(
            "test", logging.INFO, "", 0,
            "key=sk-abcdefghijklmnopqrstuvwxyz", (), None
        )
        result = formatter.format(record)
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in result

    def test_format_passes_plain_text(self):
        import logging
        formatter = RedactingFormatter("%(message)s")
        record = logging.LogRecord("test", logging.INFO, "", 0, "no secrets here", (), None)
        result = formatter.format(record)
        assert "no secrets here" in result


# ---------------------------------------------------------------------------
# TraceEvent
# ---------------------------------------------------------------------------

class TestTraceEvent:
    def test_fields_stored(self):
        ev = TraceEvent(event_type="tool_call", timestamp=1.0)
        assert ev.event_type == "tool_call"
        assert ev.timestamp == 1.0

    def test_duration_defaults_none(self):
        ev = TraceEvent(event_type="x", timestamp=0.0)
        assert ev.duration_ms is None

    def test_metadata_defaults_empty(self):
        ev = TraceEvent(event_type="x", timestamp=0.0)
        assert ev.metadata == {}

    def test_explicit_duration(self):
        ev = TraceEvent(event_type="x", timestamp=0.0, duration_ms=42.5)
        assert ev.duration_ms == pytest.approx(42.5)

    def test_explicit_metadata(self):
        ev = TraceEvent(event_type="x", timestamp=0.0, metadata={"key": "val"})
        assert ev.metadata["key"] == "val"


# ---------------------------------------------------------------------------
# SpanContext
# ---------------------------------------------------------------------------

class TestSpanContext:
    def test_fields_stored(self):
        sc = SpanContext(name="my_span", start_time=1.0, events=[])
        assert sc.name == "my_span"
        assert sc.start_time == 1.0

    def test_events_stored(self):
        ev = TraceEvent(event_type="x", timestamp=0.0)
        sc = SpanContext(name="s", start_time=0.0, events=[ev])
        assert len(sc.events) == 1


# ---------------------------------------------------------------------------
# AgentTracer
# ---------------------------------------------------------------------------

class TestAgentTracerEmpty:
    def test_empty_timeline(self):
        tracer = AgentTracer()
        assert tracer.get_timeline() == []

    def test_summary_all_zeros(self):
        tracer = AgentTracer()
        s = tracer.summary()
        assert s["total_duration_ms"] == 0.0
        assert s["tool_calls_count"] == 0
        assert s["llm_calls_count"] == 0
        assert s["total_tokens"] == 0
        assert s["error_count"] == 0


class TestAgentTracerLogCalls:
    def test_log_tool_call_adds_event(self):
        tracer = AgentTracer()
        tracer.log_tool_call("bash", {"cmd": "ls"}, "output", 10.0)
        assert len(tracer.get_timeline()) == 1

    def test_log_tool_call_event_type(self):
        tracer = AgentTracer()
        tracer.log_tool_call("bash", {}, None, 5.0)
        assert tracer.get_timeline()[0].event_type == "tool_call"

    def test_log_llm_call_adds_event(self):
        tracer = AgentTracer()
        tracer.log_llm_call("sonnet", 100, 50, 200.0)
        assert len(tracer.get_timeline()) == 1

    def test_log_llm_call_event_type(self):
        tracer = AgentTracer()
        tracer.log_llm_call("haiku", 100, 50, 50.0)
        assert tracer.get_timeline()[0].event_type == "llm_call"

    def test_log_error_adds_event(self):
        tracer = AgentTracer()
        tracer.log_error("something broke", {})
        assert len(tracer.get_timeline()) == 1

    def test_log_error_event_type(self):
        tracer = AgentTracer()
        tracer.log_error("error msg")
        assert tracer.get_timeline()[0].event_type == "error"

    def test_multiple_events(self):
        tracer = AgentTracer()
        tracer.log_tool_call("read", {}, "text", 5.0)
        tracer.log_llm_call("sonnet", 200, 100, 300.0)
        tracer.log_error("oops")
        assert len(tracer.get_timeline()) == 3


class TestAgentTracerSummary:
    def test_counts_tool_calls(self):
        tracer = AgentTracer()
        tracer.log_tool_call("bash", {}, None, 10.0)
        tracer.log_tool_call("read", {}, None, 5.0)
        assert tracer.summary()["tool_calls_count"] == 2

    def test_counts_llm_calls(self):
        tracer = AgentTracer()
        tracer.log_llm_call("sonnet", 100, 50, 200.0)
        assert tracer.summary()["llm_calls_count"] == 1

    def test_counts_errors(self):
        tracer = AgentTracer()
        tracer.log_error("err1")
        tracer.log_error("err2")
        assert tracer.summary()["error_count"] == 2

    def test_total_tokens_from_llm_calls(self):
        tracer = AgentTracer()
        tracer.log_llm_call("sonnet", 100, 50, 100.0)
        tracer.log_llm_call("haiku", 200, 80, 80.0)
        assert tracer.summary()["total_tokens"] == 430

    def test_total_duration_sums_all(self):
        tracer = AgentTracer()
        tracer.log_tool_call("bash", {}, None, 10.0)
        tracer.log_llm_call("sonnet", 100, 50, 200.0)
        assert tracer.summary()["total_duration_ms"] == pytest.approx(210.0)

    def test_start_span_returns_span_context(self):
        tracer = AgentTracer()
        span = tracer.start_span("my_span")
        assert isinstance(span, SpanContext)
        assert span.name == "my_span"


# ---------------------------------------------------------------------------
# IterationBudget
# ---------------------------------------------------------------------------

class TestIterationBudgetInitial:
    def test_used_starts_zero(self):
        b = IterationBudget(max_total=10)
        assert b.used == 0

    def test_remaining_is_max_total(self):
        b = IterationBudget(max_total=10)
        assert b.remaining == 10

    def test_not_exhausted(self):
        b = IterationBudget(max_total=10)
        assert b.exhausted is False

    def test_pressure_none_at_start(self):
        b = IterationBudget(max_total=100)
        assert b.pressure == "none"

    def test_no_pressure_message_at_start(self):
        b = IterationBudget(max_total=100)
        assert b.get_pressure_message() is None


class TestIterationBudgetConsume:
    def test_consume_returns_true_when_available(self):
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

    def test_consume_returns_false_when_exhausted(self):
        b = IterationBudget(max_total=2)
        b.consume()
        b.consume()
        assert b.consume() is False

    def test_exhausted_after_max_consumes(self):
        b = IterationBudget(max_total=3)
        for _ in range(3):
            b.consume()
        assert b.exhausted is True

    def test_remaining_never_negative(self):
        b = IterationBudget(max_total=2)
        for _ in range(5):
            b.consume()
        assert b.remaining == 0


class TestIterationBudgetRefund:
    def test_refund_decrements_used(self):
        b = IterationBudget(max_total=10)
        b.consume()
        b.consume()
        b.refund(1)
        assert b.used == 1

    def test_refund_never_below_zero(self):
        b = IterationBudget(max_total=10)
        b.refund(5)
        assert b.used == 0

    def test_refund_multiple(self):
        b = IterationBudget(max_total=10)
        for _ in range(5):
            b.consume()
        b.refund(3)
        assert b.used == 2


class TestIterationBudgetPressure:
    def test_caution_at_70_percent(self):
        b = IterationBudget(max_total=10)
        for _ in range(7):
            b.consume()
        assert b.pressure == "caution"

    def test_urgent_at_90_percent(self):
        # max_total=20: after 18 consumes, ratio=0.9, remaining=2 → urgent (not critical)
        b = IterationBudget(max_total=20)
        for _ in range(18):
            b.consume()
        assert b.pressure == "urgent"

    def test_critical_at_max_minus_1(self):
        # 1 remaining → critical (remaining <= 1)
        b = IterationBudget(max_total=5)
        for _ in range(4):
            b.consume()
        assert b.pressure == "critical"

    def test_get_pressure_message_caution(self):
        b = IterationBudget(max_total=10)
        for _ in range(7):
            b.consume()
        msg = b.get_pressure_message()
        assert msg is not None
        assert "NOTE" in msg or "caution" in msg.lower() or "wrap" in msg.lower()

    def test_get_pressure_message_urgent(self):
        # Use max_total=20 so that 18 consumes yields ratio=0.9, remaining=2 → urgent
        b = IterationBudget(max_total=20)
        for _ in range(18):
            b.consume()
        msg = b.get_pressure_message()
        assert msg is not None
        assert "URGENT" in msg

    def test_get_pressure_message_critical(self):
        b = IterationBudget(max_total=2)
        b.consume()
        b.consume()
        msg = b.get_pressure_message()
        assert msg is not None
        assert "CRITICAL" in msg


class TestIterationBudgetTimeout:
    def test_no_timeout_never_timed_out(self):
        b = IterationBudget(max_total=10, timeout_seconds=None)
        assert b.timed_out is False

    def test_elapsed_seconds_positive(self):
        b = IterationBudget(max_total=10)
        assert b.elapsed_seconds >= 0.0

    def test_timeout_blocks_consume(self):
        b = IterationBudget(max_total=10, timeout_seconds=0.001)
        time.sleep(0.01)
        assert b.consume() is False

    def test_timed_out_flag_after_timeout(self):
        b = IterationBudget(max_total=10, timeout_seconds=0.001)
        time.sleep(0.01)
        assert b.timed_out is True


class TestIterationBudgetThreadSafety:
    def test_concurrent_consumes_correct_count(self):
        b = IterationBudget(max_total=100)
        results = []
        def do_consume():
            for _ in range(10):
                results.append(b.consume())
        threads = [threading.Thread(target=do_consume) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        true_count = sum(1 for r in results if r is True)
        assert true_count == 100
        assert b.exhausted is True
