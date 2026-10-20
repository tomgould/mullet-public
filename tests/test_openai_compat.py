"""Tests for OpenAI-compatible provider."""
import logging
import pytest
from unittest.mock import patch, MagicMock
from mullet.core.providers.openai_compat import OpenAICompatibleProvider


def test_init_local():
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    assert p.name == "local"
    assert p.model == "default"


def test_init_openrouter():
    p = OpenAICompatibleProvider(base_url="https://openrouter.ai/api/v1", api_key="test")
    assert p.name == "openrouter"


def test_init_custom():
    p = OpenAICompatibleProvider(base_url="https://custom.api.com/v1", api_key="test")
    assert p.name not in ("local", "openrouter")


def test_health_check_success():
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    mock_choice = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    with patch.object(p.client.chat.completions, "create", return_value=mock_resp):
        assert p.health_check() is True


def test_health_check_failure():
    p = OpenAICompatibleProvider(base_url="http://localhost:9999/v1")
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        assert p.health_check() is False


def test_local_timeout_is_300s():
    # Local GPU generation is much slower per-token than cloud APIs — a full
    # file-write tool call from a 14B model can legitimately exceed 60s.
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    assert p.client.timeout == 300.0


def test_cloud_timeout_is_60s():
    p = OpenAICompatibleProvider(base_url="https://openrouter.ai/api/v1", api_key="test")
    assert p.client.timeout == 60.0


def test_openrouter_headers():
    p = OpenAICompatibleProvider(
        base_url="https://openrouter.ai/api/v1",
        api_key="test",
        openrouter_config={"app_name": "Mullet"},
    )
    assert p.name == "openrouter"


def test_call_with_retry_context_exceeded_strips_and_retries():
    """Context size exceeded error strips tools/prompt and retries."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    call_count = [0]
    def fail_then_succeed(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("Context size has been exceeded.")
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        return resp

    with patch.object(p.client.chat.completions, "create", side_effect=fail_then_succeed):
        result = p._call_with_retry({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "dummy"}}],
        })

    assert call_count[0] == 2
    assert result is not None


def test_call_with_retry_non_retryable_raises_immediately():
    """Non-retryable errors (e.g. auth error) are raised without retry."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    call_count = [0]
    def auth_error(**kwargs):
        call_count[0] += 1
        raise ValueError("Invalid API key")

    with patch.object(p.client.chat.completions, "create", side_effect=auth_error):
        with pytest.raises(ValueError):
            p._call_with_retry({"messages": []})

    assert call_count[0] == 1  # No retry for non-retryable error


def test_call_with_retry_wall_clock_budget_exceeded(caplog):
    """Retry loop gives up when the overall wall-clock budget is exceeded."""
    from mullet.core.providers import openai_compat as oc
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    call_count = [0]
    def always_fail(**kwargs):
        call_count[0] += 1
        raise RuntimeError("503 Service Unavailable")

    with patch.object(p.client.chat.completions, "create", side_effect=always_fail):
        with caplog.at_level(logging.WARNING, logger="mullet.core.providers.openai_compat"):
            with patch("time.monotonic", side_effect=[0.0, 0.0, 0.0, 200.0]):
                with pytest.raises(RuntimeError, match="503"):
                    p._call_with_retry({"messages": [{"role": "user", "content": "hi"}]})

    assert call_count[0] == 1
    assert any("retry wall-clock budget" in r.message for r in caplog.records)


def test_is_garbage_content_all_question_marks():
    """All-? content (LM Studio Channel Error) is detected as garbage."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    assert p._is_garbage_content("?" * 100) is True


def test_is_garbage_content_short_not_garbage():
    """Short responses below threshold are not flagged as garbage."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    assert p._is_garbage_content("???") is False


def test_is_garbage_content_normal_text():
    """Normal text is not garbage."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    assert p._is_garbage_content("Hello, how can I help you today?") is False


def test_garbage_response_triggers_retry():
    """Garbled response triggers retry in _call_with_retry."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    call_count = [0]
    def garbled_then_good(**kwargs):
        call_count[0] += 1
        resp = MagicMock()
        if call_count[0] == 1:
            resp.choices = [MagicMock(
                message=MagicMock(content="?" * 200, tool_calls=None),
                finish_reason="stop",
            )]
        else:
            resp.choices = [MagicMock(
                message=MagicMock(content="valid response", tool_calls=None),
                finish_reason="stop",
            )]
        resp.model = "test-model"
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        return resp

    with patch.object(p.client.chat.completions, "create", side_effect=garbled_then_good):
        result = p._call_with_retry({"messages": [{"role": "user", "content": "hi"}]})

    assert call_count[0] == 2
    assert result.content == "valid response"


class TestEscalationTracking:
    def test_initial_failure_count_is_zero(self):
        from mullet.core.providers.openai_compat import get_consecutive_failures
        assert get_consecutive_failures("nonexistent_provider") == 0

    def test_reset_clears_count(self):
        from mullet.core.providers import openai_compat as oc
        oc._consecutive_failures["test_prov"] = 3
        oc.reset_consecutive_failures("test_prov")
        assert oc.get_consecutive_failures("test_prov") == 0

    def test_escalation_recommended_above_threshold(self):
        from mullet.core.providers import openai_compat as oc
        oc._consecutive_failures["flaky_prov"] = oc._ESCALATION_THRESHOLD
        assert oc.is_escalation_recommended("flaky_prov") is True
        oc._consecutive_failures.pop("flaky_prov")

    def test_escalation_not_recommended_below_threshold(self):
        from mullet.core.providers import openai_compat as oc
        oc._consecutive_failures["ok_prov"] = oc._ESCALATION_THRESHOLD - 1
        assert oc.is_escalation_recommended("ok_prov") is False
        oc._consecutive_failures.pop("ok_prov")

    def test_success_resets_consecutive_failures(self):
        """After a successful call, the failure counter is reset."""
        from mullet.core.providers import openai_compat as oc
        p = oc.OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
        p.model = "test-model"
        # Pre-seed failure count
        oc._consecutive_failures[p.name] = 3

        def good_response(**kwargs):
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None), finish_reason="stop")]
            resp.model = "test-model"
            resp.usage = MagicMock(prompt_tokens=5, completion_tokens=5, total_tokens=10)
            return resp

        with patch.object(p.client.chat.completions, "create", side_effect=good_response):
            p._call_with_retry({"messages": [{"role": "user", "content": "hi"}]})

        assert oc.get_consecutive_failures(p.name) == 0
