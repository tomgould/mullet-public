"""OpenRouter withdrawing a model must not fail the task.

When OpenRouter answers that a model can't be served (the 404 "This model is
unavailable for free ... use this slug instead" reply, "No endpoints found",
an invalid model ID), OpenRouterProvider records the model as unavailable and
sends the same request to the next candidate. The HTTP layer is mocked
throughout: no request leaves the machine.
"""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion

from mullet.core.providers import model_availability
from mullet.core.providers.openrouter import OpenRouterProvider, next_openrouter_model

URL = "https://openrouter.ai/api/v1/chat/completions"
FREE_GONE = (
    "This model is unavailable for free. The paid version is available now - "
    "use this slug instead: qwen/qwen3-coder"
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(model_availability, "STORE_PATH", tmp_path / "unavailable.json")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    # Keep interaction logging off the disk.
    monkeypatch.setattr("mullet.core.logging.interaction_log.log_interaction", lambda **_k: None)


def _status_error(cls, status: int, message: str):
    request = httpx.Request("POST", URL)
    body = {"error": {"message": message, "code": status}}
    response = httpx.Response(status, request=request, content=json.dumps(body).encode())
    return cls(f"Error code: {status} - {body}", response=response, body=body)


def _completion(text: str, model: str) -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": text},
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    })


def _provider(model: str, side_effect) -> tuple[OpenRouterProvider, MagicMock]:
    provider = OpenRouterProvider(model=model)
    create = MagicMock(side_effect=side_effect)
    provider.client = MagicMock()
    provider.client.chat.completions.create = create
    return provider, create


def _sent_models(create: MagicMock) -> list[str]:
    return [c.kwargs["model"] for c in create.call_args_list]


# ---------------------------------------------------------------------------
# Detecting the error
# ---------------------------------------------------------------------------

class TestDetection:
    def test_free_model_withdrawn_404(self):
        exc = _status_error(openai.NotFoundError, 404, FREE_GONE)
        assert model_availability.is_model_unavailable_error(exc)
        assert model_availability.suggested_replacement(exc) == "qwen/qwen3-coder"

    def test_no_endpoints_found(self):
        exc = _status_error(openai.NotFoundError, 404, "No endpoints found for example/model:free.")
        assert model_availability.is_model_unavailable_error(exc)

    def test_invalid_model_id_400(self):
        exc = _status_error(openai.BadRequestError, 400, "example/nope is not a valid model ID")
        assert model_availability.is_model_unavailable_error(exc)

    def test_plain_message_without_status(self):
        assert model_availability.is_model_unavailable_error(RuntimeError(FREE_GONE))

    @pytest.mark.parametrize("cls,status,message", [
        (openai.RateLimitError, 429, "Rate limit exceeded: free-models-per-day"),
        (openai.InternalServerError, 503, "No endpoints found, provider overloaded"),
        (openai.AuthenticationError, 401, "No auth credentials found"),
        (openai.BadRequestError, 400, "max_tokens is too large"),
    ])
    def test_other_errors_are_not_matched(self, cls, status, message):
        assert not model_availability.is_model_unavailable_error(_status_error(cls, status, message))


# ---------------------------------------------------------------------------
# Falling back
# ---------------------------------------------------------------------------

class TestFallback:
    def test_withdrawn_free_model_falls_back_and_completes(self):
        provider, create = _provider("qwen/qwen3-coder:free", [
            _status_error(openai.NotFoundError, 404, FREE_GONE),
            _completion("done", "deepseek/deepseek-v4-flash"),
        ])
        result = provider.create(messages=[{"role": "user", "content": "hi"}])

        assert result.content == "done"
        assert _sent_models(create) == ["qwen/qwen3-coder:free", "deepseek/deepseek-v4-flash"]
        # The provider keeps the working model for the rest of the task.
        assert provider.model == "deepseek/deepseek-v4-flash"
        assert result.model == "deepseek/deepseek-v4-flash"

    def test_dead_model_is_remembered(self):
        provider, _ = _provider("qwen/qwen3-coder:free", [
            _status_error(openai.NotFoundError, 404, FREE_GONE),
            _completion("done", "deepseek/deepseek-v4-flash"),
        ])
        provider.create(messages=[{"role": "user", "content": "hi"}])

        assert model_availability.is_unavailable("openrouter", "qwen/qwen3-coder:free")
        stored = json.loads(model_availability._store_path().read_text())
        assert "unavailable for free" in stored["openrouter"]["qwen/qwen3-coder:free"]["message"]

    def test_falls_through_several_dead_models(self):
        provider, create = _provider("deepseek/deepseek-v4-flash", [
            _status_error(openai.NotFoundError, 404, "No endpoints found for deepseek/deepseek-v4-flash."),
            _status_error(openai.NotFoundError, 404, "No endpoints found for deepseek/deepseek-v4-pro."),
            _completion("done", "x"),
        ])
        result = provider.create(messages=[{"role": "user", "content": "hi"}])

        assert result.content == "done"
        sent = _sent_models(create)
        assert sent[:2] == ["deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro"]
        assert len(set(sent)) == 3  # never retries a model it has seen fail

    def test_routing_tier_guides_the_replacement(self):
        provider, create = _provider("deepseek/deepseek-v4-flash", [
            _status_error(openai.NotFoundError, 404, "No endpoints found for deepseek/deepseek-v4-flash."),
            _completion("done", "x"),
        ])
        provider.routing_tier = "CRITICAL"
        provider.create(messages=[{"role": "user", "content": "hi"}])
        replacement = _sent_models(create)[1]
        assert replacement == next_openrouter_model(
            "deepseek/deepseek-v4-flash", exclude={"deepseek/deepseek-v4-flash"}, tier="CRITICAL",
        )

    def test_gives_up_after_the_fallback_limit(self):
        errors = [
            _status_error(openai.NotFoundError, 404, f"No endpoints found for model {i}.")
            for i in range(OpenRouterProvider.MAX_MODEL_FALLBACKS + 1)
        ]
        provider, create = _provider("deepseek/deepseek-v4-flash", errors)
        with pytest.raises(openai.NotFoundError):
            provider.create(messages=[{"role": "user", "content": "hi"}])
        assert create.call_count == OpenRouterProvider.MAX_MODEL_FALLBACKS + 1

    def test_raises_when_no_candidate_is_left(self):
        provider, create = _provider("qwen/qwen3-coder:free", [
            _status_error(openai.NotFoundError, 404, FREE_GONE),
        ])
        with patch("mullet.core.providers.openrouter.next_openrouter_model", return_value=None):
            with pytest.raises(openai.NotFoundError):
                provider.create(messages=[{"role": "user", "content": "hi"}])
        assert create.call_count == 1

    def test_other_errors_are_raised_unchanged(self):
        provider, create = _provider("deepseek/deepseek-v4-flash", [
            _status_error(openai.AuthenticationError, 401, "No auth credentials found"),
        ])
        with pytest.raises(openai.AuthenticationError):
            provider.create(messages=[{"role": "user", "content": "hi"}])
        assert create.call_count == 1
        assert model_availability.unavailable_models("openrouter") == set()

    def test_stream_open_falls_back_too(self):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "hi"
        chunk.choices[0].delta.tool_calls = None
        chunk.choices[0].finish_reason = "stop"
        provider, create = _provider("qwen/qwen3-coder:free", [
            _status_error(openai.NotFoundError, 404, FREE_GONE),
            iter([chunk]),
        ])
        chunks = list(provider.create_stream(messages=[{"role": "user", "content": "hi"}]))
        assert [c.content for c in chunks] == ["hi"]
        assert _sent_models(create) == ["qwen/qwen3-coder:free", "deepseek/deepseek-v4-flash"]


# ---------------------------------------------------------------------------
# Choosing the next model
# ---------------------------------------------------------------------------

class TestNextModel:
    def test_free_model_prefers_other_free_candidates(self, monkeypatch):
        from mullet.core.providers import value_picker_constants as consts

        monkeypatch.setattr(consts, "OPENROUTER_FREE_TIER_CANDIDATES", [
            ("example/gone:free", 262_144), ("example/alive:free", 262_144),
        ])
        assert next_openrouter_model("example/gone:free") == "example/alive:free"

    def test_free_model_then_cheapest_paid(self):
        assert next_openrouter_model("qwen/qwen3-coder:free") == "deepseek/deepseek-v4-flash"

    def test_skips_recorded_unavailable_models(self):
        model_availability.mark_unavailable("openrouter", "deepseek/deepseek-v4-flash", "gone")
        assert next_openrouter_model("qwen/qwen3-coder:free") != "deepseek/deepseek-v4-flash"

    def test_none_when_everything_is_excluded(self):
        from mullet.core.providers.value_picker_constants import OPENROUTER_VALUE_PICK_CANDIDATES

        everything = {m for m, _ in OPENROUTER_VALUE_PICK_CANDIDATES}
        assert next_openrouter_model("deepseek/deepseek-v4-flash", exclude=everything) is None


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

class TestRecord:
    def test_entries_expire(self, monkeypatch):
        model_availability.mark_unavailable("openrouter", "example/old:free", "gone")
        assert model_availability.is_unavailable("openrouter", "example/old:free")
        later = time.time() + model_availability.TTL_SECONDS + 60
        monkeypatch.setattr(model_availability.time, "time", lambda: later)
        assert not model_availability.is_unavailable("openrouter", "example/old:free")

    def test_without_unavailable_filters_candidates(self):
        model_availability.mark_unavailable("openrouter", "b", "gone")
        assert model_availability.without_unavailable("openrouter", [("a", 1), ("b", 2)]) == [("a", 1)]
        assert model_availability.without_unavailable("kimi", [("b", 2)]) == [("b", 2)]

    def test_corrupt_file_is_ignored(self):
        path = model_availability._store_path()
        path.write_text("{not json")
        assert model_availability.unavailable_models("openrouter") == set()
        model_availability.mark_unavailable("openrouter", "a", "gone")
        assert model_availability.is_unavailable("openrouter", "a")

    def test_mcp_router_tells_the_provider_its_tier(self, monkeypatch):
        from mullet.server import mcp_handlers

        for name in ("KIMI_API_KEY", "MOONSHOT_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        registry = MagicMock()
        registry.get_best_model.return_value = "deepseek/deepseek-v4-flash"
        with patch("mullet.core.providers.model_registry.get_registry", return_value=registry):
            provider = mcp_handlers._provider_for_tier("LOW", allowed={"openrouter"})
        assert isinstance(provider, OpenRouterProvider)
        assert provider.routing_tier == "LOW"
