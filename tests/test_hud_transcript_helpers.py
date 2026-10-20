"""Tests for pure helper functions in mullet.hud.transcript."""
from __future__ import annotations

from mullet.hud.transcript import (
    _extract_target,
    _infer_provider_from_model,
    _normalise_token_count,
)


# ---------------------------------------------------------------------------
# _normalise_token_count
# ---------------------------------------------------------------------------

class TestNormaliseTokenCount:
    def test_positive_int(self) -> None:
        assert _normalise_token_count(100) == 100

    def test_zero(self) -> None:
        assert _normalise_token_count(0) == 0

    def test_negative_clamped_to_zero(self) -> None:
        assert _normalise_token_count(-5) == 0

    def test_string_integer(self) -> None:
        assert _normalise_token_count("42") == 42

    def test_none_returns_zero(self) -> None:
        assert _normalise_token_count(None) == 0

    def test_non_numeric_string_returns_zero(self) -> None:
        assert _normalise_token_count("abc") == 0

    def test_float_truncated(self) -> None:
        assert _normalise_token_count(9.9) == 9


# ---------------------------------------------------------------------------
# _extract_target
# ---------------------------------------------------------------------------

class TestExtractTarget:
    def test_read_returns_file_path(self) -> None:
        assert _extract_target("Read", {"file_path": "/src/auth.py"}) == "/src/auth.py"

    def test_write_returns_file_path(self) -> None:
        assert _extract_target("Write", {"file_path": "/tmp/out.py"}) == "/tmp/out.py"

    def test_edit_returns_file_path(self) -> None:
        assert _extract_target("Edit", {"file_path": "/src/main.py"}) == "/src/main.py"

    def test_glob_returns_pattern(self) -> None:
        assert _extract_target("Glob", {"pattern": "**/*.py"}) == "**/*.py"

    def test_grep_returns_pattern(self) -> None:
        assert _extract_target("Grep", {"pattern": "def auth"}) == "def auth"

    def test_bash_truncated_at_30_with_ellipsis(self) -> None:
        long_cmd = "pytest tests/ -v --tb=short 2>&1 | head -50"
        result = _extract_target("Bash", {"command": long_cmd})
        assert result is not None
        assert result.endswith("...")
        # The truncated part should be exactly 30 chars
        assert result[:-3] == long_cmd[:30]

    def test_bash_short_command_no_ellipsis(self) -> None:
        result = _extract_target("Bash", {"command": "git status"})
        assert result == "git status"

    def test_unknown_tool_returns_none(self) -> None:
        assert _extract_target("Agent", {"description": "explore"}) is None

    def test_none_input_returns_none(self) -> None:
        assert _extract_target("Read", None) is None

    def test_empty_input_returns_none(self) -> None:
        assert _extract_target("Read", {}) is None

    def test_bash_empty_command_returns_none(self) -> None:
        assert _extract_target("Bash", {"command": ""}) is None


# ---------------------------------------------------------------------------
# _infer_provider_from_model
# ---------------------------------------------------------------------------

class TestInferProviderFromModel:
    def test_claude_maps_to_anthropic(self) -> None:
        assert _infer_provider_from_model("claude-sonnet-4-6") == "anthropic"
        assert _infer_provider_from_model("claude-haiku-4-5") == "anthropic"
        assert _infer_provider_from_model("claude-opus-4-7") == "anthropic"

    def test_kimi_maps_to_kimi(self) -> None:
        assert _infer_provider_from_model("kimi-k2.6") == "kimi"

    def test_moonshot_maps_to_kimi(self) -> None:
        assert _infer_provider_from_model("moonshot-v1-8k") == "kimi"

    def test_gemini_maps_to_gemini(self) -> None:
        assert _infer_provider_from_model("gemini-2.5-flash") == "gemini"

    def test_gpt_maps_to_openai(self) -> None:
        assert _infer_provider_from_model("gpt-4o") == "openai"

    def test_o1_maps_to_openai(self) -> None:
        assert _infer_provider_from_model("o1-mini") == "openai"

    def test_mistral_maps_to_local(self) -> None:
        assert _infer_provider_from_model("mistral-nemo-instruct") == "local"

    def test_qwen_maps_to_local(self) -> None:
        assert _infer_provider_from_model("qwen2.5-14b") == "local"

    def test_phi_maps_to_local(self) -> None:
        assert _infer_provider_from_model("phi-3.5-mini") == "local"

    def test_llama_maps_to_local(self) -> None:
        assert _infer_provider_from_model("llama-3.2-3b") == "local"

    def test_none_returns_none(self) -> None:
        assert _infer_provider_from_model(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _infer_provider_from_model("") is None

    def test_unknown_model_returns_none(self) -> None:
        assert _infer_provider_from_model("unknown-xyz-model") is None


# ---------------------------------------------------------------------------
# _iso_to_dt
# ---------------------------------------------------------------------------

class TestIsoDt:
    def test_invalid_string_returns_none(self) -> None:
        """Lines 106-107: ValueError on bad ISO string → return None."""
        from mullet.hud.transcript import _iso_to_dt
        assert _iso_to_dt("not-a-date") is None

    def test_type_error_returns_none(self) -> None:
        """Lines 106-107: TypeError on non-string → return None."""
        from mullet.hud.transcript import _iso_to_dt
        assert _iso_to_dt(12345) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _dict_to_agent
# ---------------------------------------------------------------------------

class TestDictToAgent:
    def test_minimal_dict(self) -> None:
        """Line 150: _dict_to_agent returns AgentCall from minimal dict."""
        from mullet.hud.transcript import _dict_to_agent, AgentCall
        result = _dict_to_agent({"id": "agent-1", "status": "running"})
        assert isinstance(result, AgentCall)
        assert result.id == "agent-1"
        assert result.status == "running"
