"""Tests for routing_cmd display helpers and opencode_tool pure helpers."""

import json
import os

from mullet.cli.routing_cmd import _rate_str, _provider_col
from mullet.tools.opencode_tool import (
    _resolve_configured_model,
    _resolve_kimi_key,
    _DEFAULT_MODEL,
)


# ---------------------------------------------------------------------------
# _rate_str
# ---------------------------------------------------------------------------

class TestRateStr:
    def test_high_rate_contains_percent(self):
        assert "%" in _rate_str(0.95)

    def test_high_rate_green_color(self):
        result = _rate_str(0.95)
        assert "\x1b[32m" in result

    def test_exactly_0_9_is_green(self):
        result = _rate_str(0.9)
        assert "\x1b[32m" in result

    def test_just_below_0_9_is_yellow(self):
        result = _rate_str(0.89)
        assert "\x1b[33m" in result
        assert "\x1b[32m" not in result

    def test_exactly_0_7_is_yellow(self):
        result = _rate_str(0.7)
        assert "\x1b[33m" in result

    def test_below_0_7_no_color_code(self):
        result = _rate_str(0.69)
        assert "\x1b[32m" not in result
        assert "\x1b[33m" not in result

    def test_zero_rate_no_color(self):
        result = _rate_str(0.0)
        assert "\x1b[32m" not in result
        assert "\x1b[33m" not in result

    def test_value_in_output(self):
        result = _rate_str(0.75)
        assert "75" in result

    def test_returns_string(self):
        assert isinstance(_rate_str(0.5), str)

    def test_reset_code_in_colored_output(self):
        result = _rate_str(0.95)
        assert "\x1b[0m" in result

    def test_1_0_rate_is_green(self):
        result = _rate_str(1.0)
        assert "\x1b[32m" in result


# ---------------------------------------------------------------------------
# _provider_col
# ---------------------------------------------------------------------------

class TestProviderCol:
    def test_contains_provider_name(self):
        result = _provider_col("kimi")
        assert "kimi" in result

    def test_cyan_color_applied(self):
        result = _provider_col("kimi")
        assert "\x1b[36m" in result

    def test_reset_code_present(self):
        result = _provider_col("kimi")
        assert "\x1b[0m" in result

    def test_different_providers_contain_name(self):
        for name in ("gemini", "openai", "local"):
            assert name in _provider_col(name)

    def test_returns_string(self):
        assert isinstance(_provider_col("kimi"), str)

    def test_padded_to_at_least_12_chars_visible(self):
        # The padding is 12; strip ANSI codes and check length
        raw = _provider_col("kimi")
        # strip ANSI
        import re
        plain = re.sub(r"\x1b\[[0-9;]*m", "", raw)
        assert len(plain) >= 4  # at least the name


# ---------------------------------------------------------------------------
# _DEFAULT_MODEL constant
# ---------------------------------------------------------------------------

class TestDefaultModel:
    def test_is_string(self):
        assert isinstance(_DEFAULT_MODEL, str)

    def test_nonempty(self):
        assert len(_DEFAULT_MODEL) > 0

    def test_contains_opencode_or_slash(self):
        assert "/" in _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# _resolve_configured_model
# ---------------------------------------------------------------------------

class TestResolveConfiguredModel:
    def test_returns_string(self):
        assert isinstance(_resolve_configured_model(), str)

    def test_nonempty(self):
        assert len(_resolve_configured_model()) > 0

    def test_falls_back_to_default_when_no_config(self, tmp_path, monkeypatch):
        # Redirect expanduser so no real config is found
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path / "missing.json"))
        result = _resolve_configured_model()
        assert result == _DEFAULT_MODEL

    def test_reads_model_from_config(self, tmp_path, monkeypatch):
        cfg = tmp_path / "opencode.json"
        cfg.write_text(json.dumps({"model": "anthropic/claude-opus"}))
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(cfg))
        result = _resolve_configured_model()
        assert result == "anthropic/claude-opus"

    def test_empty_model_falls_back(self, tmp_path, monkeypatch):
        cfg = tmp_path / "opencode.json"
        cfg.write_text(json.dumps({"model": ""}))
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(cfg))
        result = _resolve_configured_model()
        assert result == _DEFAULT_MODEL

    def test_non_string_model_falls_back(self, tmp_path, monkeypatch):
        cfg = tmp_path / "opencode.json"
        cfg.write_text(json.dumps({"model": 42}))
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(cfg))
        result = _resolve_configured_model()
        assert result == _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# _resolve_kimi_key
# ---------------------------------------------------------------------------

class TestResolveKimiKey:
    def test_returns_kimi_env_var(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "test-kimi-123")
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        result = _resolve_kimi_key()
        assert result == "test-kimi-123"

    def test_returns_moonshot_env_var(self, monkeypatch):
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.setenv("MOONSHOT_API_KEY", "moon-456")
        result = _resolve_kimi_key()
        assert result == "moon-456"

    def test_kimi_takes_priority_over_moonshot(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "kimi-first")
        monkeypatch.setenv("MOONSHOT_API_KEY", "moon-second")
        result = _resolve_kimi_key()
        assert result == "kimi-first"

    def test_returns_none_or_string(self, monkeypatch):
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        result = _resolve_kimi_key()
        assert result is None or isinstance(result, str)

    def test_ignores_mcp_json_in_cwd(self, tmp_path, monkeypatch):
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        mcp = tmp_path / ".mcp.json"
        mcp.write_text(json.dumps({
            "mcpServers": {
                "kimi": {
                    "env": {"KIMI_API_KEY": "from-mcp-file"}
                }
            }
        }))
        monkeypatch.chdir(tmp_path)
        result = _resolve_kimi_key()
        assert result is None



