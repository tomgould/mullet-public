"""Tests for pure helper functions in mullet.tools.opencode_tool."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import mullet.tools.opencode_tool as _oc


_DEFAULT_MODEL = _oc._DEFAULT_MODEL  # "opencode/big-pickle"


# ---------------------------------------------------------------------------
# _resolve_configured_model
# ---------------------------------------------------------------------------

class TestResolveConfiguredModel:
    def test_returns_default_when_no_config_file(self, tmp_path) -> None:
        cfg_path = tmp_path / "opencode.json"
        with patch.object(_oc.os.path, "expanduser", return_value=str(cfg_path)):
            result = _oc._resolve_configured_model()
        assert result == _DEFAULT_MODEL

    def test_returns_model_from_config(self, tmp_path) -> None:
        cfg_path = tmp_path / "opencode.json"
        cfg_path.write_text(json.dumps({"model": "moonshot/kimi-k2.6"}))
        with patch.object(_oc.os.path, "expanduser", return_value=str(cfg_path)):
            result = _oc._resolve_configured_model()
        assert result == "moonshot/kimi-k2.6"

    def test_returns_default_when_model_key_missing(self, tmp_path) -> None:
        cfg_path = tmp_path / "opencode.json"
        cfg_path.write_text(json.dumps({"theme": "dark"}))
        with patch.object(_oc.os.path, "expanduser", return_value=str(cfg_path)):
            result = _oc._resolve_configured_model()
        assert result == _DEFAULT_MODEL

    def test_returns_default_when_model_is_empty_string(self, tmp_path) -> None:
        cfg_path = tmp_path / "opencode.json"
        cfg_path.write_text(json.dumps({"model": ""}))
        with patch.object(_oc.os.path, "expanduser", return_value=str(cfg_path)):
            result = _oc._resolve_configured_model()
        assert result == _DEFAULT_MODEL

    def test_returns_default_when_model_is_non_string(self, tmp_path) -> None:
        cfg_path = tmp_path / "opencode.json"
        cfg_path.write_text(json.dumps({"model": 42}))
        with patch.object(_oc.os.path, "expanduser", return_value=str(cfg_path)):
            result = _oc._resolve_configured_model()
        assert result == _DEFAULT_MODEL

    def test_returns_default_on_invalid_json(self, tmp_path) -> None:
        cfg_path = tmp_path / "opencode.json"
        cfg_path.write_text("not json }{")
        with patch.object(_oc.os.path, "expanduser", return_value=str(cfg_path)):
            result = _oc._resolve_configured_model()
        assert result == _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# _resolve_kimi_key
# ---------------------------------------------------------------------------

class TestResolveKimiKey:
    def test_returns_kimi_api_key_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("KIMI_API_KEY", "kimi-secret-123")
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        result = _oc._resolve_kimi_key()
        assert result == "kimi-secret-123"

    def test_returns_moonshot_api_key_from_env(self, monkeypatch) -> None:
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.setenv("MOONSHOT_API_KEY", "moonshot-secret-456")
        result = _oc._resolve_kimi_key()
        assert result == "moonshot-secret-456"

    def test_kimi_key_takes_precedence_over_moonshot(self, monkeypatch) -> None:
        monkeypatch.setenv("KIMI_API_KEY", "kimi-first")
        monkeypatch.setenv("MOONSHOT_API_KEY", "moonshot-second")
        result = _oc._resolve_kimi_key()
        assert result == "kimi-first"

    def test_ignores_mcp_json_in_cwd_and_home(self, tmp_path, monkeypatch) -> None:
        """Keys come only from the process environment, never from .mcp.json files."""
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        servers = {"mcpServers": {"mullet": {"env": {"KIMI_API_KEY": "from-mcp-file"}}}}
        (tmp_path / ".mcp.json").write_text(json.dumps(servers))
        (Path.home() / ".mcp.json").write_text(json.dumps(servers))
        monkeypatch.chdir(tmp_path)
        assert _oc._resolve_kimi_key() is None

    def test_returns_none_when_no_key_anywhere(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        assert _oc._resolve_kimi_key() is None


# ---------------------------------------------------------------------------
# _resolve_forwarded_keys
# ---------------------------------------------------------------------------

class TestResolveForwardedKeys:
    """Only known keys, and only from the process environment."""

    def _clear(self, monkeypatch) -> None:
        for key in _oc._FORWARDED_API_KEYS:
            monkeypatch.delenv(key, raising=False)

    def test_env_var_forwarded(self, monkeypatch) -> None:
        self._clear(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
        assert _oc._resolve_forwarded_keys() == {"OPENROUTER_API_KEY": "from-env"}

    def test_dotenv_files_ignored(self, tmp_path, monkeypatch) -> None:
        """Neither the project's .env nor ~/.mullet/.env is read."""
        self._clear(monkeypatch)
        (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from-project-dotenv\n")
        (Path.home() / ".mullet").mkdir()
        (Path.home() / ".mullet" / ".env").write_text("OPENAI_API_KEY=from-home-dotenv\n")
        monkeypatch.chdir(tmp_path)
        assert _oc._resolve_forwarded_keys() == {}

    def test_empty_value_omitted(self, monkeypatch) -> None:
        self._clear(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        assert "OPENROUTER_API_KEY" not in _oc._resolve_forwarded_keys()

    def test_unknown_keys_not_forwarded(self, monkeypatch) -> None:
        self._clear(monkeypatch)
        monkeypatch.setenv("FOOBAR_API_KEY", "should-be-ignored")
        monkeypatch.setenv("OPENROUTER_API_KEY", "ok")
        result = _oc._resolve_forwarded_keys()
        assert "FOOBAR_API_KEY" not in result
        assert result.get("OPENROUTER_API_KEY") == "ok"

    def test_all_eight_known_keys_resolvable(self, monkeypatch) -> None:
        for k in _oc._FORWARDED_API_KEYS:
            monkeypatch.setenv(k, f"val-{k}")
        result = _oc._resolve_forwarded_keys()
        for k in _oc._FORWARDED_API_KEYS:
            assert result.get(k) == f"val-{k}", f"missing {k}"


# ---------------------------------------------------------------------------
# _provider_from_model
# ---------------------------------------------------------------------------

class TestProviderFromModel:
    def test_opencode_prefix_is_local(self) -> None:
        assert _oc._provider_from_model("opencode/big-pickle") == "local"

    def test_empty_string_is_local(self) -> None:
        assert _oc._provider_from_model("") == "local"

    def test_local_keyword_is_local(self) -> None:
        assert _oc._provider_from_model("local/qwen3-coder") == "local"

    def test_lmstudio_is_local(self) -> None:
        assert _oc._provider_from_model("lmstudio/llama3") == "local"

    def test_moonshot_prefix_is_kimi(self) -> None:
        assert _oc._provider_from_model("moonshot/kimi-k2.6") == "kimi"

    def test_kimi_in_name_is_kimi(self) -> None:
        assert _oc._provider_from_model("kimi-k2.7-code") == "kimi"

    def test_openrouter_prefix_is_openrouter(self) -> None:
        assert _oc._provider_from_model("openrouter/deepseek/deepseek-v3.2-exp") == "openrouter"

    def test_anthropic_prefix_is_anthropic(self) -> None:
        assert _oc._provider_from_model("anthropic/claude-opus-4.1") == "anthropic"

    def test_unknown_defaults_to_openrouter(self) -> None:
        assert _oc._provider_from_model("some-unknown/model-v1") == "openrouter"

    # _merge_id3_usage and _apply_usage_update were removed along with the
    # ACP/JSON-RPC dispatch protocol they supported — opencode_run() now
    # dispatches via `opencode run --format json` (a one-shot CLI invocation
    # that actually respects --model, unlike ACP's session/new "model"
    # param) and accumulates usage directly from step_finish events inline
    # in _read_run_events(). See tests/test_opencode_tool_main.py.
