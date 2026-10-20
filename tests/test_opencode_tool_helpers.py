"""Tests for pure helper functions in mullet.tools.opencode_tool."""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest


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

    def test_reads_kimi_key_from_mcp_json(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        mcp_path = tmp_path / ".mcp.json"
        mcp_path.write_text(json.dumps({
            "mcpServers": {
                "mullet": {
                    "env": {"KIMI_API_KEY": "from-mcp-file"}
                }
            }
        }))
        # Patch both cwd and expanduser so it finds the tmp file
        monkeypatch.chdir(tmp_path)
        with patch.object(_oc.os.path, "expanduser", return_value=str(tmp_path / ".mcp.json.nonexistent")):
            result = _oc._resolve_kimi_key()
        assert result == "from-mcp-file"

    def test_reads_moonshot_key_from_mcp_json(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        mcp_path = tmp_path / ".mcp.json"
        mcp_path.write_text(json.dumps({
            "mcpServers": {
                "svc": {
                    "env": {"MOONSHOT_API_KEY": "ms-from-file"}
                }
            }
        }))
        monkeypatch.chdir(tmp_path)
        with patch.object(_oc.os.path, "expanduser", return_value=str(tmp_path / ".mcp.json.nonexistent")):
            result = _oc._resolve_kimi_key()
        assert result == "ms-from-file"

    def test_returns_none_when_no_key_anywhere(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        # Point cwd and home to paths with no .mcp.json
        monkeypatch.chdir(tmp_path)
        nonexistent = str(tmp_path / "no_such.json")
        with patch.object(_oc.os, "getcwd", return_value=str(tmp_path)):
            with patch.object(_oc.os.path, "expanduser", return_value=nonexistent):
                result = _oc._resolve_kimi_key()
        assert result is None

    def test_skips_malformed_mcp_json(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        mcp_path = tmp_path / ".mcp.json"
        mcp_path.write_text("{ broken json")
        monkeypatch.chdir(tmp_path)
        with patch.object(_oc.os.path, "expanduser", return_value=str(tmp_path / ".mcp.json.nonexistent")):
            result = _oc._resolve_kimi_key()
        assert result is None


# ---------------------------------------------------------------------------
# _read_dotenv
# ---------------------------------------------------------------------------

class TestReadDotenv:
    def test_missing_file_returns_empty_dict(self, tmp_path) -> None:
        result = _oc._read_dotenv(str(tmp_path / "no_such.env"))
        assert result == {}

    def test_empty_file_returns_empty_dict(self, tmp_path) -> None:
        p = tmp_path / "empty.env"
        p.write_text("")
        assert _oc._read_dotenv(str(p)) == {}

    def test_comments_only_returns_empty_dict(self, tmp_path) -> None:
        p = tmp_path / "comments.env"
        p.write_text("# this is a comment\n# another one\n\n")
        assert _oc._read_dotenv(str(p)) == {}

    def test_simple_key_value(self, tmp_path) -> None:
        p = tmp_path / ".env"
        p.write_text("FOO=bar\n")
        assert _oc._read_dotenv(str(p)) == {"FOO": "bar"}

    def test_export_prefix_stripped(self, tmp_path) -> None:
        p = tmp_path / ".env"
        p.write_text("export FOO=bar\n")
        assert _oc._read_dotenv(str(p)) == {"FOO": "bar"}

    def test_single_quoted_value(self, tmp_path) -> None:
        p = tmp_path / ".env"
        p.write_text("FOO='hello world'\n")
        assert _oc._read_dotenv(str(p)) == {"FOO": "hello world"}

    def test_double_quoted_value(self, tmp_path) -> None:
        p = tmp_path / ".env"
        p.write_text('FOO="hello world"\n')
        assert _oc._read_dotenv(str(p)) == {"FOO": "hello world"}

    def test_malformed_line_skipped(self, tmp_path) -> None:
        p = tmp_path / ".env"
        p.write_text("good_one=ok\nthis_line_has_no_equals\nstill_good=yes\n")
        result = _oc._read_dotenv(str(p))
        assert result == {"good_one": "ok", "still_good": "yes"}

    def test_key_whitespace_stripped(self, tmp_path) -> None:
        p = tmp_path / ".env"
        p.write_text("  FOO  =bar\n")
        assert _oc._read_dotenv(str(p)) == {"FOO": "bar"}

    def test_blank_lines_skipped(self, tmp_path) -> None:
        p = tmp_path / ".env"
        p.write_text("\n\nFOO=bar\n\n\nBAZ=qux\n")
        assert _oc._read_dotenv(str(p)) == {"FOO": "bar", "BAZ": "qux"}


# ---------------------------------------------------------------------------
# _resolve_forwarded_keys
# ---------------------------------------------------------------------------

class TestResolveForwardedKeys:
    """env-var > .env precedence, only known keys forwarded."""

    def _isolate(self, monkeypatch, tmp_path, dotenv_content: str = "") -> None:
        """Common setup: clear forwarded keys from env, point all .env candidates
        at tmp_path/.env, and chdir there (so $PWD/.env also resolves to it)."""
        for key in _oc._FORWARDED_API_KEYS:
            monkeypatch.delenv(key, raising=False)
        if dotenv_content:
            (tmp_path / ".env").write_text(dotenv_content)
        real_expanduser = os.path.expanduser
        tmp_env = str(tmp_path / ".env")

        def fake_expanduser(p):
            if p.startswith("~"):
                return tmp_env
            return real_expanduser(p)

        monkeypatch.setattr(_oc.os.path, "expanduser", fake_expanduser)
        monkeypatch.chdir(tmp_path)

    def test_env_var_wins_over_dotenv(self, tmp_path, monkeypatch) -> None:
        self._isolate(monkeypatch, tmp_path, "OPENROUTER_API_KEY=from-dotenv\n")
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
        result = _oc._resolve_forwarded_keys()
        assert result.get("OPENROUTER_API_KEY") == "from-env"

    def test_dotenv_used_when_env_absent(self, tmp_path, monkeypatch) -> None:
        self._isolate(monkeypatch, tmp_path, "OPENROUTER_API_KEY=from-dotenv\n")
        result = _oc._resolve_forwarded_keys()
        assert result.get("OPENROUTER_API_KEY") == "from-dotenv"

    def test_missing_everywhere_omitted(self, tmp_path, monkeypatch) -> None:
        self._isolate(monkeypatch, tmp_path, "")
        result = _oc._resolve_forwarded_keys()
        assert "OPENROUTER_API_KEY" not in result

    def test_unknown_keys_not_forwarded(self, tmp_path, monkeypatch) -> None:
        self._isolate(
            monkeypatch, tmp_path,
            "FOOBAR_API_KEY=should-be-ignored\nOPENROUTER_API_KEY=ok\n",
        )
        result = _oc._resolve_forwarded_keys()
        assert "FOOBAR_API_KEY" not in result
        assert result.get("OPENROUTER_API_KEY") == "ok"

    def test_all_eight_known_keys_resolvable(self, tmp_path, monkeypatch) -> None:
        body = "\n".join(f"{k}=val-{k}" for k in _oc._FORWARDED_API_KEYS) + "\n"
        self._isolate(monkeypatch, tmp_path, body)
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
