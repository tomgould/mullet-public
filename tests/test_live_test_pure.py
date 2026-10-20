"""Tests for pure helpers in mullet.cli.live_test."""
from __future__ import annotations

import json
import os
from pathlib import Path


from mullet.cli.live_test import (
    _DEFAULT_SECONDS,
    _DEFAULT_SWARM_WORKERS,
    _MCP_ENV_KEYS,
    _TRIVIAL_MAX_TOKENS,
    _TRIVIAL_PROMPT,
    _load_mcp_env_fallback,
    _parse_args,
)


class TestConstants:
    def test_default_seconds_positive(self) -> None:
        assert _DEFAULT_SECONDS > 0

    def test_default_swarm_workers_positive(self) -> None:
        assert _DEFAULT_SWARM_WORKERS > 0

    def test_trivial_max_tokens_small(self) -> None:
        assert _TRIVIAL_MAX_TOKENS <= 10

    def test_trivial_prompt_is_string(self) -> None:
        assert isinstance(_TRIVIAL_PROMPT, str)
        assert len(_TRIVIAL_PROMPT) > 0

    def test_mcp_env_keys_tuple(self) -> None:
        assert isinstance(_MCP_ENV_KEYS, tuple)
        assert "KIMI_API_KEY" in _MCP_ENV_KEYS


class TestParseArgs:
    def test_default_seconds(self) -> None:
        args = _parse_args([])
        assert args.seconds == _DEFAULT_SECONDS

    def test_explicit_seconds(self) -> None:
        args = _parse_args(["30"])
        assert args.seconds == 30

    def test_no_swarm_default_false(self) -> None:
        args = _parse_args([])
        assert args.no_swarm is False

    def test_no_swarm_flag(self) -> None:
        args = _parse_args(["--no-swarm"])
        assert args.no_swarm is True

    def test_swarm_size_default(self) -> None:
        args = _parse_args([])
        assert args.swarm_size == _DEFAULT_SWARM_WORKERS

    def test_swarm_size_explicit(self) -> None:
        args = _parse_args(["--swarm-size", "5"])
        assert args.swarm_size == 5

    def test_no_local_default_false(self) -> None:
        args = _parse_args([])
        assert args.no_local is False

    def test_no_local_flag(self) -> None:
        args = _parse_args(["--no-local"])
        assert args.no_local is True

    def test_combined_args(self) -> None:
        args = _parse_args(["60", "--no-swarm", "--swarm-size", "2"])
        assert args.seconds == 60
        assert args.no_swarm is True
        assert args.swarm_size == 2


class TestLoadMcpEnvFallback:
    def test_missing_file_is_noop(self, monkeypatch) -> None:
        monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent_home_xyz"))
        # Should not raise
        _load_mcp_env_fallback()

    def test_loads_key_from_mcp_json(self, tmp_path: Path, monkeypatch) -> None:
        mcp_json = tmp_path / ".mcp.json"
        mcp_json.write_text(json.dumps({
            "mcpServers": {
                "mullet": {
                    "env": {"KIMI_API_KEY": "test_kimi_key_abc123"}
                }
            }
        }))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        # Remove key from env if present
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        _load_mcp_env_fallback()
        assert os.environ.get("KIMI_API_KEY") == "test_kimi_key_abc123"

    def test_does_not_overwrite_existing_env(self, tmp_path: Path, monkeypatch) -> None:
        mcp_json = tmp_path / ".mcp.json"
        mcp_json.write_text(json.dumps({
            "mcpServers": {
                "mullet": {
                    "env": {"KIMI_API_KEY": "from_mcp"}
                }
            }
        }))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("KIMI_API_KEY", "existing_value")
        _load_mcp_env_fallback()
        assert os.environ.get("KIMI_API_KEY") == "existing_value"

    def test_invalid_json_is_noop(self, tmp_path: Path, monkeypatch) -> None:
        mcp_json = tmp_path / ".mcp.json"
        mcp_json.write_text("not valid json {{{")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        # Should not raise
        _load_mcp_env_fallback()
