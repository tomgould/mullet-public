"""Tests for mullet.tools.opencode_tool and mullet.cli.override_cmd helpers."""


from mullet.tools.opencode_tool import (
    _DEFAULT_MODEL,
    _resolve_configured_model,
    _resolve_kimi_key,
)
from mullet.cli.override_cmd import _PROVIDER_MODEL_DEFAULTS


# ---------------------------------------------------------------------------
# opencode_tool constants
# ---------------------------------------------------------------------------

class TestDefaultModel:
    def test_is_string(self):
        assert isinstance(_DEFAULT_MODEL, str)

    def test_is_non_empty(self):
        assert len(_DEFAULT_MODEL) > 0


# ---------------------------------------------------------------------------
# _resolve_configured_model
# ---------------------------------------------------------------------------

class TestResolveConfiguredModel:
    def test_returns_string(self, tmp_path, monkeypatch):
        # Point home to tmp_path so no real opencode config is found
        monkeypatch.setenv("HOME", str(tmp_path))
        result = _resolve_configured_model()
        assert isinstance(result, str)

    def test_fallback_to_default_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        result = _resolve_configured_model()
        assert result == _DEFAULT_MODEL

    def test_reads_model_from_config(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / ".config" / "opencode"
        cfg_dir.mkdir(parents=True)
        cfg_file = cfg_dir / "opencode.json"
        cfg_file.write_text('{"model": "moonshot/kimi-k2.6"}')
        monkeypatch.setenv("HOME", str(tmp_path))
        result = _resolve_configured_model()
        assert result == "moonshot/kimi-k2.6"

    def test_missing_model_key_falls_back(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / ".config" / "opencode"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "opencode.json").write_text('{"other_key": "value"}')
        monkeypatch.setenv("HOME", str(tmp_path))
        result = _resolve_configured_model()
        assert result == _DEFAULT_MODEL

    def test_invalid_json_falls_back(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / ".config" / "opencode"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "opencode.json").write_text("not json")
        monkeypatch.setenv("HOME", str(tmp_path))
        result = _resolve_configured_model()
        assert result == _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# _resolve_kimi_key
# ---------------------------------------------------------------------------

class TestResolveKimiKey:
    def test_kimi_api_key_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        result = _resolve_kimi_key()
        assert result == "test-kimi-key"

    def test_moonshot_api_key_fallback(self, monkeypatch):
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.setenv("MOONSHOT_API_KEY", "test-moonshot-key")
        result = _resolve_kimi_key()
        assert result == "test-moonshot-key"

    def test_no_key_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        # Also ensure no .mcp.json in cwd or home that has a key
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        result = _resolve_kimi_key()
        assert result is None

    def test_kimi_key_takes_priority_over_moonshot(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "kimi-key")
        monkeypatch.setenv("MOONSHOT_API_KEY", "moonshot-key")
        result = _resolve_kimi_key()
        assert result == "kimi-key"


# ---------------------------------------------------------------------------
# override_cmd._PROVIDER_MODEL_DEFAULTS
# ---------------------------------------------------------------------------

class TestProviderModelDefaults:
    def test_is_dict(self):
        assert isinstance(_PROVIDER_MODEL_DEFAULTS, dict)

    def test_opus_present(self):
        assert "opus" in _PROVIDER_MODEL_DEFAULTS

    def test_sonnet_present(self):
        assert "sonnet" in _PROVIDER_MODEL_DEFAULTS

    def test_haiku_present(self):
        assert "haiku" in _PROVIDER_MODEL_DEFAULTS

    def test_all_values_are_strings(self):
        assert all(isinstance(v, str) for v in _PROVIDER_MODEL_DEFAULTS.values())

    def test_all_values_non_empty(self):
        assert all(len(v) > 0 for v in _PROVIDER_MODEL_DEFAULTS.values())

    def test_sonnet_value_contains_sonnet(self):
        assert "sonnet" in _PROVIDER_MODEL_DEFAULTS["sonnet"].lower()

    def test_haiku_value_contains_haiku(self):
        assert "haiku" in _PROVIDER_MODEL_DEFAULTS["haiku"].lower()
