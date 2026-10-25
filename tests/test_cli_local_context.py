"""Tests for mullet/cli/main.get_local_loaded_context — LM Studio context probe.

Pure unit tests with mocked urllib — no real network.
"""

import json
from unittest.mock import MagicMock, patch



def _fake_urlopen(response_bytes):
    """Return a context-manager mock for urllib.request.urlopen."""
    resp = MagicMock()
    resp.read.return_value = response_bytes
    cm = MagicMock()
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    return cm


def test_returns_max_loaded_context():
    """When multiple loaded LLMs exist, return the largest context length."""
    from mullet.cli.main import get_local_loaded_context

    data = json.dumps({
        "data": [
            {"state": "loaded", "type": "llm", "loaded_context_length": 8192},
            {"state": "loaded", "type": "llm", "loaded_context_length": 131072},
            {"state": "not-loaded", "type": "llm", "loaded_context_length": 999999},
        ]
    }).encode()

    with patch("urllib.request.urlopen", return_value=_fake_urlopen(data)):
        result = get_local_loaded_context()

    assert result == 131072


def test_returns_zero_when_no_loaded_model():
    """No loaded LLM → 0."""
    from mullet.cli.main import get_local_loaded_context

    data = json.dumps({
        "data": [
            {"state": "not-loaded", "type": "llm", "loaded_context_length": 8192},
        ]
    }).encode()

    with patch("urllib.request.urlopen", return_value=_fake_urlopen(data)):
        result = get_local_loaded_context()

    assert result == 0


def test_returns_zero_when_empty_data():
    """Empty data list → 0."""
    from mullet.cli.main import get_local_loaded_context

    data = json.dumps({"data": []}).encode()
    with patch("urllib.request.urlopen", return_value=_fake_urlopen(data)):
        result = get_local_loaded_context()

    assert result == 0


def test_returns_zero_when_connection_fails():
    """Any exception → 0 (never raises, never crashes CLI)."""
    from mullet.cli.main import get_local_loaded_context

    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError()):
        result = get_local_loaded_context()

    assert result == 0


def test_returns_zero_on_malformed_response():
    """Garbage JSON → 0, not raise."""
    from mullet.cli.main import get_local_loaded_context

    with patch("urllib.request.urlopen", return_value=_fake_urlopen(b"not json at all")):
        result = get_local_loaded_context()

    assert result == 0


def test_ignores_non_llm_types():
    """Embedding models and other types should not contribute."""
    from mullet.cli.main import get_local_loaded_context

    data = json.dumps({
        "data": [
            {"state": "loaded", "type": "vlm", "loaded_context_length": 999999},
            {"state": "loaded", "type": "embedding", "loaded_context_length": 888888},
            {"state": "loaded", "type": "llm", "loaded_context_length": 4096},
        ]
    }).encode()

    with patch("urllib.request.urlopen", return_value=_fake_urlopen(data)):
        result = get_local_loaded_context()

    assert result == 4096


def test_custom_base_url_is_respected():
    """Passing a custom base_url builds the right URL."""
    from mullet.cli.main import get_local_loaded_context

    captured_url = {}

    def fake_urlopen(url, timeout=None):
        captured_url["url"] = url if isinstance(url, str) else url.full_url
        raise ConnectionRefusedError()  # don't actually hit network

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        get_local_loaded_context(base_url="http://10.0.0.5:9999")

    assert captured_url["url"] == "http://10.0.0.5:9999/api/v0/models"


# ---------------------------------------------------------------------------
# get_opencode_declared_context — opencode.json vs LM Studio loaded-context mismatch
# ---------------------------------------------------------------------------

def _write_opencode_config(tmp_path, models: dict):
    cfg_dir = tmp_path / ".config" / "opencode"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "opencode.json"
    cfg_path.write_text(json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "model": "moonshot/kimi-k2.6",
        "provider": {"lmstudio": {"models": models}},
    }))
    return cfg_path


def test_get_opencode_declared_context_returns_configured_value(tmp_path, monkeypatch):
    from mullet.cli.main import get_opencode_declared_context
    _write_opencode_config(tmp_path, {
        "qwen2.5-coder-14b-instruct": {"limit": {"context": 32768, "output": 8192}},
    })
    monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", str(tmp_path)))
    assert get_opencode_declared_context("qwen2.5-coder-14b-instruct") == 32768


def test_get_opencode_declared_context_unknown_model_returns_none(tmp_path, monkeypatch):
    from mullet.cli.main import get_opencode_declared_context
    _write_opencode_config(tmp_path, {
        "qwen2.5-coder-14b-instruct": {"limit": {"context": 32768}},
    })
    monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", str(tmp_path)))
    assert get_opencode_declared_context("some-other-model") is None


def test_get_opencode_declared_context_missing_file_returns_none(tmp_path, monkeypatch):
    from mullet.cli.main import get_opencode_declared_context
    monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", str(tmp_path)))
    assert get_opencode_declared_context("anything") is None


def test_get_opencode_declared_context_corrupt_file_returns_none(tmp_path, monkeypatch):
    from mullet.cli.main import get_opencode_declared_context
    cfg_dir = tmp_path / ".config" / "opencode"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "opencode.json").write_text("{not valid json")
    monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", str(tmp_path)))
    assert get_opencode_declared_context("anything") is None


def test_get_opencode_declared_context_no_lmstudio_provider_returns_none(tmp_path, monkeypatch):
    from mullet.cli.main import get_opencode_declared_context
    cfg_dir = tmp_path / ".config" / "opencode"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "opencode.json").write_text(json.dumps({"model": "moonshot/kimi-k2.6", "provider": {}}))
    monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", str(tmp_path)))
    assert get_opencode_declared_context("anything") is None


# ---------------------------------------------------------------------------
# _run_doctor — Local LLM check reflects the loaded-vs-declared mismatch
# ---------------------------------------------------------------------------

def _mock_local_provider(model="qwen2.5-coder-14b-instruct"):
    p = MagicMock()
    p.name = "LM Studio"
    p.model = model
    p.health_check.return_value = True
    return p


def test_doctor_warns_on_context_mismatch(capsys, fake_claude):
    from mullet.cli.main import _run_doctor
    with patch("mullet.cli.main.detect_local_provider", return_value=_mock_local_provider()), \
         patch("mullet.cli.main.get_local_loaded_context", return_value=4096), \
         patch("mullet.cli.main.get_opencode_declared_context", return_value=32768):
        _run_doctor()
    out = capsys.readouterr().out
    assert "⚠" in out
    assert "opencode.json declares" in out
    assert "context-overflow" in out


def test_doctor_ok_when_loaded_context_meets_declared(capsys, fake_claude):
    from mullet.cli.main import _run_doctor
    with patch("mullet.cli.main.detect_local_provider", return_value=_mock_local_provider()), \
         patch("mullet.cli.main.get_local_loaded_context", return_value=32768), \
         patch("mullet.cli.main.get_opencode_declared_context", return_value=32768):
        _run_doctor()
    out = capsys.readouterr().out
    assert "opencode.json declares" not in out


def test_doctor_falls_back_to_flat_floor_when_model_not_in_opencode_json(capsys, fake_claude):
    """A local model opencode.json doesn't know about at all still gets the
    old flat '< 4096 = very small' floor as a fallback signal."""
    from mullet.cli.main import _run_doctor
    with patch("mullet.cli.main.detect_local_provider", return_value=_mock_local_provider("some-other-model")), \
         patch("mullet.cli.main.get_local_loaded_context", return_value=2048), \
         patch("mullet.cli.main.get_opencode_declared_context", return_value=None):
        _run_doctor()
    out = capsys.readouterr().out
    assert "very small" in out
