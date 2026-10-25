"""Local model preference order and `lms load` settings in LMStudioManager."""

import logging
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from mullet.core.providers import lmstudio_manager as lm
from mullet.core.providers.lmstudio_manager import LMStudioManager, preferred_models


@pytest.fixture(autouse=True)
def _no_config_file(tmp_path, monkeypatch):
    """Run from an empty directory, so a checkout's own mullet.yaml isn't read."""
    monkeypatch.chdir(tmp_path)


def _model(model_id, state="not-loaded", type_="llm", ctx=None):
    m = {"id": model_id, "type": type_, "state": state}
    if ctx is not None:
        m["loaded_context_length"] = ctx
    return m


# The set LM Studio lists today: the three preferred models, the FIM coder and embeddings.
def _current_set(loaded=()):
    ids = ["qwen2.5-coder-3b-instruct", "ornith-1.5-9b", "qwen3.6-35b-a3b", "ornith-1.5-35b-a3b"]
    models = [
        _model(i, "loaded", ctx=32768) if i in loaded else _model(i)
        for i in ids
    ]
    models += [
        _model("text-embedding-nomic-embed-text-v1.5@q4_k_m", type_="embeddings"),
        _model("text-embedding-nomic-embed-text-v1.5@q8_0", type_="embeddings"),
    ]
    return models


def _pick(models, **kwargs):
    mgr = LMStudioManager()
    with patch.object(mgr, "get_models", return_value=models):
        best = mgr.pick_best_model(**kwargs)
    return best["id"] if best else None


# ---------------------------------------------------------------------------
# preferred_models(): env var, then config file, then the built-in list
# ---------------------------------------------------------------------------

def test_default_preference_order():
    assert preferred_models() == ["ornith-1.5-35b-a3b", "qwen3.6-35b-a3b", "ornith-1.5-9b"]
    assert tuple(preferred_models()) == lm.DEFAULT_PREFERRED_MODELS


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("MULLET_LOCAL_PREFERRED_MODELS", " ornith-1.5-9b , qwen3.6-35b-a3b,")
    assert preferred_models() == ["ornith-1.5-9b", "qwen3.6-35b-a3b"]


def test_config_file_overrides_default(tmp_path):
    (tmp_path / "mullet.yaml").write_text(
        "providers:\n  local:\n    type: openai_compatible\n"
        "    preferred_models: [qwen3.6-35b-a3b, ornith-1.5-9b]\n"
    )
    assert preferred_models() == ["qwen3.6-35b-a3b", "ornith-1.5-9b"]


def test_config_file_accepts_a_comma_separated_string(tmp_path):
    (tmp_path / "mullet.yaml").write_text(
        "providers:\n  local:\n    preferred_models: 'ornith-1.5-9b, ornith-1.5-35b-a3b'\n"
    )
    assert preferred_models() == ["ornith-1.5-9b", "ornith-1.5-35b-a3b"]


def test_env_var_beats_config_file(tmp_path, monkeypatch):
    (tmp_path / "mullet.yaml").write_text(
        "providers:\n  local:\n    preferred_models: [qwen3.6-35b-a3b]\n"
    )
    monkeypatch.setenv("MULLET_LOCAL_PREFERRED_MODELS", "ornith-1.5-9b")
    assert preferred_models() == ["ornith-1.5-9b"]


@pytest.mark.parametrize("text", [
    "providers: [not, a, mapping]\n",
    "providers:\n  local: just-a-string\n",
    "providers:\n  local:\n    preferred_models: 42\n",
    "providers: {local: [unclosed\n",
])
def test_unusable_config_falls_back_to_default(tmp_path, text):
    (tmp_path / "mullet.yaml").write_text(text)
    assert preferred_models() == list(lm.DEFAULT_PREFERRED_MODELS)


# ---------------------------------------------------------------------------
# pick_best_model(): loaded first, then preference order, then smallest context
# ---------------------------------------------------------------------------

def test_nothing_loaded_picks_first_preferred():
    assert _pick(_current_set()) == "ornith-1.5-35b-a3b"


@pytest.mark.parametrize("missing,expected", [
    ({"ornith-1.5-35b-a3b"}, "qwen3.6-35b-a3b"),
    ({"ornith-1.5-35b-a3b", "qwen3.6-35b-a3b"}, "ornith-1.5-9b"),
])
def test_nothing_loaded_walks_down_the_preference_list(missing, expected):
    models = [m for m in _current_set() if m["id"] not in missing]
    assert _pick(models) == expected


def test_no_preferred_model_falls_back_to_smallest_fitting_context():
    # qwen2.5-coder-3b resolves to 128K in model_info; an unknown model to 32K.
    models = [_model("qwen2.5-coder-3b-instruct"), _model("some-unknown-model")]
    assert _pick(models) == "some-unknown-model"


def test_loaded_model_is_kept_over_an_unloaded_preferred_one():
    assert _pick(_current_set(loaded={"qwen2.5-coder-3b-instruct"})) == "qwen2.5-coder-3b-instruct"


def test_several_loaded_picks_the_highest_preference():
    loaded = {"qwen2.5-coder-3b-instruct", "ornith-1.5-9b", "qwen3.6-35b-a3b"}
    assert _pick(_current_set(loaded=loaded)) == "qwen3.6-35b-a3b"


def test_env_override_changes_the_pick(monkeypatch):
    monkeypatch.setenv("MULLET_LOCAL_PREFERRED_MODELS", "ornith-1.5-9b,ornith-1.5-35b-a3b")
    assert _pick(_current_set()) == "ornith-1.5-9b"


def test_preference_ignores_quant_suffix_and_case():
    models = [_model("qwen3.6-35b-a3b"), _model("Ornith-1.5-35B-A3B@q4_k_m")]
    assert _pick(models) == "Ornith-1.5-35B-A3B@q4_k_m"


def test_embeddings_are_never_picked_even_when_preferred(monkeypatch):
    monkeypatch.setenv(
        "MULLET_LOCAL_PREFERRED_MODELS",
        "text-embedding-nomic-embed-text-v1.5@q4_k_m,ornith-1.5-9b",
    )
    models = _current_set() + [_model("nomic-embedding-llm-typed")]
    assert _pick(models) == "ornith-1.5-9b"


def test_only_embeddings_returns_none():
    models = [m for m in _current_set() if m["type"] == "embeddings"]
    assert _pick(models) is None


# ---------------------------------------------------------------------------
# ensure_loaded(): `lms load` arguments, TTL and timeout
# ---------------------------------------------------------------------------

def test_load_settings():
    assert lm.LOAD_CONTEXT_LENGTH == 32768
    assert lm.LOAD_TTL_S == 600
    assert lm.LOAD_TIMEOUT_S == 120


def test_ensure_loaded_passes_context_length_ttl_and_timeout():
    mgr = LMStudioManager()
    done = MagicMock(returncode=0, stdout="loaded", stderr="")
    with patch.object(mgr, "get_loaded_models", side_effect=[[], [{"id": "ornith-1.5-35b-a3b"}]]), \
         patch("mullet.core.providers.lmstudio_manager.subprocess.run", return_value=done) as run, \
         patch("time.sleep"):
        assert mgr.ensure_loaded("ornith-1.5-35b-a3b") is True
    run.assert_called_once()
    args, kwargs = run.call_args
    assert args[0] == [
        mgr.LMS_CLI, "load", "--ttl", "600", "--context-length", "32768",
        "--yes", "ornith-1.5-35b-a3b",
    ]
    assert kwargs["timeout"] == 120


def test_get_or_load_provider_loads_with_the_default_ttl():
    mgr = LMStudioManager()
    with patch.object(mgr, "get_loaded_models", return_value=[]), \
         patch.object(mgr, "pick_best_model", return_value={"id": "ornith-1.5-35b-a3b"}), \
         patch.object(mgr, "ensure_loaded", return_value=True) as ensure:
        assert mgr.get_or_load_provider() == ("ornith-1.5-35b-a3b", False)
    ensure.assert_called_once_with("ornith-1.5-35b-a3b", 600)


def test_slow_lms_load_still_counts_when_the_model_is_up():
    """`lms load` blocks; if it returns after the timeout with the model loaded, that's a success."""
    mgr = LMStudioManager()
    clock = iter([0, 130])
    with patch.object(mgr, "get_loaded_models", side_effect=[[], [{"id": "slow"}]]), \
         patch.object(mgr, "_run_lms_command", return_value=(True, "loaded")), \
         patch("time.time", side_effect=lambda: next(clock, 999)), \
         patch("time.sleep"):
        assert mgr.ensure_loaded("slow") is True


def test_lms_load_killed_after_the_timeout(caplog):
    caplog.set_level(logging.ERROR, logger="mullet.core.providers.lmstudio_manager")
    mgr = LMStudioManager()
    expired = subprocess.TimeoutExpired(cmd="lms load", timeout=120)
    with patch.object(mgr, "get_loaded_models", return_value=[]), \
         patch("mullet.core.providers.lmstudio_manager.subprocess.run", side_effect=expired):
        assert mgr.ensure_loaded("huge-model") is False
    assert "timed out after 120s" in caplog.text


def test_detect_local_provider_uses_the_default_ttl():
    from mullet.core.providers import local_detection as ld

    ld.clear_local_provider_cache()
    try:
        with patch("mullet.core.providers.lmstudio_manager.LMStudioManager") as MockMgr, \
             patch("mullet.core.providers.local_detection.OpenAICompatibleProvider"):
            MockMgr.return_value.get_or_load_provider.return_value = ("ornith-1.5-35b-a3b", True)
            ld.detect_local_provider()
        MockMgr.return_value.get_or_load_provider.assert_called_once_with(ttl=600)
    finally:
        ld.clear_local_provider_cache()
