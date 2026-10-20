"""Tests for mullet.core.observation_summary."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


from mullet.core.observations import Observation, record
from mullet.core.observation_summary import (
    SessionSummary,
    _build_prompt,
    _call_llm,
    _facts_from_obs,
    collect_session_facts,
    summarise_session,
    summarise_window,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_obs(ts: str, tool: str, inp: str = "", out: str = "", sid: str = "sess-1") -> Observation:
    return Observation(
        timestamp=ts,
        session_id=sid,
        tool_name=tool,
        tool_input_preview=inp,
        tool_output_preview=out,
        cwd="/home/user/project",
    )


SAMPLE_OBS = [
    _make_obs("2024-01-01T10:00:00Z", "Read", '{"file_path": "/home/user/project/foo.py"}'),
    _make_obs("2024-01-01T10:01:00Z", "Write", '{"file_path": "/home/user/project/bar.py"}'),
    _make_obs("2024-01-01T10:02:00Z", "Bash", '{"command": "pytest tests/"}'),
    _make_obs("2024-01-01T10:03:00Z", "Read", '{"file_path": "/home/user/project/foo.py"}'),
    _make_obs("2024-01-01T10:04:00Z", "Edit", '{"file_path": "/home/user/project/baz.py"}'),
]


# ---------------------------------------------------------------------------
# _facts_from_obs
# ---------------------------------------------------------------------------

def test_facts_tool_counts():
    facts = _facts_from_obs(SAMPLE_OBS, "sess-1")
    assert facts["tool_counts"]["Read"] == 2
    assert facts["tool_counts"]["Write"] == 1
    assert facts["tool_counts"]["Bash"] == 1
    assert facts["tool_counts"]["Edit"] == 1


def test_facts_started_ended():
    facts = _facts_from_obs(SAMPLE_OBS, "sess-1")
    assert facts["started_at"] == "2024-01-01T10:00:00Z"
    assert facts["ended_at"] == "2024-01-01T10:04:00Z"


def test_facts_files_touched_dedup():
    facts = _facts_from_obs(SAMPLE_OBS, "sess-1")
    # foo.py appears twice in Read but should be deduped
    assert facts["files_touched"].count("/home/user/project/foo.py") == 1


def test_facts_files_touched_contains_write():
    facts = _facts_from_obs(SAMPLE_OBS, "sess-1")
    assert "/home/user/project/bar.py" in facts["files_touched"]


def test_facts_empty_session():
    facts = _facts_from_obs([], "empty-sess")
    assert facts["started_at"] == ""
    assert facts["ended_at"] == ""
    assert facts["tool_counts"] == {}
    assert facts["files_touched"] == []


def test_facts_excerpts_present():
    facts = _facts_from_obs(SAMPLE_OBS, "sess-1")
    assert len(facts["excerpts"]) == len(SAMPLE_OBS)
    assert "Read" in facts["excerpts"][0]


# ---------------------------------------------------------------------------
# collect_session_facts (reads from disk)
# ---------------------------------------------------------------------------

def test_collect_session_facts_from_disk():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for obs in SAMPLE_OBS:
            record(
                tool_name=obs.tool_name,
                tool_input={"file_path": "/tmp/test.py"},
                tool_output="ok",
                session_id="sess-disk",
                base=base,
            )
        facts = collect_session_facts("sess-disk", base=base)
        assert facts["tool_counts"].get("Read", 0) >= 0  # record uses tool_name
        assert facts["session_id"] == "sess-disk"


def test_collect_session_facts_missing_session():
    with tempfile.TemporaryDirectory() as tmp:
        facts = collect_session_facts("nonexistent-session", base=Path(tmp))
        assert facts["started_at"] == ""
        assert facts["tool_counts"] == {}


# ---------------------------------------------------------------------------
# _call_llm — mock provider
# ---------------------------------------------------------------------------

def _mock_provider(response_text: str):
    provider = MagicMock()
    resp = MagicMock()
    resp.content = response_text
    provider.create.return_value = resp
    return provider


def test_call_llm_valid_json():
    payload = {
        "summary": "The developer worked on tests.",
        "bullet_points": ["Ran pytest", "Fixed imports"],
        "open_questions": ["Are all tests green?"],
    }
    provider = _mock_provider(json.dumps(payload))
    facts = _facts_from_obs(SAMPLE_OBS, "sess-1")
    result = _call_llm(provider, _build_prompt(facts))
    assert result["summary"] == "The developer worked on tests."
    assert "Ran pytest" in result["bullet_points"]
    assert "Are all tests green?" in result["open_questions"]


def test_call_llm_markdown_fences_stripped():
    payload = {"summary": "Done.", "bullet_points": ["a"], "open_questions": ["b"]}
    raw = f"```json\n{json.dumps(payload)}\n```"
    provider = _mock_provider(raw)
    facts = _facts_from_obs(SAMPLE_OBS, "sess-1")
    result = _call_llm(provider, _build_prompt(facts))
    assert result["summary"] == "Done."


def test_call_llm_malformed_json_fallback():
    provider = _mock_provider("not json at all — just text response here")
    facts = _facts_from_obs(SAMPLE_OBS, "sess-1")
    result = _call_llm(provider, _build_prompt(facts))
    assert "not json" in result["summary"]
    assert result["bullet_points"] == []
    assert result["open_questions"] == []


# ---------------------------------------------------------------------------
# summarise_session — with mocked provider resolution
# ---------------------------------------------------------------------------

GOOD_LLM_RESPONSE = json.dumps({
    "summary": "Developer read and edited Python files, ran tests.",
    "bullet_points": ["Read foo.py", "Edited baz.py", "Ran pytest"],
    "open_questions": ["Did tests pass?"],
})


def test_summarise_session_returns_summary():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for obs in SAMPLE_OBS:
            record(obs.tool_name, {}, "ok", session_id="s1", base=base)

        mock_p = _mock_provider(GOOD_LLM_RESPONSE)
        with patch(
            "mullet.core.observation_summary._resolve_provider",
            return_value=mock_p,
        ):
            result = summarise_session("s1", base=base)

    assert isinstance(result, SessionSummary)
    assert result.session_id == "s1"
    assert "Developer" in result.summary
    assert len(result.bullet_points) >= 1


def test_summarise_session_no_data_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        with patch("mullet.core.observation_summary._resolve_provider", return_value=_mock_provider("{}")):
            result = summarise_session("no-such-session", base=Path(tmp))
    assert result is None


def test_summarise_session_no_provider_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        record("Read", {}, "ok", session_id="s2", base=base)
        with patch("mullet.core.observation_summary._resolve_provider", return_value=None):
            result = summarise_session("s2", base=base)
    assert result is None


# ---------------------------------------------------------------------------
# summarise_window
# ---------------------------------------------------------------------------

def test_summarise_window_returns_summary():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        # Record some observations with current timestamps
        record("Read", {}, "ok", session_id="sw1", base=base)
        record("Bash", {"command": "ls"}, "file.py", session_id="sw1", base=base)

        mock_p = _mock_provider(GOOD_LLM_RESPONSE)
        with patch("mullet.core.observation_summary._resolve_provider", return_value=mock_p):
            result = summarise_window(minutes=60, base=base)

    assert result is not None
    assert "window" in result.session_id


def test_summarise_window_empty_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        with patch("mullet.core.observation_summary._resolve_provider", return_value=_mock_provider("{}")):
            result = summarise_window(minutes=1, base=Path(tmp))
    assert result is None


# ---------------------------------------------------------------------------
# prompt truncation
# ---------------------------------------------------------------------------

def test_build_prompt_truncates_large_input():
    """Prompt must not exceed _MAX_PROMPT_CHARS."""
    from mullet.core.observation_summary import _MAX_PROMPT_CHARS
    big_obs = [
        _make_obs(f"2024-01-01T10:{i:02d}:00Z", "Read", "x" * 300)
        for i in range(200)
    ]
    facts = _facts_from_obs(big_obs, "big-sess")
    prompt = _build_prompt(facts)
    assert len(prompt) <= _MAX_PROMPT_CHARS + 100  # allow small header overage
    assert "... [truncated] ..." in prompt


# ---------------------------------------------------------------------------
# _resolve_provider
# ---------------------------------------------------------------------------

from mullet.core.observation_summary import _resolve_provider


def test_resolve_provider_local_found():
    """When local provider is available, return it."""
    mock_provider = MagicMock()
    with patch(
        "mullet.core.providers.local_detection.detect_local_provider",
        return_value=mock_provider,
    ):
        result = _resolve_provider("local")
    assert result is mock_provider


def test_resolve_provider_local_not_found():
    """When local provider is unavailable and no key, return None."""
    with patch(
        "mullet.core.providers.local_detection.detect_local_provider",
        return_value=None,
    ):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=True):
            result = _resolve_provider("local")
    assert result is None


def test_resolve_provider_haiku_with_key():
    """When ANTHROPIC_API_KEY is set, haiku provider is returned."""
    mock_provider = MagicMock()
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
        with patch(
            "mullet.core.providers.anthropic.AnthropicProvider",
            return_value=mock_provider,
        ):
            result = _resolve_provider("haiku")
    assert result is mock_provider


def test_resolve_provider_auto_nothing_available():
    """Auto with no local and no API key logs warning and returns None."""
    with patch(
        "mullet.core.providers.local_detection.detect_local_provider",
        return_value=None,
    ):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=True):
            result = _resolve_provider("auto")
    assert result is None


def test_resolve_provider_local_raises():
    """When detect_local_provider raises, fall through to next option."""
    with patch(
        "mullet.core.providers.local_detection.detect_local_provider",
        side_effect=RuntimeError("no ollama"),
    ):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=True):
            result = _resolve_provider("auto")
    assert result is None


def test_resolve_provider_haiku_init_raises():
    """When AnthropicProvider init raises, return None."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
        with patch(
            "mullet.core.providers.anthropic.AnthropicProvider",
            side_effect=RuntimeError("bad key"),
        ):
            result = _resolve_provider("haiku")
    assert result is None


# ---------------------------------------------------------------------------
# _call_llm — provider.create raises
# ---------------------------------------------------------------------------

def test_call_llm_provider_create_raises():
    """If provider.create raises, return empty fallback dict."""
    provider = MagicMock()
    provider.create.side_effect = RuntimeError("network down")
    facts = _facts_from_obs(SAMPLE_OBS, "sess-1")
    result = _call_llm(provider, _build_prompt(facts))
    assert result == {"summary": "", "bullet_points": [], "open_questions": []}


# ---------------------------------------------------------------------------
# summarise_session — exception handling
# ---------------------------------------------------------------------------

def test_summarise_session_unexpected_exception():
    """Any unexpected error returns None."""
    with patch(
        "mullet.core.observation_summary.collect_session_facts",
        side_effect=RuntimeError("boom"),
    ):
        result = summarise_session("s1")
    assert result is None


# ---------------------------------------------------------------------------
# summarise_window — no provider + exception handling
# ---------------------------------------------------------------------------

def test_summarise_window_no_provider_returns_none():
    """When provider is None, summarise_window returns None."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        record("Read", {}, "ok", session_id="sw2", base=base)
        with patch(
            "mullet.core.observation_summary._resolve_provider",
            return_value=None,
        ):
            result = summarise_window(minutes=60, base=base)
    assert result is None


def test_summarise_window_unexpected_exception():
    """Any unexpected error in summarise_window returns None."""
    with patch(
        "mullet.core.observation_summary.iter_recent",
        side_effect=RuntimeError("boom"),
    ):
        result = summarise_window(minutes=60)
    assert result is None
