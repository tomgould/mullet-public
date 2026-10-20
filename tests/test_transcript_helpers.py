"""Tests for pure helper functions in mullet.hud.transcript."""

from datetime import datetime, timezone

from mullet.hud.transcript import (
    ToolCall,
    AgentCall,
    TranscriptSnapshot,
    _dt_to_iso,
    _iso_to_dt,
    _tool_to_dict,
    _dict_to_tool,
    _agent_to_dict,
    _dict_to_agent,
    _extract_target,
    _infer_provider_from_model,
    _normalise_token_count,
    _cache_path,
    _snapshot_to_dict,
    _dict_to_snapshot,
)

_NOW = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
_LATER = datetime(2024, 6, 15, 12, 0, 5, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------

class TestDtToIso:
    def test_none_returns_none(self):
        assert _dt_to_iso(None) is None

    def test_aware_datetime_returns_iso(self):
        result = _dt_to_iso(_NOW)
        assert result == "2024-06-15T12:00:00+00:00"

    def test_roundtrips_via_iso_to_dt(self):
        iso = _dt_to_iso(_NOW)
        recovered = _iso_to_dt(iso)
        assert recovered == _NOW


class TestIsoToDt:
    def test_none_returns_none(self):
        assert _iso_to_dt(None) is None

    def test_empty_string_returns_none(self):
        assert _iso_to_dt("") is None

    def test_valid_iso_returns_datetime(self):
        result = _iso_to_dt("2024-06-15T12:00:00+00:00")
        assert isinstance(result, datetime)
        assert result.year == 2024

    def test_invalid_string_returns_none(self):
        assert _iso_to_dt("not-a-date") is None

    def test_garbage_returns_none(self):
        assert _iso_to_dt("{}garbage") is None


# ---------------------------------------------------------------------------
# ToolCall serialization
# ---------------------------------------------------------------------------

class TestToolToDict:
    def _make_tool(self, **kw) -> ToolCall:
        defaults = dict(
            id="t1", name="Bash", status="completed",
            start_time=_NOW, end_time=_LATER,
            target="ls", provider="anthropic", model="claude-sonnet",
        )
        defaults.update(kw)
        return ToolCall(**defaults)

    def test_roundtrip(self):
        tool = self._make_tool()
        d = _tool_to_dict(tool)
        recovered = _dict_to_tool(d)
        assert recovered.id == tool.id
        assert recovered.name == tool.name
        assert recovered.status == tool.status
        assert recovered.provider == tool.provider
        assert recovered.model == tool.model

    def test_optional_fields_none(self):
        tool = self._make_tool(end_time=None, target=None, provider=None, model=None)
        d = _tool_to_dict(tool)
        assert d["end_time"] is None
        assert d["target"] is None

    def test_dict_has_expected_keys(self):
        tool = self._make_tool()
        d = _tool_to_dict(tool)
        assert set(d.keys()) == {"id", "name", "status", "start_time", "end_time",
                                  "target", "provider", "model"}


# ---------------------------------------------------------------------------
# AgentCall serialization
# ---------------------------------------------------------------------------

class TestAgentToDict:
    def _make_agent(self, **kw) -> AgentCall:
        defaults = dict(
            id="a1", agent_type="general-purpose", model="claude-sonnet",
            description="search for bugs", status="completed",
            start_time=_NOW, end_time=_LATER, provider="anthropic",
        )
        defaults.update(kw)
        return AgentCall(**defaults)

    def test_roundtrip(self):
        agent = self._make_agent()
        d = _agent_to_dict(agent)
        recovered = _dict_to_agent(d)
        assert recovered.id == agent.id
        assert recovered.agent_type == agent.agent_type
        assert recovered.model == agent.model
        assert recovered.description == agent.description
        assert recovered.provider == agent.provider

    def test_dict_keys(self):
        agent = self._make_agent()
        d = _agent_to_dict(agent)
        assert "id" in d
        assert "agent_type" in d
        assert "model" in d
        assert "status" in d


# ---------------------------------------------------------------------------
# _extract_target
# ---------------------------------------------------------------------------

class TestExtractTarget:
    def test_read_extracts_file_path(self):
        assert _extract_target("Read", {"file_path": "/foo/bar.py"}) == "/foo/bar.py"

    def test_write_extracts_file_path(self):
        assert _extract_target("Write", {"file_path": "/a/b.txt"}) == "/a/b.txt"

    def test_edit_extracts_file_path(self):
        assert _extract_target("Edit", {"file_path": "src/app.py"}) == "src/app.py"

    def test_glob_extracts_pattern(self):
        assert _extract_target("Glob", {"pattern": "*.py"}) == "*.py"

    def test_grep_extracts_pattern(self):
        assert _extract_target("Grep", {"pattern": "def foo"}) == "def foo"

    def test_bash_short_command_returned_verbatim(self):
        result = _extract_target("Bash", {"command": "ls -la"})
        assert result == "ls -la"

    def test_bash_long_command_truncated_at_30(self):
        cmd = "x" * 50
        result = _extract_target("Bash", {"command": cmd})
        assert result is not None
        assert result.endswith("...")
        assert len(result) == 33  # 30 chars + "..."

    def test_bash_exactly_30_chars_no_ellipsis(self):
        cmd = "a" * 30
        result = _extract_target("Bash", {"command": cmd})
        assert result == cmd
        assert "..." not in result

    def test_unknown_tool_returns_none(self):
        assert _extract_target("Agent", {"description": "do stuff"}) is None

    def test_no_input_returns_none(self):
        assert _extract_target("Read", None) is None

    def test_empty_input_returns_none(self):
        assert _extract_target("Read", {}) is None


# ---------------------------------------------------------------------------
# _infer_provider_from_model
# ---------------------------------------------------------------------------

class TestInferProviderFromModel:
    def test_none_returns_none(self):
        assert _infer_provider_from_model(None) is None

    def test_empty_string_returns_none(self):
        assert _infer_provider_from_model("") is None

    def test_claude_maps_to_anthropic(self):
        assert _infer_provider_from_model("claude-sonnet-4-6") == "anthropic"
        assert _infer_provider_from_model("claude-opus-4") == "anthropic"

    def test_kimi_maps_to_kimi(self):
        assert _infer_provider_from_model("kimi-k2-0711-preview") == "kimi"

    def test_moonshot_maps_to_kimi(self):
        assert _infer_provider_from_model("moonshot-v1-8k") == "kimi"

    def test_gemini_maps_to_gemini(self):
        assert _infer_provider_from_model("gemini-2.0-flash") == "gemini"

    def test_gpt_maps_to_openai(self):
        assert _infer_provider_from_model("gpt-4o") == "openai"

    def test_local_llm_names(self):
        for model in ("mistral-7b", "qwen2.5-7b", "phi-3", "gemma-2b",
                      "llama-3.1-8b", "deepseek-coder", "codestral-22b", "nemo"):
            assert _infer_provider_from_model(model) == "local", f"Expected local for {model}"

    def test_unknown_model_returns_none(self):
        assert _infer_provider_from_model("some-unknown-model-xyz") is None


# ---------------------------------------------------------------------------
# _normalise_token_count
# ---------------------------------------------------------------------------

class TestNormaliseTokenCount:
    def test_positive_int(self):
        assert _normalise_token_count(500) == 500

    def test_zero(self):
        assert _normalise_token_count(0) == 0

    def test_negative_becomes_zero(self):
        assert _normalise_token_count(-10) == 0

    def test_string_int(self):
        assert _normalise_token_count("123") == 123

    def test_none_returns_zero(self):
        assert _normalise_token_count(None) == 0

    def test_garbage_returns_zero(self):
        assert _normalise_token_count("not-a-number") == 0

    def test_float_truncated(self):
        assert _normalise_token_count(3.9) == 3


# ---------------------------------------------------------------------------
# _cache_path
# ---------------------------------------------------------------------------

class TestCachePath:
    def test_returns_string(self):
        result = _cache_path("/path/to/file.jsonl", "/tmp/cache")
        assert isinstance(result, str)

    def test_different_paths_produce_different_digests(self):
        a = _cache_path("/path/a.jsonl", "/cache")
        b = _cache_path("/path/b.jsonl", "/cache")
        assert a != b

    def test_same_path_same_digest(self):
        a = _cache_path("/path/file.jsonl", "/cache")
        b = _cache_path("/path/file.jsonl", "/cache")
        assert a == b

    def test_cache_dir_is_prefix(self):
        result = _cache_path("/some/file.jsonl", "/my/cache")
        assert result.startswith("/my/cache")

    def test_result_ends_with_json(self):
        result = _cache_path("/x.jsonl", "/cache")
        assert result.endswith(".json")


# ---------------------------------------------------------------------------
# TranscriptSnapshot round-trip
# ---------------------------------------------------------------------------

class TestSnapshotRoundtrip:
    def test_empty_snapshot_roundtrips(self):
        snap = TranscriptSnapshot()
        d = _snapshot_to_dict(snap)
        recovered = _dict_to_snapshot(d)
        assert recovered.session_tokens_in == 0
        assert recovered.running_tools == []
        assert recovered.todos == []

    def test_snapshot_with_tool_roundtrips(self):
        tool = ToolCall(
            id="t99", name="Read", status="completed",
            start_time=_NOW, end_time=_LATER,
            target="/foo.py", provider="anthropic", model="claude-opus",
        )
        snap = TranscriptSnapshot(completed_tools=[tool], session_tokens_in=1000)
        d = _snapshot_to_dict(snap)
        recovered = _dict_to_snapshot(d)
        assert len(recovered.completed_tools) == 1
        assert recovered.completed_tools[0].name == "Read"
        assert recovered.session_tokens_in == 1000
