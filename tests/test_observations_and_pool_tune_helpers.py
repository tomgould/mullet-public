"""Tests for mullet.core.observations and mullet.cli.pool_tune pure helpers."""

import json

from mullet.core.observations import (
    _compact,
    _make_input_preview,
    _make_output_preview,
    _fit_line,
    _parse_line,
    record,
    iter_session,
    iter_recent,
    search,
    Observation,
    _INPUT_LIMIT,
    _OUTPUT_LIMIT,
    _LINE_HARD_LIMIT,
)
from mullet.cli.pool_tune import _recommend_inter_batch_sleep


# ---------------------------------------------------------------------------
# _compact
# ---------------------------------------------------------------------------

class TestCompact:
    def test_single_spaces_unchanged(self):
        assert _compact("hello world") == "hello world"

    def test_collapses_multiple_spaces(self):
        assert _compact("hello   world") == "hello world"

    def test_collapses_tabs(self):
        assert _compact("hello\tworld") == "hello world"

    def test_collapses_newlines(self):
        assert _compact("hello\nworld") == "hello world"

    def test_strips_leading_trailing(self):
        assert _compact("  hello  ") == "hello"

    def test_empty_string(self):
        assert _compact("") == ""

    def test_only_whitespace(self):
        assert _compact("   \t\n  ") == ""

    def test_mixed_whitespace(self):
        assert _compact("a \t \n b") == "a b"


# ---------------------------------------------------------------------------
# _make_input_preview
# ---------------------------------------------------------------------------

class TestMakeInputPreview:
    def test_simple_dict_serialized(self):
        result = _make_input_preview({"key": "value"})
        assert "key" in result
        assert "value" in result

    def test_truncates_to_input_limit(self):
        big = {"data": "x" * 1000}
        result = _make_input_preview(big)
        assert len(result) <= _INPUT_LIMIT + 10  # compact strips some chars

    def test_whitespace_collapsed(self):
        d = {"a": "hello\n   world"}
        result = _make_input_preview(d)
        assert "  " not in result

    def test_non_serializable_falls_back_to_str(self):
        # A dict with a non-JSON-serializable value (we patch to verify)
        # Use an object that json.dumps raises on
        class Unserializable:
            pass
        d = {"x": Unserializable()}
        result = _make_input_preview(d)
        assert isinstance(result, str)

    def test_empty_dict(self):
        result = _make_input_preview({})
        assert result == "{}"


# ---------------------------------------------------------------------------
# _make_output_preview
# ---------------------------------------------------------------------------

class TestMakeOutputPreview:
    def test_simple_string_returned(self):
        result = _make_output_preview("hello")
        assert result == "hello"

    def test_truncates_to_output_limit(self):
        long_str = "x" * 5000
        result = _make_output_preview(long_str)
        assert len(result) <= _OUTPUT_LIMIT + 10

    def test_whitespace_collapsed(self):
        result = _make_output_preview("hello\n   world")
        assert "  " not in result

    def test_non_string_coerced(self):
        result = _make_output_preview(12345)
        assert "12345" in result

    def test_empty_string(self):
        assert _make_output_preview("") == ""


# ---------------------------------------------------------------------------
# _fit_line
# ---------------------------------------------------------------------------

class TestFitLine:
    def _make_obs_dict(self, output_preview="short"):
        return {
            "timestamp": "2025-01-01T00:00:00Z",
            "session_id": "s1",
            "tool_name": "Read",
            "tool_input_preview": "{}",
            "tool_output_preview": output_preview,
            "cwd": "/tmp",
        }

    def test_short_line_unchanged(self):
        d = self._make_obs_dict("short output")
        result = _fit_line(d)
        assert "short output" in result

    def test_long_line_trimmed(self):
        # Create a line that will be > _LINE_HARD_LIMIT
        long_output = "x" * 2000
        d = self._make_obs_dict(long_output)
        result = _fit_line(d)
        assert len(result) <= _LINE_HARD_LIMIT

    def test_result_is_valid_json(self):
        long_output = "y" * 2000
        d = self._make_obs_dict(long_output)
        result = _fit_line(d)
        parsed = json.loads(result)
        assert "tool_output_preview" in parsed

    def test_short_line_is_valid_json(self):
        d = self._make_obs_dict("hello")
        result = _fit_line(d)
        parsed = json.loads(result)
        assert parsed["tool_name"] == "Read"


# ---------------------------------------------------------------------------
# _parse_line
# ---------------------------------------------------------------------------

class TestParseLine:
    def _valid_line(self):
        return json.dumps({
            "timestamp": "2025-01-01T00:00:00Z",
            "session_id": "s1",
            "tool_name": "Bash",
            "tool_input_preview": "{}",
            "tool_output_preview": "output",
            "cwd": "/home/user",
        })

    def test_valid_json_returns_observation(self):
        result = _parse_line(self._valid_line())
        assert isinstance(result, Observation)

    def test_valid_json_fields_correct(self):
        result = _parse_line(self._valid_line())
        assert result.tool_name == "Bash"
        assert result.session_id == "s1"

    def test_invalid_json_returns_none(self):
        assert _parse_line("not json at all") is None

    def test_empty_string_returns_none(self):
        assert _parse_line("") is None

    def test_missing_field_returns_none(self):
        d = {"timestamp": "2025-01-01T00:00:00Z", "session_id": "s1"}
        assert _parse_line(json.dumps(d)) is None


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

class TestRecord:
    def test_returns_observation(self, tmp_path):
        obs = record("Read", {"file": "x.py"}, "file contents", "sess1", base=tmp_path)
        assert isinstance(obs, Observation)

    def test_observation_tool_name(self, tmp_path):
        obs = record("Write", {"content": "hi"}, "ok", "sess2", base=tmp_path)
        assert obs.tool_name == "Write"

    def test_creates_jsonl_file(self, tmp_path):
        record("Bash", {"command": "ls"}, "a b c", "sess3", base=tmp_path)
        jsonl = tmp_path / ".mullet" / "observations" / "sess3.jsonl"
        assert jsonl.exists()

    def test_file_contains_json_line(self, tmp_path):
        record("Edit", {"path": "f.py"}, "done", "sess4", base=tmp_path)
        jsonl = tmp_path / ".mullet" / "observations" / "sess4.jsonl"
        lines = [l for l in jsonl.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["tool_name"] == "Edit"

    def test_multiple_records_append(self, tmp_path):
        for i in range(3):
            record("Bash", {}, f"output{i}", "sess5", base=tmp_path)
        jsonl = tmp_path / ".mullet" / "observations" / "sess5.jsonl"
        lines = [l for l in jsonl.read_text().splitlines() if l.strip()]
        assert len(lines) == 3

    def test_session_id_stored(self, tmp_path):
        obs = record("Read", {}, "out", "my-session", base=tmp_path)
        assert obs.session_id == "my-session"

    def test_timestamp_is_iso(self, tmp_path):
        obs = record("Read", {}, "out", "sess6", base=tmp_path)
        assert "T" in obs.timestamp
        assert obs.timestamp.endswith("Z")


# ---------------------------------------------------------------------------
# iter_session
# ---------------------------------------------------------------------------

class TestIterSession:
    def test_empty_session_yields_nothing(self, tmp_path):
        results = list(iter_session("no-such-session", base=tmp_path))
        assert results == []

    def test_yields_recorded_observations(self, tmp_path):
        record("Read", {}, "out", "s1", base=tmp_path)
        record("Write", {}, "ok", "s1", base=tmp_path)
        results = list(iter_session("s1", base=tmp_path))
        assert len(results) == 2

    def test_different_sessions_isolated(self, tmp_path):
        record("Read", {}, "out", "sa", base=tmp_path)
        record("Write", {}, "out", "sb", base=tmp_path)
        sa_results = list(iter_session("sa", base=tmp_path))
        assert len(sa_results) == 1
        assert sa_results[0].tool_name == "Read"

    def test_yields_observation_objects(self, tmp_path):
        record("Bash", {}, "done", "s2", base=tmp_path)
        result = next(iter_session("s2", base=tmp_path))
        assert isinstance(result, Observation)

    def test_skips_malformed_lines(self, tmp_path):
        record("Read", {}, "ok", "s3", base=tmp_path)
        # Inject a bad line
        jsonl = tmp_path / ".mullet" / "observations" / "s3.jsonl"
        with open(jsonl, "a") as f:
            f.write("THIS IS NOT JSON\n")
        results = list(iter_session("s3", base=tmp_path))
        assert len(results) == 1  # only the valid one


# ---------------------------------------------------------------------------
# iter_recent
# ---------------------------------------------------------------------------

class TestIterRecent:
    def test_empty_dir_yields_nothing(self, tmp_path):
        results = list(iter_recent(base=tmp_path))
        assert results == []

    def test_yields_all_sessions(self, tmp_path):
        record("Read", {}, "out", "r1", base=tmp_path)
        record("Write", {}, "out", "r2", base=tmp_path)
        results = list(iter_recent(limit=100, base=tmp_path))
        assert len(results) == 2

    def test_limit_respected(self, tmp_path):
        for i in range(5):
            record("Bash", {}, f"out{i}", f"rsess{i}", base=tmp_path)
        results = list(iter_recent(limit=2, base=tmp_path))
        assert len(results) == 2


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_returns_empty_when_no_match(self, tmp_path):
        record("Read", {"file": "test.py"}, "file content here", "sr1", base=tmp_path)
        results = search("nonexistent_xyz_token", base=tmp_path)
        assert results == []

    def test_matches_input_preview(self, tmp_path):
        record("Read", {"file": "important_file.py"}, "output", "sr2", base=tmp_path)
        results = search("important_file", base=tmp_path)
        assert len(results) == 1

    def test_matches_output_preview(self, tmp_path):
        record("Bash", {"cmd": "ls"}, "special_output_token here", "sr3", base=tmp_path)
        results = search("special_output_token", base=tmp_path)
        assert len(results) == 1

    def test_case_insensitive(self, tmp_path):
        record("Read", {"f": "MyFile.py"}, "output", "sr4", base=tmp_path)
        results = search("myfile", base=tmp_path)
        assert len(results) == 1

    def test_limit_respected(self, tmp_path):
        for i in range(5):
            record("Read", {"target": "match_this"}, "out", f"sl{i}", base=tmp_path)
        results = search("match_this", limit=2, base=tmp_path)
        assert len(results) <= 2

    def test_returns_observation_objects(self, tmp_path):
        record("Read", {"f": "find_me.txt"}, "out", "sr5", base=tmp_path)
        results = search("find_me", base=tmp_path)
        assert all(isinstance(r, Observation) for r in results)


# ---------------------------------------------------------------------------
# _recommend_inter_batch_sleep
# ---------------------------------------------------------------------------

class TestRecommendInterBatchSleep:
    def test_none_returns_2_0(self):
        assert _recommend_inter_batch_sleep(None) == 2.0

    def test_above_8192_returns_0_0(self):
        assert _recommend_inter_batch_sleep(8193) == 0.0

    def test_exactly_8192_returns_0_5(self):
        # 8192 is NOT > 8192, so falls to next branch
        assert _recommend_inter_batch_sleep(8192) == 0.5

    def test_above_4096_returns_0_5(self):
        assert _recommend_inter_batch_sleep(5000) == 0.5

    def test_exactly_4096_returns_1_0(self):
        # 4096 is NOT > 4096
        assert _recommend_inter_batch_sleep(4096) == 1.0

    def test_below_4096_returns_1_0(self):
        assert _recommend_inter_batch_sleep(2048) == 1.0

    def test_zero_returns_1_0(self):
        assert _recommend_inter_batch_sleep(0) == 1.0

    def test_very_large_returns_0_0(self):
        assert _recommend_inter_batch_sleep(32768) == 0.0
