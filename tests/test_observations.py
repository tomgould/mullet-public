"""Tests for mullet.core.observations — record, query, truncation, concurrency."""
import json
import threading
from unittest.mock import patch


from mullet.core.observations import (
    Observation,
    _LINE_HARD_LIMIT,
    _INPUT_LIMIT,
    _OUTPUT_LIMIT,
    iter_recent,
    iter_session,
    observations_dir,
    record,
    search,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(tmp_path, tool="Read", inp=None, out="ok", sid="ses1"):
    return record(
        tool_name=tool,
        tool_input=inp or {"file": "/tmp/x.py"},
        tool_output=out,
        session_id=sid,
        base=tmp_path,
    )


# ---------------------------------------------------------------------------
# Basic record / directory creation
# ---------------------------------------------------------------------------

def test_observations_dir_created(tmp_path):
    d = observations_dir(base=tmp_path)
    assert d.is_dir()
    assert d == tmp_path / ".mullet" / "observations"


def test_record_returns_observation(tmp_path):
    obs = _rec(tmp_path)
    assert isinstance(obs, Observation)
    assert obs.tool_name == "Read"
    assert obs.session_id == "ses1"


def test_record_writes_jsonl(tmp_path):
    _rec(tmp_path, sid="abc")
    path = tmp_path / ".mullet" / "observations" / "abc.jsonl"
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    d = json.loads(lines[0])
    assert d["tool_name"] == "Read"
    assert d["session_id"] == "abc"


def test_record_appends_multiple(tmp_path):
    for i in range(5):
        _rec(tmp_path, out=f"out{i}", sid="multi")
    path = tmp_path / ".mullet" / "observations" / "multi.jsonl"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 5


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

def test_input_preview_truncated(tmp_path):
    long_val = "x" * 2000
    obs = record("Read", {"k": long_val}, "out", "t1", base=tmp_path)
    assert len(obs.tool_input_preview) <= _INPUT_LIMIT + 20  # +slack for JSON wrapper


def test_output_preview_truncated(tmp_path):
    long_out = "y" * 5000
    obs = record("Bash", {}, long_out, "t2", base=tmp_path)
    assert len(obs.tool_output_preview) <= _OUTPUT_LIMIT + 5


def test_line_hard_limit(tmp_path):
    huge_out = "z" * 10_000
    _rec(tmp_path, out=huge_out, sid="cap")
    path = tmp_path / ".mullet" / "observations" / "cap.jsonl"
    line = path.read_text().strip()
    assert len(line) <= _LINE_HARD_LIMIT


# ---------------------------------------------------------------------------
# iter_session
# ---------------------------------------------------------------------------

def test_iter_session_yields_in_order(tmp_path):
    for i in range(3):
        _rec(tmp_path, out=f"val{i}", sid="ord")
    results = list(iter_session("ord", base=tmp_path))
    assert len(results) == 3
    assert results[0].tool_output_preview == "val0"
    assert results[2].tool_output_preview == "val2"


def test_iter_session_missing_file(tmp_path):
    results = list(iter_session("nonexistent", base=tmp_path))
    assert results == []


# ---------------------------------------------------------------------------
# iter_recent
# ---------------------------------------------------------------------------

def test_iter_recent_across_sessions(tmp_path):
    _rec(tmp_path, sid="s1", out="alpha")
    _rec(tmp_path, sid="s2", out="beta")
    results = list(iter_recent(limit=50, base=tmp_path))
    outputs = {r.tool_output_preview for r in results}
    assert "alpha" in outputs
    assert "beta" in outputs


def test_iter_recent_respects_limit(tmp_path):
    for i in range(10):
        _rec(tmp_path, sid="big", out=f"x{i}")
    results = list(iter_recent(limit=3, base=tmp_path))
    assert len(results) <= 3


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_finds_match(tmp_path):
    _rec(tmp_path, inp={"file": "/home/user/secret.py"}, sid="srch")
    _rec(tmp_path, inp={"cmd": "ls"}, out="nothing special", sid="srch")
    results = search("secret", base=tmp_path)
    assert len(results) == 1
    assert "secret" in results[0].tool_input_preview


def test_search_case_insensitive(tmp_path):
    _rec(tmp_path, out="Hello World", sid="ci")
    results = search("hello world", base=tmp_path)
    assert len(results) == 1


def test_search_no_match_returns_empty(tmp_path):
    _rec(tmp_path, out="nothing here", sid="nm")
    assert search("xyzzy_not_found", base=tmp_path) == []


def test_search_limit(tmp_path):
    for i in range(10):
        _rec(tmp_path, out=f"needle{i}", sid="lim")
    results = search("needle", limit=5, base=tmp_path)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Unicode / bytes safety
# ---------------------------------------------------------------------------

def test_unicode_input_and_output(tmp_path):
    obs = record("Read", {"path": "/tmp/\u4e2d\u6587.py"}, "\u2603 snowman", "uni", base=tmp_path)
    assert "\u4e2d\u6587" in obs.tool_input_preview or len(obs.tool_input_preview) > 0
    path = tmp_path / ".mullet" / "observations" / "uni.jsonl"
    d = json.loads(path.read_text().strip())
    assert d["session_id"] == "uni"


# ---------------------------------------------------------------------------
# Concurrency — atomic appends
# ---------------------------------------------------------------------------

def test_concurrent_writes_no_lost_lines(tmp_path):
    n = 30
    errors = []

    def writer(i):
        try:
            record("Bash", {"i": i}, f"output{i}", "conc", base=tmp_path)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread errors: {errors}"
    path = tmp_path / ".mullet" / "observations" / "conc.jsonl"
    lines = [l for l in path.read_text().strip().splitlines() if l.strip()]
    assert len(lines) == n, f"Expected {n} lines, got {len(lines)}"
    # Verify every line is valid JSON
    for line in lines:
        json.loads(line)


def test_record_swallows_io_error(tmp_path):
    """Lines 108-109: exception inside record's try block → swallowed, returns obs."""
    with patch("mullet.core.observations.observations_dir", side_effect=OSError("disk full")):
        obs = record("Read", {"f": "x.py"}, "ok", "ses99", base=tmp_path)
    assert obs.tool_name == "Read"  # still returns the Observation


def test_iter_recent_skips_unreadable_file(tmp_path):
    """Lines 153-154: exception when opening a session file → continue to next."""
    obs_dir = tmp_path / ".mullet" / "observations"
    obs_dir.mkdir(parents=True)
    good = obs_dir / "good.jsonl"
    bad = obs_dir / "bad.jsonl"
    good.write_text('{"timestamp":"2024-01-01T00:00:00Z","session_id":"s","tool_name":"Read","tool_input_preview":"","tool_output_preview":"","cwd":"/"}\n')
    bad.write_text("data")
    bad.chmod(0o000)
    try:
        results = list(iter_recent(limit=10, base=tmp_path))
        assert len(results) == 1
        assert results[0].tool_name == "Read"
    finally:
        bad.chmod(0o644)
