"""Tests for mullet.core.decisions — extraction, persistence, search."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mullet.core.decisions import (
    Decision,
    _classify_line,
    _extract_from_lines,
    _is_false_positive,
    decisions_path,
    load,
    persist,
    scan_all,
    scan_docs,
    scan_git_log,
    scan_progress,
    search,
)


# ---------------------------------------------------------------------------
# _classify_line
# ---------------------------------------------------------------------------

class TestClassifyLine:
    def test_decision_trigger(self):
        result = _classify_line("We decided to use PostgreSQL for persistence.")
        assert result is not None
        kind, kws = result
        assert kind == "decision"
        assert any("decided" in kw.lower() for kw in kws)

    def test_chose_trigger(self):
        result = _classify_line("The team chose FastAPI because of async support.")
        assert result is not None
        kind, _ = result
        assert kind == "decision"

    def test_rejection_trigger(self):
        result = _classify_line("We rejected the idea of using Redis for primary storage.")
        assert result is not None
        kind, kws = result
        assert kind == "rejection"
        assert any("rejected" in kw.lower() for kw in kws)

    def test_ruled_out_trigger(self):
        result = _classify_line("We ruled out GraphQL because the team lacks familiarity with it.")
        assert result is not None
        kind, _ = result
        # "because" also matches rationale but decision priority wins? No — "ruled out" is rejection
        # priority order: decision > rationale > rejection
        # "ruled out" → rejection; "because" → rationale; rationale wins over rejection
        assert kind in ("rejection", "rationale")

    def test_rationale_trigger(self):
        result = _classify_line("Because performance was critical we went with the faster option.")
        assert result is not None
        kind, kws = result
        assert kind == "rationale"
        assert any(kw.lower() == "because" for kw in kws)

    def test_case_insensitive(self):
        result = _classify_line("DECIDED TO switch from YAML to TOML for config files.")
        assert result is not None
        assert result[0] == "decision"

    def test_line_too_short(self):
        assert _classify_line("decided") is None

    def test_line_too_long(self):
        long_line = "decided to " + "x" * 401
        assert _classify_line(long_line) is None

    def test_markdown_only_line(self):
        assert _classify_line("## -----") is None
        assert _classify_line("* * * ") is None

    def test_no_match(self):
        assert _classify_line("This is a plain sentence with no trigger words.") is None

    def test_multiple_triggers_decision_wins(self):
        # "decided" (decision) + "because" (rationale) → decision wins
        result = _classify_line("We decided to use Nginx because it is fast and well-supported.")
        assert result is not None
        kind, kws = result
        assert kind == "decision"
        assert len(kws) >= 2

    def test_keyword_collected(self):
        result = _classify_line("We decided to go with Rust for the CLI tooling.")
        assert result is not None
        _, kws = result
        assert len(kws) >= 1


# ---------------------------------------------------------------------------
# Post-match filter (_is_false_positive / _classify_line)
# ---------------------------------------------------------------------------

class TestFalsePositiveFilter:
    def test_pytest_output_rejected(self):
        # "skipped" appears but it's in pytest output context → reject
        result = _classify_line("Full suite: 3584 passed, 2 skipped, 0 failed after the run.")
        assert result is None, "pytest output line should be filtered"

    def test_pytest_failed_count_rejected(self):
        result = _classify_line("The run decided: 10 passed, 3 failed in total.")
        assert result is None, "line with digit+failed should be rejected"

    def test_keyword_in_inline_code_rejected(self):
        # keyword is inside backticks only → reject
        result = _classify_line("Use the `decided` flag to toggle this behaviour in production code.")
        assert result is None, "keyword only in inline code span should be filtered"

    def test_keyword_outside_inline_code_accepted(self):
        # keyword appears both inside code AND in plain prose → accept
        result = _classify_line("We decided to use the `decided` flag for legacy compatibility reasons.")
        assert result is not None, "keyword appears in prose, should not be filtered"

    def test_high_nonalnum_ratio_rejected(self):
        # Line is mostly symbols / data
        result = _classify_line("decided |------|------|------|------|------|------|------| end")
        assert result is None, "high non-alnum ratio should be rejected"

    def test_too_few_words_rejected(self):
        # Only 4 alpha words total
        result = _classify_line("We decided now: done.")
        assert result is None, "fewer than MIN_WORD_COUNT alpha words should be rejected"

    def test_genuine_decision_passes_filter(self):
        result = _classify_line("We decided to adopt Poetry for dependency management across all services.")
        assert result is not None, "genuine decision sentence should pass filter"

    def test_is_false_positive_direct_pytest(self):
        assert _is_false_positive("Full suite: 10 passed, 2 skipped, 0 failed", "skipped")

    def test_is_false_positive_direct_clean(self):
        assert not _is_false_positive(
            "We decided to use Postgres for all relational data storage needs", "decided"
        )


# ---------------------------------------------------------------------------
# _extract_from_lines
# ---------------------------------------------------------------------------

class TestExtractFromLines:
    def test_basic_extraction(self):
        lines = [
            "Some preamble text that says nothing.",
            "We decided to use SQLite for the local cache.",
            "This line matches nothing.",
            "We rejected Mongo because of licensing concerns.",
        ]
        results = _extract_from_lines(lines, "test.md", "docs")
        assert len(results) == 2
        assert results[0].kind == "decision"
        assert results[1].kind in ("rejection", "rationale")

    def test_skips_code_blocks(self):
        lines = [
            "```python",
            "decided_value = True  # decided to hardcode",
            "```",
            "We decided to use async/await throughout the codebase.",
        ]
        results = _extract_from_lines(lines, "test.md", "docs")
        # Code block line skipped, only the last line matches
        assert len(results) == 1
        assert "async/await" in results[0].excerpt

    def test_line_numbers_assigned(self):
        lines = ["nothing", "We decided to keep line numbers accurate."]
        results = _extract_from_lines(lines, "x.md", "docs")
        assert len(results) == 1
        assert results[0].line_number == 2

    def test_source_fields(self):
        lines = ["We chose Python 3.11 as the minimum supported version."]
        results = _extract_from_lines(lines, "docs/README.md", "docs", timestamp="2024-01-01T00:00:00Z")
        assert results[0].source_path == "docs/README.md"
        assert results[0].source_type == "docs"
        assert results[0].timestamp == "2024-01-01T00:00:00Z"

    def test_id_is_stable_hash(self):
        lines = ["We decided to use pytest for all testing."]
        r1 = _extract_from_lines(lines, "x.md", "docs")
        r2 = _extract_from_lines(lines, "x.md", "docs")
        assert r1[0].id == r2[0].id

    def test_excerpt_at_max_length(self):
        # A line of exactly 400 chars should be accepted and stored verbatim.
        body = "We decided to " + "x" * 386  # total = 400 chars
        lines = [body]
        results = _extract_from_lines(lines, "x.md", "docs")
        assert len(results) == 1
        assert len(results[0].excerpt) == 400


# ---------------------------------------------------------------------------
# scan_docs
# ---------------------------------------------------------------------------

class TestScanDocs:
    def test_recursive_scan(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        sub = docs / "sub"
        sub.mkdir()
        (docs / "main.md").write_text("We decided to use Postgres.\n")
        (sub / "inner.md").write_text("We chose SQLite for testing.\n")

        results = scan_docs(docs)
        assert len(results) == 2

    def test_source_type_roadmap(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "ROADMAP.md").write_text("We decided to implement feature X.\n")

        results = scan_docs(docs)
        assert results[0].source_type == "roadmap"

    def test_source_type_docs(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "architecture.md").write_text("We decided to use microservices.\n")

        results = scan_docs(docs)
        assert results[0].source_type == "docs"

    def test_missing_dir_returns_empty(self, tmp_path):
        assert scan_docs(tmp_path / "nonexistent") == []

    def test_relative_path_in_source(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "notes.md").write_text("We picked JSONL for log storage.\n")
        results = scan_docs(docs)
        assert results[0].source_path.startswith("docs/")


# ---------------------------------------------------------------------------
# scan_progress
# ---------------------------------------------------------------------------

class TestScanProgress:
    def test_extracts_from_progress(self, tmp_path):
        pf = tmp_path / "progress.txt"
        pf.write_text(
            "Session 1:\n"
            "We decided to keep the CLI minimal.\n"
            "Nothing interesting here.\n"
        )
        results = scan_progress(pf)
        assert len(results) == 1
        assert results[0].source_type == "progress"

    def test_missing_file_returns_empty(self, tmp_path):
        assert scan_progress(tmp_path / "nope.txt") == []


# ---------------------------------------------------------------------------
# scan_git_log
# ---------------------------------------------------------------------------

_FAKE_GIT_LOG = """\
abc123def456789012345678901234567890abcd|2024-06-15T10:00:00+00:00|feat: decided to use async workers
We decided to refactor the pool manager for async support.
Because the previous sync approach was a bottleneck.

def456abc123789012345678901234567890def0|2024-05-01T08:00:00+00:00|chore: version bump
Just a version bump, no decisions here.

"""


class TestScanGitLog:
    def test_extracts_from_commit_body(self, tmp_path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = _FAKE_GIT_LOG

        with patch("mullet.core.decisions.subprocess.run", return_value=mock_result):
            results = scan_git_log(limit=10, repo_dir=tmp_path)

        assert len(results) >= 1
        sources = {r.source_type for r in results}
        assert "commit" in sources

    def test_timestamp_captured(self, tmp_path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = _FAKE_GIT_LOG

        with patch("mullet.core.decisions.subprocess.run", return_value=mock_result):
            results = scan_git_log(limit=10, repo_dir=tmp_path)

        ts_results = [r for r in results if r.timestamp]
        assert ts_results, "Expected at least one result with a timestamp"

    def test_non_git_dir_returns_empty(self, tmp_path):
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stdout = ""

        with patch("mullet.core.decisions.subprocess.run", return_value=mock_result):
            assert scan_git_log(repo_dir=tmp_path) == []

    def test_file_not_found_returns_empty(self, tmp_path):
        with patch("mullet.core.decisions.subprocess.run", side_effect=FileNotFoundError):
            assert scan_git_log(repo_dir=tmp_path) == []


# ---------------------------------------------------------------------------
# persist + load (roundtrip)
# ---------------------------------------------------------------------------

def _make_decision(**kwargs) -> Decision:
    defaults = dict(
        id="aabbccdd11223344",
        source_path="docs/test.md",
        source_type="docs",
        line_number=1,
        timestamp="2024-01-01T00:00:00Z",
        excerpt="We decided to use pytest.",
        kind="decision",
        keywords=["decided"],
    )
    defaults.update(kwargs)
    return Decision(**defaults)


class TestPersistLoad:
    def test_roundtrip(self, tmp_path):
        d = _make_decision()
        n = persist([d], base=tmp_path)
        assert n == 1

        loaded = load(base=tmp_path)
        assert len(loaded) == 1
        assert loaded[0].id == d.id
        assert loaded[0].excerpt == d.excerpt

    def test_dedup_by_id(self, tmp_path):
        d = _make_decision()
        persist([d], base=tmp_path)
        # Persist again — same id, should not duplicate
        n = persist([d], base=tmp_path)
        assert n == 1

    def test_new_id_adds_to_existing(self, tmp_path):
        d1 = _make_decision(id="aaaa0001", excerpt="We decided to use pytest.")
        d2 = _make_decision(id="bbbb0002", excerpt="We chose Redis for caching.")
        persist([d1], base=tmp_path)
        persist([d2], base=tmp_path)
        loaded = load(base=tmp_path)
        assert len(loaded) == 2

    def test_load_missing_file(self, tmp_path):
        assert load(base=tmp_path) == []

    def test_jsonl_format(self, tmp_path):
        d = _make_decision()
        persist([d], base=tmp_path)
        path = decisions_path(tmp_path)
        raw_lines = path.read_text().splitlines()
        assert len(raw_lines) == 1
        obj = json.loads(raw_lines[0])
        assert obj["id"] == d.id


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def _seed(self, tmp_path: Path) -> None:
        decisions = [
            _make_decision(
                id="s001",
                excerpt="We decided to use pytest for testing.",
                kind="decision",
                timestamp="2024-03-01T00:00:00Z",
            ),
            _make_decision(
                id="s002",
                excerpt="We rejected using unittest because it is verbose.",
                kind="rejection",
                timestamp="2024-01-01T00:00:00Z",
            ),
            _make_decision(
                id="s003",
                excerpt="Because speed matters we chose async patterns.",
                kind="rationale",
                timestamp="2024-06-01T00:00:00Z",
            ),
            _make_decision(
                id="s004",
                excerpt="We decided to keep the API minimal.",
                kind="decision",
                timestamp=None,
            ),
        ]
        persist(decisions, base=tmp_path)

    def test_substring_match(self, tmp_path):
        self._seed(tmp_path)
        results = search("pytest", base=tmp_path)
        assert len(results) == 1
        assert "pytest" in results[0].excerpt

    def test_case_insensitive_query(self, tmp_path):
        self._seed(tmp_path)
        results = search("PYTEST", base=tmp_path)
        assert len(results) == 1

    def test_filter_by_kind(self, tmp_path):
        self._seed(tmp_path)
        results = search("", limit=10, kind="rejection", base=tmp_path)
        assert all(r.kind == "rejection" for r in results)

    def test_limit_applied(self, tmp_path):
        self._seed(tmp_path)
        results = search("", limit=2, base=tmp_path)
        assert len(results) <= 2

    def test_timestamp_ordering_newest_first(self, tmp_path):
        self._seed(tmp_path)
        results = search("decided", base=tmp_path, limit=10)
        ts_results = [r for r in results if r.timestamp]
        timestamps = [r.timestamp for r in ts_results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_none_timestamps_last(self, tmp_path):
        self._seed(tmp_path)
        results = search("decided", base=tmp_path, limit=10)
        seen_none = False
        for r in results:
            if r.timestamp is None:
                seen_none = True
            elif seen_none:
                pytest.fail("Timestamped result appeared after a None-timestamp result")

    def test_no_results(self, tmp_path):
        self._seed(tmp_path)
        assert search("zzznomatch", base=tmp_path) == []


# ---------------------------------------------------------------------------
# Coverage gaps
# ---------------------------------------------------------------------------

class TestClassifyLineCoverage:
    def test_no_keywords_match_after_passing_filters(self):
        """Line 141 (in _classify_line): passes length/markdown checks but no pattern matches."""
        line = "This sentence is definitely long enough to pass the minimum length filter."
        assert _classify_line(line) is None

    def test_markdown_only_long_line(self):
        """Line 141: markdown-only line that is long enough triggers early None."""
        line = "# " + "-" * 20
        assert _classify_line(line) is None


class TestScanDocsCoverage:
    def test_oserror_on_md_read(self, tmp_path, monkeypatch):
        """Line 245-246: OSError while reading an .md file is skipped gracefully."""
        docs = tmp_path / "docs"
        docs.mkdir()
        good = docs / "good.md"
        good.write_text("We decided to use SQLite.\n")
        bad = docs / "bad.md"
        bad.write_text("We decided to use Postgres.\n")

        original_read_text = Path.read_text

        def _patched_read_text(self, *args, **kwargs):
            if self.name == "bad.md":
                raise OSError("permission denied")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", _patched_read_text)
        results = scan_docs(docs)
        assert len(results) == 1
        assert "SQLite" in results[0].excerpt

    def test_scan_docs_defaults_to_cwd(self, tmp_path, monkeypatch):
        """Line 230: scan_docs with no argument scans <cwd>/docs."""
        monkeypatch.chdir(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "notes.md").write_text("We decided to use pytest.\n")
        results = scan_docs()
        assert len(results) == 1
        assert results[0].source_type == "docs"


class TestScanProgressCoverage:
    def test_oserror_on_progress_read(self, tmp_path, monkeypatch):
        """Lines 262-263: OSError while reading progress.txt returns empty list."""
        pf = tmp_path / "progress.txt"
        pf.write_text("We decided to keep it simple.\n")

        monkeypatch.setattr(Path, "read_text", lambda *a, **kw: (_ for _ in ()).throw(OSError("fail")))
        assert scan_progress(pf) == []

    def test_progress_file_outside_cwd(self, tmp_path, monkeypatch):
        """Line 265: progress file not relative to cwd uses absolute path as source_path."""
        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "outside_progress"
        outside.mkdir()
        pf = outside / "progress.txt"
        pf.write_text("We decided to go with plan B.\n")
        results = scan_progress(pf)
        assert len(results) == 1
        assert results[0].source_path == str(pf)


class TestScanAllCoverage:
    def test_scan_all_combines_sources(self, tmp_path, monkeypatch):
        """Lines 323-336: scan_all merges docs, progress, and dedups by id."""
        docs = tmp_path / "docs"
        docs.mkdir()
        # Two identical lines → same id, should be deduped to 1
        (docs / "notes.md").write_text("We decided to use pytest.\nWe decided to use pytest.\n")
        mullet = tmp_path / ".mullet"
        mullet.mkdir()
        (mullet / "progress.txt").write_text("We chose Redis instead of Memcached for caching.\n")

        monkeypatch.chdir(tmp_path)

        with patch("mullet.core.decisions.scan_git_log", return_value=[]):
            results = scan_all(base=tmp_path)

        assert len(results) == 2
        excerpts = {r.excerpt for r in results}
        assert "We decided to use pytest." in excerpts
        assert "We chose Redis instead of Memcached for caching." in excerpts


class TestPersistCoverage:
    def test_persist_existing_file_jsondecodeerror(self, tmp_path):
        """Lines 360-361: corrupted existing decisions.jsonl is ignored during persist."""
        path = decisions_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not-json-at-all\n")
        d = _make_decision()
        n = persist([d], base=tmp_path)
        assert n == 1
        loaded = load(base=tmp_path)
        assert len(loaded) == 1

    def test_persist_tmp_write_failure_cleanup(self, tmp_path, monkeypatch):
        """Lines 375-377: tmp file is cleaned up and exception re-raised on write failure."""
        d = _make_decision()
        called_unlink = False

        original_unlink = Path.unlink

        def _tracking_unlink(self, *args, **kwargs):
            nonlocal called_unlink
            if self.suffix == ".tmp":
                called_unlink = True
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", _tracking_unlink)
        monkeypatch.setattr(
            "builtins.open",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")),
        )

        with pytest.raises(OSError, match="disk full"):
            persist([d], base=tmp_path)

        assert called_unlink

    def test_persist_ignores_empty_lines_in_existing(self, tmp_path):
        """Existing file with empty lines is handled gracefully."""
        path = decisions_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n\n")
        d = _make_decision()
        n = persist([d], base=tmp_path)
        assert n == 1


class TestLoadCoverage:
    def test_load_corrupted_jsonl(self, tmp_path):
        """Lines 394-395: OSError/JSONDecodeError during load returns empty list silently."""
        path = decisions_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("bad json line\n")
        assert load(base=tmp_path) == []

    def test_load_ignores_empty_lines(self, tmp_path):
        """File with empty lines is handled gracefully."""
        path = decisions_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = _make_decision()
        persist([d], base=tmp_path)
        with open(path, "a") as f:
            f.write("\n\n")
        loaded = load(base=tmp_path)
        assert len(loaded) == 1


class TestFalsePositiveCoverage:
    def test_table_separator_with_short_keyword(self):
        """Line 74: table separator line with only short alpha words is rejected."""
        line = "| we use |-------|"
        assert _classify_line(line) is None


class TestScanGitLogCoverage:
    def test_malformed_header_skipped(self, tmp_path):
        """Malformed git header with only one pipe falls through to body accumulation."""
        fake_log = (
            "abc123def456789012345678901234567890abcd|2024-06-15T10:00:00+00:00|feat: initial\n"
            "def456abc123789012345678901234567890def0|2024-06-16T10:00:00+00:00\n"
            "We decided to use something.\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_log

        with patch("mullet.core.decisions.subprocess.run", return_value=mock_result):
            results = scan_git_log(limit=10, repo_dir=tmp_path)

        assert len(results) == 1
        assert results[0].source_type == "commit"
