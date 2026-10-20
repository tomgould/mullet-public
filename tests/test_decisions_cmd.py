"""Tests for mullet.cli.decisions_cmd — dispatch."""
from unittest.mock import patch

from mullet.cli.decisions_cmd import run, _kind_str, _source_str, _print_decision, _HAS_COLOR
from mullet.core.decisions import Decision


def _make_decision(source_type="commit", kind="decided", excerpt="Use X over Y"):
    return Decision(
        id="abc123",
        source_path="docs/foo.md",
        source_type=source_type,
        line_number=None,
        timestamp=None,
        excerpt=excerpt,
        kind=kind,
    )


class TestRunDispatch:
    def test_no_args_shows_usage(self, capsys):
        rc = run([])
        out = capsys.readouterr().out
        assert "Usage" in out
        assert rc == 0

    def test_unknown_action_returns_1(self, capsys):
        rc = run(["doesnotexist"])
        assert rc == 1

    def test_list_with_empty_decisions(self, capsys):
        with patch("mullet.cli.decisions_cmd.load", return_value=[]):
            rc = run(["list"])
        capsys.readouterr()
        assert rc == 0

    def test_search_requires_query(self, capsys):
        rc = run(["search"])
        assert rc == 1

    def test_search_with_no_results(self, capsys):
        with patch("mullet.cli.decisions_cmd.dec_search", return_value=[]):
            rc = run(["search", "nonexistent topic"])
        capsys.readouterr()
        assert rc == 0

    def test_scan_calls_scan_all(self, capsys):
        decisions = [_make_decision("commit"), _make_decision("doc")]
        with (
            patch("mullet.cli.decisions_cmd.scan_all", return_value=decisions) as mock_scan,
            patch("mullet.cli.decisions_cmd.persist", return_value=len(decisions)),
        ):
            rc = run(["scan"])
        mock_scan.assert_called_once()
        assert rc == 0

    def test_list_prints_decisions(self, capsys):
        decisions = [_make_decision(excerpt="Use Kimi over Gemini for cost")]
        with patch("mullet.cli.decisions_cmd.load", return_value=decisions):
            rc = run(["list"])
        out = capsys.readouterr().out
        assert "Use Kimi" in out
        assert rc == 0


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------

class TestKindStr:
    def test_known_kind(self):
        out = _kind_str("decision")
        assert "decision" in out
        if _HAS_COLOR:
            assert "\033[" in out

    def test_unknown_kind_plain(self):
        assert _kind_str("unknown") == "unknown"


class TestSourceStr:
    def test_known_source(self):
        out = _source_str("docs", "README.md")
        assert "docs:README.md" in out
        if _HAS_COLOR:
            assert "\033[" in out

    def test_unknown_source_plain(self):
        assert _source_str("unknown", "file.txt") == "unknown:file.txt"


class TestPrintDecision:
    def test_prints_excerpt(self, capsys):
        d = Decision(
            id="id1",
            source_path="docs/foo.md",
            source_type="docs",
            line_number=1,
            timestamp="2026-05-02T10:00:00Z",
            excerpt="Use X over Y",
            kind="decision",
        )
        _print_decision(d)
        out = capsys.readouterr().out
        assert "Use X over Y" in out

    def test_handles_none_timestamp(self, capsys):
        d = Decision(
            id="id1",
            source_path="docs/foo.md",
            source_type="docs",
            line_number=1,
            timestamp=None,
            excerpt="No timestamp here",
            kind="rejection",
        )
        _print_decision(d)
        out = capsys.readouterr().out
        assert "No timestamp here" in out

    def test_long_excerpt_truncated(self, capsys):
        long_excerpt = "A" * 300
        d = Decision(
            id="id1",
            source_path="docs/foo.md",
            source_type="docs",
            line_number=1,
            timestamp="2026-05-02T10:00:00Z",
            excerpt=long_excerpt,
            kind="rationale",
        )
        _print_decision(d)
        out = capsys.readouterr().out
        assert "A" * 200 in out
        assert len(out) < len(long_excerpt) + 100


# ---------------------------------------------------------------------------
# Missing coverage: scan empty, list filter excludes all, search with results
# ---------------------------------------------------------------------------

def test_scan_empty_prints_no_decisions(capsys):
    """Lines 72-73: scan_all returns [] → 'No decisions found.' + return 0."""
    with patch("mullet.cli.decisions_cmd.scan_all", return_value=[]):
        rc = run(["scan"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "No decisions found" in out


def test_list_filter_excludes_all_decisions(capsys):
    """Lines 124-125: decisions exist but filter removes all → 'No matching decisions.'"""
    d = _make_decision(kind="decision")
    d_with_ts = Decision(
        id="ts1", source_path="docs/foo.md", source_type="docs",
        line_number=1, timestamp="2026-01-01T00:00:00Z",
        excerpt="use sqlite", kind="decision",
    )
    with patch("mullet.cli.decisions_cmd.load", return_value=[d_with_ts]):
        rc = run(["list", "--kind", "rejection"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "No matching decisions" in out


def test_search_with_results_calls_print_decision(capsys):
    """Lines 151-154: dec_search returns results → _print_decision called + return 0."""
    d = _make_decision(excerpt="Use Kimi over Gemini for long context tasks")
    with patch("mullet.cli.decisions_cmd.dec_search", return_value=[d]):
        rc = run(["search", "Kimi"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Kimi" in out
