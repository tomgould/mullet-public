"""Tests for the deterministic (non-LLM) diff-risk classifier.

Covers each RiskFlag category individually, safe/no-flag cases that must
stay silent, the identical-content no-op case, and classify_git_diff
integration against a real temporary git repository.
"""
import subprocess

import pytest

from mullet.core.verification.diff_risk import RiskFlag, classify_diff, classify_git_diff


def _categories(flags):
    return {f.category for f in flags}


# ---------------------------------------------------------------------------
# One test per category (a) through (g)
# ---------------------------------------------------------------------------

class TestCategoriesFlag:
    def test_literal_change_numeric_test_assertion_example(self):
        """toHaveLength(2) -> toHaveLength(1): numeric literal changed."""
        old = "expect(x).toHaveLength(2);"
        new = "expect(x).toHaveLength(1);"
        flags = classify_diff(old, new, "example.ts")
        assert "literal_change" in _categories(flags)

    def test_literal_change_string_postcode_example(self):
        """A quoted string literal's contents changing (e.g. a corrupted postcode)."""
        old = "const postcode = 'SW1A 1AA';"
        new = "const postcode = 'SW1A 1AB';"
        flags = classify_diff(old, new, "example.ts")
        matches = [f for f in flags if f.category == "literal_change"]
        assert matches, f"expected literal_change, got categories {_categories(flags)}"

    def test_quote_flattening(self):
        """Curly quotes flattened to straight ASCII quotes."""
        old = "const s = ‘hello’;"
        new = "const s = 'hello';"
        flags = classify_diff(old, new, "example.ts")
        assert "quote_flattening" in _categories(flags)

    def test_new_control_flow(self):
        """A new && introduces control flow that wasn't in the original line."""
        old = "return x;"
        new = "return x && y;"
        flags = classify_diff(old, new, "example.ts")
        assert "new_control_flow" in _categories(flags)

    def test_new_control_flow_if_statement(self):
        """A new if ( guard introduces control flow that wasn't there before."""
        old = "doThing();"
        new = "if (ready) doThing();"
        flags = classify_diff(old, new, "example.ts")
        assert "new_control_flow" in _categories(flags)

    def test_assertion_on_literal(self):
        """A nonsensical non-null assertion applied directly to a string literal."""
        old = "const x = foo();"
        new = "const x = 'literal'!;"
        flags = classify_diff(old, new, "example.ts")
        assert "assertion_on_literal" in _categories(flags)

    def test_call_args_changed(self):
        """A fabricated extra call argument."""
        old = "buildApp('example.com');"
        new = "buildApp('example.com', extraArg);"
        flags = classify_diff(old, new, "example.ts")
        assert "call_args_changed" in _categories(flags)

    def test_comment_changed(self):
        """An existing comment's text silently changed."""
        old = "const x = 1; // original note"
        new = "const x = 1; // different note"
        flags = classify_diff(old, new, "example.ts")
        assert "comment_changed" in _categories(flags)

    def test_test_assertion_changed_replaces_call_args_changed(self):
        """A silently-changed test expectation is tagged test_assertion_changed,
        not (also/instead) generic call_args_changed for that same call."""
        old = "expect(items).toHaveLength(2);"
        new = "expect(items).toHaveLength(1);"
        flags = classify_diff(old, new, "example.spec.ts")
        cats = _categories(flags)
        assert "test_assertion_changed" in cats
        assert "call_args_changed" not in cats


# ---------------------------------------------------------------------------
# Safe / no-flag cases
# ---------------------------------------------------------------------------

class TestSafeCasesStaySilent:
    def test_pure_type_annotation_addition_is_silent(self):
        old = "function f(x) {"
        new = "function f(x: string) {"
        flags = classify_diff(old, new, "example.ts")
        assert flags == []

    def test_pure_whitespace_change_is_silent(self):
        old = "  return x;"
        new = "    return x;"
        flags = classify_diff(old, new, "example.ts")
        assert flags == []

    def test_pure_comment_addition_is_silent(self):
        old = "const x = 1;"
        new = "const x = 1; // note"
        flags = classify_diff(old, new, "example.ts")
        assert "comment_changed" not in _categories(flags)

    def test_pure_comment_removal_is_silent(self):
        old = "const x = 1; // note"
        new = "const x = 1;"
        flags = classify_diff(old, new, "example.ts")
        assert "comment_changed" not in _categories(flags)

    def test_as_const_cast_addition_does_not_flag_literal_change(self):
        old = "const n = 2;"
        new = "const n = 2 as const;"
        flags = classify_diff(old, new, "example.ts")
        assert "literal_change" not in _categories(flags)

    def test_bare_non_null_assertion_after_identifier_is_silent(self):
        """A ! immediately after an identifier (not a literal) must not
        trigger assertion_on_literal."""
        old = "const y = x;"
        new = "const y = x!;"
        flags = classify_diff(old, new, "example.ts")
        assert "assertion_on_literal" not in _categories(flags)


# ---------------------------------------------------------------------------
# Identical content
# ---------------------------------------------------------------------------

def test_identical_content_returns_empty_list():
    assert classify_diff("same", "same") == []


# ---------------------------------------------------------------------------
# classify_git_diff integration tests against a real temp git repo
# ---------------------------------------------------------------------------

def _init_git_repo(path):
    subprocess.run(["git", "init"], cwd=path, capture_output=True, text=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path, capture_output=True, text=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=path, capture_output=True, text=True, check=True,
    )


def _commit_file(path, filename, content):
    file_path = path / filename
    file_path.write_text(content)
    subprocess.run(["git", "add", filename], cwd=path, capture_output=True, text=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add " + filename],
        cwd=path, capture_output=True, text=True, check=True,
    )
    return file_path


class TestClassifyGitDiffIntegration:
    def test_modified_committed_file_flags_literal_change(self, tmp_path):
        _init_git_repo(tmp_path)
        _commit_file(tmp_path, "postcode.ts", "const postcode = 'SW1A 1AA';\n")

        (tmp_path / "postcode.ts").write_text("const postcode = 'SW1A 1AB';\n")

        flags = classify_git_diff(str(tmp_path), "postcode.ts")
        assert "literal_change" in _categories(flags)

    def test_new_uncommitted_file_does_not_raise_or_flag(self, tmp_path):
        _init_git_repo(tmp_path)
        # Need at least one commit so HEAD exists, but the target file itself
        # is never committed — `git show HEAD:<file>` must fail cleanly.
        _commit_file(tmp_path, "unrelated.ts", "const a = 1;\n")

        (tmp_path / "brand_new.ts").write_text("const totallyNew = 42;\n")

        flags = classify_git_diff(str(tmp_path), "brand_new.ts")
        assert flags == []

    def test_to_have_length_change_end_to_end(self, tmp_path):
        _init_git_repo(tmp_path)
        _commit_file(
            tmp_path,
            "example.spec.ts",
            "expect(items).toHaveLength(2);\n",
        )

        (tmp_path / "example.spec.ts").write_text("expect(items).toHaveLength(1);\n")

        flags = classify_git_diff(str(tmp_path), "example.spec.ts")
        assert "test_assertion_changed" in _categories(flags)

    def test_no_head_at_all_does_not_raise(self, tmp_path):
        """A git repo with zero commits (no HEAD yet) must not raise."""
        _init_git_repo(tmp_path)
        (tmp_path / "fresh.ts").write_text("const x = 1;\n")

        flags = classify_git_diff(str(tmp_path), "fresh.ts")
        assert flags == []
