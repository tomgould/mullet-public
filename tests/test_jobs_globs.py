"""Tests for mullet.jobs.globs path patterns."""

import pytest

from mullet.jobs.globs import compile_pattern, matches, matches_any


@pytest.mark.parametrize(
    "pattern, path, expected",
    [
        ("**", "a/b/c.py", True),
        ("src/**", "src", True),
        ("src/**", "src/a/b.py", True),
        ("src/**", "srcx/a.py", False),
        ("src/**/*.py", "src/a.py", True),
        ("src/**/*.py", "src/a/b/c.py", True),
        ("src/**/*.py", "src/a/b/c.txt", False),
        ("src/*.py", "src/a/b.py", False),
        ("CLAUDE.md", "CLAUDE.md", True),
        ("CLAUDE.md", "docs/deep/CLAUDE.md", True),
        ("/CLAUDE.md", "docs/CLAUDE.md", False),
        (".git", ".git", True),
        (".git/**", ".git/config", True),
        (".git/**", ".github/workflows/ci.yml", False),
        (".github/workflows/**", ".github/workflows/ci.yml", True),
        (".env.*", ".env.local", True),
        (".env.*", "app/.env.production", True),
        (".env", ".envrc", False),
        ("node_modules/", "node_modules/x/index.js", True),
        ("test_?.py", "tests/test_a.py", True),
        ("test_?.py", "tests/test_ab.py", False),
        ("[ab].py", "a.py", True),
        ("[!ab].py", "c.py", True),
        ("[!ab].py", "a.py", False),
        ("*.lock", "deep/Cargo.lock", True),
        ("a+b(c).py", "a+b(c).py", True),
    ],
)
def test_matches(pattern, path, expected):
    assert matches(path, pattern) is expected


def test_star_never_crosses_a_separator():
    assert not matches("a/b", "a*b")
    assert not matches("a/b", "a[!x]b")


def test_matches_any():
    assert matches_any("pkg/uv.lock", ["*.json", "uv.lock"])
    assert not matches_any("pkg/main.py", ["*.json", "uv.lock"])
    assert not matches_any("anything", [])


def test_leading_slash_on_path_is_ignored():
    assert matches("/src/a.py", "src/*.py")


def test_empty_pattern_rejected():
    with pytest.raises(ValueError):
        compile_pattern("  ")
