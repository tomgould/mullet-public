"""Tests for mullet.core.router.auto_swarm."""

import os
from unittest.mock import patch

import pytest

from mullet.core.router.auto_swarm import (
    _is_doc,
    _language_count,
    _detect_file_mentions_in_task,
    _should_swarm,
)


# --- _is_doc ---------------------------------------------------------------

def test_is_doc_known_extensions():
    assert _is_doc("README.md")
    assert _is_doc("guide.rst")
    assert _is_doc("notes.txt")
    assert _is_doc("page.adoc")
    assert _is_doc("index.html")


def test_is_doc_uppercase_extension():
    assert _is_doc("README.MD")
    assert _is_doc("GUIDE.RST")


def test_is_doc_not_doc():
    assert not _is_doc("main.py")
    assert not _is_doc("script.js")
    assert not _is_doc("data.json")
    assert not _is_doc("config")
    assert not _is_doc("Makefile")


# --- _language_count ---------------------------------------------------------

def test_language_count_empty():
    assert _language_count([]) == 0


def test_language_count_single_language():
    assert _language_count(["a.py", "b.py"]) == 1


def test_language_count_multiple_languages():
    assert _language_count(["a.py", "b.js", "c.go"]) == 3


def test_language_count_ignores_docs_and_data():
    files = ["a.md", "b.rst", "c.json", "d.yaml", "e.yml", "f.toml", "g.ini", "h.cfg", "i.txt"]
    assert _language_count(files) == 0


def test_language_count_mixed():
    files = ["a.py", "b.md", "c.js", "d.json", "e.rs"]
    assert _language_count(files) == 3  # py, js, rs


def test_language_count_no_extension():
    assert _language_count(["Makefile", "Dockerfile"]) == 0


def test_language_count_case_insensitive():
    assert _language_count(["A.PY", "b.JS"]) == 2


# --- _detect_file_mentions_in_task -------------------------------------------

def test_detect_file_mentions_none():
    assert _detect_file_mentions_in_task("just some plain text") == 0


def test_detect_file_mentions_single():
    assert _detect_file_mentions_in_task("review foo.py") == 1


def test_detect_file_mentions_multiple():
    task = "review src/foo.py, lib/bar.js, and tests/test_go.go"
    assert _detect_file_mentions_in_task(task) == 3


def test_detect_file_mentions_with_paths():
    task = "fix a/b/c.tsx and d/e/f.jsx"
    assert _detect_file_mentions_in_task(task) == 2


def test_detect_file_mentions_does_not_match_words():
    task = "the python code and java script"
    assert _detect_file_mentions_in_task(task) == 0


def test_detect_file_mentions_doc_extensions():
    task = "update README.md and CHANGELOG.rst"
    assert _detect_file_mentions_in_task(task) == 2


def test_detect_file_mentions_caches_regex():
    # first call compiles regex
    count1 = _detect_file_mentions_in_task("foo.py")
    # second call uses cached regex
    count2 = _detect_file_mentions_in_task("bar.js")
    assert count1 == 1
    assert count2 == 1


# --- _should_swarm -----------------------------------------------------------

def test_should_swarm_critical_tier():
    result, reason = _should_swarm("task", [], tier="CRITICAL")
    assert result is True
    assert "tier=CRITICAL" in reason


def test_should_swarm_critical_tier_lowercase():
    result, reason = _should_swarm("task", [], tier="critical")
    assert result is True
    assert "tier=critical" in reason


def test_should_swarm_prompt_len_above_threshold():
    long_task = "x" * 3001
    result, reason = _should_swarm(long_task, [])
    assert result is True
    assert "prompt_len=" in reason


def test_should_swarm_prompt_len_param():
    result, reason = _should_swarm("short", [], prompt_len=5000)
    assert result is True
    assert "prompt_len=5000" in reason


def test_should_swarm_multi_language():
    files = ["a.py", "b.js"]
    result, reason = _should_swarm("task", files)
    assert result is True
    assert "multi-language" in reason


def test_should_swarm_doc_files():
    files = ["a.md", "b.rst"]
    result, reason = _should_swarm("task", files)
    assert result is True
    assert "doc_file_count=2" in reason


def test_should_swarm_file_count():
    files = ["a.py", "b.py"]
    result, reason = _should_swarm("task", files)
    assert result is True
    assert "file_count=2" in reason


def test_should_swarm_file_count_five():
    files = ["a.py", "b.py", "c.py", "d.py", "e.py"]
    result, reason = _should_swarm("task", files)
    assert result is True
    assert "file_count=5" in reason


def test_should_swarm_mentioned_files_no_files_param():
    task = "review foo.py and bar.js"
    result, reason = _should_swarm(task, [])
    assert result is True
    assert "task mentions" in reason


def test_should_swarm_mentioned_files_not_enough():
    task = "review foo.py"
    result, reason = _should_swarm(task, [])
    assert result is False
    assert reason == ""


def test_should_swarm_no_signals():
    result, reason = _should_swarm("short task", ["a.py"])
    assert result is False
    assert reason == ""


def test_should_swarm_ignores_mentions_when_files_passed():
    task = "review foo.py and bar.js and baz.go and qux.rs and quux.java"
    files = ["a.py"]
    result, reason = _should_swarm(task, files)
    # file_count=1 < _MIN_FILES_FOR_SWARM, prompt short, no docs, 1 language
    assert result is False
    assert reason == ""


@patch("mullet.core.router.auto_swarm.os.path.getsize", side_effect=OSError("no file"))
def test_should_swarm_oserror_on_file_size(_mock_getsize):
    # When files don't exist, OSError is caught and prompt_len falls back to task len
    result, reason = _should_swarm("x" * 500, ["missing.py"])
    assert result is False
    assert reason == ""


@patch("mullet.core.router.auto_swarm.os.path.getsize", return_value=1000)
def test_should_swarm_file_size_contributes_to_prompt_len(mock_getsize):
    task = "x" * 1000
    # prompt_len = 1000 + 1000 = 2000, which is < _MIN_PROMPT_LEN_FOR_SWARM
    result, reason = _should_swarm(task, ["a.py"])
    assert result is False
    mock_getsize.assert_called_once_with("a.py")


@patch("mullet.core.router.auto_swarm.os.path.getsize", return_value=2000)
def test_should_swarm_file_size_pushes_over_threshold(_mock_getsize):
    task = "x" * 1500
    # prompt_len = 1500 + 2000 = 3500 > _MIN_PROMPT_LEN_FOR_SWARM
    result, reason = _should_swarm(task, ["a.py"])
    assert result is True
    assert "prompt_len=" in reason


def test_should_swarm_none_prompt_len_with_files():
    # prompt_len=None triggers file size calculation, but no files means just task len
    result, reason = _should_swarm("short", [])
    assert result is False


def test_should_swarm_doc_files_need_at_least_two():
    files = ["a.md"]
    result, reason = _should_swarm("task", files)
    assert result is False


def test_should_swarm_multi_language_needs_at_least_two_files():
    files = ["a.py"]
    result, reason = _should_swarm("task", files)
    assert result is False


def test_should_swarm_prompt_len_exactly_at_threshold():
    result, reason = _should_swarm("x" * 3000, [])
    assert result is True
    assert "prompt_len=3000" in reason
