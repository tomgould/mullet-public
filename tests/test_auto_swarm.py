"""Tests for the auto-swarm routing heuristic."""

from mullet.core.router.auto_swarm import _should_swarm, _language_count, _is_doc, _detect_file_mentions_in_task


# ---------------------------------------------------------------------------
# _is_doc
# ---------------------------------------------------------------------------

def test_is_doc_markdown():
    assert _is_doc("README.md") is True


def test_is_doc_rst():
    assert _is_doc("docs/guide.rst") is True


def test_is_doc_python():
    assert _is_doc("src/main.py") is False


def test_is_doc_no_extension():
    assert _is_doc("Makefile") is False


# ---------------------------------------------------------------------------
# _language_count
# ---------------------------------------------------------------------------

def test_language_count_single():
    assert _language_count(["a.py", "b.py", "c.py"]) == 1


def test_language_count_multi():
    assert _language_count(["a.py", "b.ts", "c.go"]) == 3


def test_language_count_excludes_docs():
    assert _language_count(["a.md", "b.rst", "c.txt"]) == 0


def test_language_count_excludes_config():
    assert _language_count(["a.json", "b.yaml", "c.toml"]) == 0


def test_language_count_mixed():
    # py + ts = 2, json/md excluded
    assert _language_count(["a.py", "b.ts", "c.json", "d.md"]) == 2


# ---------------------------------------------------------------------------
# _should_swarm — no signals
# ---------------------------------------------------------------------------

def test_no_swarm_empty_files():
    ok, reason = _should_swarm("simple task", [], prompt_len=100)
    assert ok is False
    assert reason == ""


def test_no_swarm_small_prompt_few_files():
    ok, reason = _should_swarm("small task", ["a.py"], prompt_len=500)
    assert ok is False
    assert reason == ""


def test_no_swarm_standard_tier():
    ok, _ = _should_swarm("task", ["a.py"], prompt_len=100, tier="STANDARD")
    assert ok is False


# ---------------------------------------------------------------------------
# _should_swarm — signal: file count
# ---------------------------------------------------------------------------

def test_swarm_on_two_files():
    files = ["a.py", "b.py"]
    ok, reason = _should_swarm("review all", files)
    assert ok is True
    assert "file_count=2" in reason


def test_swarm_on_five_files():
    files = [f"file{i}.py" for i in range(5)]
    ok, reason = _should_swarm("review all", files)
    assert ok is True
    assert "file_count=5" in reason


def test_swarm_on_many_files():
    files = [f"f{i}.py" for i in range(20)]
    ok, _ = _should_swarm("audit", files)
    assert ok is True


def test_no_swarm_one_file():
    files = ["f0.py"]
    ok, _ = _should_swarm("audit", files)
    assert ok is False


# ---------------------------------------------------------------------------
# _should_swarm — signal: CRITICAL tier
# ---------------------------------------------------------------------------

def test_swarm_on_critical_tier():
    ok, reason = _should_swarm("complex refactor", ["a.py"], tier="CRITICAL")
    assert ok is True
    assert "tier=CRITICAL" in reason


def test_swarm_on_critical_tier_case_insensitive():
    ok, _ = _should_swarm("complex refactor", [], tier="critical")
    assert ok is True


def test_no_swarm_advanced_tier():
    ok, _ = _should_swarm("task", ["a.py"], tier="ADVANCED")
    assert ok is False


# ---------------------------------------------------------------------------
# _should_swarm — signal: prompt length
# ---------------------------------------------------------------------------

def test_swarm_on_long_prompt():
    ok, reason = _should_swarm("x" * 8_000, [])
    assert ok is True
    assert "prompt_len=8000" in reason


def test_no_swarm_short_prompt():
    ok, _ = _should_swarm("x" * 2_999, [])
    assert ok is False


# ---------------------------------------------------------------------------
# _should_swarm — signal: multi-language
# ---------------------------------------------------------------------------

def test_swarm_multi_language():
    ok, reason = _should_swarm("task", ["a.py", "b.ts"])
    assert ok is True
    assert "multi-language" in reason


def test_no_swarm_single_language_one_file():
    ok, _ = _should_swarm("task", ["a.py"])
    assert ok is False


# ---------------------------------------------------------------------------
# _should_swarm — signal: doc set
# ---------------------------------------------------------------------------

def test_swarm_doc_set():
    files = ["docs/a.md", "docs/b.rst", "docs/c.txt"]
    ok, reason = _should_swarm("update docs", files)
    assert ok is True
    assert "doc_file_count=3" in reason


def test_swarm_two_doc_files():
    ok, reason = _should_swarm("review docs", ["a.md", "b.md"])
    assert ok is True
    assert "doc_file_count=2" in reason


# ---------------------------------------------------------------------------
# force_single bypass (tested via mcp_handlers)
# ---------------------------------------------------------------------------

def test_force_single_param_exists_in_handler(monkeypatch):
    """handle_mullet_run should accept force_single without error."""
    # force_single=True + no provider = error about provider, not about force_single
    from mullet.server.mcp_handlers import handle_mullet_run
    monkeypatch.setattr(
        "mullet.server.mcp_handlers._resolve_provider", lambda **kw: None
    )
    result = handle_mullet_run({"task": "hello", "force_single": True})
    # Should hit the "No LLM provider" path, not crash on unknown param
    assert "error" in result
    assert "force_single" not in result.get("error", "")


def test_detect_no_mentions():
    assert _detect_file_mentions_in_task("hello world") == 0


def test_detect_single_mention():
    assert _detect_file_mentions_in_task("fix foo.py") == 1


def test_detect_multiple_mentions():
    assert _detect_file_mentions_in_task("fix foo.py and bar.ts") == 2


def test_detect_with_directory():
    assert _detect_file_mentions_in_task("fix src/foo.py") == 1


def test_detect_various_extensions():
    assert _detect_file_mentions_in_task("a.go b.rs c.java") == 3


def test_detect_no_false_positives():
    assert _detect_file_mentions_in_task("python code without paths") == 0


def test_detect_mixed_with_regular_text():
    task = "Please review src/main.py and then check tests/test_main.py for errors"
    assert _detect_file_mentions_in_task(task) == 2


def test_detect_no_match_without_extension():
    assert _detect_file_mentions_in_task("foo bar baz qux") == 0
