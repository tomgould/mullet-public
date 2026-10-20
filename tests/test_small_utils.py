"""Unit tests for small utility modules: ansi_strip, knowledge.context."""



# --- ansi_strip ------------------------------------------------------------


def test_strip_ansi_plain_text_passthrough():
    from mullet.core.utils.ansi_strip import strip_ansi
    assert strip_ansi("hello world") == "hello world"


def test_strip_ansi_removes_csi_sequences():
    """CSI sequences like \\x1b[31m (red) must be stripped."""
    from mullet.core.utils.ansi_strip import strip_ansi
    colored = "\x1b[31mRED TEXT\x1b[0m"
    assert strip_ansi(colored) == "RED TEXT"


def test_strip_ansi_removes_multiple_sequences():
    from mullet.core.utils.ansi_strip import strip_ansi
    s = "\x1b[1m\x1b[32mbold green\x1b[0m ordinary \x1b[33myellow\x1b[0m"
    assert strip_ansi(s) == "bold green ordinary yellow"


def test_strip_ansi_removes_osc_sequences():
    """OSC sequences terminated by \\x07 (BEL) — e.g. hyperlinks."""
    from mullet.core.utils.ansi_strip import strip_ansi
    # \x1b]8;;https://example.com\x07Link\x1b]8;;\x07
    s = "\x1b]8;;https://example.com\x07Link\x1b]8;;\x07"
    assert strip_ansi(s) == "Link"


def test_strip_ansi_preserves_literal_non_ansi_text():
    from mullet.core.utils.ansi_strip import strip_ansi
    s = "path/to/file.py:line 42"
    assert strip_ansi(s) == s


def test_strip_ansi_empty_string():
    from mullet.core.utils.ansi_strip import strip_ansi
    assert strip_ansi("") == ""


# --- knowledge.context -----------------------------------------------------


def test_build_system_prompt_without_knowledge():
    """No knowledge context — returns base prompt unchanged."""
    from mullet.core.knowledge.context import build_system_prompt
    out = build_system_prompt("You are helpful.")
    assert out == "You are helpful."


def test_build_system_prompt_with_project_and_markdown():
    from mullet.core.knowledge.context import build_system_prompt
    from mullet.core.knowledge.manager import KnowledgeContext

    kc = KnowledgeContext.__new__(KnowledgeContext)
    kc.markdown = "## Project notes\nUse Python 3.10+."
    out = build_system_prompt("base", knowledge_context=kc, project_name="MyProj")
    assert "base" in out
    assert "MyProj" in out
    assert "Project notes" in out


def test_build_system_prompt_with_empty_markdown():
    """Empty markdown → no injection."""
    from mullet.core.knowledge.context import build_system_prompt
    from mullet.core.knowledge.manager import KnowledgeContext

    kc = KnowledgeContext.__new__(KnowledgeContext)
    kc.markdown = ""
    out = build_system_prompt("base", knowledge_context=kc, project_name="X")
    assert out == "base"


def test_estimate_tokens_four_chars_per_token():
    from mullet.core.knowledge.context import estimate_tokens
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("x" * 400) == 100


def test_truncate_to_budget_passthrough_within_budget():
    from mullet.core.knowledge.context import truncate_to_budget
    text = "hello"
    out = truncate_to_budget(text, max_tokens=100)
    assert out == text


def test_truncate_to_budget_truncates_when_over():
    from mullet.core.knowledge.context import truncate_to_budget
    text = "x" * 1000
    out = truncate_to_budget(text, max_tokens=10)  # 10*4 = 40 chars
    assert len(out) > 40  # includes the truncation marker
    assert "[...context truncated" in out
    assert out.startswith("x" * 40)
