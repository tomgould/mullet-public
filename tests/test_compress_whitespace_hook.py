"""Tests for mullet/hooks/compress_whitespace.py (and its scripts/hooks shim)."""

import json
import subprocess
import sys
from pathlib import Path


HOOK = Path(__file__).parent.parent / "scripts" / "hooks" / "compress_whitespace.py"

from mullet.hooks.compress_whitespace import compress


# --- Unit tests for compress() ---

def test_prose_collapses_double_spaces():
    assert compress("hello  world") == "hello world"


def test_prose_collapses_many_spaces():
    assert compress("a   b    c") == "a b c"


def test_single_spaces_unchanged():
    assert compress("hello world") == "hello world"


def test_empty_string():
    assert compress("") == ""


def test_leading_indent_preserved():
    assert compress("    hello  world") == "    hello world"


def test_tab_indent_preserved():
    assert compress("\thello  world") == "\thello world"


def test_code_block_untouched():
    code = "```python\ndef foo():    pass\n    x  =  1\n```"
    assert compress(code) == code


def test_code_block_inline_fence_untouched():
    prompt = "some  prose\n```\ndef x():  y  =  1\n```\nmore  prose"
    result = compress(prompt)
    assert "some prose" in result
    assert "def x():  y  =  1" in result
    assert "more prose" in result


def test_markdown_table_lines_untouched():
    table = "| col1  | col2  |\n| a  | b  |"
    assert compress(table) == table


def test_mixed_table_and_prose():
    text = "intro  text\n| a  | b  |\noutro  text"
    result = compress(text)
    assert "intro text" in result
    assert "| a  | b  |" in result
    assert "outro text" in result


def test_unclosed_fence_returns_unchanged():
    # Odd number of fences — bail out, don't corrupt.
    prompt = "hello  world\n```\nunclosed"
    assert compress(prompt) == prompt


def test_multiple_code_blocks():
    prompt = "prose  a\n```\ncode  1\n```\nprose  b\n```\ncode  2\n```\nprose  c"
    result = compress(prompt)
    assert "prose a" in result
    assert "code  1" in result
    assert "prose b" in result
    assert "code  2" in result
    assert "prose c" in result


def test_purely_code_block():
    prompt = "```\ndef foo():    pass\n```"
    assert compress(prompt) == prompt


# --- Integration test: hook via stdin/stdout ---

def _run_hook(prompt: str) -> tuple[str, str, int]:
    payload = json.dumps({"prompt": prompt})
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input=payload,
        capture_output=True,
        text=True,
    )
    return result.stdout, result.stderr, result.returncode


def test_hook_exits_zero():
    _, _, code = _run_hook("hello  world")
    assert code == 0


def test_hook_outputs_compressed_json():
    stdout, _, _ = _run_hook("hello  world")
    data = json.loads(stdout)
    assert data["prompt"] == "hello world"


def test_hook_no_output_when_unchanged():
    stdout, _, _ = _run_hook("hello world")
    assert stdout.strip() == ""


def test_hook_logs_saved_chars_to_stderr():
    _, stderr, _ = _run_hook("hello  world")
    assert "saved" in stderr
    assert "chars" in stderr


def test_hook_empty_prompt_no_output():
    stdout, _, _ = _run_hook("")
    assert stdout.strip() == ""


def test_hook_handles_user_prompt_key():
    payload = json.dumps({"user_prompt": "hello  world"})
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input=payload,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    assert data["prompt"] == "hello world"
