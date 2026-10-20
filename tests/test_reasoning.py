"""Tests for reasoning/thinking block extraction."""

from mullet.core.agent.reasoning import (
    strip_think_blocks,
    extract_reasoning,
    extract_content_or_reasoning,
)


# --- strip_think_blocks ---

def test_strip_none():
    assert strip_think_blocks(None) == ""


def test_strip_empty():
    assert strip_think_blocks("") == ""


def test_strip_no_blocks():
    assert strip_think_blocks("hello world") == "hello world"


def test_strip_think_block():
    result = strip_think_blocks("<think>reasoning here</think> answer")
    assert result == "answer"


def test_strip_thinking_block():
    result = strip_think_blocks("<thinking>deep thought</thinking>the real answer")
    assert result == "the real answer"


def test_strip_reasoning_block():
    result = strip_think_blocks("<reasoning>analysis</reasoning>conclusion")
    assert result == "conclusion"


def test_strip_scratchpad():
    result = strip_think_blocks("<REASONING_SCRATCHPAD>notes</REASONING_SCRATCHPAD>final")
    assert result == "final"


def test_strip_multiline():
    text = """<think>
    Step 1: analyze
    Step 2: compute
    Step 3: verify
    </think>
    The answer is 42."""
    result = strip_think_blocks(text)
    assert "42" in result
    assert "Step 1" not in result


def test_strip_orphaned_close_tag():
    result = strip_think_blocks("some reasoning</think>actual answer")
    assert "actual answer" in result


def test_strip_orphaned_thinking_close():
    result = strip_think_blocks("reasoning</thinking>answer")
    assert "answer" in result


def test_strip_case_insensitive():
    result = strip_think_blocks("<THINKING>loud thinking</THINKING>quiet answer")
    assert result == "quiet answer"


def test_strip_multiple_blocks():
    text = "<think>first</think>visible<think>second</think>also visible"
    result = strip_think_blocks(text)
    assert "visible" in result
    assert "also visible" in result
    assert "first" not in result
    assert "second" not in result


# --- extract_reasoning ---

def test_extract_reasoning_none():
    assert extract_reasoning(None) == ""


def test_extract_reasoning_no_blocks():
    assert extract_reasoning("just plain text") == ""


def test_extract_reasoning_single():
    result = extract_reasoning("<think>my reasoning</think>answer")
    assert result == "my reasoning"


def test_extract_reasoning_multiple():
    text = "<think>step 1</think>middle<thinking>step 2</thinking>end"
    result = extract_reasoning(text)
    assert "step 1" in result
    assert "step 2" in result


# --- extract_content_or_reasoning ---

def test_extract_plain_content():
    msg = {"content": "plain answer"}
    assert extract_content_or_reasoning(msg) == "plain answer"


def test_extract_strips_think_blocks():
    msg = {"content": "<think>reasoning</think>the answer"}
    assert extract_content_or_reasoning(msg) == "the answer"


def test_extract_falls_back_to_reasoning_field():
    msg = {"content": "", "reasoning": "structured reasoning output"}
    assert extract_content_or_reasoning(msg) == "structured reasoning output"


def test_extract_reasoning_content_field():
    msg = {"content": None, "reasoning_content": "deepseek reasoning"}
    assert extract_content_or_reasoning(msg) == "deepseek reasoning"


def test_extract_reasoning_details_array():
    msg = {
        "content": "",
        "reasoning_details": [
            {"summary": "First conclusion"},
            {"content": "Second conclusion"},
        ],
    }
    result = extract_content_or_reasoning(msg)
    assert "First conclusion" in result
    assert "Second conclusion" in result


def test_extract_prefers_content_over_reasoning():
    msg = {
        "content": "real answer",
        "reasoning": "should not use this",
    }
    assert extract_content_or_reasoning(msg) == "real answer"


def test_extract_empty_everything():
    msg = {"content": ""}
    assert extract_content_or_reasoning(msg) == ""


def test_extract_content_only_think_blocks():
    """If content is ONLY think blocks, fall back to reasoning fields."""
    msg = {
        "content": "<think>all reasoning no answer</think>",
        "reasoning": "fallback reasoning",
    }
    result = extract_content_or_reasoning(msg)
    assert result == "fallback reasoning"
