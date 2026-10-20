"""Tests for conversation export."""

import json
from mullet.core.agent.export import export_json, export_markdown
from mullet.core.agent.types import AgentResult, TokenUsage


def _make_result() -> AgentResult:
    return AgentResult(
        completed=True,
        final_response="The answer is 42.",
        messages=[
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "what is the answer?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "compute", "arguments": '{"x": 42}'}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"result": 42}'},
            {"role": "assistant", "content": "The answer is 42."},
        ],
        tool_calls_made=1,
        iterations_used=2,
        tokens_used=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        provider_used="local",
        tier_used="LOW",
    )


def test_export_json():
    result = _make_result()
    output = export_json(result, "what is the answer?")
    data = json.loads(output)
    assert data["task"] == "what is the answer?"
    assert data["completed"] is True
    assert data["tokens"]["total"] == 150
    assert len(data["messages"]) == 5


def test_export_json_to_file(tmp_path):
    result = _make_result()
    path = str(tmp_path / "export.json")
    export_json(result, "test", path)
    data = json.loads((tmp_path / "export.json").read_text())
    assert data["provider"] == "local"


def test_export_markdown():
    result = _make_result()
    md = export_markdown(result, "what is the answer?")
    assert "# Mullet Conversation Export" in md
    assert "what is the answer?" in md
    assert "local" in md
    assert "42" in md
    assert "Tool call" in md or "Tool Result" in md


def test_export_markdown_to_file(tmp_path):
    result = _make_result()
    path = str(tmp_path / "export.md")
    export_markdown(result, "test", path)
    md = (tmp_path / "export.md").read_text()
    assert "# Mullet" in md


def test_export_includes_metadata():
    result = _make_result()
    output = export_json(result, "test")
    data = json.loads(output)
    assert "mullet_version" in data
    assert "exported_at" in data
    assert data["tier"] == "LOW"
