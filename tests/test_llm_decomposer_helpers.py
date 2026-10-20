"""Tests for mullet.core.orchestration.llm_decomposer — pure parsing helpers."""

import json
import pytest

from mullet.core.orchestration.llm_decomposer import (
    DecompositionError,
    _extract_json_array,
    _parse_task_list,
)
from mullet.core.orchestration.swarm_types import SwarmTask


# ---------------------------------------------------------------------------
# _extract_json_array
# ---------------------------------------------------------------------------

class TestExtractJsonArray:
    def test_simple_array(self):
        raw = '[{"id": "t1", "instruction": "do something"}]'
        assert _extract_json_array(raw) == raw

    def test_array_with_preamble(self):
        raw = 'Here is the array:\n[{"id": "t1"}]'
        result = _extract_json_array(raw)
        assert result.startswith("[")
        assert result.endswith("]")

    def test_array_with_trailing_notes(self):
        raw = '[{"id": "t1"}]\nSome notes here with [brackets] ignored.'
        result = _extract_json_array(raw)
        assert result == '[{"id": "t1"}]'

    def test_nested_objects(self):
        raw = '[{"id": "t1", "data": {"key": "value"}}]'
        result = _extract_json_array(raw)
        parsed = json.loads(result)
        assert parsed[0]["data"]["key"] == "value"

    def test_empty_array(self):
        result = _extract_json_array("[]")
        assert result == "[]"

    def test_empty_array_with_whitespace(self):
        result = _extract_json_array("  [ ]  ")
        assert result.strip() == "[ ]"

    def test_string_containing_brackets_not_confused(self):
        raw = '[{"id": "t1", "instruction": "handle [case] here"}]'
        result = _extract_json_array(raw)
        parsed = json.loads(result)
        assert "[case]" in parsed[0]["instruction"]

    def test_escaped_quotes_in_string(self):
        raw = r'[{"id": "t1", "instruction": "say \"hello\""}]'
        result = _extract_json_array(raw)
        parsed = json.loads(result)
        assert 'say "hello"' in parsed[0]["instruction"]

    def test_no_array_raises(self):
        with pytest.raises(DecompositionError, match="No JSON array"):
            _extract_json_array('{"id": "t1"}')

    def test_unclosed_array_raises(self):
        with pytest.raises(DecompositionError, match="Unclosed JSON array"):
            _extract_json_array('[{"id": "t1"')

    def test_multiple_top_level_arrays_takes_first(self):
        raw = '[{"id": "t1"}] [{"id": "t2"}]'
        result = _extract_json_array(raw)
        parsed = json.loads(result)
        assert parsed[0]["id"] == "t1"

    def test_markdown_fenced_array(self):
        raw = '```json\n[{"id": "t1"}]\n```'
        # _extract_json_array doesn't strip fences — it finds the [ directly
        result = _extract_json_array(raw)
        assert "[" in result


# ---------------------------------------------------------------------------
# _parse_task_list
# ---------------------------------------------------------------------------

class TestParseTaskList:
    def test_valid_single_task(self):
        raw = json.dumps([{"id": "t1", "instruction": "Do auth"}])
        tasks = _parse_task_list(raw, "markdown report")
        assert len(tasks) == 1
        assert tasks[0].id == "t1"
        assert tasks[0].instruction == "Do auth"

    def test_valid_multiple_tasks(self):
        raw = json.dumps([
            {"id": "t1", "instruction": "Task 1"},
            {"id": "t2", "instruction": "Task 2"},
        ])
        tasks = _parse_task_list(raw, "report")
        assert len(tasks) == 2

    def test_returns_swarm_task_objects(self):
        raw = json.dumps([{"id": "t1", "instruction": "X"}])
        tasks = _parse_task_list(raw, "report")
        assert isinstance(tasks[0], SwarmTask)

    def test_default_output_shape_applied(self):
        raw = json.dumps([{"id": "t1", "instruction": "X"}])
        tasks = _parse_task_list(raw, "json output")
        assert tasks[0].expected_output_shape == "json output"

    def test_explicit_output_shape_overrides_default(self):
        raw = json.dumps([{
            "id": "t1",
            "instruction": "X",
            "expected_output_shape": "EDIT blocks",
        }])
        tasks = _parse_task_list(raw, "report")
        assert tasks[0].expected_output_shape == "EDIT blocks"

    def test_context_refs_populated(self):
        raw = json.dumps([{
            "id": "t1",
            "instruction": "X",
            "context_refs": ["auth.py", "models.py"],
        }])
        tasks = _parse_task_list(raw, "report")
        assert tasks[0].context_refs == ["auth.py", "models.py"]

    def test_missing_context_refs_defaults_empty(self):
        raw = json.dumps([{"id": "t1", "instruction": "X"}])
        tasks = _parse_task_list(raw, "report")
        assert tasks[0].context_refs == []

    def test_empty_array_returns_empty_list(self):
        tasks = _parse_task_list("[]", "report")
        assert tasks == []

    def test_markdown_fence_stripped(self):
        raw = "```json\n" + json.dumps([{"id": "t1", "instruction": "X"}]) + "\n```"
        tasks = _parse_task_list(raw, "report")
        assert len(tasks) == 1

    def test_markdown_fence_without_lang_stripped(self):
        raw = "```\n" + json.dumps([{"id": "t1", "instruction": "X"}]) + "\n```"
        tasks = _parse_task_list(raw, "report")
        assert len(tasks) == 1

    def test_trailing_notes_after_array_ignored(self):
        raw = json.dumps([{"id": "t1", "instruction": "X"}]) + "\nHere are some notes."
        tasks = _parse_task_list(raw, "report")
        assert len(tasks) == 1

    def test_missing_id_raises(self):
        raw = json.dumps([{"instruction": "X"}])
        with pytest.raises(DecompositionError, match="missing required keys"):
            _parse_task_list(raw, "report")

    def test_missing_instruction_raises(self):
        raw = json.dumps([{"id": "t1"}])
        with pytest.raises(DecompositionError, match="missing required keys"):
            _parse_task_list(raw, "report")

    def test_non_dict_item_raises(self):
        raw = json.dumps(["not a dict"])
        with pytest.raises(DecompositionError, match="not an object"):
            _parse_task_list(raw, "report")

    def test_not_array_raises(self):
        raw = json.dumps({"id": "t1", "instruction": "X"})
        with pytest.raises(DecompositionError):
            _parse_task_list(raw, "report")

    def test_invalid_json_raises(self):
        with pytest.raises(DecompositionError):
            _parse_task_list("[not valid json", "report")


# ---------------------------------------------------------------------------
# DecompositionError
# ---------------------------------------------------------------------------

class TestDecompositionError:
    def test_is_exception(self):
        assert issubclass(DecompositionError, Exception)

    def test_message_preserved(self):
        err = DecompositionError("something failed")
        assert "something failed" in str(err)
