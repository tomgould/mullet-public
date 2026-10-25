"""Tests for llm_decomposer pure helpers."""

import pytest

from mullet.core.orchestration.llm_decomposer import (
    DecompositionError,
    _extract_json_array,
    _parse_task_list,
)


# ---------------------------------------------------------------------------
# DecompositionError
# ---------------------------------------------------------------------------

class TestDecompositionError:
    def test_is_exception(self):
        assert isinstance(DecompositionError("oops"), Exception)

    def test_message_preserved(self):
        e = DecompositionError("bad json")
        assert "bad json" in str(e)

    def test_can_be_raised(self):
        with pytest.raises(DecompositionError):
            raise DecompositionError("test")


# ---------------------------------------------------------------------------
# _extract_json_array
# ---------------------------------------------------------------------------

class TestExtractJsonArray:
    def test_simple_array(self):
        result = _extract_json_array('[{"id": "t1"}]')
        assert result == '[{"id": "t1"}]'

    def test_array_with_prefix_text(self):
        result = _extract_json_array('Here is the response:\n[{"id": "t1"}]')
        assert result.startswith('[')
        assert result.endswith(']')

    def test_nested_objects_preserved(self):
        raw = '[{"id": "t1", "data": {"key": "val"}}]'
        result = _extract_json_array(raw)
        assert '"key"' in result

    def test_empty_array_returned(self):
        result = _extract_json_array('[]')
        assert result == '[]'

    def test_no_bracket_raises(self):
        with pytest.raises(DecompositionError):
            _extract_json_array("no brackets here")

    def test_unclosed_bracket_raises(self):
        with pytest.raises(DecompositionError):
            _extract_json_array("[{unclosed}")

    def test_string_with_bracket_not_counted(self):
        raw = '[{"id": "t[1]"}]'
        result = _extract_json_array(raw)
        assert result == raw

    def test_trailing_text_after_array_ignored(self):
        raw = '[{"id": "t1"}] extra text here'
        result = _extract_json_array(raw)
        assert result == '[{"id": "t1"}]'

    def test_returns_string(self):
        assert isinstance(_extract_json_array('[1,2,3]'), str)


# ---------------------------------------------------------------------------
# _parse_task_list
# ---------------------------------------------------------------------------

class TestParseTaskList:
    DEFAULT_SHAPE = "markdown report"

    def test_valid_json_array_parsed(self):
        raw = '[{"id": "t1", "instruction": "Do thing"}]'
        tasks = _parse_task_list(raw, self.DEFAULT_SHAPE)
        assert len(tasks) == 1
        assert tasks[0].id == "t1"
        assert tasks[0].instruction == "Do thing"

    def test_multiple_tasks_parsed(self):
        raw = '[{"id": "a", "instruction": "Task A"}, {"id": "b", "instruction": "Task B"}]'
        tasks = _parse_task_list(raw, self.DEFAULT_SHAPE)
        assert len(tasks) == 2

    def test_default_output_shape_applied(self):
        raw = '[{"id": "t1", "instruction": "Do thing"}]'
        tasks = _parse_task_list(raw, "json output")
        assert tasks[0].expected_output_shape == "json output"

    def test_explicit_output_shape_preserved(self):
        raw = '[{"id": "t1", "instruction": "Do", "expected_output_shape": "yaml"}]'
        tasks = _parse_task_list(raw, self.DEFAULT_SHAPE)
        assert tasks[0].expected_output_shape == "yaml"

    def test_context_refs_default_empty(self):
        raw = '[{"id": "t1", "instruction": "Do"}]'
        tasks = _parse_task_list(raw, self.DEFAULT_SHAPE)
        assert tasks[0].context_refs == []

    def test_context_refs_preserved(self):
        raw = '[{"id": "t1", "instruction": "Do", "context_refs": ["a.py"]}]'
        tasks = _parse_task_list(raw, self.DEFAULT_SHAPE)
        assert "a.py" in tasks[0].context_refs

    def test_empty_array_returns_empty_list(self):
        tasks = _parse_task_list('[]', self.DEFAULT_SHAPE)
        assert tasks == []

    def test_markdown_fence_stripped(self):
        raw = '```json\n[{"id": "t1", "instruction": "x"}]\n```'
        tasks = _parse_task_list(raw, self.DEFAULT_SHAPE)
        assert len(tasks) == 1

    def test_missing_id_raises(self):
        raw = '[{"instruction": "no id"}]'
        with pytest.raises(DecompositionError):
            _parse_task_list(raw, self.DEFAULT_SHAPE)

    def test_missing_instruction_raises(self):
        raw = '[{"id": "t1"}]'
        with pytest.raises(DecompositionError):
            _parse_task_list(raw, self.DEFAULT_SHAPE)

    def test_non_array_raises(self):
        raw = '{"id": "t1", "instruction": "x"}'
        with pytest.raises(DecompositionError):
            _parse_task_list(raw, self.DEFAULT_SHAPE)

    def test_invalid_json_raises(self):
        with pytest.raises(DecompositionError):
            _parse_task_list("[not valid json}", self.DEFAULT_SHAPE)

    def test_returns_list(self):
        result = _parse_task_list('[]', self.DEFAULT_SHAPE)
        assert isinstance(result, list)
