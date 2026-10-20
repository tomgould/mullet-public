"""Tests for llm_decomposer pure helpers: _extract_json_array, _parse_task_list, DECOMPOSE_PROMPT."""

import pytest

from mullet.core.orchestration.llm_decomposer import (
    DecompositionError,
    _extract_json_array,
    _parse_task_list,
    DECOMPOSE_PROMPT,
)


# ---------------------------------------------------------------------------
# _extract_json_array
# ---------------------------------------------------------------------------

class TestExtractJsonArray:
    def test_simple_array_extracted(self):
        raw = '[{"id": "t1", "instruction": "do it"}]'
        result = _extract_json_array(raw)
        assert result == raw

    def test_trailing_text_ignored(self):
        raw = '[{"a": 1}] trailing notes here'
        result = _extract_json_array(raw)
        assert result == '[{"a": 1}]'

    def test_leading_text_ignored(self):
        raw = 'some intro text\n[{"a": 1}]'
        result = _extract_json_array(raw)
        assert result == '[{"a": 1}]'

    def test_nested_brackets_handled(self):
        raw = '[{"refs": ["file.py", "other.py"]}]'
        result = _extract_json_array(raw)
        assert result == raw

    def test_second_array_after_trailing_excluded(self):
        raw = '[{"a": 1}] notes [not this one]'
        result = _extract_json_array(raw)
        assert result == '[{"a": 1}]'

    def test_empty_array_returned(self):
        result = _extract_json_array('[]')
        assert result == '[]'

    def test_no_array_raises(self):
        with pytest.raises(DecompositionError):
            _extract_json_array('no array here')

    def test_unclosed_array_raises(self):
        with pytest.raises(DecompositionError):
            _extract_json_array('[{"a": 1}')

    def test_string_with_brackets_not_confused(self):
        raw = '[{"msg": "this has [brackets] inside strings"}]'
        result = _extract_json_array(raw)
        assert result == raw

    def test_returns_string(self):
        result = _extract_json_array('[{"x": 1}]')
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _parse_task_list
# ---------------------------------------------------------------------------

class TestParseTaskList:
    def _one_task(self, extra=None):
        task = {"id": "t1", "instruction": "do the thing"}
        if extra:
            task.update(extra)
        import json
        return json.dumps([task])

    def test_returns_list(self):
        result = _parse_task_list(self._one_task(), "report")
        assert isinstance(result, list)

    def test_empty_array_returns_empty(self):
        result = _parse_task_list("[]", "report")
        assert result == []

    def test_task_id_set(self):
        tasks = _parse_task_list(self._one_task(), "report")
        assert tasks[0].id == "t1"

    def test_task_instruction_set(self):
        tasks = _parse_task_list(self._one_task(), "report")
        assert tasks[0].instruction == "do the thing"

    def test_default_output_shape_used_when_absent(self):
        tasks = _parse_task_list(self._one_task(), "my default shape")
        assert tasks[0].expected_output_shape == "my default shape"

    def test_explicit_output_shape_preserved(self):
        tasks = _parse_task_list(self._one_task({"expected_output_shape": "json blob"}), "default")
        assert tasks[0].expected_output_shape == "json blob"

    def test_context_refs_defaults_empty(self):
        tasks = _parse_task_list(self._one_task(), "report")
        assert tasks[0].context_refs == []

    def test_context_refs_set_when_present(self):
        tasks = _parse_task_list(self._one_task({"context_refs": ["a.py", "b.py"]}), "report")
        assert tasks[0].context_refs == ["a.py", "b.py"]

    def test_markdown_fence_stripped(self):
        fenced = '```json\n[{"id": "t1", "instruction": "test"}]\n```'
        tasks = _parse_task_list(fenced, "report")
        assert len(tasks) == 1
        assert tasks[0].id == "t1"

    def test_missing_id_raises(self):
        raw = '[{"instruction": "no id"}]'
        with pytest.raises(DecompositionError):
            _parse_task_list(raw, "report")

    def test_missing_instruction_raises(self):
        import json
        raw = json.dumps([{"id": "t1"}])
        with pytest.raises(DecompositionError):
            _parse_task_list(raw, "report")

    def test_invalid_json_raises(self):
        with pytest.raises(DecompositionError):
            _parse_task_list('[{invalid json}]', "report")

    def test_multiple_tasks_parsed(self):
        import json
        raw = json.dumps([
            {"id": "t1", "instruction": "first"},
            {"id": "t2", "instruction": "second"},
        ])
        tasks = _parse_task_list(raw, "report")
        assert len(tasks) == 2

    def test_task_ids_preserved(self):
        import json
        raw = json.dumps([
            {"id": "step-001", "instruction": "a"},
            {"id": "step-002", "instruction": "b"},
        ])
        tasks = _parse_task_list(raw, "report")
        assert [t.id for t in tasks] == ["step-001", "step-002"]


# ---------------------------------------------------------------------------
# DECOMPOSE_PROMPT
# ---------------------------------------------------------------------------

class TestDecomposePrompt:
    def test_is_string(self):
        assert isinstance(DECOMPOSE_PROMPT, str)

    def test_nonempty(self):
        assert len(DECOMPOSE_PROMPT) > 100

    def test_has_goal_placeholder(self):
        assert "{goal}" in DECOMPOSE_PROMPT

    def test_has_max_placeholder(self):
        assert "{max}" in DECOMPOSE_PROMPT

    def test_has_context_section_placeholder(self):
        assert "{context_section}" in DECOMPOSE_PROMPT

    def test_references_json_array(self):
        assert "JSON" in DECOMPOSE_PROMPT.upper() or "json" in DECOMPOSE_PROMPT

    def test_mentions_id_key(self):
        assert '"id"' in DECOMPOSE_PROMPT

    def test_mentions_instruction_key(self):
        assert '"instruction"' in DECOMPOSE_PROMPT
