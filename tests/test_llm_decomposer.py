"""Tests for LLMDecomposer module."""

import pytest
from unittest.mock import Mock

from mullet.core.orchestration.llm_decomposer import (
    LLMDecomposer,
    DecompositionError,
    _parse_task_list,
    DECOMPOSE_PROMPT
)
from mullet.core.orchestration.swarm_types import SwarmContext


class TestParseTaskList:
    """Tests for _parse_task_list function."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        raw = '[{"id": "t1", "instruction": "task 1"}, {"id": "t2", "instruction": "task 2"}]'
        tasks = _parse_task_list(raw, "markdown report")
        
        assert len(tasks) == 2
        assert tasks[0].id == "t1"
        assert tasks[0].instruction == "task 1"
        assert tasks[0].expected_output_shape == "markdown report"
        assert tasks[1].id == "t2"
        assert tasks[1].instruction == "task 2"

    def test_parse_with_output_shape_override(self):
        """Test that output shape from items overrides default."""
        raw = '[{"id": "t1", "instruction": "task 1", "expected_output_shape": "json"}]'
        tasks = _parse_task_list(raw, "markdown report")
        
        assert tasks[0].expected_output_shape == "json"


class TestLLMDecomposer:
    """Tests for LLMDecomposer class."""

    def test_decompose_valid_json(self):
        """Test decomposition with valid JSON response."""
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = '[{"id": "analyze", "instruction": "Analyze data"}, {"id": "report", "instruction": "Write report"}]'
        mock_provider.create.return_value = mock_response
        
        decomposer = LLMDecomposer(mock_provider)
        context = SwarmContext()
        result = decomposer.decompose("Analyze customer feedback", context)
        
        assert len(result) == 2
        assert result[0].id == "analyze"
        assert result[0].instruction == "Analyze data"
        assert result[1].id == "report"
        assert result[1].instruction == "Write report"

    def test_decompose_empty_array(self):
        """Test decomposition returns empty array when not swarmable."""
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = '[]'
        mock_provider.create.return_value = mock_response
        
        decomposer = LLMDecomposer(mock_provider)
        context = SwarmContext()
        result = decomposer.decompose("Simple task", context)
        
        assert result == []

    def test_decompose_json_in_prose(self):
        """Test JSON extraction from prose response."""
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = '''
        Based on your goal, here are the sub-tasks:
        [{"id": "step1", "instruction": "First step", "expected_output_shape": "json"}]
        These tasks should cover the requirements.
        '''
        mock_provider.create.return_value = mock_response
        
        decomposer = LLMDecomposer(mock_provider)
        context = SwarmContext()
        result = decomposer.decompose("Complex goal", context)
        
        assert len(result) == 1
        assert result[0].id == "step1"
        assert result[0].expected_output_shape == "json"

    def test_decompose_invalid_json_raises(self):
        """Test that invalid JSON raises DecompositionError."""
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = 'This is not JSON at all'
        mock_provider.create.return_value = mock_response
        
        decomposer = LLMDecomposer(mock_provider)
        context = SwarmContext()
        
        with pytest.raises(DecompositionError):
            decomposer.decompose("Test goal", context)

    def test_decompose_missing_instruction_key_raises(self):
        """Test that missing instruction key raises DecompositionError."""
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = '[{"id": "t1"}]'  # Missing instruction
        mock_provider.create.return_value = mock_response
        
        decomposer = LLMDecomposer(mock_provider)
        context = SwarmContext()
        
        with pytest.raises(DecompositionError):
            decomposer.decompose("Test goal", context)

    def test_decompose_caps_at_max_subtasks(self):
        """Test that tasks are capped at max_subtasks."""
        mock_provider = Mock()
        mock_response = Mock()
        # Create 25 tasks
        tasks = [{"id": f"t{i}", "instruction": f"task {i}"} for i in range(25)]
        mock_response.content = str(tasks).replace("'", '"')
        mock_provider.create.return_value = mock_response
        
        decomposer = LLMDecomposer(mock_provider, max_subtasks=5)
        context = SwarmContext()
        result = decomposer.decompose("Big goal", context)
        
        assert len(result) == 5

    def test_decompose_output_shape_from_context(self):
        """Test output shape defaults to context.output_shape."""
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = '[{"id": "t1", "instruction": "task"}]'
        mock_provider.create.return_value = mock_response
        
        decomposer = LLMDecomposer(mock_provider)
        context = SwarmContext(output_shape="unified diff")
        result = decomposer.decompose("Test", context)
        
        assert result[0].expected_output_shape == "unified diff"

    def test_decompose_output_shape_from_item(self):
        """Test output shape from item overrides context."""
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = '[{"id": "t1", "instruction": "task", "expected_output_shape": "json"}]'
        mock_provider.create.return_value = mock_response
        
        decomposer = LLMDecomposer(mock_provider)
        context = SwarmContext(output_shape="markdown report")
        result = decomposer.decompose("Test", context)
        
        assert result[0].expected_output_shape == "json"

    def test_decompose_prompt_contains_goal(self):
        """Test that the prompt sent to create() contains the goal."""
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = '[]'
        mock_provider.create.return_value = mock_response
        
        decomposer = LLMDecomposer(mock_provider)
        context = SwarmContext()
        decomposer.decompose("My special test goal", context)
        
        # Check the arguments passed to create
        create_call_args = mock_provider.create.call_args
        messages = create_call_args[1]['messages']
        prompt = messages[0]['content']
        
        assert "My special test goal" in prompt
        assert "My special test goal" in DECOMPOSE_PROMPT.format(goal="My special test goal", max=20, context_section="")