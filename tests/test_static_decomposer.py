"""Tests for StaticDecomposer."""
from mullet.core.orchestration.static_decomposer import StaticDecomposer
from mullet.core.orchestration.swarm_types import SwarmContext


class TestStaticDecomposer:
    """Test cases for StaticDecomposer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.decomposer = StaticDecomposer()
        
    def test_decompose_files_list(self):
        """Test that a list of files produces correct tasks."""
        files = ["file1.py", "file2.py", "file3.py"]
        context = SwarmContext(files=files)
        
        tasks = self.decomposer.decompose("analyze code", context)
        
        assert len(tasks) == 3
        assert tasks[0].id == "task-000"
        assert tasks[1].id == "task-001"
        assert tasks[2].id == "task-002"
        
        # Check context_refs for files
        for task, file in zip(tasks, files):
            assert task.context_refs == [file]
            
    def test_decompose_items_list(self):
        """Test that a list of items produces correct tasks."""
        items = ["feature A", "feature B"]
        context = SwarmContext(items=items)
        
        tasks = self.decomposer.decompose("describe feature", context)
        
        assert len(tasks) == 2
        assert tasks[0].id == "task-000"
        assert tasks[1].id == "task-001"
        
        # Items don't get context_refs
        for task in tasks:
            assert task.context_refs == []
            
    def test_decompose_empty_returns_empty(self):
        """Test that empty context returns empty list."""
        context = SwarmContext()
        
        tasks = self.decomposer.decompose("do something", context)
        
        assert tasks == []
        
    def test_decompose_files_takes_priority_over_items(self):
        """Test that files take priority over items."""
        context = SwarmContext(
            files=["file1.py", "file2.py"],
            items=["item1", "item2", "item3"]
        )
        
        tasks = self.decomposer.decompose("analyze", context)
        
        assert len(tasks) == 2
        # Should use files, not items
        assert "file1.py" in tasks[0].instruction
        assert "file2.py" in tasks[1].instruction
        
    def test_decompose_task_ids_sequential(self):
        """Test that task IDs are sequential with zero-padding."""
        files = ["a.py", "b.py", "c.py", "d.py", "e.py", "f.py", "g.py", "h.py", "i.py", "j.py"]
        context = SwarmContext(files=files)
        
        tasks = self.decomposer.decompose("test", context)
        
        assert len(tasks) == 10
        expected_ids = ["task-000", "task-001", "task-002", "task-003", "task-004",
                       "task-005", "task-006", "task-007", "task-008", "task-009"]
        actual_ids = [task.id for task in tasks]
        assert actual_ids == expected_ids
        
    def test_decompose_instruction_contains_goal(self):
        """Test that instruction starts with the goal."""
        goal = "Generate comprehensive documentation"
        files = ["main.py"]
        context = SwarmContext(files=files)
        
        tasks = self.decomposer.decompose(goal, context)
        
        assert len(tasks) == 1
        assert tasks[0].instruction.startswith(goal)
        
    def test_decompose_instruction_contains_target(self):
        """Test that instruction contains the target path."""
        target = "src/components/button.py"
        context = SwarmContext(files=[target])
        
        tasks = self.decomposer.decompose("document", context)
        
        assert len(tasks) == 1
        assert f"Target: {target}" in tasks[0].instruction
        
    def test_decompose_output_shape_forwarded(self):
        """Test that custom output_shape is forwarded to SwarmTask."""
        custom_shape = "structured JSON"
        context = SwarmContext(
            files=["test.py"],
            output_shape=custom_shape
        )
        
        tasks = self.decomposer.decompose("analyze", context)
        
        assert len(tasks) == 1
        assert tasks[0].expected_output_shape == custom_shape
        
    def test_decompose_deduplicates_files(self):
        """Test that duplicate files are removed."""
        files = ["file1.py", "file2.py", "file1.py", "file3.py", "file2.py"]
        context = SwarmContext(files=files)
        
        tasks = self.decomposer.decompose("analyze", context)
        
        # Should have 3 unique files
        assert len(tasks) == 3
        
        # Check that all tasks have unique targets
        targets = [task.instruction.split("Target: ")[1] for task in tasks]
        assert len(set(targets)) == 3
        assert "file1.py" in targets
        assert "file2.py" in targets
        assert "file3.py" in targets
        
    def test_decompose_context_refs_only_for_files(self):
        """Test that context_refs are only set for files, not items."""
        # Test with files
        file_context = SwarmContext(files=["file1.py"])
        file_tasks = self.decomposer.decompose("analyze", file_context)
        assert file_tasks[0].context_refs == ["file1.py"]
        
        # Test with items
        item_context = SwarmContext(items=["some description"])
        item_tasks = self.decomposer.decompose("analyze", item_context)
        assert item_tasks[0].context_refs == []
        
    def test_decompose_deduplication_preserves_order(self):
        """Test that deduplication preserves original order."""
        files = ["first.py", "second.py", "first.py", "third.py", "second.py"]
        context = SwarmContext(files=files)
        
        tasks = self.decomposer.decompose("analyze", context)
        
        # Should preserve order of first occurrence
        expected_order = ["first.py", "second.py", "third.py"]
        actual_order = [task.instruction.split("Target: ")[1] for task in tasks]
        assert actual_order == expected_order