"""Additional tests for auto_swarm enhancements (file-size + file-mention detection)."""


from mullet.core.router.auto_swarm import (
    _should_swarm,
    _detect_file_mentions_in_task,
)


def test_detect_file_mentions_in_task_five_paths():
    task = "Please review these files: src/a.py, src/b.py, c.py, tests/d.py, and e.py"
    assert _detect_file_mentions_in_task(task) >= 5


def test_detect_file_mentions_in_task_none():
    task = "What is the meaning of life?"
    assert _detect_file_mentions_in_task(task) == 0


def test_detect_file_mentions_ignores_words_with_dots():
    task = "The end of the story. This is nice. Not really."
    assert _detect_file_mentions_in_task(task) == 0


def test_should_swarm_task_mentions_many_files_without_param():
    """Caller forgot files=[...] but the task lists many file paths — fire swarm."""
    task = "Review these: src/foo.py src/bar.py src/baz.py src/qux.py src/quux.py"
    fire, reason = _should_swarm(task, files=[])
    assert fire is True
    assert "mentions" in reason.lower()


def test_should_swarm_no_files_no_mentions():
    """Minimal prompt with no files and no mentions — don't fire."""
    fire, reason = _should_swarm("Hello, world.", files=[])
    assert fire is False
    assert reason == ""


def test_should_swarm_includes_file_sizes_in_prompt_len(tmp_path):
    """Large file contents should push prompt_len past the threshold even with file_count=1."""
    big = tmp_path / "big.py"
    big.write_text("x = 1\n" * 2000)  # ~12KB, beyond the 8000 threshold
    fire, reason = _should_swarm("Short task", files=[str(big)])
    assert fire is True
    assert "prompt_len" in reason


def test_should_swarm_file_count_still_wins():
    """Five files always trip the heuristic regardless of size or tier."""
    files = [f"f{i}.py" for i in range(5)]
    fire, reason = _should_swarm("task", files=files)
    assert fire is True
    assert "file_count" in reason
