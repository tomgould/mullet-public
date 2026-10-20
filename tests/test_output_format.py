from mullet.core.agent.output_format import AgentOutput

def test_agent_output_success_no_changes_no_commands():
    output = AgentOutput(
        response_text="Task completed successfully.",
        files_changed=[],
        commands_run=[],
        tokens_used=100,
        duration_s=1.5,
        success=True
    )
    summary = output.format_summary()
    assert "Agent Run SUCCESS in 1.50 seconds using 100 tokens." in summary
    assert "Response: Task completed successfully." in summary
    assert "Files Changed:" not in summary
    assert "Commands Run:" not in summary

def test_agent_output_failure_with_changes_and_commands():
    output = AgentOutput(
        response_text="Failed to complete task.",
        files_changed=["file1.py", "file2.txt"],
        commands_run=["ls -l", "git status"],
        tokens_used=200,
        duration_s=10.0,
        success=False
    )
    summary = output.format_summary()
    assert "Agent Run FAILURE in 10.00 seconds using 200 tokens." in summary
    assert "Response: Failed to complete task." in summary
    assert "Files Changed:" in summary
    assert "  - file1.py" in summary
    assert "  - file2.txt" in summary
    assert "Commands Run:" in summary
    assert "  - ls -l" in summary
    assert "  - git status" in summary

def test_agent_output_no_response_text():
    output = AgentOutput(
        response_text="",
        files_changed=[],
        commands_run=[],
        tokens_used=50,
        duration_s=0.5,
        success=True
    )
    summary = output.format_summary()
    assert "Agent Run SUCCESS in 0.50 seconds using 50 tokens." in summary
    assert "Response:" not in summary

def test_agent_output_only_files_changed():
    output = AgentOutput(
        response_text="",
        files_changed=["new_file.py"],
        commands_run=[],
        tokens_used=75,
        duration_s=2.1,
        success=True
    )
    summary = output.format_summary()
    assert "Agent Run SUCCESS in 2.10 seconds using 75 tokens." in summary
    assert "Files Changed:" in summary
    assert "  - new_file.py" in summary
    assert "Commands Run:" not in summary

def test_agent_output_only_commands_run():
    output = AgentOutput(
        response_text="",
        files_changed=[],
        commands_run=["pytest"],
        tokens_used=120,
        duration_s=3.3,
        success=True
    )
    summary = output.format_summary()
    assert "Agent Run SUCCESS in 3.30 seconds using 120 tokens." in summary
    assert "Commands Run:" in summary
    assert "  - pytest" in summary
    assert "Files Changed:" not in summary

def test_agent_output_zero_tokens_and_duration():
    output = AgentOutput(
        response_text="Empty run.",
        files_changed=[],
        commands_run=[],
        tokens_used=0,
        duration_s=0.0,
        success=True
    )
    summary = output.format_summary()
    assert "Agent Run SUCCESS in 0.00 seconds using 0 tokens." in summary
    assert "Response: Empty run." in summary

def test_agent_output_multiple_files_and_commands():
    output = AgentOutput(
        response_text="Complex operation completed.",
        files_changed=["a.txt", "b.md", "c.json"],
        commands_run=["cmd1", "cmd2 -f", "cmd3 arg"],
        tokens_used=300,
        duration_s=15.75,
        success=True
    )
    summary = output.format_summary()
    assert "Agent Run SUCCESS in 15.75 seconds using 300 tokens." in summary
    assert "Response: Complex operation completed." in summary
    assert "Files Changed:" in summary
    assert "  - a.txt" in summary
    assert "  - b.md" in summary
    assert "  - c.json" in summary
    assert "Commands Run:" in summary
    assert "  - cmd1" in summary
    assert "  - cmd2 -f" in summary
    assert "  - cmd3 arg" in summary

def test_agent_output_failure_no_details():
    output = AgentOutput(
        response_text="",
        files_changed=[],
        commands_run=[],
        tokens_used=10,
        duration_s=0.1,
        success=False
    )
    summary = output.format_summary()
    assert "Agent Run FAILURE in 0.10 seconds using 10 tokens." in summary
    assert "Response:" not in summary
    assert "Files Changed:" not in summary
    assert "Commands Run:" not in summary
