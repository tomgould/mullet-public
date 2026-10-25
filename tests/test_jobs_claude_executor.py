"""Tests for the Claude Code executor, using a fake `claude` script.

The fake records how it was called and replays a scripted stream, so these
tests check the command line, settings, environment and result mapping
without calling Anthropic.
"""

import json
import os
import stat
import time
from pathlib import Path

import pytest

from mullet.jobs.executors.base import Budget, NodeContext, Task
from mullet.jobs.executors.claude_code import (
    ClaudeCodeExecutor, ExecutorUnavailable, build_prompt,
)
from mullet.jobs.types import NodeSpec

from tests.jobs_helpers import FAKE_CLAUDE

@pytest.fixture
def fake(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "claude"
    script.write_text(FAKE_CLAUDE)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)

    def scenario(**kwargs):
        (bin_dir / "scenario.json").write_text(json.dumps(kwargs))

    def call():
        return json.loads((bin_dir / "call.json").read_text())

    scenario(structured={"status": "done", "summary": "did it"})
    return {"binary": str(script), "scenario": scenario, "call": call}


@pytest.fixture
def work(tmp_path):
    w = tmp_path / "work"
    w.mkdir()
    return w


def make(tmp_path, fake, **kwargs):
    return ClaudeCodeExecutor(tmp_path / "logs", binary=fake["binary"],
                              bash_commands=["python3 runtests.py"], **kwargs)


TASK = Task("Add subtraction", NodeSpec(id="n1", goal="add sub()", scope=["app.py", "tests/**"],
                                         acceptance=["python3 runtests.py"]))


def test_command_has_every_safety_flag_and_prompt_goes_on_stdin(tmp_path, fake, work):
    make(tmp_path, fake).run(TASK, work, Budget(max_turns=7))
    call = fake["call"]()
    argv = call["argv"]
    for flag in ("-p", "--restricted", "--strict-mcp-config", "--disable-slash-commands"):
        assert flag in argv
    # A worker's session is kept so a failed attempt can be resumed.
    assert "--no-session-persistence" not in argv and "--resume" not in argv
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert argv[argv.index("--permission-prompts") + 1] == "none"
    assert argv[argv.index("--max-turns") + 1] == "7"
    assert argv[argv.index("--tools") + 1] == "Read,Edit,Write,Glob,Grep,Bash"
    assert "add sub()" in call["prompt"]
    assert not any("add sub()" in a for a in argv)
    assert call["cwd"] == str(work)


def test_settings_turn_the_sandbox_on_fail_closed(tmp_path, fake, work):
    make(tmp_path, fake).run(TASK, work, Budget())
    settings = json.loads((tmp_path / "logs" / "claude-001" / "settings.json").read_text())
    sandbox = settings["sandbox"]
    assert sandbox["enabled"] and sandbox["failIfUnavailable"]
    assert sandbox["allowUnsandboxedCommands"] is False
    assert sandbox["autoAllowBashIfSandboxed"] is False
    assert sandbox["network"] == {"allowedDomains": [], "strictAllowlist": True}
    assert sandbox["filesystem"]["denyRead"][0] == "~/"
    assert str(work) in sandbox["filesystem"]["allowRead"]
    allow = settings["permissions"]["allow"]
    assert "Edit(./**)" in allow and "Bash(python3 runtests.py:*)" in allow
    assert not any(rule in ("Bash", "Bash(*)") for rule in allow)
    assert settings["permissions"]["blockReadsOutsideWorkingDirectories"] is True


def test_environment_never_carries_provider_keys(tmp_path, fake, work, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-pass")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-should-not-pass")
    monkeypatch.setenv("MULLET_DELEGATION", "on")
    make(tmp_path, fake).run(TASK, work, Budget())
    env = fake["call"]()["env"]
    assert "ANTHROPIC_API_KEY" not in env
    assert "OPENROUTER_API_KEY" not in env
    assert "MULLET_DELEGATION" not in env
    assert env["HOME"] == os.environ["HOME"]


def test_each_attempt_gets_a_short_private_tmpdir_that_is_removed(tmp_path, fake, work):
    make(tmp_path, fake).run(TASK, work, Budget())
    tmpdir = fake["call"]()["env"]["TMPDIR"]
    # Short enough for the sandbox's Unix sockets, and not the shared /tmp/claude-<uid>.
    assert tmpdir.startswith("/tmp/mullet-") and len(tmpdir) < 40
    assert not Path(tmpdir).exists()


def test_api_billing_passes_only_the_anthropic_key(tmp_path, fake, work, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-for-api-billing")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-should-not-pass")
    make(tmp_path, fake, billing="api").run(TASK, work, Budget())
    env = fake["call"]()["env"]
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-for-api-billing"
    assert "OPENROUTER_API_KEY" not in env


def test_api_billing_without_a_key_refuses(tmp_path, fake, work, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ExecutorUnavailable):
        make(tmp_path, fake, billing="api").run(TASK, work, Budget())


def test_success_returns_the_claim_usage_and_transcript(tmp_path, fake, work):
    fake["scenario"](write={"app.py": "def sub(a, b):\n    return a - b\n"},
                     structured={"status": "done", "summary": "added sub"})
    result = make(tmp_path, fake).run(TASK, work, Budget())
    assert result.exit_code == 0 and result.error is None
    assert result.claim == {"status": "done", "summary": "added sub"}
    assert result.usage["total_cost_usd"] == 0.12 and result.usage["num_turns"] == 3
    assert result.usage["api_key_source"] == "none"
    assert (work / "app.py").exists()
    lines = Path(result.transcript_path).read_text().splitlines()
    assert json.loads(lines[-1])["type"] == "result"


def test_running_out_of_turns_is_an_error_not_a_claim(tmp_path, fake, work):
    fake["scenario"](subtype="error_max_turns", is_error=True, exit=1,
                     errors=["Reached maximum number of turns (7)"])
    result = make(tmp_path, fake).run(TASK, work, Budget())
    assert result.exit_code == 1
    assert result.claim is None
    assert "maximum number of turns" in result.error


def test_a_rejected_rate_limit_pauses_until_reset(tmp_path, fake, work):
    resets = time.time() + 7200
    fake["scenario"](rate_limit={"status": "rejected", "resetsAt": resets, "rateLimitType": "five_hour"},
                     is_error=True, exit=1)
    result = make(tmp_path, fake).run(TASK, work, Budget())
    assert result.paused_until == pytest.approx(resets)
    assert result.error == "usage limit reached"


def test_rate_limit_info_is_recorded_when_allowed(tmp_path, fake, work):
    info = {"status": "allowed", "unifiedWindows": {"five_hour": {"utilization": 0.4}}}
    fake["scenario"](rate_limit=info, structured={"status": "done", "summary": "ok"})
    result = make(tmp_path, fake).run(TASK, work, Budget())
    assert result.paused_until is None
    assert result.usage["rate_limit"]["unifiedWindows"]["five_hour"]["utilization"] == 0.4


def test_wall_clock_limit_kills_the_run(tmp_path, fake, work):
    fake["scenario"](sleep=60)
    started = time.monotonic()
    result = make(tmp_path, fake).run(TASK, work, Budget(wall_clock_s=1))
    assert time.monotonic() - started < 20
    assert result.exit_code is None and result.error == "timed out"


def test_split_request_is_passed_through(tmp_path, fake, work):
    steps = [{"goal": "add sub"}, {"goal": "add tests", "scope": ["tests/**"]}]
    fake["scenario"](structured={"status": "split", "summary": "too big", "split": steps})
    result = make(tmp_path, fake).run(TASK, work, Budget())
    assert result.split_request == steps


def test_a_run_without_a_result_is_an_executor_crash_not_a_failed_attempt(tmp_path, fake, work):
    Path(fake["binary"]).write_text("#!/bin/sh\nexit 0\n")
    result = make(tmp_path, fake).run(TASK, work, Budget())
    assert result.error == "the CLI ended without a result (exit 0)"
    assert result.aborted
    assert result.claim is None


def test_dependency_dirs_are_write_protected_for_the_worker(tmp_path, fake, work):
    make(tmp_path, fake, deps_paths=[".venv", "node_modules/"]).run(TASK, work, Budget())
    settings = json.loads((tmp_path / "logs" / "claude-001" / "settings.json").read_text())
    deny = settings["permissions"]["deny"]
    for path in (".venv", "node_modules"):
        assert f"Edit(./{path}/**)" in deny and f"Write(./{path}/**)" in deny
        assert str(work / path) in settings["sandbox"]["filesystem"]["denyWrite"]


def test_socket_directories_and_runtime_dir_are_out_of_reach(tmp_path, fake, work, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    make(tmp_path, fake).run(TASK, work, Budget())
    settings = json.loads((tmp_path / "logs" / "claude-001" / "settings.json").read_text())
    assert {"~/", "/run", "/var/run", "/var/snap"} <= set(settings["sandbox"]["filesystem"]["denyRead"])
    assert "XDG_RUNTIME_DIR" not in fake["call"]()["env"]


def test_a_rejection_is_kept_even_if_a_later_event_allows(tmp_path, fake, work):
    resets = time.time() + 3600
    fake["scenario"](rate_limits=[
        {"status": "rejected", "resetsAt": resets, "rateLimitType": "five_hour"},
        {"status": "allowed", "rateLimitType": "seven_day"},
    ], is_error=True, exit=1)
    result = make(tmp_path, fake).run(TASK, work, Budget())
    assert result.paused_until == pytest.approx(resets)


def test_a_rejection_without_a_reset_time_still_pauses(tmp_path, fake, work):
    fake["scenario"](rate_limit={"status": "rejected", "resetsAt": None}, is_error=True, exit=1)
    result = make(tmp_path, fake).run(TASK, work, Budget())
    assert result.paused_until is not None and result.paused_until > time.time()


def test_missing_cli_is_refused(tmp_path, work, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    executor = ClaudeCodeExecutor(tmp_path / "logs")
    with pytest.raises(ExecutorUnavailable):
        executor.run(TASK, work, Budget())


def test_a_worker_run_reports_its_session_and_resume_continues_it(tmp_path, fake, work):
    executor = make(tmp_path, fake)
    first = executor.run(TASK, work, Budget())
    assert first.session_id == "sess-1"
    task = Task(TASK.job_goal, TASK.node, feedback=["old feedback", "G5 failed: test_app::test_add"])
    second = executor.resume(task, work, Budget(), "sess-1")
    call = fake["call"]()
    argv = call["argv"]
    assert argv[argv.index("--resume") + 1] == "sess-1"
    assert "--no-session-persistence" not in argv and "--settings" in argv and "--max-turns" in argv
    assert "G5 failed: test_app::test_add" in call["prompt"] and "old feedback" not in call["prompt"]
    assert "one command per call" in call["prompt"]
    assert second.session_id == "sess-1"


def test_an_ask_session_is_kept_only_when_asked_to_be(tmp_path, fake, work):
    fake["scenario"](structured={"verdict": "approve", "reasons": []})
    result = make(tmp_path, fake).ask("review this", work, {"type": "object"}, Budget())
    assert "--no-session-persistence" in fake["call"]()["argv"]
    assert result.session_id is None
    result = make(tmp_path, fake).ask("review this", work, {"type": "object"}, Budget(), persist=True)
    assert "--no-session-persistence" not in fake["call"]()["argv"]
    assert result.session_id is not None


@pytest.mark.parametrize("reask_attempts, kept", [(0, False), (1, True), (None, True)],
                         ids=["off", "on", "default"])
def test_the_runners_reviewer_keeps_its_session_only_while_reasking_is_on(tmp_path, fake, work, monkeypatch,
                                                                          reask_attempts, kept):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from mullet.jobs.clone import ChangedFile, Snapshot
    from mullet.jobs.runner import JobRunner

    config = tmp_path / "config"
    config.mkdir()
    if reask_attempts is not None:
        (config / "config.yaml").write_text(f"jobs:\n  reask_attempts: {reask_attempts}\n")
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(config))
    fake["scenario"](structured={"verdict": "approve", "findings": [], "reasons": []})
    executor = make(tmp_path, fake)
    runner = SimpleNamespace(_reviewer=None, _executor=executor, _rate_limit=None, job_id="j",
                             _worker_slot=lambda e: nullcontext(), _record_limits=lambda usage: None,
                             ledger=SimpleNamespace(add_event=lambda *a: None))
    job = SimpleNamespace(goal="Goal", budget={}, base_sha=None)
    reviewer = JobRunner._make_reviewer(runner, job, SimpleNamespace(review_standards=None), executor, force=True)
    snap = Snapshot(tree="t", base="b", diff="diff --git a/app.py b/app.py\n+y\n",
                    files=[ChangedFile("app.py", "M", "100644", "100644")])
    ctx = SimpleNamespace(node=NodeSpec(id="n1", goal="g", scope=["app.py"]), snapshot=snap,
                          clone=SimpleNamespace(work=work))
    assert reviewer(ctx).status.value == "pass"
    assert ("--no-session-persistence" not in fake["call"]()["argv"]) is kept


def test_prompt_is_the_brief_and_files_are_preloaded_only_when_asked(tmp_path, fake, work):
    (work / "app.py").write_text("def add(a, b):\n    return a + b\n")
    (work / "tests").mkdir()
    (work / "tests" / "test_app.py").write_text("def test_add():\n    assert True\n")
    ctx = NodeContext(parent_goal="arithmetic", plan_notes="mind the sign", files=["app.py", "tests/test_app.py"],
                      checks={"test_full": "python3 runtests.py"})
    task = Task("Add subtraction", TASK.node, context=ctx)
    make(tmp_path, fake).run(task, work, Budget())
    prompt = fake["call"]()["prompt"]
    for text in ("Task brief from Mullet", "arithmetic", "mind the sign", "1: def add(a, b):", "Repository map"):
        assert text in prompt
    assert "## Pre-loaded files" not in prompt and "return a + b" not in prompt
    make(tmp_path, fake, preload_files=True).run(task, work, Budget())
    prompt = fake["call"]()["prompt"]
    assert "## Pre-loaded files" in prompt and "### app.py (in scope)" in prompt and "return a + b" in prompt
    assert (tmp_path / "logs" / "claude-002" / "prompt.md").read_text() == prompt


def test_prompt_carries_scope_acceptance_feedback_and_rules():
    task = Task("Goal", NodeSpec(id="n1", goal="do x", scope=["src/**"], acceptance=["make test"]),
                feedback=["G5 failed: test_app::test_add"])
    prompt = build_prompt(task, ["make test"])
    assert "src/**" in prompt and "make test" in prompt
    assert "G5 failed: test_app::test_add" in prompt
    assert "Don't weaken, skip or delete tests" in prompt
    assert "one command per call" in prompt
    assert "one command per call" not in build_prompt(task)


def test_prompt_rules_ask_for_small_test_first_changes_inside_scope():
    prompt = build_prompt(TASK)
    assert "smallest change" in prompt
    assert "one test" in prompt and "one behaviour at a time" in prompt
    assert "test helpers" in prompt
    assert "commented-out code" in prompt and "TODO" in prompt
    assert "new test file" in prompt and '"blocked"' in prompt


# Idle limit and grace window: short limits, never real minutes.


def test_a_silent_worker_is_stopped_as_a_provider_style_abort(tmp_path, fake, work):
    fake["scenario"](sleep=60)
    started = time.monotonic()
    result = make(tmp_path, fake).run(TASK, work, Budget(wall_clock_s=120, idle_s=1.0))
    assert time.monotonic() - started < 20
    assert result.aborted and result.provider_error and result.idle_stopped
    assert result.error == "stopped: no output for 1 second"
    assert result.usage["idle_stopped"] is True


def test_a_worker_that_hangs_after_its_result_is_killed_and_judged_normally(tmp_path, fake, work):
    fake["scenario"](structured={"status": "done", "summary": "did it"}, sleep_after=60)
    started = time.monotonic()
    result = make(tmp_path, fake).run(TASK, work, Budget(wall_clock_s=120, idle_s=30.0, grace_s=0.5))
    assert time.monotonic() - started < 20
    assert not result.aborted and not result.provider_error and result.error is None
    assert result.claim == {"status": "done", "summary": "did it"}
    assert result.exit_code is None and result.usage["exited_after_grace"] is True


def test_a_worker_that_keeps_printing_is_never_idle(tmp_path, fake, work):
    fake["scenario"](structured={"status": "done", "summary": "slow"}, ticks=8, tick_s=0.3)
    result = make(tmp_path, fake).run(TASK, work, Budget(wall_clock_s=120, idle_s=1.0, grace_s=5.0))
    assert result.error is None and not result.aborted and result.exit_code == 0
    assert result.claim["summary"] == "slow" and "idle_stopped" not in result.usage


def test_stderr_output_counts_as_output(tmp_path, fake, work):
    fake["scenario"](structured={"status": "done", "summary": "noisy"}, ticks=8, tick_s=0.3, tick_to="stderr")
    result = make(tmp_path, fake).run(TASK, work, Budget(wall_clock_s=120, idle_s=1.0))
    assert result.error is None and not result.aborted and result.claim["summary"] == "noisy"
