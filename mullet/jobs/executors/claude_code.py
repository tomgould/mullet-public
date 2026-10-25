"""Attempt a node with the user's own Claude Code CLI in headless mode.

Mullet never touches Claude credentials. It runs the user's installed,
logged-in `claude` binary the way the user would, with:

- `--restricted`: file tools confined to the clone; user settings, hooks and
  plugins ignored; writes to git and settings files refused;
- `--strict-mcp-config` with no servers, so no MCP tools, Mullet's included;
- a settings file that turns on Claude Code's own sandbox for Bash, fail
  closed: no unsandboxed retries, no network, home unreadable, and only the
  listed commands allowed;
- a minimal environment. `ANTHROPIC_API_KEY` is passed only when the job is
  set to bill the API, because in print mode the key silently replaces the
  subscription, and Mullet's other provider keys are never passed at all.

Each attempt's prompt is Mullet's task brief (`mullet.jobs.context`): the
goals, the planner's notes, what earlier nodes changed, the checks, a symbol
outline and a repo map. Claude explores well on its own, so file contents
are only pre-loaded when `preload_files` is set. Worker sessions are kept so
the runner can `resume` one with gate feedback; planner and reviewer sessions
(`ask`) are kept only while `jobs.reask_attempts` is above 0.

What the run says about its own work is returned as a claim. The gates
decide whether it did the work.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mullet.jobs import children, context
from mullet.jobs.executors.base import Budget, ExecResult, ExecutorUnavailable, Task, Watch, idle_text

__all__ = ["ClaudeCodeExecutor", "ExecutorUnavailable", "build_prompt", "build_resume_prompt"]

RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["done", "blocked", "split"]},
        "summary": {"type": "string"},
        "split": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "scope": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["goal"],
            },
        },
    },
    "required": ["status", "summary"],
}

TOOLS = "Read,Edit,Write,Glob,Grep,Bash"
READ_ONLY_TOOLS = "Read,Glob,Grep"

# Read-only commands every worker may run, on top of the profile's own checks.
READ_ONLY_BASH: tuple[str, ...] = (
    "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "diff", "sort", "uniq", "tree",
    "git status", "git diff", "git log", "git show", "git ls-files",
)

# XDG_RUNTIME_DIR is left out on purpose: it points at sockets (D-Bus, the
# keyring, rootless Docker) that a worker has no business near.
_ENV_KEEP: tuple[str, ...] = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM",
    "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME",
    "CLAUDE_CONFIG_DIR", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy",
    "no_proxy", "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "SSL_CERT_DIR",
)

# Never readable from the worker's Bash: home, and where privileged sockets live.
_DENY_READ: tuple[str, ...] = ("~/", "/run", "/var/run", "/var/snap")

_PROMPT_RULES = """\
Rules:
- Your current directory is a private copy of the repository. Work only there.
- Change only files that match the scope above. Leave lockfiles, CI config,
  .claude/, CLAUDE.md, .mcp.json and .env files alone unless this step says otherwise.
- Any new test file must match the scope too. If the step needs one outside it,
  answer "blocked" and say so rather than writing it elsewhere.
- Don't commit or run git commands that change anything. Mullet records your work.
- Make the smallest change that meets the goal. Write one test, then the code
  for it, one behaviour at a time.
- Don't invent new test helpers or seams just to make code testable. Leave no
  commented-out code and no TODO comments.
- Don't weaken, skip or delete tests to get them passing.
- Run the project's checks before you finish if you can.
- Mullet checks your work itself after you stop, so report honestly.
- If the step is too big to do well in one go, stop and answer with status
  "split" and a list of smaller steps. If you can't do it, answer "blocked"
  and say why. Otherwise answer "done" with a short summary."""


# Claude's context window, for sizing pre-loaded files when no model is named.
_DEFAULT_CONTEXT = 200_000


def _rules(bash_commands: list[str] | None) -> str:
    rules = _PROMPT_RULES
    if bash_commands:
        rules += ("\n\nIn Bash you may run commands that start with one of these, one command per call, "
                  "with no pipes, redirects, `&&` or `cd` (you are already in the repo):\n"
                  + "\n".join(f"  {c}" for c in bash_commands))
    return rules


def build_prompt(task: Task, bash_commands: list[str] | None = None, work: Path | None = None,
                 preload: list[context.ContextFile] | None = None) -> str:
    """The prompt for one attempt: Mullet's brief, then any pre-loaded files."""
    brief = context.build_brief(
        task, work or Path("."), attached=preload or None, rules=_rules(bash_commands),
        attached_note="Their contents follow the brief, so you don't need to read them again:")
    if preload and work is not None:
        brief += "\n## Pre-loaded files\n\n" + context.preload_block(work, preload) + "\n"
    return brief


def build_resume_prompt(task: Task, bash_commands: list[str] | None = None) -> str:
    """What a resumed session is told: the newest gate feedback and the rules again."""
    latest = task.feedback[-1] if task.feedback else "Mullet's checks failed, with no details recorded."
    return (
        "Mullet ran its checks on your work and they failed. The files are as you left them. "
        "Fix the problems below, then stop and report as before.\n\n"
        f"{latest}\n\n{_rules(bash_commands)}"
    )


def build_settings(work: Path, bash_commands: list[str], read_paths: list[Path],
                   deps_paths: list[str] | None = None) -> dict[str, Any]:
    """Claude Code settings for a worker: its own sandbox on, fail closed.

    Dependency directories are gitignored, so changes there never show in a
    snapshot; the worker must not be able to write to them at all.
    """
    allow = ["Read(./**)", "Edit(./**)", "Write(./**)", "Glob", "Grep"]
    allow += [f"Bash({cmd}:*)" for cmd in dict.fromkeys([*READ_ONLY_BASH, *bash_commands]) if cmd.strip()]
    deps = [p.strip("/") for p in deps_paths or [] if p.strip("/")]
    deny = ["WebFetch", "WebSearch"]
    for path in deps:
        deny += [f"Edit(./{path}/**)", f"Write(./{path}/**)", f"Edit(./{path})", f"Write(./{path})"]
    return {
        "permissions": {
            "allow": allow,
            "deny": deny,
            "blockReadsOutsideWorkingDirectories": True,
        },
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "allowUnsandboxedCommands": False,
            "autoAllowBashIfSandboxed": False,
            "filesystem": {
                "denyRead": list(_DENY_READ),
                "allowRead": [str(work), *(str(p) for p in read_paths)],
                "denyWrite": [str(work / path) for path in deps],
            },
            "network": {"allowedDomains": [], "strictAllowlist": True},
        },
    }


def build_readonly_settings(cwd: Path, read_paths: list[Path]) -> dict[str, Any]:
    """Settings for a read-only session: no edits, no Bash, sandbox still fail closed."""
    settings = build_settings(cwd, [], read_paths)
    settings["permissions"]["allow"] = ["Read(./**)", "Glob", "Grep"]
    settings["permissions"]["deny"] = ["Edit", "Write", "NotebookEdit", "Bash", "WebFetch", "WebSearch"]
    return settings


def build_env(billing: Literal["subscription", "api"]) -> dict[str, str]:
    """The environment for the CLI: only what it needs, never Mullet's provider keys."""
    env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
    if billing == "api":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ExecutorUnavailable("billing is set to api but ANTHROPIC_API_KEY is not set")
        env["ANTHROPIC_API_KEY"] = key
    return env


@dataclass
class _StreamState:
    lines: int = 0
    session_id: str | None = None
    api_key_source: str | None = None
    model: str | None = None
    rate_limit: dict[str, Any] = field(default_factory=dict)
    rejected: dict[str, Any] | None = None  # kept even if a later event says "allowed"
    result: dict[str, Any] | None = None
    watch: Watch | None = None


def _consume(stream, transcript_path: Path, state: _StreamState) -> None:
    with open(transcript_path, "w", encoding="utf-8") as transcript:
        for line in stream:
            transcript.write(line)
            state.lines += 1
            if state.watch is not None:
                state.watch.output()
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("type")
            if event.get("session_id") and not state.session_id:
                state.session_id = str(event["session_id"])
            if kind == "system" and event.get("subtype") == "init":
                state.api_key_source = event.get("apiKeySource")
                state.model = event.get("model")
            elif kind == "rate_limit_event":
                info = event.get("rate_limit_info") or {}
                state.rate_limit = info
                if info.get("status") == "rejected":
                    state.rejected = info
            elif kind == "result":
                state.result = event
                if state.watch is not None:
                    state.watch.finished()


class ClaudeCodeExecutor:
    """Runs `claude -p` in the clone: a new session per attempt, or a resumed one after a failure."""

    name = "claude-code"

    def __init__(self, logs_dir: Path, *, bash_commands: list[str] | None = None,
                 read_paths: list[Path] | None = None, deps_paths: list[str] | None = None,
                 model: str | None = None, billing: Literal["subscription", "api"] = "subscription",
                 binary: str | None = None, preload_files: bool = False,
                 context_share: float = context.BUDGET_SHARE):
        self.logs_dir = logs_dir
        self.bash_commands = list(bash_commands or [])
        self.read_paths = list(read_paths or [])
        self.deps_paths = list(deps_paths or [])
        self.model = model
        self.billing = billing
        self.binary = binary or shutil.which("claude")
        self.preload_files = preload_files
        self.context_share = context_share
        self._seq = 0

    def command(self, settings_path: Path, budget: Budget, *, tools: str = TOOLS,
                schema: dict[str, Any] | None = None, model: str | None = None,
                persist: bool = False, resume: str | None = None) -> list[str]:
        """The full CLI invocation. The prompt goes on stdin, not the command line.

        Worker sessions are kept (`persist`), so a failed attempt can be
        resumed, and so are planner and reviewer sessions that may be asked
        again for a reply they couldn't use; other sessions leave nothing behind.
        """
        if not self.binary:
            raise ExecutorUnavailable("the claude CLI is not on PATH")
        argv = [
            self.binary, "-p", "--restricted", "--strict-mcp-config", "--disable-slash-commands",
            "--tools", tools, "--settings", str(settings_path),
            "--permission-mode", "dontAsk", "--permission-prompts", "none",
            "--output-format", "stream-json", "--verbose",
            "--max-turns", str(budget.max_turns),
            "--json-schema", json.dumps(schema or RESULT_SCHEMA),
        ]
        if not persist:
            argv.append("--no-session-persistence")
        if resume:
            argv += ["--resume", resume]
        if model or self.model:
            argv += ["--model", model or self.model]
        return argv

    def run(self, task: Task, work: Path, budget: Budget) -> ExecResult:
        """Attempt `task` in `work` and return what the run reported."""
        preload = None
        if self.preload_files:
            length = None if self.model else _DEFAULT_CONTEXT
            tokens = context.context_budget(self.model, self.context_share, context_length=length)
            preload = context.select_context_files(task, work, tokens)
        settings = build_settings(work, self.bash_commands, self.read_paths, self.deps_paths)
        return self._launch(build_prompt(task, self.bash_commands, work, preload), work, settings, budget,
                            tools=TOOLS, schema=RESULT_SCHEMA, model=None, label="claude", persist=True)

    def resume(self, task: Task, work: Path, budget: Budget, session_id: str) -> ExecResult:
        """Continue `session_id` on the tree it left, with the newest gate feedback."""
        settings = build_settings(work, self.bash_commands, self.read_paths, self.deps_paths)
        return self._launch(build_resume_prompt(task, self.bash_commands), work, settings, budget,
                            tools=TOOLS, schema=RESULT_SCHEMA, model=None, label="claude",
                            persist=True, resume=session_id)

    def ask(self, prompt: str, cwd: Path, schema: dict[str, Any], budget: Budget, *,
            model: str | None = None, label: str = "ask", persist: bool = False,
            resume: str | None = None) -> ExecResult:
        """A read-only session (Read, Glob, Grep) that answers `prompt` as JSON matching `schema`.

        Used by the planner and the reviewer. The answer is in `claim`. With
        `persist` the session is kept (its id in `session_id`) so `resume` can
        continue it: the planner and reviewer ask again that way for a reply
        they couldn't use.
        """
        return self._launch(prompt, cwd, build_readonly_settings(cwd, self.read_paths), budget,
                            tools=READ_ONLY_TOOLS, schema=schema, model=model, label=label,
                            persist=persist, resume=resume)

    def _launch(self, prompt: str, cwd: Path, settings: dict[str, Any], budget: Budget, *,
                tools: str, schema: dict[str, Any], model: str | None, label: str,
                persist: bool = False, resume: str | None = None) -> ExecResult:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        while True:  # a resumed job reuses logs_dir, so skip numbers already used
            self._seq += 1
            attempt_dir = self.logs_dir / f"{label}-{self._seq:03d}"
            try:
                attempt_dir.mkdir()
                break
            except FileExistsError:
                continue
        settings_path = attempt_dir / "settings.json"
        settings_path.write_text(json.dumps(settings, indent=2))
        transcript = attempt_dir / "transcript.jsonl"
        (attempt_dir / "prompt.md").write_text(prompt)
        argv = self.command(settings_path, budget, tools=tools, schema=schema, model=model,
                            persist=persist, resume=resume)
        env = build_env(self.billing)
        work = cwd
        # Claude Code's sandbox lets Bash write to its per-user temp directory
        # (normally /tmp/claude-<uid>, shared by every session). A private,
        # per-attempt TMPDIR moves that writable area somewhere nothing else
        # uses. It must stay short: the sandbox puts Unix sockets in it, and
        # socket paths are limited to about 108 bytes.
        private_tmp = Path(tempfile.mkdtemp(prefix="mullet-", dir="/tmp" if os.path.isdir("/tmp") else None))
        env["TMPDIR"] = str(private_tmp)

        state = _StreamState(watch=Watch(budget.idle_s, budget.grace_s))
        started = time.monotonic()
        started_wall = time.time()
        stop: _Stop = None
        children.check_not_stopping()
        stderr_path = attempt_dir / "stderr.log"
        with open(stderr_path, "wb") as stderr:
            proc = subprocess.Popen(
                argv, cwd=work, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=stderr, text=True, encoding="utf-8", errors="replace", start_new_session=True,
            )
            reader = threading.Thread(target=_consume, args=(proc.stdout, transcript, state), daemon=True)
            reader.start()
            with children.track(proc):
                try:
                    proc.stdin.write(prompt)
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
                try:
                    stop = _wait(proc, started, budget, state.watch, stderr_path)
                    if stop is not None:
                        _kill_group(proc)
                except BaseException:  # Ctrl-C or a stop: take the CLI and its children with us
                    _kill_group(proc)
                    shutil.rmtree(private_tmp, ignore_errors=True)
                    raise
            reader.join(timeout=30)
        children.kill_in_dir(private_tmp, since=started_wall)
        shutil.rmtree(private_tmp, ignore_errors=True)
        result = self._result(proc.returncode, stop, state, transcript, time.monotonic() - started, budget)
        if persist:
            result.session_id = state.session_id
        return result

    def _result(self, code: int | None, stop: _Stop, state: _StreamState, transcript: Path,
                duration: float, budget: Budget) -> ExecResult:
        timed_out = stop == "timeout"
        if stop == "grace":
            # It gave its final reply and then didn't exit: killed, and judged like any finished run.
            code = None
        res = state.result or {}
        usage = {
            "duration_s": round(duration, 1),
            "num_turns": res.get("num_turns"),
            "total_cost_usd": res.get("total_cost_usd"),
            "usage": res.get("usage"),
            "model_usage": res.get("modelUsage"),
            "api_key_source": state.api_key_source,
            "model": state.model,
            "rate_limit": state.rate_limit,
        }
        claim = res.get("structured_output") if isinstance(res.get("structured_output"), dict) else None
        split = claim.get("split") if claim and claim.get("status") == "split" else None
        paused_until = None
        if state.rejected is not None:
            try:
                paused_until = float(state.rejected.get("resetsAt"))
            except (TypeError, ValueError):
                paused_until = time.time() + 3600
        error = None
        aborted = provider_error = idle_stopped = False
        if stop == "grace":
            usage["exited_after_grace"] = True
        if timed_out:
            error = "timed out"
        elif paused_until is not None:
            error = "usage limit reached"
        elif stop == "idle":
            # Like a provider stop: the CLI hung, so it doesn't use up an attempt.
            error, aborted, provider_error, idle_stopped = f"stopped: {idle_text(budget.idle_s or 0)}", True, True, True
            usage["idle_stopped"] = True
        elif not res:
            # A crash, a lost login or a killed process: not the worker's
            # doing, so it shouldn't use up an attempt.
            error = f"the CLI ended without a result (exit {code})"
            aborted = True
        elif res.get("is_error") or code not in (0, None):
            reasons = res.get("errors") or [res.get("subtype") or f"exit {code}"]
            error = "; ".join(str(r) for r in reasons)
        return ExecResult(
            exit_code=None if timed_out or idle_stopped else code, transcript_path=str(transcript), usage=usage,
            claim=claim, split_request=split, paused_until=paused_until, aborted=aborted, error=error,
            provider_error=provider_error, idle_stopped=idle_stopped,
        )


_Stop = Literal["timeout", "idle", "grace"] | None


def _wait(proc: subprocess.Popen, started: float, budget: Budget, watch: Watch, stderr_path: Path,
          poll_s: float = 0.25) -> _Stop:
    """Wait for the CLI to exit; say why it must be stopped instead, if it must."""
    while True:
        try:
            proc.wait(timeout=poll_s)
            return None
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        if now - started > budget.wall_clock_s:
            return "timeout"
        watch.poll_file(stderr_path, now)
        verdict = watch.verdict(now)
        if verdict is not None:
            return verdict


def _kill_group(proc: subprocess.Popen, grace: float = 10.0) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue
