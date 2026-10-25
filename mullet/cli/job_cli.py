"""CLI dispatcher for ``mullet job``: plan, approve, run and hand back gated jobs.

Wired into ``mullet/cli/main.py`` through the same ``sys.argv[1]`` early
branch as ``mullet preset``. The job model lives in ``mullet.jobs``; see
docs/MULLET2.md.
"""

from __future__ import annotations

import argparse
import fcntl
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import psutil
import yaml

from mullet.jobs import sandbox
from mullet.jobs.clone import GitError, JobClone, resolve_commit
from mullet.jobs.executors.base import ExecutorUnavailable
from mullet.jobs.ladder import ORDERS, Ladder, MachineState, RungChoice, plan_rungs, spend_state
from mullet.jobs.ledger import JobRecord, Ledger
from mullet.jobs.paths import job_dir
from mullet.jobs.planner import PLAN_KEYS, PlanError, plan_job, validate_plan
from mullet.jobs.profile import (
    Profile, load_profile, profile_path, repo_id, save_profile, user_profile, whole_check_commands,
)
from mullet.jobs.report import build_report
from mullet.jobs.runner import (
    ExecutorNotAllowed, JobRunner, check_executor_allowed, claude_executor, claude_usage, default_executor,
    executor_name, job_max_spend, solo_rung,
)
from mullet.jobs.types import JobStatus, NodeSpec

_EXECUTOR_NAMES = {"claude": "claude-code", "claude-code": "claude-code", "opencode": "opencode"}

_RUNNABLE = (JobStatus.APPROVED, JobStatus.PAUSED, JobStatus.BLOCKED)


class CliError(RuntimeError):
    """A problem to report to the user and exit 1."""


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise CliError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def _repo_root(path: str) -> Path:
    return Path(_git(Path(path).expanduser().resolve(), "rev-parse", "--show-toplevel"))


def _runner_alive(job_id: str) -> bool:
    """True while some runner holds this job's lock, whether started in the background or not."""
    lock = job_dir(job_id) / "runner.lock"
    if not lock.exists():
        return False
    fd = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _runner_pid(job_id: str) -> int | None:
    """The pid of the live runner for this job, checked against its start time, or None."""
    if not _runner_alive(job_id):
        return None
    try:
        pid_text, started = (job_dir(job_id) / "runner.pid").read_text().split()
        pid = int(pid_text)
        if abs(psutil.Process(pid).create_time() - float(started)) > 0.01:
            return None
    except (OSError, ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    return pid


def _load_nodes(path: str) -> tuple[list[tuple[NodeSpec, str | None]], str]:
    """A hand-written or edited plan, held to the same rules as the planner's, and its notes."""
    try:
        data = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise CliError(f"{path}: {exc}") from exc
    items = data.get("nodes") if isinstance(data, dict) else data
    notes = data.get("notes") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise CliError(f"{path}: expected a list of nodes (or a mapping with a 'nodes' list)")
    try:
        nodes = validate_plan({"nodes": [dict(i) if isinstance(i, dict) else i for i in items]})
    except PlanError as exc:
        raise CliError(f"{path}: {exc}") from exc
    return nodes, str(notes).strip() if notes else ""


def check_opencode_model(model: str | None) -> None:
    """Refuse an opencode job without a model opencode actually offers."""
    from mullet.jobs.executors.opencode import OpencodeExecutor

    if not model:
        raise CliError("--executor opencode needs --executor-model provider/model (see `opencode models`)")
    with tempfile.TemporaryDirectory(prefix="mullet-opencode-") as tmp:
        try:
            OpencodeExecutor(Path(tmp), model, home=Path(tmp) / "home").validate_model()
        except ExecutorUnavailable as exc:
            raise CliError(str(exc)) from exc


def _gates_line(profile: dict) -> list[str]:
    checks = profile.get("checks", {})
    lines = []
    for name in ("lint", "typecheck", "build", "test_full", "test_targeted"):
        value = checks.get(name)
        lines.append(f"  {name}: {value}" if value else f"  {name}: not configured")
    if not checks.get("test_full"):
        lines.append("  With no test command, every node will be reported as unverified, never passed.")
    return lines


def _dollars(text: str) -> float:
    """An argparse type for a spending cap: a finite amount (NaN or inf would switch the cap off)."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an amount: {text!r}") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"not an amount: {text!r}")
    return value


def make_ladder() -> Ladder:
    """The ladder as the user's config sets it up (tests replace this)."""
    return Ladder()


def ladder_preview(job: JobRecord, ladder: Ladder, attempts_usage: list | None = None,
                   specs: list[NodeSpec] | None = None,
                   cap: float | None = None) -> tuple[RungChoice, dict[str, RungChoice]]:
    """The rungs a mechanical step of `job` would get now, and each of `specs`' own, with reasons.

    LM Studio and the public check (cached for an hour) are only asked when their rung is configured.
    """
    profile = Profile.model_validate(job.profile)
    config = ladder.config
    budget = ladder.budget()
    machine = MachineState(False, "not checked")
    public = None
    if config.local_model:
        machine = ladder.machine(config)
    if config.free_model:
        public = ladder.public(Path(job.repo_path), job.base_sha, profile)
    if cap is None:
        cap = float((job.budget or {}).get("max_spend", config.max_spend))
    spend = spend_state(attempts_usage or [], cap, config)

    def plan(spec: NodeSpec) -> RungChoice:
        return plan_rungs(spec, profile, budget, machine, spend, config=config, public=public,
                          has_key=ladder.has_key)

    probe = NodeSpec(id="probe", goal="probe", kind="mechanical")
    return plan(probe), {spec.id: plan(spec) for spec in specs or []}


def show(ledger: Ledger, job_id: str, ladder: Ladder | None = None) -> str:
    """A job's plan, budget and checks, for review before approval."""
    job = ledger.get_job(job_id)
    budget = job.budget or {}
    worker = executor_name(job)
    worker_model = budget.get("executor_model") or (budget.get("model") if worker == "claude-code" else None)
    on_ladder = bool(budget.get("ladder"))
    if on_ladder:
        worker_line = (f"Worker: the delegation ladder (spending cap ${job_max_spend(ledger, job, 0.0):.2f}); "
                       "the planner, reviewer, design and analysis steps are Claude")
    else:
        solo = solo_rung(job)
        worker_line = (f"Worker: {worker}" + (f" with {worker_model}" if worker_model else "")
                       + ("; files pre-loaded" if worker == "opencode" or budget.get("preload_files") else "")
                       + (f"; spending cap ${job_max_spend(ledger, job, 0.0):.2f}" if solo and solo.paid else ""))
    lines = [
        f"Job {job.id}  [{_state_label(job, ledger.nodes(job.id))}]" + (f"  {job.error}" if job.error else ""),
        f"Goal: {job.goal}",
        f"Repo: {job.repo_path} at {job.base_ref} ({job.base_sha[:12]})",
        f"Attempts per node: {job.max_attempts}; per attempt: {budget.get('max_turns', 60)} turns, "
        f"{int(float(budget.get('wall_clock_s', 3600)) // 60)} minutes; "
        f"model: {budget.get('model') or 'Claude Code default'}; billing: {budget.get('billing', 'subscription')}; "
        f"review (G7): {'on' if budget.get('review') else 'off'}",
        worker_line,
    ]
    starts: dict[str, RungChoice] = {}
    nodes = ledger.nodes(job_id)
    parents = {n.parent_id for n in nodes if n.parent_id}
    if on_ladder:
        active = ladder or make_ladder()
        leaves = [n.spec for n in nodes if n.node_id not in parents]
        choice, starts = ladder_preview(job, active, [a.usage for a in ledger.attempts(job_id)], leaves,
                                        cap=job_max_spend(ledger, job, active.config.max_spend))
        lines.append(active.budget().describe())
        lines.append("Ladder for mechanical steps now: " + (" > ".join(r.label() for r in choice.rungs) or "none"))
        lines += [f"  skipped {label}: {why}" for label, why in choice.skipped]
    lines += ["", "Plan:"]
    for node in nodes:
        spec = node.spec
        indent = "    " if node.parent_id else "  "
        lines.append(f"{indent}{spec.id} [{node.status.value}] ({spec.kind}): {spec.goal}")
        mine = starts.get(spec.id)
        if mine is not None and mine.rungs:
            lines.append(f"{indent}  starts on: {mine.rungs[0].label()} ({mine.group}; then "
                         + (" > ".join(r.name for r in mine.rungs[1:]) or "nothing") + ")")
        lines.append(f"{indent}  scope: {', '.join(spec.scope)}")
        if spec.acceptance:
            lines.append(f"{indent}  must pass: {'; '.join(spec.acceptance)}")
        if spec.allow_protected:
            lines.append(f"{indent}  may change protected: {', '.join(spec.allow_protected)}")
        if spec.allow_test_weakening:
            lines.append(f"{indent}  may remove or skip tests")
        defaults = NodeSpec(id="x", goal="x")
        if (spec.max_files, spec.max_lines) != (defaults.max_files, defaults.max_lines):
            lines.append(f"{indent}  limits: {spec.max_files} files, {spec.max_lines} lines")
    lines += ["", "Checks Mullet will run (from the repo profile):", *_gates_line(job.profile)]
    if job.profile.get("setup"):
        lines.append("  setup: " + "; ".join(job.profile["setup"]))
    lines.append(f"  data class: {job.profile.get('data_class', 'private')}")
    return "\n".join(lines)


def decide_ladder(args: argparse.Namespace, profile: Profile, configured: bool, stored: Path) -> bool:
    """Whether a new job uses the delegation ladder.

    On by default when the user's config has a `ladder:` key, the profile
    allows opencode and no single executor was asked for; `--ladder` and
    `--no-ladder` decide outright.
    """
    if args.ladder is False:
        return False
    if args.ladder is None:
        return configured and args.executor is None and "opencode" in profile.allowed_executors
    if args.executor == "opencode":
        raise CliError("--ladder chooses the worker for each step; drop --executor opencode, or use --no-ladder")
    if "opencode" not in profile.allowed_executors:
        raise CliError("the ladder sends mechanical steps to opencode workers, which this repo's profile doesn't "
                       f"allow. Add 'opencode' to allowed_executors in your stored profile ({stored}) first; "
                       f"this repo's data class is {profile.data_class}.")
    if profile.ladder == "claude-only":
        print("Note: this repo's profile says ladder: claude-only, so every step stays on Claude.")
    return True


def cmd_plan(args: argparse.Namespace, ledger: Ledger) -> int:
    repo = _repo_root(args.repo)
    base_sha = resolve_commit(repo, args.ref)
    base_ref = args.ref
    if args.ref == "HEAD":
        base_ref = subprocess.run(["git", "-C", str(repo), "symbolic-ref", "--short", "-q", "HEAD"],
                                  capture_output=True, text=True).stdout.strip() or base_sha
    if _git(repo, "status", "--porcelain"):
        print("Note: the repo has uncommitted changes. The job starts from the last commit and won't see them.")
    profile = load_profile(repo, base_sha)
    stored = profile_path(repo_id(repo))
    if not stored.exists():
        save_profile(profile)
        print(f"Saved a detected profile to {stored}. Edit it to change setup or checks, then plan again.")
    executor = _EXECUTOR_NAMES[args.executor or "claude"]
    ladder_config = make_ladder().config
    use_ladder = decide_ladder(args, profile, ladder_config.configured, stored)
    max_spend = ladder_config.max_spend if args.max_spend is None else args.max_spend
    if max_spend < 0:
        raise CliError("--max-spend can't be negative")
    budget = {"max_turns": args.max_turns, "wall_clock_s": args.minutes * 60,
              "model": args.model, "billing": args.billing, "review": not args.no_review,
              "executor": executor, "executor_model": args.executor_model, "preload_files": args.preload,
              "ladder": use_ladder, "max_spend": max_spend,
              # Commands the worker may run as written even with `;`: only what the
              # user's stored profile (or detection) wrote, never the repo's own file.
              "trusted_commands": [c for c in whole_check_commands(profile)
                                   if c in whole_check_commands(user_profile(repo))]}
    # The executor, the model and any plan file are checked before a job
    # exists, so a bad one leaves nothing behind.
    try:
        check_executor_allowed(executor, profile)
    except ExecutorNotAllowed as exc:
        raise CliError(f"{exc} (stored profile: {stored})") from exc
    if executor == "opencode":
        check_opencode_model(args.executor_model)
    nodes, notes = _load_nodes(args.plan_file) if args.plan_file else (None, "")
    job_id = ledger.create_job(str(repo), args.goal, base_ref, base_sha, profile.model_dump(),
                               budget, max_attempts=args.max_attempts)
    if nodes is None and args.single:
        nodes = [(NodeSpec(id="n1", goal=args.goal, scope=args.scope or ["**"]), None)]
    elif nodes is None:
        try:
            nodes, notes = _run_planner(ledger, job_id, profile, args.model)
        except (PlanError, ExecutorUnavailable, GitError) as exc:
            ledger.set_job_status(job_id, JobStatus.FAILED, f"planning failed: {exc}")
            raise CliError(f"planning failed: {exc}") from exc
        except KeyboardInterrupt:
            ledger.set_job_status(job_id, JobStatus.FAILED, "planning interrupted")
            raise
    ledger.set_plan(job_id, nodes)
    if notes:
        # Kept for the workers too: each node's brief carries the planner's notes.
        ledger.add_event(job_id, "plan_notes", {"notes": notes})
    ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
    print(show(ledger, job_id))
    if notes:
        print(f"\nPlanner's notes: {notes}")
    print(f"\nApprove with: mullet job approve {job_id}")
    print(f"To change the plan: mullet job show {job_id} --yaml > plan.yaml, edit it, then "
          f"mullet job plan \"<goal>\" --plan-file plan.yaml (repeat any --no-review, --model, "
          "--executor, --executor-model, --max-attempts, --no-ladder, --max-spend you used)")
    return 0


def _run_planner(ledger: Ledger, job_id: str, profile, model: str | None):
    """Clone the repo for the job and ask a read-only session for a task tree. The planner is always Claude."""
    if not shutil.which("claude"):
        raise ExecutorUnavailable("the claude CLI is not on PATH; use --single or --plan-file instead")
    job = ledger.get_job(job_id)
    branch = f"mullet/{job_id}"
    clone = JobClone.create(Path(job.repo_path), job.base_sha, job_dir(job_id) / "clone", branch)
    ledger.set_job_paths(job_id, clone.work, clone.git_dir, branch)
    files = [p for p in clone.git("ls-tree", "-r", "-z", "--name-only", "HEAD").stdout.split("\0") if p]
    executor = claude_executor(job, profile, job_dir(job_id) / "logs" / "planner")
    print("Planning with Claude Code (read-only); this can take a few minutes...", flush=True)

    def ask(prompt, cwd, schema, budget):
        result = executor.ask(prompt, cwd, schema, budget, model=model, label="plan")
        # The planner's Claude work goes in the ledger and keeps the budget figures fresh.
        ledger.add_event(job_id, "claude_usage", claude_usage("planner", result.usage))
        JobRunner._record_limits(result.usage)
        return result

    nodes, notes, _ = plan_job(job.goal, clone.work, profile, files, ask)
    return nodes, notes


def cmd_show(args: argparse.Namespace, ledger: Ledger) -> int:
    if args.yaml:
        nodes = []
        for node in ledger.nodes(args.job_id):
            data = {k: v for k, v in node.spec.model_dump(exclude_defaults=True).items() if k in PLAN_KEYS}
            data.setdefault("id", node.spec.id)
            data.setdefault("goal", node.spec.goal)
            data.setdefault("kind", node.spec.kind)
            data.setdefault("scope", node.spec.scope)
            data["parent"] = node.parent_id
            nodes.append(data)
        notes = [e.data.get("notes") for e in ledger.events(args.job_id) if e.kind == "plan_notes"]
        plan = {"nodes": nodes, **({"notes": notes[-1]} if notes and notes[-1] else {})}
        print(yaml.safe_dump(plan, sort_keys=False, width=100))
    else:
        print(show(ledger, args.job_id))
    return 0


def cmd_approve(args: argparse.Namespace, ledger: Ledger) -> int:
    ledger.approve(args.job_id)
    print(f"Approved. Start it with: mullet job run {args.job_id}")
    return 0


def preflight(job_status: JobStatus, job: JobRecord | None = None) -> None:
    """Refuse to start a job this machine can't run safely.

    RUNNING is accepted because the caller has already checked that no runner
    holds the job: that is what a runner killed mid-job leaves behind. With
    `job`, its executor is checked too: allowed by the profile, installed,
    and for opencode, a model opencode offers. Claude Code is needed for a
    Claude worker and for the G7 reviewer.
    """
    if job_status not in (*_RUNNABLE, JobStatus.RUNNING):
        raise CliError(f"the job is {job_status.value}; only approved, paused or blocked jobs can run")
    check = sandbox.check()
    if not check.ok:
        raise CliError(f"Mullet's sandbox is unavailable: {check.reason}")
    worker = executor_name(job) if job else "claude-code"
    if job is not None:
        profile = Profile.model_validate(job.profile)
        try:
            check_executor_allowed(worker, profile)
        except ExecutorNotAllowed as exc:
            raise CliError(str(exc)) from exc
        if worker == "opencode":
            from mullet.jobs.executors.opencode import nested_sandbox_ok

            if not shutil.which("opencode"):
                raise CliError("opencode is not on PATH; install it first (npm i -g opencode-ai)")
            if not nested_sandbox_ok():
                raise CliError("opencode needs a nested sandbox for the worker's shell, and bubblewrap "
                               "can't start one inside another here")
            if not shutil.which("socat"):
                raise CliError("opencode reaches its provider through socat (for example: sudo apt install socat)")
            try:
                default_executor(job, profile, job_dir(job.id) / "logs" / "executor").validate_model()
            except ExecutorUnavailable as exc:
                raise CliError(str(exc)) from exc
    if worker == "claude-code" or job is None or (job.budget or {}).get("review"):
        if not shutil.which("claude"):
            raise CliError("the claude CLI is not on PATH; install Claude Code and log in first")
        if sys.platform == "linux" and not shutil.which("socat"):
            raise CliError("Claude Code's own sandbox needs socat on Linux (for example: sudo apt install socat)")


def cmd_run(args: argparse.Namespace, ledger: Ledger) -> int:
    job = ledger.get_job(args.job_id)
    if _runner_alive(args.job_id):
        raise CliError("a runner is already working on this job")
    preflight(job.status, job)
    if args.max_spend is not None:
        if not (job.budget or {}).get("ladder") and executor_name(job) != "opencode":
            raise CliError("--max-spend applies to jobs planned with the ladder or with --executor opencode")
        if args.max_spend < 0:
            raise CliError("--max-spend can't be negative")
        ledger.add_event(args.job_id, "max_spend", {"value": float(args.max_spend)})
        print(f"Spending cap for this job is now ${args.max_spend:.2f}.")
    if args.foreground:
        from mullet.jobs.runner import main as runner_main
        return runner_main([args.job_id])
    logs = job_dir(args.job_id) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / "runner.log", "ab") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "mullet.jobs.runner", args.job_id],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=job_dir(args.job_id),
        )
    print(f"Running in the background (pid {proc.pid}). Log: {logs / 'runner.log'}")
    print(f"Follow it with: mullet job tail {args.job_id} -f")
    return 0


def _state_label(job, nodes) -> str:
    """The job's status, with a finished job that has unverified nodes marked so it doesn't read as fully done."""
    if job.status is JobStatus.DONE and any(n.status.value == "unverified" for n in nodes):
        return "done (unverified)"
    return job.status.value


def _status_line(ledger: Ledger, job_id: str) -> str:
    job = ledger.get_job(job_id)
    nodes = ledger.nodes(job_id)
    passed = sum(1 for n in nodes if n.status.value == "passed")
    unverified = sum(1 for n in nodes if n.status.value == "unverified")
    counts = f"{passed} passed" + (f", {unverified} unverified" if unverified else "")
    running = " (runner alive)" if _runner_alive(job_id) else ""
    goal = job.goal if len(job.goal) <= 60 else job.goal[:57] + "..."
    return f"{job.id}  {_state_label(job, nodes):<17} {counts} (of {len(nodes)} nodes){running}  {goal}"


def cmd_status(args: argparse.Namespace, ledger: Ledger) -> int:
    if not args.job_id:
        return cmd_list(args, ledger)
    print(_status_line(ledger, args.job_id))
    job = ledger.get_job(args.job_id)
    if job.error:
        print(f"  {job.error}")
    for node in ledger.nodes(args.job_id):
        tries = ledger.attempt_count(args.job_id, node.node_id)
        print(f"  {node.node_id:<20} {node.status.value:<11} attempts {tries}/{job.max_attempts}")
    return 0


def cmd_list(args: argparse.Namespace, ledger: Ledger) -> int:
    jobs = ledger.list_jobs(limit=getattr(args, "limit", 20) or 20)
    if not jobs:
        print("No jobs yet. Start one with: mullet job plan \"<goal>\"")
    for job in jobs:
        print(_status_line(ledger, job.id))
    return 0


def _print_events(ledger: Ledger, job_id: str, since: int) -> int:
    for event in ledger.events(job_id, since_id=since):
        stamp = time.strftime("%H:%M:%S", time.localtime(event.ts))
        detail = {k: v for k, v in event.data.items() if k not in ("runs", "details")}
        print(f"{stamp} {event.kind} {detail}")
        since = event.id
    return since


def cmd_tail(args: argparse.Namespace, ledger: Ledger) -> int:
    since = _print_events(ledger, args.job_id, 0)
    while args.follow:
        time.sleep(2)
        since = _print_events(ledger, args.job_id, since)
        if not _runner_alive(args.job_id):
            break
    return 0


def cmd_stop(args: argparse.Namespace, ledger: Ledger) -> int:
    if not _runner_alive(args.job_id):
        raise CliError("no runner is working on this job")
    pid = _runner_pid(args.job_id)
    if not pid:
        raise CliError("a runner holds this job but its pid can't be confirmed; stop it with Ctrl-C where it runs")
    os.kill(pid, signal.SIGTERM)
    print("Asked the runner to stop. The job will be paused; resume it with: "
          f"mullet job resume {args.job_id}")
    return 0


def cmd_report(args: argparse.Namespace, ledger: Ledger) -> int:
    print(build_report(ledger, args.job_id))
    return 0


def cmd_discard(args: argparse.Namespace, ledger: Ledger) -> int:
    if _runner_alive(args.job_id):
        raise CliError("stop the job first: mullet job stop " + args.job_id)
    job = ledger.get_job(args.job_id)
    if job.status is JobStatus.RUNNING:
        ledger.set_job_status(args.job_id, JobStatus.PAUSED, "runner gone")
    clone_root = job_dir(args.job_id) / "clone"
    if clone_root.exists():
        JobClone(clone_root).remove()
    ledger.set_job_status(args.job_id, JobStatus.DISCARDED)
    print("Discarded. The ledger keeps its record; the clone is deleted.")
    return 0


def cmd_merge(args: argparse.Namespace, ledger: Ledger) -> int:
    job = ledger.get_job(args.job_id)
    if job.status is not JobStatus.DONE:
        raise CliError(f"the job is {job.status.value}; only a finished job can be handed back")
    unverified = [n.node_id for n in ledger.nodes(args.job_id) if n.status.value == "unverified"]
    if unverified and not args.allow_unverified:
        raise CliError(f"these nodes could not be fully checked: {', '.join(unverified)}. "
                       "Review them, then hand the branch back with --allow-unverified")
    clone = JobClone(job_dir(args.job_id) / "clone")
    print(clone.git("diff", "--stat", job.base_sha, "HEAD").stdout)
    if unverified:
        print(f"Warning: these nodes could not be fully checked: {', '.join(unverified)}")
    target = args.branch or job.branch or f"mullet/{job.id}"
    repo = Path(job.repo_path)
    if subprocess.run(["git", "-C", str(repo), "check-ref-format", "--branch", target],
                      capture_output=True).returncode != 0:
        raise CliError(f"{target!r} is not a valid branch name")
    if subprocess.run(["git", "-C", str(repo), "show-ref", "--verify", "--quiet", f"refs/heads/{target}"]).returncode == 0:
        raise CliError(f"branch {target} already exists in {repo}; pick another with --branch")
    if not args.yes:
        answer = input(f"Copy this as branch {target} into {job.repo_path}? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Nothing changed.")
            return 1
    try:
        sha = clone.fetch_into(repo, target)
    except GitError as exc:
        raise CliError(f"could not copy the branch: {exc}") from exc
    ledger.set_job_status(args.job_id, JobStatus.MERGED)
    print(f"Branch {target} ({sha[:12]}) is now in your repo. Nothing else changed.")
    print(f"Review and merge it yourself, for example: git merge --no-ff {target}")
    return 0


def _model_status(model: str) -> str:
    """Whether opencode offers `model`, from `opencode models` (once per provider per process)."""
    from mullet.jobs.executors.opencode import OpencodeExecutor

    if not shutil.which("opencode"):
        return "opencode is not on PATH"
    with tempfile.TemporaryDirectory(prefix="mullet-ladder-") as tmp:
        try:
            OpencodeExecutor(Path(tmp), model, home=Path(tmp) / "home").validate_model()
        except ExecutorUnavailable as exc:
            return f"not usable: {exc}"
    return "offered by opencode"


def _night_line(ladder: Ladder) -> str:
    """'Night mode: on until 07:30' or 'Night mode: off'; off when night mode can't be read."""
    try:
        state = ladder.night()
    except Exception:
        return "Night mode: off"
    if not state.on or state.until is None:
        return "Night mode: off"
    return "Night mode: on until " + datetime.fromtimestamp(state.until).strftime("%H:%M")


def ladder_report(ladder: Ladder, repo: Path | None, *, offline: bool = False,
                  model_status: Callable[[str], str] | None = None) -> str:
    """`mullet job ladder`: the configured rungs, which are usable now, and Claude's budget."""
    from mullet.jobs.ladder import config_path

    config = ladder.config
    check_model = model_status or _model_status
    source = config.source or str(config_path())
    lines = [
        "Delegation ladder " + (f"(from {source})" if config.configured else
                                f"(defaults: {source} has no ladder: section, so new jobs don't use it "
                                "unless planned with --ladder)"),
        ladder.budget().describe(),
        f"Thresholds: below {config.claude_low:g} paced weekly pressure Claude comes before the paid rungs; "
        f"above {config.claude_high:g} "
        "Claude is only the last resort.",
        f"Spending cap per job on paid rungs: ${config.max_spend:.2f}; at most {config.max_below_claude} "
        f"attempts below Claude before its turn; free and local attempts get {config.cheap_minutes:g} minutes",
        _night_line(ladder),
        "Order by Claude's budget (mechanical / checkable design / analysis, other design and integration):",
        *(f"  {level:<7} " + " / ".join(" > ".join(orders[g]) for g in ("mechanical", "checkable", "claude-first"))
          for level, orders in ORDERS.items()),
        "",
        "Rungs:",
    ]
    configured = {r.name for r in config.rungs()}
    for name in ("free", "local"):
        if name not in configured:
            lines.append(f"  {name:<11} not configured (ladder.{name}); skipped")
    for rung in config.rungs():
        notes = []
        if rung.name == "claude":
            notes.append("your own Claude Code; " + ("claude CLI found" if shutil.which("claude")
                                                    else "the claude CLI is not on PATH"))
        else:
            notes.append("key: found" if ladder.has_key(rung) or rung.name == "local" else "key: missing")
            if rung.name == "local":
                notes.append(ladder.machine(config).reason if not offline else "LM Studio not checked (--offline)")
            if not offline:
                notes.append(check_model(str(rung.model)))
            notes.append("paid" if rung.paid else "no charge")
        lines.append(f"  {rung.name:<11} {rung.model or '':<44} " + "; ".join(notes))
    if config.problems:
        lines += ["", "Problems in the config:", *(f"  {p}" for p in config.problems)]
    if repo is not None:
        try:
            profile = load_profile(repo)
        except Exception as exc:  # a broken profile shouldn't hide the rest
            lines += ["", f"This repo ({repo}): can't read its profile: {exc}"]
        else:
            lines += ["", f"This repo ({repo}): data class {profile.data_class}; allowed executors "
                      f"{', '.join(profile.allowed_executors)}; ladder {profile.ladder}"]
            if "opencode" not in profile.allowed_executors:
                lines.append("  Only Claude can work on it: add 'opencode' to allowed_executors in your stored "
                             f"profile ({profile_path(profile.repo_id)}) to use the other rungs.")
            if config.free_model:
                if offline:
                    lines.append("  public check: not run (--offline)")
                else:
                    try:
                        head = _git(repo, "rev-parse", "HEAD")
                        found = ladder.public(repo, head, profile)
                        lines.append(f"  public check: {'passes' if found.public else 'fails'}: {found.reason}")
                    except CliError as exc:
                        lines.append(f"  public check: can't run: {exc}")
    return "\n".join(lines)


def probe_credit() -> dict:
    """Each paid provider's own balance now (the network; tests replace this)."""
    from mullet.jobs.reconcile import probe_providers

    return probe_providers()


def credit_lines(ladder: Ladder, probe: Callable[[], dict] | None = None, threshold: float | None = None) -> list[str]:
    """The credit line for the paid rungs that have a key, and a warning for any under the threshold.

    Nothing is probed (and no lines come back) when no paid rung has a key.
    """
    from mullet.cli.doctor_checks import low_credit, low_credit_text, warn_below_usd
    from mullet.jobs.notify import status_line
    from mullet.jobs.reconcile import take

    try:
        paid = [r for r in ladder.config.rungs() if r.paid and ladder.has_key(r)]
    except Exception:
        return []
    if not paid:
        return []
    names = {r.name for r in paid}
    found = take(probe or probe_credit)
    snapshot = {k: v for k, v in found.items() if k in names}
    line = status_line(snapshot, None)
    if not line:
        return []
    limit = warn_below_usd() if threshold is None else threshold
    return [line, *("Warning: " + low_credit_text(n, r, limit) for n, r in low_credit(snapshot, limit))]


def cmd_retry(args: argparse.Namespace, ledger: Ledger) -> int:
    """Give a blocked step fresh attempts, on the whole ladder again or on one named rung."""
    from mullet.jobs.ladder import RUNG_ORDER

    job = ledger.get_job(args.job_id)
    node = ledger.get_node(args.job_id, args.node_id)
    if _runner_alive(args.job_id):
        raise CliError("a runner is working on this job; stop it first: mullet job stop " + args.job_id)
    if job.status not in (JobStatus.BLOCKED, JobStatus.PAUSED):
        raise CliError(f"the job is {job.status.value}; only a blocked or paused job's steps can be retried")
    if node.status.value not in ("blocked", "failed", "pending"):
        raise CliError(f"step {node.node_id} is {node.status.value}; only a blocked or failed step can be retried")
    on_ladder = bool((job.budget or {}).get("ladder"))
    if args.rung:
        if not on_ladder:
            raise CliError("--rung needs a job planned with the ladder; a plain job's blocked step gets one "
                           "more attempt each time it is resumed")
        name, _, model = args.rung.partition(":")
        if name not in RUNG_ORDER:
            raise CliError(f"{args.rung!r} is not a rung ({', '.join(RUNG_ORDER)}, or name:model)")
        if model:
            known = [r.model for r in make_ladder().config.rungs() if r.name == name and r.model]
            if model not in known and f"openrouter/{model}" not in known:
                raise CliError(f"{args.rung!r}: your ladder config has no {name} model {model!r} "
                               f"(configured: {', '.join(known) or 'none'})")
    if not on_ladder:
        print(f"Step {node.node_id} will get one more attempt when the job resumes.")
    else:
        attempts = ledger.attempts(args.job_id, node.node_id)
        ledger.add_event(args.job_id, "node_retry", {
            "node_id": node.node_id, "rung": args.rung, "after_attempt": attempts[-1].id if attempts else 0})
        where = f"on {args.rung}" if args.rung else "on the ladder from its start rung"
        print(f"Step {node.node_id} gets fresh attempts {where}, starting fresh (no session carried over).")
    print(f"Resume the job with: mullet job resume {args.job_id}")
    return 0


def cmd_ladder(args: argparse.Namespace, ledger: Ledger) -> int:
    try:
        repo: Path | None = _repo_root(args.repo)
    except CliError:
        repo = None
    ladder = make_ladder()
    print(ladder_report(ladder, repo, offline=args.offline))
    if not args.offline:
        lines = credit_lines(ladder)
        if lines:
            print("\n" + "\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The ``mullet job`` argument parser."""
    parser = argparse.ArgumentParser(prog="mullet job", description="Gated, sandboxed coding jobs.")
    sub = parser.add_subparsers(dest="action", required=True)

    plan = sub.add_parser("plan", help="create a job and show its plan for approval")
    plan.add_argument("goal")
    plan.add_argument("--repo", default=".")
    plan.add_argument("--ref", default="HEAD", help="commit or branch the job starts from")
    plan.add_argument("--plan-file", help="YAML list of nodes to use instead of asking the planner")
    plan.add_argument("--single", action="store_true", help="one node for the whole goal, no planner")
    plan.add_argument("--scope", action="append", help="with --single: path pattern the node may change")
    plan.add_argument("--no-review", action="store_true", help="skip the G7 reviewer for this job")
    plan.add_argument("--max-attempts", type=int, default=3)
    plan.add_argument("--max-turns", type=int, default=60)
    plan.add_argument("--minutes", type=int, default=60, help="wall-clock limit per attempt")
    plan.add_argument("--model", help="Claude model for the planner and reviewer (and a Claude worker)")
    plan.add_argument("--billing", choices=("subscription", "api"), default="subscription")
    plan.add_argument("--executor", choices=("claude", "opencode"), default=None,
                      help="one worker for every node (default claude); opencode must be in the profile's "
                           "allowed_executors. Giving it turns the ladder off unless --ladder is also given")
    plan.add_argument("--ladder", dest="ladder", action="store_true", default=None,
                      help="send each mechanical step up the delegation ladder (the default when your config "
                           "has a ladder: section and the profile allows opencode)")
    plan.add_argument("--no-ladder", dest="ladder", action="store_false",
                      help="one executor for every node, as before the ladder")
    plan.add_argument("--max-spend", type=_dollars, default=None,
                      help="dollars the job may spend on paid rungs, or on a paid --executor opencode model "
                           "(default: ladder.max_spend, $2)")
    plan.add_argument("--executor-model", help="the worker's model; for opencode, a provider/model id")
    plan.add_argument("--preload", action="store_true",
                      help="also pre-load the files in scope into a Claude worker's prompt (opencode always does)")
    plan.set_defaults(fn=cmd_plan)

    show_p = sub.add_parser("show", help="show a job's plan and checks")
    show_p.add_argument("job_id")
    show_p.add_argument("--yaml", action="store_true", help="print the plan as an editable plan file")
    show_p.set_defaults(fn=cmd_show)

    for name, fn, text in (
        ("approve", cmd_approve, "approve a planned job"),
        ("stop", cmd_stop, "stop a running job; it is paused, not lost"),
        ("report", cmd_report, "print the job report"),
        ("discard", cmd_discard, "delete a job's clone and mark it discarded"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("job_id")
        p.set_defaults(fn=fn)

    for name in ("run", "resume"):
        p = sub.add_parser(name, help="start or resume a job in the background")
        p.add_argument("job_id")
        p.add_argument("--foreground", action="store_true")
        p.add_argument("--max-spend", type=_dollars, default=None,
                       help="ladder and opencode jobs: a new spending cap in dollars, e.g. after the job "
                            "paused at its cap")
        p.set_defaults(fn=cmd_run)

    status = sub.add_parser("status", help="one job's progress, or all jobs")
    status.add_argument("job_id", nargs="?")
    status.set_defaults(fn=cmd_status)

    listing = sub.add_parser("list", help="recent jobs")
    listing.add_argument("--limit", type=int, default=20)
    listing.set_defaults(fn=cmd_list)

    tail = sub.add_parser("tail", help="a job's events")
    tail.add_argument("job_id")
    tail.add_argument("-f", "--follow", action="store_true")
    tail.set_defaults(fn=cmd_tail)

    retry = sub.add_parser("retry", help="give a blocked step fresh attempts, then resume the job")
    retry.add_argument("job_id")
    retry.add_argument("node_id")
    retry.add_argument("--rung", help="ladder jobs: the rung to retry on (claude, kimi, openrouter, "
                                      "openrouter:<model>, local, free); default: the ladder again from the start")
    retry.set_defaults(fn=cmd_retry)

    ladder_p = sub.add_parser("ladder", help="the delegation ladder: rungs, which work now, Claude's budget")
    ladder_p.add_argument("--repo", default=".", help="repo to run the public check and profile checks for")
    ladder_p.add_argument("--offline", action="store_true",
                          help="skip `opencode models`, LM Studio and GitHub; only config and keys")
    ladder_p.set_defaults(fn=cmd_ladder)

    merge = sub.add_parser("merge", help="copy a finished job's branch into your repo")
    merge.add_argument("job_id")
    merge.add_argument("--branch", help="name for the branch in your repo")
    merge.add_argument("--yes", action="store_true")
    merge.add_argument("--allow-unverified", action="store_true",
                       help="hand the branch back even though some nodes could not be fully checked "
                            "(unverified); without this flag merge refuses")
    merge.set_defaults(fn=cmd_merge)
    return parser


def cmd_job(argv: list[str]) -> int:
    """Entry point used by ``mullet job ...``."""
    args = build_parser().parse_args(argv)
    try:
        with Ledger() as ledger:
            return args.fn(args, ledger)
    except (CliError, KeyError, ValueError) as exc:
        print(f"mullet job: {exc}", file=sys.stderr)
        return 1
