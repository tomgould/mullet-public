"""Run an approved job: one node at a time, gates after every attempt.

For each node the runner resets the clone to where the node starts, lets the
executor attempt it, snapshots and materializes the result, runs the gates
and records every result. The ledger judges the attempt. Only a passed or
unverified attempt is committed, and the commit is exactly the tree the
gates judged. A failed attempt is retried with the gate output as feedback
until the node runs out of attempts. When the executor can resume a
session, the retry carries on the failed attempt's session on the tree it
left, to fix rather than restart; the node's last attempt always starts
fresh. Either way the gates judge against the node's start.

Each attempt gets a context pack (`mullet.jobs.context`): the job and
parent goals, the planner's notes, what earlier nodes changed and the
repo's files, which the executor turns into its brief.

A job planned with the delegation ladder (`mullet.jobs.ladder`) picks each
attempt's executor with `executor_for`: one attempt per rung, cheapest
suitable first and Claude last, with one executor instance per rung for the
job. A provider error on a rung below the top moves the step up without
using an attempt, and a session is never resumed on another rung. Every
attempt records its rung and model in the ledger.

A gate that crashes, or a required review that gives no verdict, says
nothing about the work, so it is never judged: the checks run again on the
same work up to `MAX_EXECUTOR_CRASHES` times in a row, then the job pauses
without committing it or using an attempt, and a resume checks that work
again rather than asking the worker for it anew.

A usage limit pauses the job until the limit resets, without using up an
attempt. The owner is notified (`mullet.jobs.notify`) when a job blocks,
pauses for a usage limit or finishes. A stop request kills the executor and every check and leaves the
job paused, so a resume carries on from the last committed node. After the
last node the full test suite and build run once more on the final tree.

Run detached with `python -m mullet.jobs.runner <job id>`; `mullet job run`
does that for you.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from mullet.jobs import children, context, reconcile
from mullet.jobs.budget import earliest_start, headroom as budget_headroom, load_budget
from mullet.jobs.clone import JobClone, Snapshot
from mullet.jobs.executors.base import (
    Budget, ExecResult, Executor, ExecutorUnavailable, NodeContext, Task, with_worker_limits,
)
from mullet.jobs.gates import (
    Baseline, GateContext, GateFn, GatePaused, GateRunner, gate_build, gate_scope, gate_static,
    gate_test_weakening, gate_tests, run_check_gate, run_gates,
)
from mullet.jobs.ladder import (
    BudgetState, Ladder, LadderConfig, MachineState, PublicCheck, Rung, RungChoice, Tried, apply_night,
    is_risky, load_config, model_rung, next_rung, plan_rungs, spend_state, step_group, step_order,
)
from mullet.jobs.night import NightState, SETTINGS as NIGHT_SETTINGS
from mullet.jobs.ledger import AttemptRecord, JobRecord, Ledger, NodeRecord
from mullet.jobs.notify import notify_job, status_line
from mullet.jobs.paths import job_dir, jobs_home
from mullet.jobs.profile import Profile, expand_home, safe_env, whole_check_commands
from mullet.jobs.settings import load_settings as load_job_settings
from mullet.jobs.text import strip_control
from mullet.jobs.types import (
    AttemptOutcome, GateResult, GateStatus, JobStatus, NodeSpec, NodeStatus,
)

logger = logging.getLogger(__name__)

FEEDBACK_LIMIT = 6000
MAX_EXECUTOR_CRASHES = 3
_TAIL_LIMIT = 2500


class JobBusy(RuntimeError):
    """Another runner already holds this job."""


class StopRequested(Exception):
    """A person asked the job to stop."""


class SetupFailed(RuntimeError):
    """The profile's setup commands failed, so nothing can be judged."""


@dataclass
class RunnerOptions:
    """Knobs for one run. The defaults suit unattended jobs."""

    pause_poll_s: float = 30.0
    max_pause_s: float = 8 * 24 * 3600
    pause_margin_s: float = 60.0
    pause_at_utilization: float = 0.95


@contextmanager
def _flock(path: Path, *, blocking: bool) -> Iterator[int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise JobBusy(f"{path} is held by another runner") from exc
        yield fd
    finally:
        os.close(fd)


DEFAULT_API_WORKERS = 3


def api_worker_slots() -> int:
    """How many opencode workers may run at once across all jobs (``MULLET_API_WORKERS``, default 3)."""
    try:
        return max(1, int(os.environ.get("MULLET_API_WORKERS", DEFAULT_API_WORKERS)))
    except ValueError:
        return DEFAULT_API_WORKERS


def job_budget(job: JobRecord) -> Budget:
    """The per-attempt budget stored with the job."""
    data = job.budget or {}
    defaults = Budget()
    return Budget(max_turns=int(data.get("max_turns", defaults.max_turns)),
                  wall_clock_s=float(data.get("wall_clock_s", defaults.wall_clock_s)))


# A junit argument such as `--junitxml="$MULLET_JUNIT"`. Claude Code won't
# match a Bash rule that expands a variable, so a worker's prefixes leave it
# out; only the gates need the XML.
_JUNIT_ARG = re.compile(r"""\s+(?:-{1,2}[\w-]+[= ])?["']?\$\{?MULLET_JUNIT\}?["']?""")


def profile_commands(profile: Profile) -> list[str]:
    """The profile's check commands, as prefixes a worker may run in Bash."""
    checks = profile.checks
    out = []
    for command in (checks.test_full, checks.test_targeted, checks.lint, checks.typecheck, checks.build):
        if command:
            out.append(_JUNIT_ARG.sub("", command.split("{tests}")[0]).strip())
    return [c for c in dict.fromkeys(out) if c]


def profile_exact_commands(profile: Profile, job: JobRecord | None = None) -> list[str]:
    """The check commands an opencode worker may run exactly as written, even with `;` in them.

    Only commands the user's stored profile (or Mullet's detection) wrote,
    recorded at plan time as `trusted_commands`, and still in the job's
    profile: a repo's own `.mullet/project.yaml` can't add a runnable one.
    A job planned before this was recorded gets none.
    """
    trusted = set((job.budget or {}).get("trusted_commands") or []) if job is not None else set()
    return [c for c in whole_check_commands(profile) if c in trusted]


def job_max_spend(ledger: Ledger, job: JobRecord, default: float) -> float:
    """A ladder job's spending cap: as last raised with `mullet job resume --max-spend`, else as planned."""
    raised = [e.data for e in ledger.events(job.id) if e.kind == "max_spend"]
    if raised and raised[-1].get("value") is not None:
        return float(raised[-1]["value"])
    return float((job.budget or {}).get("max_spend", default))


def ladder_on(job: JobRecord) -> bool:
    """True for a job planned with the delegation ladder."""
    return bool((job.budget or {}).get("ladder"))


def solo_rung(job: JobRecord) -> Rung | None:
    """The rung a non-ladder opencode job's one model counts as (`model_rung`), or None for any other job.

    It decides whether the job's attempts are paid, so held to its spending cap
    and the shared limits, and whether the repo must be public (a `:free` model).
    """
    if ladder_on(job) or executor_name(job) != "opencode":
        return None
    return model_rung(str((job.budget or {}).get("executor_model") or ""))


EXECUTORS: tuple[str, ...] = ("claude-code", "opencode")


class ExecutorNotAllowed(RuntimeError):
    """The repo's profile doesn't allow the executor this job asks for."""


def executor_name(job: JobRecord) -> str:
    """The executor a job was planned with."""
    return str((job.budget or {}).get("executor") or "claude-code")


def check_executor_allowed(name: str, profile: Profile) -> None:
    """Refuse an executor the profile doesn't list in `allowed_executors`.

    Only the user's stored profile can add to that list, so a repo can't
    send its own code to a model the user hasn't chosen for it.
    """
    if name not in EXECUTORS:
        raise ExecutorNotAllowed(f"unknown executor {name!r}; use one of {', '.join(EXECUTORS)}")
    if name not in profile.allowed_executors:
        raise ExecutorNotAllowed(
            f"this repo's profile doesn't allow the {name} executor (allowed: "
            f"{', '.join(profile.allowed_executors) or 'none'}). To allow it, add {name!r} to "
            f"allowed_executors in your stored profile for {profile.repo_id}; this repo's data class "
            f"is {profile.data_class}, so its code would go to that executor's model provider.")


def claude_executor(job: JobRecord, profile: Profile, logs: Path, *, worker: bool = False):
    """The Claude Code executor: always the planner and reviewer, and the worker unless the job says otherwise."""
    from mullet.jobs.executors.claude_code import ClaudeCodeExecutor

    data = job.budget or {}
    model = (data.get("executor_model") or data.get("model")) if worker else data.get("model")
    return ClaudeCodeExecutor(
        logs, bash_commands=profile_commands(profile),
        read_paths=[expand_home(p) for p in profile.expose_ro], deps_paths=list(profile.deps_paths),
        model=model, billing=data.get("billing", "subscription"),
        preload_files=bool(data.get("preload_files", False)),
    )


def default_executor(job: JobRecord, profile: Profile, logs: Path) -> Executor:
    """The executor the job was planned with, set up from its budget and profile."""
    name = executor_name(job)
    check_executor_allowed(name, profile)
    if name == "opencode":
        data = job.budget or {}
        model = str(data.get("executor_model") or "")
        # An OpenRouter model gets the same host routing as the ladder's OpenRouter rungs.
        routed = model.startswith("openrouter/") and not model.endswith(":free")
        return opencode_executor(job, profile, logs, model, home=job_dir(job.id) / "opencode",
                                 request_fields=load_config().request_fields(Rung("openrouter", model))
                                 if routed else None)
    return claude_executor(job, profile, logs, worker=True)


def opencode_executor(job: JobRecord, profile: Profile, logs: Path, model: str, *, home: Path,
                      upstream: str | None = None, request_fields: dict[str, Any] | None = None) -> Executor:
    """An opencode worker for `model`, set up from the job's profile."""
    from mullet.jobs.executors.opencode import OpencodeExecutor

    return OpencodeExecutor(
        logs, model, home=home, bash_commands=profile_commands(profile),
        read_paths=[expand_home(p) for p in profile.expose_ro], deps_paths=list(profile.deps_paths),
        shell_env=safe_env(profile.env), formatter=profile.formatter, lsp=profile.opencode_lsp,
        upstream=upstream, exact_commands=profile_exact_commands(profile, job),
        request_fields=request_fields,
    )


BASELINE_SHARE_S = 24 * 3600.0


def baseline_key(job: JobRecord, profile: Profile) -> str:
    """What a G0 baseline depends on: the repo, base commit, profile and copy_in files' contents."""
    digest = hashlib.sha256()
    for part in (job.repo_path, job.base_sha, json.dumps(profile.model_dump(mode="json"), sort_keys=True)):
        digest.update(part.encode() + b"\0")
    for rel in profile.copy_in:
        try:
            digest.update((Path(job.repo_path) / rel).read_bytes())
        except OSError:
            digest.update(b"missing")
        digest.update(b"\0")
    return digest.hexdigest()[:32]


def shared_baseline(path: Path, sha: str, *, now: float) -> Baseline | None:
    """The baseline another job stored at `path` for `sha` within the last day, or None."""
    try:
        data = json.loads(path.read_text())
        if data.get("sha") != sha or now - float(data.get("at", 0)) > BASELINE_SHARE_S:
            return None
        return Baseline.from_dict(data)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def rung_executor(job: JobRecord, profile: Profile, rung: Rung, logs: Path, *,
                  config: LadderConfig | None = None) -> Executor:
    """The executor for one rung of the ladder: Claude Code, or opencode on the rung's model.

    Each opencode rung has its own logs and home, so sessions never cross rungs.
    """
    if rung.name == "claude":
        return claude_executor(job, profile, logs / "executor", worker=True)
    return opencode_executor(job, profile, logs / f"executor-{rung.slug}", str(rung.model),
                             home=job_dir(job.id) / "opencode" / rung.slug, upstream=rung.upstream,
                             request_fields=(config or LadderConfig()).request_fields(rung))


@dataclass
class RungPick:
    """The rung the next attempt at a step uses, or None (with the reason) when none is left."""

    rung: Rung | None
    final: bool  # the step's last try: it starts fresh, and blocks the step if it fails
    choice: RungChoice
    budget: BudgetState
    reason: str = ""

    @property
    def moves_up(self) -> bool:
        """True when a broken run on this rung passes the step on instead of counting as a crash.

        Every rung but Claude: Claude's own failures keep the crash count, so
        three in a row stop the job.
        """
        return bool(self.rung and self.rung.name != "claude")


class SpendPaused(Exception):
    """The job has no spending room left: pause rather than spend more.

    On a ladder job this is the cap reached while Claude's week is tight, so
    the remaining steps aren't all handed to Claude. `reason` says which limit
    stopped a non-ladder opencode job.
    """

    def __init__(self, node_id: str, reason: str = ""):
        super().__init__(node_id)
        self.reason = reason


def claude_usage(role: str, usage: dict) -> dict:
    """What the ledger keeps of a planner or reviewer session: turns, tokens, the CLI's cost estimate."""
    tokens = usage.get("usage") if isinstance(usage.get("usage"), dict) else {}
    keep = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    return {"role": role, "num_turns": usage.get("num_turns"), "model": usage.get("model"),
            "total_cost_usd": usage.get("total_cost_usd"),
            "tokens": {k: tokens[k] for k in keep if isinstance(tokens.get(k), (int, float))}}


CLAIM_CHARS = 1500
CLAIM_STATUSES = ("done", "blocked", "split")


def claim_status(claim: Any) -> str:
    """What a worker said about its step: done, blocked, split, or unknown."""
    status = str((claim if isinstance(claim, dict) else {}).get("status") or "").strip().lower()
    return status if status in CLAIM_STATUSES else "unknown"


def _quoted_claim(summary: str) -> list[str]:
    """The worker's summary as quoted lines: no control characters, capped, and never a trailer.

    Every line starts with "> ", so none can parse as a git trailer such as
    `Signed-off-by:` or `Co-Authored-By:`.
    """
    text = strip_control(summary.replace("\t", " ")).strip()
    if len(text) > CLAIM_CHARS:
        text = text[:CLAIM_CHARS].rstrip() + " ... (cut)"
    return [f"> {line}".rstrip() for line in text.splitlines()] or ["> (no summary)"]


def _commit_message(spec: NodeSpec, results: list[GateResult] | None = None, claim: Any = None,
                    body: str = "full") -> str:
    """A passed step's commit message: the subject, and with `body` "full" the gates and the worker's claim.

    `body` is the `jobs.commit_body` setting; "minimal" gives the subject alone.
    """
    first = " ".join(spec.goal.split())
    subject = (first[:69] + "...") if len(first) > 72 else first
    if body == "minimal":
        return subject + "\n"
    passed = [r.gate for r in results or [] if r.status is GateStatus.PASS]
    lines = [subject, "", f"Mullet step {spec.id} ({spec.kind}) passed gates {', '.join(passed) or 'none'}."]
    if isinstance(claim, dict):
        summary = claim.get("summary")
        lines += ["", f"----- worker's claim, not verified (status: {claim_status(claim)}) -----",
                  *_quoted_claim(summary if isinstance(summary, str) else ""),
                  "----- end of worker's claim -----"]
    return "\n".join(lines) + "\n"


def feedback_for(results: list[GateResult], exec_result: ExecResult | None = None,
                 redact: list[str] | None = None) -> str:
    """What to tell the next attempt about why this one failed.

    `redact` lists strings (from copy_in files) that test output may have
    printed; they are replaced before anything reaches the model.
    """
    text = _feedback_text(results, exec_result)
    for value in redact or []:
        text = text.replace(value, "[redacted]")
    return text


def _feedback_text(results: list[GateResult], exec_result: ExecResult | None) -> str:
    lines = []
    if exec_result is not None and exec_result.error:
        lines.append(f"The previous run ended with: {exec_result.error}")
    for r in results:
        if r.status is not GateStatus.FAIL:
            continue
        lines.append(f"{r.gate} failed: {r.summary}")
        if r.gate == "G7" and ("reason_lines" in r.details or "finding_lines" in r.details):
            # Reasons first, so the length cap below can't cut them off behind 20 findings.
            if r.details.get("reason_lines"):
                lines.append("  The reviewer's reasons, in its words:")
                lines += [f"  - {item}" for item in r.details["reason_lines"][:20]]
            if r.details.get("finding_lines"):
                lines.append("  The reviewer's findings on the changed files (path:line: problem), in its words:")
                lines += [f"  - {item}" for item in r.details["finding_lines"][:20]]
            continue
        for key in ("problems", "regressions"):
            items = r.details.get(key)
            if items:
                if key == "problems" and r.gate == "G7" and r.details.get("findings"):
                    lines.append("  The reviewer's findings on the changed files (path:line: problem), in its words:")
                lines += [f"  - {item}" for item in items[:20]]
        tail = r.details.get("tail") or next((f.get("tail") for f in r.details.get("failed", []) if f.get("tail")), "")
        if tail:
            lines.append("Output (last part):\n" + tail[-_TAIL_LIMIT:])
    return "\n".join(lines)[:FEEDBACK_LIMIT]


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _limit_reset(rate_limit: dict, threshold: float, now: float) -> float | None:
    """When to resume if a usage window is at or past `threshold`, else None."""
    windows = rate_limit.get("unifiedWindows") or {}
    resets = [
        _number(w.get("resetsAt")) for w in windows.values()
        if isinstance(w, dict) and _number(w.get("utilization")) >= threshold
        and _number(w.get("resetsAt")) > now
    ]
    return max(resets) if resets else None


class JobRunner:
    """Moves one approved job through its nodes."""

    def __init__(self, ledger: Ledger, job_id: str, executor: Executor | None = None,
                 options: RunnerOptions | None = None, *, reviewer: GateFn | None = None,
                 sleep: Callable[[float], None] = time.sleep, now: Callable[[], float] = time.time,
                 ladder: Ladder | None = None, rung_executors: Callable[[Rung], Executor] | None = None,
                 notify: Callable[..., object] | None = None,
                 snapshot: Callable[[], dict] | None = None):
        """`ladder` and `rung_executors` replace the user's ladder config and the per-rung
        executors of a ladder job; `notify` replaces `notify_job`. All three are for tests.
        `snapshot` reads each paid provider's own usage or balance; it runs when the job starts
        and stops so the report can compare the ledger with the providers. Without it nothing is probed."""
        self.ledger = ledger
        self._snapshot_fn = snapshot
        self.job_id = job_id
        self.options = options or RunnerOptions()
        self.root = job_dir(job_id)
        self.logs = self.root / "logs"
        self._executor = executor
        self._reviewer = reviewer
        self._stop = threading.Event()
        self._sleep = sleep
        self._now = now
        self._rate_limit: dict = {}
        self._ladder = ladder
        self._night_config: LadderConfig | None = None
        self._rung_factory = rung_executors
        self._rung_cache: dict[str, Executor] = {}
        self._public: PublicCheck | None = None
        self._notify_fn = notify or notify_job
        self._job: JobRecord | None = None
        self._profile: Profile | None = None
        self._forced_review: GateFn | None = None

    def request_stop(self) -> None:
        """Stop as soon as possible: kill running children, start no more, pause the job.

        Not safe to call from a signal handler (it takes locks); `main()` sets
        a flag in the handler and calls this from a watcher thread.
        """
        self._stop.set()
        children.request_stop()

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise StopRequested()

    def run(self) -> JobStatus:
        """Run the job until it is done, blocked, paused or stopped."""
        self.logs.mkdir(parents=True, exist_ok=True)
        with _flock(self.root / "runner.lock", blocking=False):
            # The start (in clock ticks since boot) lets `mullet job stop` tell this process from a recycled pid.
            (self.root / "runner.pid").write_text(children.pid_record())
            children.kill_stale(self.root / "children.json")
            children.configure(self.root / "children.json")
            try:
                self._take_snapshot("start")
                return self._run()
            finally:
                self._take_snapshot("end")
                children.configure(None)
                (self.root / "runner.pid").unlink(missing_ok=True)

    def _take_snapshot(self, when: str) -> None:
        """Record each paid provider's own figure, for the report's comparison with the ledger."""
        if self._snapshot_fn is None:
            return
        found = reconcile.take(self._snapshot_fn)
        if found:
            self.ledger.add_event(self.job_id, "provider_snapshot",
                                  {"when": when, "at": self._now(), "providers": found})

    def _set_status(self, status: JobStatus, error: str | None = None) -> None:
        if self.ledger.get_job(self.job_id).status is not status:
            self.ledger.set_job_status(self.job_id, status, error)
            if status is JobStatus.BLOCKED:
                self._notify("blocked", error or "")
            elif status is JobStatus.DONE:
                self._notify("finished", self._done_detail())

    def _notify(self, event: str, detail: str) -> None:
        """Tell the job's owner; a failure to is logged and never changes the job."""
        try:
            status = self._status_line()
            self._notify_fn(event, self.job_id, self.ledger.get_job(self.job_id).goal, detail,
                            **({"status": status} if status else {}))
        except Exception as exc:
            logger.warning("job %s: notification failed: %s", self.job_id, exc)

    def _status_line(self) -> str:
        """Provider credit and Claude's weekly use, for notifications. Empty without a provider probe."""
        if self._snapshot_fn is None:
            return ""
        windows = [w for w in (self._rate_limit.get("unifiedWindows") or {}).values() if isinstance(w, dict)]
        weekly = max(windows, key=lambda w: _number(w.get("resetsAt")), default=None)
        pct = None if weekly is None or weekly.get("utilization") is None else _number(weekly["utilization"]) * 100
        return status_line(reconcile.take(self._snapshot_fn), pct)

    def _done_detail(self) -> str:
        nodes = self.ledger.nodes(self.job_id)
        passed = sum(1 for n in nodes if n.status is NodeStatus.PASSED)
        unverified = sum(1 for n in nodes if n.status is NodeStatus.UNVERIFIED)
        return f"{passed} of {len(nodes)} steps passed" + (f", {unverified} unverified" if unverified else "")

    def _block(self, error: str) -> None:
        # A paused job can only be blocked by way of running again.
        if self.ledger.get_job(self.job_id).status is JobStatus.PAUSED:
            self.ledger.set_job_status(self.job_id, JobStatus.RUNNING)
        self._set_status(JobStatus.BLOCKED, error)

    def _run(self) -> JobStatus:
        job = self.ledger.get_job(self.job_id)
        if job.status in (JobStatus.APPROVED, JobStatus.PAUSED, JobStatus.BLOCKED):
            self.ledger.set_job_status(self.job_id, JobStatus.RUNNING)
        elif job.status is not JobStatus.RUNNING:
            raise ValueError(f"job {self.job_id} is {job.status.value}; only approved, paused or blocked jobs run")
        recovered = self.ledger.recover_interrupted(self.job_id)
        if recovered:
            self.ledger.add_event(self.job_id, "resumed", {"interrupted_attempts": recovered})
        for node in self.ledger.nodes(self.job_id):
            if node.status is NodeStatus.BLOCKED:
                self.ledger.unblock_node(self.job_id, node.node_id)
        try:
            return self._drive(job)
        except StopRequested:
            self._set_status(JobStatus.PAUSED, "stopped")
            return JobStatus.PAUSED
        except SpendPaused as paused:
            spend = self._spend()
            why = (paused.reason if paused.reason else
                   f"spending cap reached (${spend.spent:.2f} of ${spend.cap:.2f}) and Claude's weekly window is tight")
            self._set_status(JobStatus.PAUSED, (
                f"{why}; step {paused} is waiting. Raise the cap with: mullet job resume {self.job_id} "
                "--max-spend <dollars>, or resume later"))
            self.ledger.add_event(self.job_id, "spend_paused", {"node_id": str(paused), "spent": spend.spent,
                                                               "cap": spend.cap})
            self._notify("spend", "")
            return JobStatus.PAUSED
        except GatePaused as paused:
            logger.warning("job %s paused: %s", self.job_id, paused)
            self._set_status(JobStatus.PAUSED, (
                f"a check couldn't run, so nothing was judged: {paused}. Fix the cause, then run "
                f"mullet job resume {self.job_id} to run the checks again"))
            self._notify("paused", f"a check couldn't run, so nothing was judged. Fix the cause, then run: "
                                   f"mullet job resume {self.job_id}")
            return JobStatus.PAUSED
        except Exception as exc:  # every failure leaves the job resumable, never "done"
            logger.error("job %s blocked: %s", self.job_id, exc)
            (self.logs / "runner-error.log").write_text(traceback.format_exc())
            self._block(f"{type(exc).__name__}: {exc}")
            return JobStatus.BLOCKED

    def _drive(self, job: JobRecord) -> JobStatus:
        profile = Profile.model_validate(job.profile)
        clone = self._clone(job)
        gates = GateRunner(clone, profile, self.logs / "gates", cache_dir=self.root / "cache")
        self._return_to_recorded_head(job, clone, gates)
        self._job, self._profile = job, profile
        check_executor_allowed(executor_name(job), profile)
        solo = solo_rung(job)
        if solo is not None:
            # The ladder's prices, spending limits and public check apply to a single opencode model too.
            self._ladder = self._ladder or Ladder()
            self._check_free_model(solo)
        self._setup(gates)
        baseline = self._baseline(job, gates)
        if ladder_on(job):
            # One executor per rung, made when a step first reaches it; the
            # Claude rung's stands in for the job's single worker below.
            self._ladder = self._ladder or Ladder()
            for problem in self._ladder.config.problems:
                logger.warning("ladder config: %s", problem)
            executor = self._executor or self._rung_executor(Rung("claude"))
        else:
            executor = self._executor or default_executor(job, profile, self.logs / "executor")
        reviewer = self._make_reviewer(job, profile, executor)
        nodes = self.ledger.nodes(self.job_id)
        kids: dict[str, list[NodeRecord]] = {n.node_id: [] for n in nodes}
        for n in nodes:
            if n.parent_id:
                kids[n.parent_id].append(n)

        def process(node: NodeRecord) -> NodeStatus:
            """Children first, then the node itself: a leaf's work or a parent's integration."""
            current = self.ledger.get_node(self.job_id, node.node_id)
            if current.status in (NodeStatus.PASSED, NodeStatus.UNVERIFIED):
                return current.status
            if not kids[node.node_id]:
                return self._run_node(job, current, clone, gates, baseline, executor, reviewer)
            opened = self._open_parent(current, clone)
            for child in kids[node.node_id]:
                if process(child) is NodeStatus.BLOCKED:
                    return NodeStatus.BLOCKED
            current = self.ledger.get_node(self.job_id, node.node_id)
            return self._run_node(job, current, clone, gates, baseline, executor, reviewer,
                                  integration_base=opened)

        for root in (n for n in nodes if n.parent_id is None):
            if process(root) is NodeStatus.BLOCKED:
                blocked = next(n for n in self.ledger.nodes(self.job_id) if n.status is NodeStatus.BLOCKED)
                why = blocked.block_reason if ladder_on(job) and blocked.block_reason else "ran out of attempts"
                self._set_status(JobStatus.BLOCKED, f"node {blocked.node_id} {why}")
                return JobStatus.BLOCKED
        return self._final_check(job, clone, gates, baseline)

    def _risk_reviewer(self, job: JobRecord, profile: Profile, executor: Executor,
                       reviewer: GateFn | None = None) -> GateFn:
        """G7 for a step that must be reviewed whatever the job says: it must pass, or the attempt doesn't.

        A one-way step (in any job) and, in a ladder job, a step that deletes,
        kills or sends data are reviewed whoever did the work: by the job's
        `reviewer` when review is on, by a forced one when it is off. A block
        is a failure. A review with no verdict pauses the job, as any required
        review's does (`run_gates`), and so does an approval the reviewer only
        gave when asked again for its verdict: a step this hard to undo needs a
        review that answered in full, so a person decides instead.
        """
        if self._forced_review is None:
            base = reviewer or self._make_reviewer(job, profile, executor, force=True)
            assert base is not None

            def strict(ctx: GateContext) -> GateResult:
                result = base(ctx)
                if result.status is GateStatus.PASS and (result.details or {}).get("reasked"):
                    return GateResult("G7", GateStatus.NOT_RUN, "the safety review of a risky or one-way step "
                                      "approved it only when asked again for its verdict; check the step and "
                                      "resume", result.details, result.log_path, result.duration_s)
                if result.status is not GateStatus.FAIL:
                    return result
                return GateResult("G7", GateStatus.FAIL, "the safety review of a risky or one-way step "
                                  f"didn't pass: {result.summary}", result.details, result.log_path,
                                  result.duration_s)
            self._forced_review = strict
        return self._forced_review

    def _make_reviewer(self, job: JobRecord, profile: Profile, executor: Executor, *,
                       force: bool = False) -> GateFn | None:
        if self._reviewer is not None or not (force or (job.budget or {}).get("review")):
            return self._reviewer
        import inspect

        from mullet.jobs.review import make_reviewer
        from mullet.jobs.settings import load_settings
        if self._executor is None:
            # The reviewer is always Claude, on the job's model, whoever does the work.
            executor = claude_executor(job, profile, self.logs / "review")
        if not hasattr(executor, "ask"):
            raise RuntimeError("review is on for this job, but its executor can't run a reviewer")
        model = (job.budget or {}).get("review_model")
        try:
            params = inspect.signature(executor.ask).parameters
        except (TypeError, ValueError):
            params = {}
        # A review with no verdict is asked again in its own session, so the
        # session is kept only when the executor can resume it and asking again is on.
        reask_attempts = load_settings().reask_attempts if {"persist", "resume"} <= set(params) else 0
        keep = {"persist": True} if reask_attempts else {}

        def reask(prompt: str, session_id: str, cwd: Path, schema: dict, budget: Budget) -> ExecResult:
            return executor.ask(prompt, cwd, schema, budget, model=model, label="review", persist=True,
                                resume=session_id)

        def note_usage(usage: dict) -> None:
            self._rate_limit = usage.get("rate_limit") or self._rate_limit
            self._record_limits(usage)
            self.ledger.add_event(self.job_id, "claude_usage", claude_usage("reviewer", usage))

        return make_reviewer(
            job.goal,
            lambda prompt, cwd, schema, budget: executor.ask(prompt, cwd, schema, budget, model=model,
                                                             label="review", **keep),
            slot=lambda: self._worker_slot(executor), on_usage=note_usage,
            standards_path=profile.review_standards, base_sha=job.base_sha,
            reask=reask if reask_attempts else None, reask_attempts=reask_attempts,
        )

    def _open_parent(self, node: NodeRecord, clone: JobClone) -> str:
        """Where a parent's combined change starts: recorded once, before its first child."""
        for event in self.ledger.events(self.job_id):
            if event.kind == "parent_opened" and event.data.get("node_id") == node.node_id:
                return event.data["start_sha"]
        start = clone.head()
        self.ledger.add_event(self.job_id, "parent_opened", {"node_id": node.node_id, "start_sha": start})
        return start

    def _clone(self, job: JobRecord) -> JobClone:
        if job.clone_dir:
            return JobClone(Path(job.clone_dir).parent)
        branch = job.branch or f"mullet/{job.id}"
        clone = JobClone.create(Path(job.repo_path), job.base_sha, self.root / "clone", branch)
        self.ledger.set_job_paths(self.job_id, clone.work, clone.git_dir, branch)
        return clone

    def _return_to_recorded_head(self, job: JobRecord, clone: JobClone, gates: GateRunner) -> None:
        """Put the branch back on the last commit the ledger knows about.

        A runner killed between committing a node and recording it leaves the
        branch one commit ahead of the ledger. That commit was judged but not
        recorded, so it is dropped and the node runs again.
        """
        # The latest recorded commit belongs to the most recently finished
        # attempt; with a tree, plan order isn't completion order.
        recorded = job.base_sha
        for attempt in self.ledger.attempts(self.job_id):
            if attempt.end_sha and attempt.outcome in (AttemptOutcome.PASSED, AttemptOutcome.UNVERIFIED):
                recorded = attempt.end_sha
        if clone.head() != recorded:
            self.ledger.add_event(self.job_id, "head_reset", {"from": clone.head(), "to": recorded})
        clone.reset_to(recorded, keep=gates.keep)

    def _latest_event(self, kind: str) -> dict | None:
        found = [e.data for e in self.ledger.events(self.job_id) if e.kind == kind]
        return found[-1] if found else None

    def _setup(self, gates: GateRunner) -> None:
        done = self._latest_event("setup")
        if done and done.get("ok"):
            gates.expected_deps = done.get("deps_fingerprint")
            return
        runs = gates.setup()
        ok = all(r.ok for r in runs)
        fingerprint = gates.deps_fingerprint() if ok else None
        self.ledger.add_event(self.job_id, "setup", {
            "ok": ok, "runs": [r.to_dict() for r in runs], "deps_fingerprint": fingerprint})
        gates.expected_deps = fingerprint
        self._check_stop()
        if not ok:
            failed = next(r for r in runs if not r.ok)
            raise SetupFailed(f"setup command failed: {failed.command} (log: {failed.log_path})")

    def _baseline(self, job: JobRecord, gates: GateRunner) -> Baseline:
        stored = self._latest_event("baseline")
        if stored:
            return Baseline.from_dict(stored)
        # Jobs from the same repo, base commit and profile share one G0 run
        # (for a day): the lock makes a second job wait for the first one's.
        cache = jobs_home() / "baselines"
        cache.mkdir(parents=True, exist_ok=True)
        key = baseline_key(job, gates.profile)
        with open(cache / f"{key}.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            baseline = shared_baseline(cache / f"{key}.json", job.base_sha, now=self._now())
            if baseline is None:
                head = gates.clone.head()
                try:
                    baseline = gates.baseline(job.base_sha, Path(job.repo_path))
                except Exception as exc:
                    raise GatePaused(f"G0 (the baseline run) crashed: {type(exc).__name__}: {exc}") from exc
                gates.clone.reset_to(head, keep=gates.keep)
                self._check_stop()
                tmp = cache / f"{key}.json.tmp"
                tmp.write_text(json.dumps({"at": self._now(), **baseline.to_dict()}))
                os.replace(tmp, cache / f"{key}.json")
            else:
                self.ledger.add_event(self.job_id, "baseline_shared", {"key": key})
        self.ledger.add_event(self.job_id, "baseline", baseline.to_dict())
        return baseline

    def _node_context(self, job: JobRecord, node: NodeRecord, profile: Profile, clone: JobClone) -> NodeContext:
        """What the executor's brief says about the job around this node."""
        parent = self.ledger.get_node(self.job_id, node.parent_id).spec.goal if node.parent_id else None
        notes = (self._latest_event("plan_notes") or {}).get("notes") or ""
        return context.gather(clone, job.base_sha, profile, parent_goal=parent, plan_notes=str(notes))

    def _resumable(self, node_id: str) -> tuple[str, str, str | None] | None:
        """(session id, tree, rung) of the node's latest attempt, if it failed its gates.

        Lets a restarted runner carry on fixing where the last one stopped.
        Anything else (an aborted or paused attempt, no session, no recorded
        tree) means the next attempt starts fresh. The rung (None outside a
        ladder job) keeps a session from being resumed on another executor.
        """
        attempts = self.ledger.attempts(self.job_id, node_id)
        if not attempts or attempts[-1].outcome is not AttemptOutcome.FAILED:
            return None
        retry = self._retry(node_id)
        if retry and int(retry.get("after_attempt") or 0) >= attempts[-1].id:
            return None  # a retry starts fresh
        usage = attempts[-1].usage or {}
        session = usage.get("session_id")
        g1 = next((r for r in self.ledger.gate_results(attempts[-1].id) if r.gate == "G1"), None)
        tree = g1.details.get("tree") if g1 else None
        return (str(session), str(tree), usage.get("rung_key")) if session and tree else None

    def _pending_check(self, node_id: str, start_sha: str, clone: JobClone) -> tuple[AttemptRecord, str] | None:
        """The node's latest attempt and the tree it left, if the job paused because its checks couldn't run.

        A resumed job checks that work again instead of asking for it anew.
        A retry, a different start or a tree git no longer has means starting fresh.
        """
        attempts = self.ledger.attempts(self.job_id, node_id)
        if not attempts or attempts[-1].outcome is not AttemptOutcome.PAUSED:
            return None
        latest = attempts[-1]
        retry = self._retry(node_id)
        if retry and int(retry.get("after_attempt") or 0) >= latest.id:
            return None
        paused = [e.data for e in self.ledger.events(self.job_id)
                  if e.kind == "checks_paused" and e.data.get("attempt_id") == latest.id]
        if not paused or paused[-1].get("start_sha") != start_sha:
            return None
        tree = str(paused[-1].get("tree") or "")
        if not tree or clone.git("cat-file", "-e", f"{tree}^{{tree}}", check=False).returncode != 0:
            return None
        return latest, tree

    def _carry_over(self, attempt_id: int, earlier: AttemptRecord) -> ExecResult:
        """Record on a recheck who made the work. Its cost stays on the earlier attempt, so it isn't counted twice."""
        usage = {k: v for k, v in (earlier.usage or {}).items() if k in ("rung", "rung_key", "model")}
        usage["rechecks_attempt"] = earlier.id
        self.ledger.record_executor_result(attempt_id, earlier.exit_code, usage, earlier.claim,
                                           earlier.transcript_path)
        return ExecResult(exit_code=earlier.exit_code, claim=earlier.claim)

    def _judge(self, ctx: GateContext, reviewer: GateFn | None, fix_ctx: GateContext | None) -> list[GateResult]:
        """The gates' results (with a parent's fix checked on its own), rerun on the same work if they can't finish.

        A usage limit is raised at once. A crash or a review with no verdict is
        raised after `MAX_EXECUTOR_CRASHES` runs in a row.
        """
        tries = 0
        while True:
            try:
                results = run_gates(ctx, reviewer)
                return self._check_fix(results, fix_ctx) if fix_ctx is not None else results
            except GatePaused as paused:
                tries += 1
                if paused.until is not None or tries >= MAX_EXECUTOR_CRASHES:
                    raise
                logger.warning("job %s: checks of %s rerun: %s", self.job_id, ctx.node.id, paused)
                self.ledger.add_event(self.job_id, "checks_rerun", {"node_id": ctx.node.id,
                                                                    "reason": str(paused)[:1000]})
                self._check_stop()

    # The delegation ladder

    def _rung_executor(self, rung: Rung) -> Executor:
        """The job's one executor for `rung`, made on first use. Raises ExecutorUnavailable."""
        if rung.key not in self._rung_cache:
            if self._rung_factory is not None:
                made = self._rung_factory(rung)
            elif self._executor is not None:
                made = self._executor  # a runner given one executor uses it on every rung
            else:
                assert self._job is not None and self._profile is not None
                made = rung_executor(self._job, self._profile, rung, self.logs,
                                     config=self._ladder.config if self._ladder else None)
            self._rung_cache[rung.key] = made
        return self._rung_cache[rung.key]

    def _public_state(self) -> PublicCheck:
        """The public check for the free rung: cached per repo and base for an hour (the ladder's cache).

        A GitHub failure skips the free rung for this attempt only. Each new
        answer is kept in the job's event log.
        """
        assert self._job is not None and self._profile is not None and self._ladder is not None
        try:
            found = self._ladder.public(Path(self._job.repo_path), self._job.base_sha, self._profile)
        except Exception as exc:  # the free rung fails closed, for this attempt
            found = PublicCheck(False, f"the public check failed: {type(exc).__name__}: {exc}", transient=True)
        if self._public is None or (found.public, found.reason, found.checked_at) != (
                self._public.public, self._public.reason, self._public.checked_at):
            self.ledger.add_event(self.job_id, "public_check", found.to_dict())
        self._public = found
        return found

    def _check_free_model(self, rung: Rung) -> None:
        """Refuse a free model unless the repo is public already, as the ladder's free rung does.

        Raises ExecutorNotAllowed, which blocks the job before any attempt; a
        resume asks again.
        """
        if rung.name != "free":
            return
        found = self._public_state()
        if not found.public:
            raise ExecutorNotAllowed(f"{rung.model} is a free model, which may only see a repo that is public "
                                     f"already, and this one isn't verified: {found.reason}")

    def _max_spend(self) -> float:
        """The job's spending cap: as planned, or as last raised with `mullet job resume --max-spend`."""
        assert self._job is not None and self._ladder is not None
        return job_max_spend(self.ledger, self._job, self._ladder.config.max_spend)

    def _spend(self):
        """The job's spend against its cap, with the shared budget limits folded in.

        The cap drops to what the envelope has left (this job's own spend included), and
        per-rung limits are carried along so a rung with none left is skipped.
        """
        assert self._ladder is not None
        spend = spend_state([a.usage for a in self.ledger.attempts(self.job_id)], self._max_spend(),
                            self._ladder.config)
        limits = load_budget()
        for problem in limits.problems:
            logger.warning("budget: %s", problem)
        now = self._now()
        since = earliest_start(limits, now)
        if since is None:
            return spend
        room = budget_headroom(limits, self.ledger.attempt_usages_since(since), self._ladder.config, now)
        if room.left is not None and spend.spent + max(0.0, room.left) < spend.cap:
            spend.cap = spend.spent + max(0.0, room.left)
            spend.reason = room.reason
        spend.rung_left, spend.rung_reasons = room.rung_left, room.rung_reasons
        return spend

    def _night_state(self) -> NightState:
        """The current night state, or off when the ladder's night hook fails."""
        assert self._ladder is not None
        try:
            return self._ladder.night()
        except Exception as exc:
            logger.warning("ladder night check failed: %s", exc)
            return NightState()

    def _maybe_record_night(self, state: NightState) -> None:
        """Record a night event when its on/off value changes; a job whose first step sees it off records nothing."""
        last = self._latest_event("night")
        if last is None and not state.on:
            return
        if last is not None and bool(last.get("on")) == state.on:
            return
        self.ledger.add_event(self.job_id, "night", {
            "on": state.on, "until": state.until, "reason": state.reason})

    def _rung_choice(self, spec: NodeSpec, integration: bool, *,
                     forced: bool = False) -> tuple[RungChoice, BudgetState]:
        """The rungs the step may use now. `forced` (a retry on a named rung) sets aside the
        step's kind and Claude's budget level, never the profile, the keys or the cap."""
        assert self._ladder is not None and self._profile is not None
        config = self._ladder.config
        night_state = self._night_state()
        effective_config = apply_night(config, night_state)
        self._night_config = effective_config
        budget = self._ladder.budget()
        if forced:
            spec = spec.model_copy(update={"kind": "mechanical"})
            integration = False
            budget = BudgetState("medium", budget.weekly_pressure)
        group, _ = step_group(spec, effective_config, integration=integration)
        order = step_order(budget.level, group, night=night_state.on)
        if self._profile.ladder == "claude-only":
            order = ("claude",)
        machine = MachineState(False, "not needed for this step")
        public = None
        # Only ask LM Studio and GitHub when their rung is in this step's order.
        if effective_config.local_model and "local" in order and (night_state.on or effective_config.local_by_day):
            wait_s = NIGHT_SETTINGS.local_headroom_wait_s if night_state.on else 0.0
            try:
                machine = self._ladder.machine(effective_config, wait_s=wait_s)
            except Exception as exc:
                machine = MachineState(False, f"the machine check failed: {type(exc).__name__}: {exc}")
        if effective_config.free_model and "free" in order:
            public = self._public_state()
        choice = plan_rungs(spec, self._profile, budget, machine, self._spend(), config=effective_config,
                            public=public, integration=integration, has_key=self._ladder.has_key,
                            night=night_state.on)
        self._maybe_record_night(night_state)
        return choice, budget

    def _retry(self, node_id: str) -> dict | None:
        """The latest `mullet job retry` for the node: fresh attempts after an attempt id, maybe on one rung."""
        found = [e for e in self.ledger.events(self.job_id) if e.kind == "node_retry" and e.data.get("node_id") == node_id]
        return {**found[-1].data, "event_id": found[-1].id} if found else None

    def _tried(self, node_id: str, after: int = 0) -> list[Tried]:
        """The node's attempts after attempt id `after` (a retry starts afresh), as the rung choice sees them."""
        out = []
        for attempt in self.ledger.attempts(self.job_id, node_id):
            if attempt.id <= after:
                continue
            usage = attempt.usage or {}
            counted = attempt.outcome in (AttemptOutcome.PASSED, AttemptOutcome.UNVERIFIED, AttemptOutcome.FAILED)
            key = usage.get("rung_key") or f"{attempt.executor}:"  # an integration check, or an older attempt
            moved = attempt.outcome is AttemptOutcome.ABORTED and bool(usage.get("moved_up"))
            if counted or moved:
                out.append(Tried(key, str(usage.get("rung") or attempt.executor), usage.get("model"), counted,
                                 attempt.outcome is AttemptOutcome.FAILED, moved))
        return out

    def _unblocks(self, node_id: str, since_event: int = 0) -> int:
        return sum(1 for e in self.ledger.events(self.job_id, since_id=since_event)
                   if e.kind == "node_status" and e.data.get("node_id") == node_id and e.data.get("from") == "blocked"
                   and e.data.get("to") == "pending")

    @staticmethod
    def _names(rung: Rung, wanted: str) -> bool:
        """True when `wanted` (a rung name, rung key, or name:model) names `rung`."""
        if wanted in (rung.name, rung.key):
            return True
        name, _, model = wanted.partition(":")
        return bool(model) and rung.name == name and rung.model in (model, f"openrouter/{model}")

    def executor_for(self, node: NodeRecord, *, integration: bool = False) -> RungPick:
        """The rung, and so the executor, for the next attempt at `node`.

        `next_rung` walks the step's order (from Claude's budget level and the
        step's group): one attempt per rung, Claude's share, no model twice
        after it failed the gates, Claude's turn after `max_below_claude`. The
        position comes from the ledger, so a resumed job carries on where it
        was, and the rungs are chosen afresh each time (the budget, machine and
        spend change during a job). After `mullet job retry` only later
        attempts count; a retry that names a rung gives it the job's attempts,
        as long as the profile, keys and spending cap still allow it.
        """
        assert self._job is not None and self._ladder is not None
        retry = self._retry(node.node_id)
        after = int(retry["after_attempt"]) if retry else 0
        since = int(retry["event_id"]) if retry else 0
        unblocks = max(0, self._unblocks(node.node_id, since) - (1 if retry else 0))
        forced = retry.get("rung") if retry else None
        choice, budget = self._rung_choice(node.spec, integration, forced=bool(forced))
        tried = self._tried(node.node_id, after)
        if forced:
            pick = next((r for r in choice.rungs if self._names(r, forced)), None)
            if pick is None:
                reasons = "; ".join(f"{label}: {why}" for label, why in choice.skipped if forced.split(":")[0] in label)
                return RungPick(None, True, choice, budget, f"can't retry on {forced}: {reasons or 'not configured'}")
            done = sum(1 for t in tried if t.rung_key == pick.key and t.counted)
            tries = self._job.max_attempts + unblocks
            if done >= tries:
                return RungPick(None, True, choice, budget, f"ran out of attempts retrying on {forced}")
            return RungPick(pick, done + 1 >= tries, choice, budget)
        if not choice.rungs:
            reasons = "; ".join(f"{label}: {why}" for label, why in choice.skipped)
            return RungPick(None, True, choice, budget, f"has no rung that can run it ({reasons})")
        found = next_rung(choice, tried, budget, max_attempts=self._job.max_attempts,
                          config=self._night_config or self._ladder.config, extra_claude_tries=unblocks)
        choice.skipped += found.skipped
        return RungPick(found.rung, found.final, choice, budget, found.reason)

    @contextmanager
    def _rung_lease(self, rung: Rung | None) -> Iterator[None]:
        """The GPU lease for a local attempt, never waited for: a busy GPU moves the step up."""
        if rung is None or rung.name != "local":
            yield
            return
        from mullet.jobs import machine

        with machine.gpu_lease(f"job {self.job_id} local step", wait=False):
            yield

    @staticmethod
    def _restore_failed_tree(clone: JobClone, gates: GateRunner, start_sha: str, tree: str) -> None:
        """Put back the tree a failed attempt left, with the index at the node's start.

        The worker then sees its own earlier change in `git status` and
        `git diff`, and the gates still judge everything against the start.
        """
        clone.materialize(tree, keep=gates.keep)
        clone.git("read-tree", start_sha)

    def _past_feedback(self, node: NodeRecord, redact: list[str]) -> list[str]:
        out = []
        for attempt in self.ledger.attempts(self.job_id, node.node_id):
            if attempt.outcome is AttemptOutcome.FAILED:
                text = feedback_for(self.ledger.gate_results(attempt.id), redact=redact)
                if text:
                    out.append(text)
        return out[-2:]

    @contextmanager
    def _worker_slot(self, worker: Executor) -> Iterator[None]:
        """Hold a machine-wide worker slot, giving up if a stop arrives while waiting.

        Claude workers share one slot, since they share one subscription's
        quota. opencode workers, billed per token by their provider, share
        ``api_worker_slots()`` slots of their own, so they run alongside each
        other and alongside Claude.
        """
        home = jobs_home()
        paths = ([home / f"api-worker-{i}.lock" for i in range(api_worker_slots())] if worker.name == "opencode"
                 else [home / "worker.lock"])
        with ExitStack() as stack:
            while not any(self._try_slot(stack, path) for path in paths):
                self._check_stop()
                self._sleep(min(self.options.pause_poll_s, 2.0))
            self._check_stop()
            yield

    @staticmethod
    def _try_slot(stack: ExitStack, path: Path) -> bool:
        try:
            stack.enter_context(_flock(path, blocking=False))
        except JobBusy:
            return False
        return True

    def _wait_for_limit(self, until: float) -> None:
        resume_at = min(until + self.options.pause_margin_s, self._now() + self.options.max_pause_s)
        when = time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(resume_at))
        self._set_status(JobStatus.PAUSED, f"usage limit; resumes at {when}")
        self.ledger.add_event(self.job_id, "waiting_for_limit", {"until": resume_at})
        self._notify("paused", f"a usage limit; it carries on by itself after {when}")
        while self._now() < resume_at:
            self._check_stop()
            self._sleep(max(0.0, min(self.options.pause_poll_s, resume_at - self._now())))
        self._check_stop()
        self._set_status(JobStatus.RUNNING)
        self._rate_limit = {}

    def _integration_spec(self, node: NodeRecord) -> NodeSpec:
        """What a parent's combined check allows: everything its children were allowed.

        Scope, protected paths and test removal are the union of the parent's
        and its descendants', and the size limits their sum, so work a child
        was approved to do never fails its parent's check.
        """
        below = [n.spec for n in self.ledger.nodes(self.job_id) if self._descends_from(n, node.node_id)]
        own = node.spec
        return own.model_copy(update={
            "scope": list(dict.fromkeys([*own.scope, *(p for s in below for p in s.scope)])),
            "allow_protected": list(dict.fromkeys([*own.allow_protected, *(p for s in below for p in s.allow_protected)])),
            "allow_test_weakening": own.allow_test_weakening or any(s.allow_test_weakening for s in below),
            "max_files": max(own.max_files, sum(s.max_files for s in below)),
            "max_lines": max(own.max_lines, sum(s.max_lines for s in below)),
            "expects_change": any(s.expects_change for s in below),
        })

    @staticmethod
    def _check_fix(results: list[GateResult], fix_ctx: GateContext) -> list[GateResult]:
        """Judge a parent's fix on its own, under the parent's own permissions.

        The combined check can't see a fix that deletes a child's test file
        (it exists in neither end of the combined diff), so the fix's own
        change also goes through G2, G3 and G6. A check that crashes raises
        `GatePaused`, as it would in `run_gates`.
        """
        out = list(results)
        for gate_id, fn in (("G2", gate_scope), ("G3", gate_static), ("G6", gate_test_weakening)):
            try:
                own = fn(fix_ctx)
            except Exception as exc:  # Mullet's own bug: neither a pass nor the worker's failure
                raise GatePaused(f"{gate_id} crashed checking the fix: {type(exc).__name__}: {exc}") from exc
            if own.status is GateStatus.FAIL:
                for i, r in enumerate(out):
                    if r.gate == gate_id:
                        out[i] = GateResult(gate_id, GateStatus.FAIL, f"in the fix itself: {own.summary}",
                                            own.details, own.log_path, r.duration_s + own.duration_s)
        return out

    def _descends_from(self, node: NodeRecord, ancestor: str) -> bool:
        parents = {n.node_id: n.parent_id for n in self.ledger.nodes(self.job_id)}
        current = node.parent_id
        while current:
            if current == ancestor:
                return True
            current = parents.get(current)
        return False

    def _run_node(self, job: JobRecord, node: NodeRecord, clone: JobClone, gates: GateRunner,
                  baseline: Baseline, executor: Executor, reviewer: GateFn | None = None,
                  integration_base: str | None = None) -> NodeStatus:
        """Attempt a node until it passes or runs out of attempts.

        A leaf is attempted by the executor and judged on its own change. A
        parent (`integration_base` set) is judged on everything since its first
        child started, with the full test suite. Its first attempt runs the
        gates only; later ones let the executor fix what they found.

        In a ladder job each attempt's executor comes from `executor_for`: one
        attempt per rung, then the next. A provider error or a broken run on a
        rung below the top moves up without using an attempt or counting as a
        crash, and a session is only resumed on the rung that started it.
        """
        spec = self._integration_spec(node) if integration_base else node.spec
        start_sha = clone.head()
        gate_base = integration_base or start_sha
        redact = gates.copy_in_values(Path(job.repo_path))
        feedback = self._past_feedback(node, redact)
        timeouts = gates.profile.timeouts
        budget = with_worker_limits(job_budget(job), load_job_settings(), (timeouts.check, timeouts.test_full))
        node_context = self._node_context(job, node, gates.profile, clone)
        ladder = ladder_on(job)
        if spec.door == "one-way" or (ladder and self._ladder is not None and is_risky(spec, self._ladder.config)):
            # A one-way step, and code that deletes, kills or sends data, is reviewed whoever writes it.
            reviewer = self._risk_reviewer(job, gates.profile, executor, reviewer)
        resume_from = self._resumable(spec.id) if ladder or hasattr(executor, "resume") else None
        solo = None if ladder else solo_rung(job)
        pending = self._pending_check(spec.id, start_sha, clone)
        crashes = 0
        while True:
            self._check_stop()
            reset_at = _limit_reset(self._rate_limit, self.options.pause_at_utilization, self._now())
            if reset_at is not None:
                self._wait_for_limit(reset_at)
            rechecking, pending = pending, None
            use_executor = integration_base is None or any(
                a.outcome is AttemptOutcome.FAILED for a in self.ledger.attempts(self.job_id, spec.id))
            pick: RungPick | None = None
            worker = executor
            if rechecking is not None:
                # The job paused because this work's checks couldn't run: check it again, don't redo it.
                name = rechecking[0].executor
                use_executor = name != "integration-check"
                last = False
            elif use_executor and ladder:
                pick = self.executor_for(self.ledger.get_node(self.job_id, spec.id),
                                         integration=integration_base is not None)
                if pick.rung is None:
                    self.ledger.block_node(self.job_id, spec.id, pick.reason)
                    return NodeStatus.BLOCKED
                if pick.rung.name == "claude" and pick.choice.capped and pick.budget.level == "high":
                    # Over the cap with Claude's week tight: don't hand every
                    # remaining step to Claude; let the owner decide.
                    raise SpendPaused(spec.id)
                self.ledger.add_event(self.job_id, "ladder_choice", {
                    "node_id": spec.id, "rung": pick.rung.key, "final": pick.final,
                    "budget": pick.budget.level, "weekly_pressure": pick.budget.weekly_pressure,
                    **pick.choice.to_dict()})
                name = pick.rung.executor
                last = pick.final
            else:
                name = executor.name if use_executor else "integration-check"
                # The last attempt always starts fresh from the node's start.
                last = self.ledger.attempt_count(self.job_id, spec.id) + 1 >= job.max_attempts
                if use_executor and solo is not None and solo.paid:
                    no_room = self._no_room(solo)
                    if no_room:
                        raise SpendPaused(spec.id, no_room)
            rung = pick.rung if pick else None
            resume = resume_from if use_executor and not last else None
            if resume and ladder and (rung is None or resume[2] != rung.key):
                resume = None  # never carry a session across executors
            attempt_id = self.ledger.start_attempt(self.job_id, spec.id, name, start_sha)
            finished = False
            try:
                clone.reset_to(start_sha, keep=gates.keep)
                result = ExecResult(exit_code=0)
                if rechecking is not None:
                    result = self._carry_over(attempt_id, rechecking[0])
                elif use_executor:
                    unusable = None
                    if rung is not None:
                        try:
                            worker = self._rung_executor(rung)
                        except ExecutorUnavailable as exc:
                            unusable = str(exc)
                    if resume and not hasattr(worker, "resume"):
                        resume = None
                    task = Task(job.goal, spec, list(feedback), context=node_context, label=f"{job.id}:{spec.id}")
                    if resume and unusable is None:
                        self._restore_failed_tree(clone, gates, start_sha, resume[1])
                        self.ledger.add_event(self.job_id, "session_resumed", {
                            "attempt_id": attempt_id, "node_id": spec.id, "session_id": resume[0]})
                    attempt_started = time.time()
                    if unusable is not None:
                        result = ExecResult(exit_code=None, aborted=True, provider_error=True, error=unusable)
                    else:
                        if rung is not None or solo is not None:
                            self._arm(worker, rung or solo, attempt_id)
                        result = self._attempt(worker, task, clone.work, self._attempt_budget(budget, rung),
                                               resume, rung)
                    children.kill_in_dir(clone.work, since=attempt_started)
                    self._rate_limit = result.usage.get("rate_limit") or {}
                    self._record_limits(result.usage)
                    usage = {**result.usage, "session_id": result.session_id, "resumed": bool(resume)}
                    if rung is not None:
                        usage.update(rung=rung.name, rung_key=rung.key,
                                     model=result.usage.get("model") or rung.model,
                                     provider_error=result.provider_error,
                                     moved_up=bool(result.aborted and pick is not None and pick.moves_up))
                    elif solo is not None:
                        # Kept so the job's spend, the shared limits and the report count it.
                        usage.update(rung=solo.name, rung_key=solo.key, model=result.usage.get("model") or solo.model)
                    self.ledger.record_executor_result(attempt_id, result.exit_code, usage,
                                                       result.claim, result.transcript_path)
                    if unusable is None:
                        self._record_claim(attempt_id, spec.id, result)
                self._check_stop()
                if result.paused_until is not None:
                    self.ledger.pause_attempt(attempt_id, result.error or "usage limit")
                    finished = True
                    resume_from = None  # the session saw work that was just thrown away
                    clone.reset_to(start_sha, keep=gates.keep)
                    self._wait_for_limit(result.paused_until)
                    continue
                if result.aborted:
                    finished = True
                    resume_from = None
                    if rung is not None and pick is not None and pick.moves_up:
                        # One attempt per rung: this one couldn't work here, so the
                        # next rung takes the step. Not an attempt, not a crash.
                        kind = "provider error" if result.provider_error else "executor failed"
                        self.ledger.abort_attempt(attempt_id, f"{kind} on {rung.label()}: {result.error}")
                        self.ledger.add_event(self.job_id, "rung_moved_up", {
                            "attempt_id": attempt_id, "node_id": spec.id, "rung": rung.key,
                            "provider_error": result.provider_error, "error": str(result.error)[:500]})
                        clone.reset_to(start_sha, keep=gates.keep)
                        continue
                    # The executor itself broke (crash, lost login): not counted
                    # against the node, but three in a row stop the job.
                    self.ledger.abort_attempt(attempt_id, f"executor failed: {result.error}")
                    clone.reset_to(start_sha, keep=gates.keep)
                    crashes += 1
                    if crashes >= MAX_EXECUTOR_CRASHES:
                        raise RuntimeError(f"the executor failed {crashes} times in a row: {result.error}")
                    continue
                delta = (clone.compare(start_sha, rechecking[1]) if rechecking is not None
                         else clone.snapshot(start_sha))
                # The judged change is built from the same tree that gets committed.
                judged = delta if gate_base == start_sha else clone.compare(gate_base, delta.tree)
                clone.materialize(judged.tree, keep=gates.keep)
                ctx = GateContext(gates, spec, judged, baseline, integration_base is not None, Path(job.repo_path),
                                  tests_retired=self._tests_retired(spec.id))
                fix_ctx = (GateContext(gates, node.spec, delta, baseline, False, Path(job.repo_path))
                           if integration_base and use_executor and not delta.empty else None)
                try:
                    results = self._judge(ctx, reviewer, fix_ctx)
                except GatePaused as paused:
                    clone.reset_to(start_sha, keep=gates.keep)
                    resume_from = None
                    if paused.until is not None:
                        self.ledger.pause_attempt(attempt_id, str(paused))
                        finished = True
                        self._wait_for_limit(paused.until)
                        continue
                    # Mullet's own checks broke or gave no verdict. That says nothing
                    # about the work, so it is neither judged nor committed nor held
                    # against the worker: the job pauses, and a resume checks it again.
                    self.ledger.pause_attempt(attempt_id, f"checks couldn't run: {paused}")
                    self.ledger.add_event(self.job_id, "checks_paused", {
                        "attempt_id": attempt_id, "node_id": spec.id, "start_sha": start_sha,
                        "tree": delta.tree, "reason": str(paused)[:1000]})
                    finished = True
                    raise
                crashes = 0
                self._check_stop()
                for r in results:
                    self.ledger.record_gate(attempt_id, r)
                outcome = self.ledger.judge_attempt(attempt_id)
                if outcome in (AttemptOutcome.PASSED, AttemptOutcome.UNVERIFIED):
                    if delta.empty:  # an analysis node, or an integration check that changed nothing
                        clone.reset_to(start_sha, keep=gates.keep)
                        sha = start_sha
                    else:
                        message = _commit_message(spec, results, result.claim,
                                                  body=load_job_settings().commit_body)
                        sha = clone.commit_snapshot(delta, message, keep=gates.keep)
                    self.ledger.finish_attempt(attempt_id, end_sha=sha)
                    finished = True
                    return NodeStatus(outcome.value)
                # A ladder step blocks when no rung is left (checked at the top
                # of the loop), not when the job's attempt count runs out.
                self.ledger.finish_attempt(attempt_id, block_when_out=not ladder)
                finished = True
                clone.reset_to(start_sha, keep=gates.keep)
                feedback.append(feedback_for(results, result, redact))
                feedback = feedback[-2:]
                # Gates failed on real work: the next attempt (if not the
                # last) carries on that session from the tree it left.
                resumable = use_executor and result.session_id and hasattr(worker, "resume")
                resume_from = (result.session_id, delta.tree, rung.key if rung else None) if resumable else None
                if self.ledger.get_node(self.job_id, spec.id).status is NodeStatus.BLOCKED:
                    return NodeStatus.BLOCKED
            except BaseException:
                if not finished:
                    reason = "stopped" if self._stop.is_set() else "runner error"
                    self.ledger.abort_attempt(attempt_id, reason)
                    clone.reset_to(start_sha, keep=gates.keep)
                raise

    def _attempt_budget(self, budget: Budget, rung: Rung | None) -> Budget:
        """The job's per-attempt budget, with free and local attempts held to the effective
        `cheap_minutes` (doubled at night)."""
        if rung is None or rung.name not in ("free", "local") or self._ladder is None:
            return budget
        cheap = (self._night_config or self._ladder.config).cheap_minutes
        return replace(budget, wall_clock_s=min(budget.wall_clock_s, cheap * 60))

    def _record_claim(self, attempt_id: int, node_id: str, result: ExecResult) -> None:
        """Keep what the worker said it did (done, blocked, split, unknown), so it can be counted.

        It is a record only: the gates still judge the attempt.
        """
        self.ledger.add_event(self.job_id, "worker_claim", {
            "attempt_id": attempt_id, "node_id": node_id, "status": claim_status(result.claim),
            "aborted": result.aborted, "idle_stopped": result.idle_stopped})

    def _arm(self, worker: Executor, rung: Rung, attempt_id: int) -> None:
        """Before an opencode attempt: its share of the job's spend, its price, and live usage recording.

        The usage so far goes into the ledger after every step, so a killed
        attempt keeps its cost; the worker stops itself once its running cost
        passes what is left of the job's spending cap.
        """
        if not hasattr(worker, "on_progress") or self._ladder is None:
            return
        config = self._ladder.config
        fields = {"rung": rung.name, "rung_key": rung.key, "model": rung.model}

        def progress(usage: dict) -> None:
            self.ledger.record_progress(attempt_id, {**usage, **fields, "in_progress": True})

        worker.on_progress = progress  # type: ignore[attr-defined]
        worker.price = config.price(rung.model)  # type: ignore[attr-defined]
        worker.price_configured = bool(rung.model) and rung.model in config.prices  # type: ignore[attr-defined]
        worker.spend_limit_usd = self._allowance(rung)  # type: ignore[attr-defined]

    def _allowance(self, rung: Rung) -> float | None:
        """What one attempt on `rung` may spend: the least of the job's room under its cap (lowered
        by the envelope) and the rung's room under budget.per_rung. None for an unpaid rung."""
        if not rung.paid:
            return None
        spend = self._spend()
        allowance = max(0.0, spend.cap - spend.spent)
        return min(allowance, max(0.0, spend.rung_left.get(rung.name, allowance)))

    def _no_room(self, rung: Rung) -> str:
        """Why a paid `rung` can't start another attempt (the reasons `plan_rungs` gives), or "" while it can."""
        spend = self._spend()
        if spend.capped:
            return spend.reason or f"the job's spending cap is reached (${spend.spent:.2f} of ${spend.cap:.2f})"
        if spend.rung_left.get(rung.name, 1.0) <= 0:
            return spend.rung_reasons.get(rung.name) or f"the {rung.name} spending limit is reached"
        return ""

    @staticmethod
    def _record_limits(usage: dict) -> None:
        """Keep Claude's rate-limit figures fresh from every `claude -p` stream (worker, planner, reviewer)."""
        info = usage.get("rate_limit") if isinstance(usage, dict) else None
        if info:
            from mullet.hud.usage import record_stream_limits

            try:
                record_stream_limits(info)
            except Exception as exc:
                logger.warning("couldn't record Claude's rate limits: %s", exc)

    def _attempt(self, worker: Executor, task: Task, work: Path, budget: Budget,
                 resume: tuple[str, str, str | None] | None, rung: Rung | None) -> ExecResult:
        """One run of `worker`, in the machine-wide worker slot (and the GPU lease for a local rung).

        On a ladder rung, a worker that can't start here (no sandbox, an unknown
        model, an unreachable provider, a busy GPU) is a provider error, so the
        step moves up; outside the ladder it stops the job as before.
        """
        from mullet.jobs.machine import MachineBusy

        with self._worker_slot(worker):
            try:
                with self._rung_lease(rung):
                    if resume:
                        return worker.resume(task, work, budget, resume[0])
                    return worker.run(task, work, budget)
            except (ExecutorUnavailable, MachineBusy) as exc:
                if rung is None:
                    raise
                return ExecResult(exit_code=None, aborted=True, provider_error=True,
                                  error=f"{type(exc).__name__}: {exc}")

    def _tests_retired(self, node_id: str) -> bool:
        """Whether another step that already passed was allowed to remove tests."""
        return any(n.spec.allow_test_weakening and n.status is NodeStatus.PASSED and n.node_id != node_id
                   for n in self.ledger.nodes(self.job_id))

    def _final_check(self, job: JobRecord, clone: JobClone, gates: GateRunner, baseline: Baseline) -> JobStatus:
        head = clone.head()
        if head == job.base_sha:
            self.ledger.add_event(self.job_id, "final_check", {"skipped": "no commits"})
            self._set_status(JobStatus.DONE)
            return JobStatus.DONE
        tree = clone.git("rev-parse", f"{head}^{{tree}}").stdout.strip()
        nodes = self.ledger.nodes(self.job_id)
        node = NodeSpec(id="final-check", goal="final check of the whole job",
                        allow_test_weakening=any(n.spec.allow_test_weakening for n in nodes))
        ctx = GateContext(gates, node, Snapshot(tree=tree, base=job.base_sha), baseline,
                          full_tests=True, source_repo=Path(job.repo_path))
        results = [run_check_gate(ctx, "G4", gate_build), run_check_gate(ctx, "G5", gate_tests)]
        clone.reset_to(head, keep=gates.keep)
        self._check_stop()
        self.ledger.add_event(self.job_id, "final_check", {
            "head": head,
            "results": [{"gate": r.gate, "status": r.status.value, "summary": r.summary} for r in results],
            # When a node may remove tests, tests missing from the final run
            # can't be told apart from removed ones, so they aren't checked.
            "lost_tests_checked": not node.allow_test_weakening,
        })
        failed = [r for r in results if r.status is GateStatus.FAIL]
        if failed:
            self._set_status(JobStatus.BLOCKED, f"final check failed: {failed[0].gate} {failed[0].summary}")
            return JobStatus.BLOCKED
        self._set_status(JobStatus.DONE)
        return JobStatus.DONE


def main(argv: list[str] | None = None) -> int:
    """Entry point for a detached runner: `python -m mullet.jobs.runner <job id>`."""
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m mullet.jobs.runner <job id>", file=sys.stderr)
        return 64
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with Ledger() as ledger:
        runner = JobRunner(ledger, args[0], snapshot=reconcile.probe_providers)
        stop_wanted = threading.Event()

        # The handler only sets a flag: stopping takes locks the interrupted
        # main thread may be holding. A watcher thread does the real work.
        def on_signal(signum, frame):
            stop_wanted.set()

        def watch() -> None:
            stop_wanted.wait()
            runner.request_stop()

        threading.Thread(target=watch, daemon=True).start()
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        try:
            status = runner.run()
        except JobBusy as exc:
            print(str(exc), file=sys.stderr)
            return 75
    return {JobStatus.DONE: 0, JobStatus.PAUSED: 3}.get(status, 1)


if __name__ == "__main__":
    sys.exit(main())
