"""Create and run a single-node delegation job for the MCP ``mullet_opencode`` tool.

The job is planned, approved and run in the foreground, and its work reaches
the caller's working tree only when the node passed its gates. Every outside
lookup is passed in or kept small enough for tests to replace.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

from mullet.jobs import handback
from mullet.jobs.clone import resolve_commit
from mullet.jobs.ladder import Ladder, LadderConfig, Rung, attempt_cost, is_risky
from mullet.jobs.ledger import JobStatus, Ledger
from mullet.jobs.paths import job_dir
from mullet.jobs.profile import load_profile, profile_path, repo_id, save_profile
from mullet.jobs.report import build_report
from mullet.jobs.types import AttemptOutcome, NodeSpec, NodeStatus

logger = logging.getLogger(__name__)

_ALLOWED_PROVIDERS = {"kimi", "openrouter", "free", "local", "claude"}


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def _repo_root(path: Path) -> Path:
    return Path(_git(path, "rev-parse", "--show-toplevel")).resolve()


def _base_ref(repo: Path, base_sha: str) -> str:
    ref = subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "--short", "-q", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    return ref or base_sha


def _resolve_repo(cwd: str | None, paths: list[str] | None) -> Path:
    if cwd:
        return _repo_root(Path(cwd))
    if paths:
        first = Path(paths[0])
        if first.is_absolute():
            start = first.parent if first.is_file() else first
            return _repo_root(start)
    return _repo_root(Path(os.getcwd()))


def _path_base(cwd: str | None) -> Path:
    return Path(cwd).resolve() if cwd else Path(os.getcwd()).resolve()


def _repo_relative_paths(paths: list[str] | None, repo: Path, cwd: str | None) -> list[str]:
    if not paths:
        return ["**"]
    repo = repo.resolve()
    base = _path_base(cwd)
    out: list[str] = []
    for p in paths:
        source = Path(p)
        if source.is_absolute():
            resolved = source.resolve()
        else:
            resolved = (base / source).resolve()
        try:
            rel = resolved.relative_to(repo)
        except ValueError as exc:
            raise ValueError(f"path {p!r} is outside the repo {repo}") from exc
        out.append(str(rel))
    return out


def restrict_ladder(config: LadderConfig, providers: list[str]) -> LadderConfig:
    """Return a copy of ``config`` with only the named provider rungs configured.

    ``providers`` may name ``kimi``, ``openrouter``, ``free``, ``local`` and/or
    ``claude``. Unknown names raise ``ValueError``. The Claude rung is handled
    by the profile's ``allowed_executors``, not the config, so this only
    removes opencode rungs from the config.
    """
    unknown = [p for p in providers if p not in _ALLOWED_PROVIDERS]
    if unknown:
        raise ValueError(f"unknown providers: {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    if "kimi" not in providers:
        kwargs["kimi_model"] = None
    if "openrouter" not in providers:
        kwargs["openrouter_models"] = []
    if "free" not in providers:
        kwargs["free_model"] = None
    if "local" not in providers:
        kwargs["local_model"] = None
    return dataclasses.replace(config, **kwargs)


def create_delegation_job(
    ledger: Ledger,
    *,
    task: str,
    paths: list[str] | None,
    cwd: str | None,
    timeout_min: int,
    providers: list[str] | None = None,
    ladder: Ladder | None = None,
) -> str:
    """Create and approve a single-node mechanical job for ``mullet_opencode``.

    The repo is ``cwd`` if given, else the git repo containing the first path,
    else the current working directory. The job uses the delegation ladder, a
    single mechanical node whose scope is the repo-relative paths (or the whole
    repo), and review on only when the ladder's risky-step rule says so. The
    tool call itself is recorded as the approval.
    """
    timeout_min = max(1, min(60, int(timeout_min)))
    repo = _resolve_repo(cwd, paths)
    base_sha = resolve_commit(repo, "HEAD")
    base_ref = _base_ref(repo, base_sha)

    profile = load_profile(repo, base_sha)
    stored = profile_path(repo_id(repo))
    if not stored.exists():
        save_profile(profile)

    ladder = ladder or Ladder()
    config = ladder.config
    if providers:
        config = restrict_ladder(config, providers)
        if "claude" not in providers:
            profile = profile.model_copy(
                update={"allowed_executors": [e for e in profile.allowed_executors if e != "claude-code"]}
            )

    scope = _repo_relative_paths(paths, repo, cwd)
    node = NodeSpec(id="n1", goal=task, kind="mechanical", scope=scope)
    review = is_risky(node, config)

    budget = {
        "ladder": True,
        "max_spend": config.max_spend,
        "wall_clock_s": timeout_min * 60,
        "review": review,
        "max_turns": 60,
        "executor": "opencode" if (providers and "claude" not in providers) else "claude-code",
    }

    job_id = ledger.create_job(
        str(repo), task, base_ref, base_sha, profile.model_dump(), budget=budget, max_attempts=3,
    )
    ledger.set_plan(job_id, [(node, None)])
    ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
    ledger.approve(job_id)
    ledger.add_event(job_id, "mullet_opencode_approved", {"source": "mullet_opencode"})
    return job_id


def _preflight(ledger: Ledger, job_id: str) -> str | None:
    """The reason this machine can't run the job (sandbox, executors), or None."""
    from mullet.cli.job_cli import CliError, preflight

    job = ledger.get_job(job_id)
    try:
        preflight(job.status, job)
    except CliError as exc:
        return str(exc)
    return None


def _run_in_thread(ledger_path: Path, job_id: str, runner_kwargs: dict[str, Any],
                   started: list[Any], errors: list[BaseException]) -> None:
    """Run the job with its own ledger connection (sqlite connections stay on one thread)."""
    from mullet.jobs.runner import JobRunner

    ledger = Ledger(ledger_path)
    try:
        runner = JobRunner(ledger, job_id, **runner_kwargs)
        started.append(runner)
        runner.run()
    except BaseException as exc:  # reported by the caller, never lost with the thread
        errors.append(exc)
    finally:
        ledger.close()


def _gate_lines(ledger: Ledger, attempt_id: int) -> list[str]:
    lines = []
    for result in ledger.gate_results(attempt_id):
        status = result.status.value
        line = f"{result.gate} {'passed' if status == 'pass' else status.replace('_', ' ')}"
        if status == "fail":
            first = (result.summary or "").strip().splitlines()
            line = f"{result.gate} failed" + (f": {first[0]}" if first else "")
        lines.append(line)
    return lines


def _branch_files(ledger: Ledger, job_id: str) -> list[str]:
    job = ledger.get_job(job_id)
    if not (job.branch and job.git_dir and job.clone_dir):
        return []
    try:
        return handback._changed_paths(job)
    except (handback.HandBackError, OSError):
        return []


def run_delegation(
    ledger: Ledger,
    job_id: str,
    caller_repo: Path,
    *,
    timeout_s: float,
    ladder: Ladder | None = None,
    rung_executors: Callable[[Rung], Any] | None = None,
    reviewer: Any = None,
    check: bool = True,
    stop_grace_s: float = 30.0,
) -> dict[str, Any]:
    """Run an approved delegation job in the foreground and hand back its work if it passed.

    The runner works in a thread with no signal handlers, since the MCP server
    may not be on the main thread. On timeout it is asked to stop and given
    ``stop_grace_s`` to pause the job. The caller's tree is only touched when
    the single node PASSED.
    """
    ladder = ladder or Ladder()
    status: str
    reason: str | None = None
    files: list[str] | None = None

    problem = _preflight(ledger, job_id) if check else None
    if problem:
        status, reason = "blocked", problem
    else:
        runner_kwargs: dict[str, Any] = {"ladder": ladder, "rung_executors": rung_executors,
                                         "reviewer": reviewer}
        started: list[Any] = []
        errors: list[BaseException] = []
        thread = threading.Thread(
            target=_run_in_thread, args=(ledger.path, job_id, runner_kwargs, started, errors),
            name=f"mullet-delegate-{job_id}", daemon=True,
        )
        thread.start()
        thread.join(timeout_s)
        if thread.is_alive():
            if started:
                started[0].request_stop()
            thread.join(stop_grace_s)
            status = "timeout"
            reason = f"the job did not finish within {timeout_s:g} seconds"
            if thread.is_alive():
                reason += "; it was asked to stop but is still winding down"
        elif errors:
            status, reason = "failed", f"{type(errors[0]).__name__}: {errors[0]}"
        else:
            status, reason, files = _outcome(ledger, job_id, caller_repo)

    report_path = job_dir(job_id) / "report.md"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(build_report(ledger, job_id, config=ladder.config))
    except Exception as exc:  # a report that can't be written never hides the outcome
        logger.warning("job %s: report not written: %s", job_id, exc)

    attempts = ledger.attempts(job_id)
    passing = [a for a in attempts if a.outcome is AttemptOutcome.PASSED]
    source = passing[-1] if passing else (attempts[-1] if attempts else None)
    usage = (source.usage or {}) if source else {}
    cost = sum((attempt_cost(a.usage, ladder.config)[0] for a in attempts), 0.0)
    gates = _gate_lines(ledger, attempts[-1].id) if attempts else []
    if files is None:
        files = _branch_files(ledger, job_id)

    return {
        "status": status,
        "reason": reason,
        "job_id": job_id,
        "report_path": str(report_path),
        "rung": usage.get("rung"),
        "model": usage.get("model"),
        "cost": round(float(cost), 2),
        "gates": gates,
        "files_changed": list(files),
    }


def _outcome(ledger: Ledger, job_id: str, caller_repo: Path) -> tuple[str, str | None, list[str] | None]:
    """Map the finished job's single node to a status, applying the diff only if it PASSED."""
    job = ledger.get_job(job_id)
    nodes = ledger.nodes(job_id)
    node = nodes[0] if nodes else None
    if node is not None and node.status is NodeStatus.PASSED and job.status is JobStatus.DONE:
        back = handback.apply_job_diff(job, caller_repo)
        if back.applied:
            return "applied", None, back.files
        return "passed_not_applied", back.reason, back.files
    if node is not None and node.status is NodeStatus.UNVERIFIED:
        return "unverified", "a required check could not run, so the work was not handed back", None
    if node is not None and node.status is NodeStatus.BLOCKED:
        return "blocked", node.block_reason or job.error, None
    return "failed", job.error or (f"the job ended {job.status.value}"), None


def delegate(
    ledger: Ledger,
    *,
    task: str,
    paths: list[str] | None = None,
    cwd: str | None = None,
    timeout_min: int = 20,
    providers: list[str] | None = None,
    ladder: Ladder | None = None,
    rung_executors: Callable[[Rung], Any] | None = None,
    reviewer: Any = None,
    check: bool = True,
    stop_grace_s: float = 30.0,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Create a delegation job and run it; ``timeout_s`` overrides ``timeout_min`` (for tests)."""
    ladder = ladder or Ladder()
    job_id = create_delegation_job(
        ledger, task=task, paths=paths, cwd=cwd, timeout_min=timeout_min,
        providers=providers, ladder=ladder,
    )
    if providers:
        ladder = dataclasses.replace(ladder, config=restrict_ladder(ladder.config, providers))
    job = ledger.get_job(job_id)
    if timeout_s is None:
        timeout_s = float(job.budget["wall_clock_s"])
    return run_delegation(
        ledger, job_id, Path(job.repo_path), timeout_s=timeout_s, ladder=ladder,
        rung_executors=rung_executors, reviewer=reviewer, check=check, stop_grace_s=stop_grace_s,
    )
