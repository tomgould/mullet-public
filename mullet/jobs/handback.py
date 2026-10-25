"""Bring a finished Mullet job's changes back into the caller's working tree.

The job is read-only here: every write happens in the caller's repository,
and only after the job's gates have passed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from mullet.jobs.clone import _git_env
from mullet.jobs.ledger import JobRecord


class HandBackError(RuntimeError):
    """Something went wrong while handing a job's changes back."""


@dataclass(frozen=True)
class HandBack:
    """Result of trying to apply a job's diff to the caller's working tree."""

    applied: bool
    reason: str | None
    files: list[str]


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], check: bool = True,
         stdin: bytes | str | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd, cwd=cwd, env=env, input=stdin, capture_output=True, text=True, errors="replace",
    )
    if check and proc.returncode != 0:
        raise HandBackError(
            f"{' '.join(cmd[:6])}... failed ({proc.returncode}): {proc.stderr.strip()[:500]}"
        )
    return proc


# The worker chose the file names, so a name like `:(exclude)x` or `*.py`
# must be read as a path, never as pathspec magic or a glob.
_LITERAL = {"GIT_LITERAL_PATHSPECS": "1"}


def _job_env() -> dict[str, str]:
    """An isolated environment for commands against the job's private clone."""
    return _git_env(_LITERAL, isolated=True)


def _caller_env() -> dict[str, str]:
    """An environment that keeps the caller's git config, but never prompts."""
    return _git_env(_LITERAL, isolated=False)


def _changed_paths(job: JobRecord) -> list[str]:
    """Paths that differ between the job's base and its branch.

    Renames are reported as a deletion of the old path and an addition of the
    new one so both sides can be checked for cleanliness in the caller tree.
    """
    assert job.git_dir is not None
    assert job.clone_dir is not None
    proc = _run(
        ["git", f"--git-dir={job.git_dir}", f"--work-tree={job.clone_dir}",
         "diff", "--name-only", "--no-renames", job.base_sha, job.branch or "HEAD"],
        cwd=Path(job.clone_dir), env=_job_env(),
    )
    return [line for line in proc.stdout.splitlines() if line]


def _patch(job: JobRecord) -> str:
    """The binary-safe diff between the job's base and its branch."""
    assert job.git_dir is not None
    assert job.clone_dir is not None
    proc = _run(
        ["git", f"--git-dir={job.git_dir}", f"--work-tree={job.clone_dir}",
         "diff", "--binary", job.base_sha, job.branch or "HEAD"],
        cwd=Path(job.clone_dir), env=_job_env(),
    )
    return proc.stdout


def _status_files(text: str) -> list[str]:
    """File paths from `git status --porcelain`, including both sides of a rename."""
    files: list[str] = []
    for line in text.splitlines():
        if len(line) < 3:
            continue
        rest = line[2:].lstrip()
        if " -> " in rest:
            files.extend(part.strip() for part in rest.split(" -> "))
        else:
            files.append(rest)
    return files


def _check_clean(caller_repo: Path, paths: list[str]) -> str | None:
    """Return a reason if any target path is dirty or untracked in the caller tree."""
    if not paths:
        return None
    proc = _run(
        ["git", "-C", str(caller_repo), "status", "--porcelain", "--", *paths],
        cwd=caller_repo, env=_caller_env(), check=False,
    )
    if proc.stdout.strip():
        dirty = _status_files(proc.stdout)
        return "caller has uncommitted or untracked changes in: " + ", ".join(sorted(set(dirty)))
    return None


def _base_still_valid(caller_repo: Path, base_sha: str, paths: list[str]) -> str | None:
    """Return a reason if the caller's HEAD no longer matches base for the target paths."""
    if not paths:
        return None
    known = _run(
        ["git", "-C", str(caller_repo), "rev-parse", "--verify", f"{base_sha}^{{commit}}"],
        cwd=caller_repo, env=_caller_env(), check=False,
    )
    if known.returncode != 0:
        return f"the caller's repository does not know the job's base commit {base_sha}"
    same = _run(
        ["git", "-C", str(caller_repo), "diff", "--quiet", base_sha, "HEAD", "--", *paths],
        cwd=caller_repo, env=_caller_env(), check=False,
    )
    if same.returncode != 0:
        return "the caller's branch has moved on for these files since the job's base: " + ", ".join(sorted(paths))
    return None


def apply_job_diff(job: JobRecord, caller_repo: Path) -> HandBack:
    """Apply the diff of ``job`` to ``caller_repo`` if it is safe to do so.

    The caller's index is restored afterwards so the changes are left unstaged.
    The caller's HEAD, commits and refs are never changed.
    """
    if not job.branch:
        return HandBack(applied=False, reason="the job has no branch", files=[])
    if not job.git_dir or not job.clone_dir:
        return HandBack(applied=False, reason="the job has no clone", files=[])

    paths = _changed_paths(job)
    if not paths:
        return HandBack(applied=False, reason="the job changed no files", files=[])

    reason = _check_clean(caller_repo, paths)
    if reason:
        return HandBack(applied=False, reason=reason, files=paths)

    reason = _base_still_valid(caller_repo, job.base_sha, paths)
    if reason:
        return HandBack(applied=False, reason=reason, files=paths)

    patch = _patch(job)
    if not patch.strip():
        return HandBack(applied=False, reason="the job changed no files", files=[])

    check = _run(
        ["git", "-C", str(caller_repo), "apply", "--3way", "--check"],
        cwd=caller_repo, env=_caller_env(), check=False, stdin=patch,
    )
    if check.returncode != 0:
        return HandBack(applied=False, reason=check.stderr.strip(), files=paths)

    applied = _run(
        ["git", "-C", str(caller_repo), "apply", "--3way"],
        cwd=caller_repo, env=_caller_env(), check=False, stdin=patch,
    )
    if applied.returncode != 0:
        return HandBack(applied=False, reason=applied.stderr.strip(), files=paths)

    _run(
        ["git", "-C", str(caller_repo), "reset", "-q", "--", *paths],
        cwd=caller_repo, env=_caller_env(),
    )

    return HandBack(applied=True, reason=None, files=paths)
