"""Gates: the checks that decide whether a node attempt passed.

Mullet runs every gate itself. Executors never report what they changed:
the gates read the snapshot `JobClone.snapshot()` took of the work tree, and
the profile's commands run in the sandbox against exactly that tree. Before
each check gate the work tree is materialized again, so neither the executor
nor an earlier check can leave anything behind that the snapshot lacks.

G0 is the baseline on the pinned base commit. It records which tests passed
and failed there, so later gates can tell a regression from an old failure.

    G1 real change    the snapshot has changes iff the node is meant to make them
    G2 scope          changed paths stay inside the node's scope, out of protected
                      paths and under size limits; no symlink escapes the repo,
                      even through other symlinks; no nested repos
    G3 static         no secrets on added lines; diff-risk flags (blocking only
                      for mechanical nodes); no mass deletion
    G6 test weakening no assertions, tests or test files removed, no skips added
    G4 build          lint, typecheck and build still pass where they passed on base
    G5 tests          every test that passed on base and ran still passes, no new
                      failures, a report was produced; failures are rerun, and a
                      test is excused as flaky only if it passed in one of the runs

The cheap static gates run first. After the first failure the rest are
recorded as skipped. A gate that crashes raises `GatePaused`, and so does a
reviewer that gives no verdict: a bug or an outage in Mullet's own checks says
nothing about the change, so it is never a verdict. Work the gates refuse to
run against (`UnsafeWorkTree`, an unsafe copy_in path) is the change's fault
and fails the gate.
"""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

from mullet.core.verification.diff_risk import classify_diff
from mullet.jobs import machine, sandbox
from mullet.jobs.clone import JobClone, Snapshot
from mullet.jobs.globs import matches_any
from mullet.jobs.profile import Profile, expand_home, safe_env
from mullet.jobs.types import GateResult, GateStatus, NodeSpec

BUILD_CHECKS: tuple[str, ...] = ("lint", "typecheck", "build")
FLAKY_RERUNS = 2
MAX_TARGETED_TESTS = 200
MAX_FILE_BYTES = 5 * 1024 * 1024
_MAX_JUNIT_BYTES = 50 * 1024 * 1024
_MAX_TEST_FILE_BYTES = 10 * 1024 * 1024
_MAX_SYMLINK_HOPS = 40
_DIFF_LINE_LIMIT = 20000

# Package caches live in the job directory, so setup never writes to the
# user's own caches and nothing one job downloads reaches another.
_JOB_CACHES = {
    "XDG_CACHE_HOME": "xdg", "PIP_CACHE_DIR": "pip", "UV_CACHE_DIR": "uv",
    "npm_config_cache": "npm", "YARN_CACHE_FOLDER": "yarn", "COMPOSER_CACHE_DIR": "composer",
    "GOMODCACHE": "gomod", "GOCACHE": "gobuild", "CARGO_HOME": "cargo",
}


class CopyInError(RuntimeError):
    """A copy_in file couldn't be placed. `unsafe` means the work tree is to blame."""

    def __init__(self, message: str, *, unsafe: bool):
        super().__init__(message)
        self.unsafe = unsafe


class UnsafeWorkTree(RuntimeError):
    """The work tree holds something the gates refuse to run against."""


class GatePaused(Exception):
    """A gate couldn't finish for reasons outside the change (a usage limit, a crash, no verdict).

    It is not a verdict: the runner sets the attempt aside without counting
    it. `until` is when a usage limit resets, or None for a crash.
    """

    def __init__(self, reason: str, until: float | None = None):
        super().__init__(reason)
        self.until = until


@dataclass
class JunitReport:
    """Test ids (`classname::name`) by outcome."""

    passed: set[str]
    failed: set[str]
    skipped: set[str]


def parse_junit(path: Path) -> JunitReport | None:
    """Read a junit XML report, or None if there isn't a readable one."""
    try:
        if not path.is_file() or path.stat().st_size > _MAX_JUNIT_BYTES:
            return None
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    report = JunitReport(set(), set(), set())
    for case in root.iter("testcase"):
        test_id = f"{case.get('classname', '')}::{case.get('name', '')}"
        tags = {child.tag for child in case}
        if tags & {"failure", "error"}:
            report.failed.add(test_id)
        elif "skipped" in tags:
            report.skipped.add(test_id)
        else:
            report.passed.add(test_id)
    return report


@dataclass
class CheckRun:
    """One run of a profile command inside the sandbox."""

    name: str
    command: str
    exit_code: int | None
    timed_out: bool
    duration_s: float
    log_path: str
    # None when the run produced no junit report.
    failing_tests: set[str] | None = None
    passing_tests: set[str] | None = None
    skipped_tests: set[str] | None = None
    tail: str = ""

    @property
    def ok(self) -> bool:
        """True when the command exited 0 within its time limit."""
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict:
        """JSON-ready form, without the output tail."""
        data = asdict(self)
        data.pop("tail")
        for key in ("failing_tests", "passing_tests", "skipped_tests"):
            value = getattr(self, key)
            data[key] = sorted(value) if value is not None else None
        return data

    @classmethod
    def from_dict(cls, data: dict) -> CheckRun:
        """Rebuild a run stored with `to_dict`."""
        sets = {k: set(data[k]) if data.get(k) is not None else None
                for k in ("failing_tests", "passing_tests", "skipped_tests")}
        return cls(**{**data, **sets})


@dataclass
class Baseline:
    """G0: how each configured check behaved on the base commit."""

    sha: str
    runs: dict[str, CheckRun] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-ready form for the ledger."""
        return {"sha": self.sha, "runs": {k: v.to_dict() for k, v in self.runs.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> Baseline:
        """Rebuild a baseline stored with `to_dict`."""
        return cls(sha=data["sha"], runs={k: CheckRun.from_dict(v) for k, v in data["runs"].items()})


def _require_real_dirs(root: Path, rel: str) -> None:
    """Raise unless every existing component of `root/rel` is a real directory."""
    current = root
    for part in PurePosixPath(rel).parts:
        current = current / part
        if current.is_symlink():
            raise UnsafeWorkTree(f"{rel}: {current.relative_to(root)} is a symlink")
        if current.exists() and not current.is_dir():
            raise UnsafeWorkTree(f"{rel}: {current.relative_to(root)} is not a directory")


class GateRunner:
    """Runs a profile's commands against one job clone, always in the sandbox."""

    def __init__(self, clone: JobClone, profile: Profile, logs_dir: Path, cache_dir: Path | None = None):
        self.clone = clone
        self.profile = profile
        self.logs_dir = logs_dir
        self.cache_dir = cache_dir or clone.root / "cache"
        self.expected_deps: str | None = None  # fingerprint taken after setup
        self._seq = 0

    @property
    def keep(self) -> list[str]:
        """Untracked paths that survive resets: the dependency directories."""
        return list(self.profile.deps_paths)

    def deps_fingerprint(self) -> str:
        """A hash over every entry in the dependency directories.

        Those directories are gitignored, so no snapshot sees them change.
        The hash covers each entry's path, mode, size, inode and ctime; ctime
        can't be set from user space, so any change to a file shows up.
        """
        digest = hashlib.sha256()
        work = self.clone.work
        for rel in sorted(self.profile.deps_paths):
            root = work / rel
            if not root.exists() or root.is_symlink():
                digest.update(f"{rel}\0absent-or-link\n".encode())
                continue
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                dirnames.sort()
                for name in sorted([*dirnames, *filenames]):
                    path = os.path.join(dirpath, name)
                    st = os.lstat(path)
                    entry = f"{os.path.relpath(path, work)}\0{st.st_mode}\0{st.st_size}\0{st.st_ino}\0{st.st_ctime_ns}\n"
                    digest.update(entry.encode("utf-8", "surrogateescape"))
        return digest.hexdigest()

    def check_deps(self) -> None:
        """Refuse dependency paths that were swapped for something else or changed since setup."""
        for rel in self.profile.deps_paths:
            _require_real_dirs(self.clone.work, rel)
        if self.expected_deps is not None and self.deps_fingerprint() != self.expected_deps:
            raise UnsafeWorkTree("the dependency directories changed after setup")

    def spec(self, out_dir: Path, *, network: bool = False, deps_writable: bool = False,
             extra_writable: list[Path] | None = None) -> sandbox.SandboxSpec:
        """The sandbox for a command: the clone, `out_dir` and the job cache writable, little else."""
        work = self.clone.work
        env = safe_env(self.profile.env)  # Mullet's own settings below, and in the sandbox, win
        for var, name in _JOB_CACHES.items():
            path = self.cache_dir / name
            path.mkdir(parents=True, exist_ok=True)
            env[var] = str(path)
        return sandbox.SandboxSpec(
            cwd=work,
            writable=[work, out_dir, self.cache_dir, *(extra_writable or [])],
            readonly_overlays=[] if deps_writable else [work / p for p in self.profile.deps_paths],
            # The git dir is readable so tools that ask git questions still work.
            expose_ro=[*(expand_home(p) for p in self.profile.expose_ro), self.clone.git_dir],
            network=network,
            env=env,
        )

    def _out_dir(self, name: str) -> Path:
        # A resumed job reuses logs_dir, so skip numbers an earlier run used.
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        while True:
            self._seq += 1
            out = self.logs_dir / f"{self._seq:03d}-{name}"
            try:
                out.mkdir()
                return out
            except FileExistsError:
                continue

    def run_command(self, name: str, command: str, timeout: float, *,
                    tests: list[str] | None = None, network: bool = False,
                    deps_writable: bool = False, extra_writable: list[Path] | None = None) -> CheckRun:
        """Run one profile command and collect its exit code and junit results."""
        out = self._out_dir(name)
        junit = out / "junit.xml"
        if tests is not None:
            # ./ stops a file named like an option (-k, --x) being read as one.
            command = command.replace("{tests}", " ".join(shlex.quote(f"./{t}") for t in tests))
        spec = self.spec(out, network=network, deps_writable=deps_writable, extra_writable=extra_writable)
        spec.env["MULLET_JUNIT"] = str(junit)
        whole_suite = name.startswith("test_full")
        with machine.suite_lease() if whole_suite else contextlib.nullcontext():
            result = sandbox.run_shell(command, spec, timeout=timeout, log_path=out / "output.log")
        report = parse_junit(junit) if self.profile.checks.junit else None
        return CheckRun(
            name=name, command=command, exit_code=result.exit_code, timed_out=result.timed_out,
            duration_s=result.duration_s, log_path=str(result.log_path),
            failing_tests=report.failed if report else None,
            passing_tests=report.passed if report else None,
            skipped_tests=report.skipped if report else None,
            tail=result.tail,
        )

    def setup(self) -> list[CheckRun]:
        """Run the profile's setup commands, the only ones allowed network access."""
        writable = [p for p in (expand_home(s) for s in self.profile.setup_writable) if p.is_dir()]
        runs = []
        for i, command in enumerate(self.profile.setup):
            run = self.run_command(f"setup{i + 1}", command, self.profile.timeouts.setup,
                                   network=True, deps_writable=True, extra_writable=writable)
            runs.append(run)
            if not run.ok:
                break
        return runs

    def timeout_for(self, name: str) -> float:
        """The time limit for a named check."""
        return self.profile.timeouts.test_full if name.startswith("test") else self.profile.timeouts.check

    def baseline(self, sha: str, source_repo: Path | None = None) -> Baseline:
        """G0: run every configured check on `sha`, each on a freshly reset tree."""
        base = Baseline(sha=sha)
        for name in (*BUILD_CHECKS, "test_full"):
            command = getattr(self.profile.checks, name)
            if not command:
                continue
            self.clone.reset_to(sha, keep=self.keep)
            with self.copied_in(source_repo):
                base.runs[name] = self.run_command(f"base-{name}", command, self.timeout_for(name))
        self.clone.reset_to(sha, keep=self.keep)
        return base

    def copy_in_values(self, source_repo: Path | None) -> list[str]:
        """Strings from the copy_in files that must never reach a model.

        Test output can print a test .env; anything sent back to the next
        attempt as feedback is scrubbed of these first. Each line counts, and
        so does the value after `=` or `:` on it.
        """
        values: set[str] = set()
        for rel in self.profile.copy_in if source_repo else []:
            try:
                text = (Path(source_repo) / rel).read_text(errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip()
                for candidate in (line, re.split(r"[=:]", line, maxsplit=1)[-1].strip().strip("'\"")):
                    if len(candidate) >= 6:
                        values.add(candidate)
        return sorted(values, key=len, reverse=True)

    @contextmanager
    def copied_in(self, source_repo: Path | None) -> Iterator[None]:
        """Place the profile's copy_in files in the clone for the duration.

        Files are only ever created, never overwritten, and never through a
        symlink, so a planted `.env -> ~/.bashrc` can't be written through.
        """
        placed: list[Path] = []
        try:
            for rel in self.profile.copy_in:
                if source_repo is None:
                    raise CopyInError(f"{rel}: no source repo to copy it from", unsafe=False)
                placed.append(self._place(Path(source_repo) / rel, rel))
            yield
        finally:
            for path in placed:
                try:
                    path.unlink()
                except OSError:
                    pass

    def _place(self, src: Path, rel: str) -> Path:
        if not src.is_file():
            raise CopyInError(f"{rel}: not a file in the source repo", unsafe=False)
        work = self.clone.work
        try:
            _require_real_dirs(work, str(PurePosixPath(rel).parent))
        except UnsafeWorkTree as exc:
            raise CopyInError(str(exc), unsafe=True) from exc
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(work / rel, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError as exc:
            raise CopyInError(f"{rel}: already exists in the work tree, refusing to overwrite it",
                              unsafe=True) from exc
        with os.fdopen(fd, "wb") as out, open(src, "rb") as data:
            shutil.copyfileobj(data, out)
        return work / rel


@dataclass
class GateContext:
    """Everything the gates need to judge one attempt."""

    runner: GateRunner
    node: NodeSpec
    snapshot: Snapshot
    baseline: Baseline
    full_tests: bool = False
    source_repo: Path | None = None
    # An earlier step of the job was allowed to remove tests, so tests missing
    # from a full run may be ones it removed rather than ones this step lost.
    tests_retired: bool = False

    @property
    def clone(self) -> JobClone:
        """The job clone."""
        return self.runner.clone

    @property
    def profile(self) -> Profile:
        """The repo profile."""
        return self.runner.profile


GateFn = Callable[[GateContext], GateResult]


def _timed(gate_id: str, fn: Callable[[], GateResult]) -> GateResult:
    started = time.monotonic()
    try:
        result = fn()
    except GatePaused:
        raise
    except CopyInError as exc:
        status = GateStatus.FAIL if exc.unsafe else GateStatus.NOT_RUN
        result = GateResult(gate_id, status, f"checks could not run: {exc}")
    except UnsafeWorkTree as exc:
        result = GateResult(gate_id, GateStatus.FAIL, f"unsafe work tree: {exc}")
    except Exception as exc:  # Mullet's own bug: neither a pass nor the worker's failure
        raise GatePaused(f"{gate_id} crashed: {type(exc).__name__}: {exc}") from exc
    result.duration_s = time.monotonic() - started
    return result


def run_check_gate(ctx: GateContext, gate_id: str, fn: GateFn) -> GateResult:
    """Run one command-running gate against a freshly materialized snapshot."""
    def go() -> GateResult:
        ctx.clone.materialize(ctx.snapshot.tree, keep=ctx.runner.keep)
        ctx.runner.check_deps()
        with ctx.runner.copied_in(ctx.source_repo):
            return fn(ctx)
    return _timed(gate_id, go)


def run_gates(ctx: GateContext, reviewer: GateFn | None = None) -> list[GateResult]:
    """Run G1 to G6 in order, then G7 if a reviewer is given; every result, skipped ones included.

    The reviewer runs last, on a freshly materialized snapshot with no copy_in
    files in place, and only when everything else passed. A given reviewer is
    required, so one that answers neither approve nor block raises
    `GatePaused` rather than leaving the attempt unverified.
    """
    static: list[tuple[str, GateFn]] = [
        ("G1", gate_real_change), ("G2", gate_scope), ("G3", gate_static), ("G6", gate_test_weakening),
    ]
    checks: list[tuple[str, GateFn]] = [("G4", gate_build), ("G5", gate_tests)]
    review: list[tuple[str, GateFn]] = [("G7", reviewer)] if reviewer else []
    order = static + checks + review
    results: list[GateResult] = []
    for i, (gate_id, fn) in enumerate(order):
        if i < len(static):
            result = _timed(gate_id, lambda fn=fn: fn(ctx))
        elif gate_id == "G7":
            def reviewed(fn=fn) -> GateResult:
                ctx.clone.materialize(ctx.snapshot.tree, keep=ctx.runner.keep)
                return fn(ctx)
            result = _timed(gate_id, reviewed)
            if result.status not in (GateStatus.PASS, GateStatus.FAIL):
                raise GatePaused(f"G7 gave no verdict: {result.summary}")
        else:
            result = run_check_gate(ctx, gate_id, fn)
        results.append(result)
        if result.status is GateStatus.FAIL:
            results += [GateResult(g, GateStatus.SKIPPED, "not run because an earlier gate failed")
                        for g, _ in order[i + 1:]]
            break
    return results


def gate_real_change(ctx: GateContext) -> GateResult:
    """G1: the attempt changed files if, and only if, the node should."""
    snap = ctx.snapshot
    details = {"files": [f.path for f in snap.files], "tree": snap.tree}
    if snap.empty and ctx.node.expects_change:
        return GateResult("G1", GateStatus.FAIL, "no files changed, but this node is meant to change code", details)
    if not snap.empty and not ctx.node.expects_change:
        return GateResult("G1", GateStatus.FAIL,
                          f"{len(snap.files)} files changed by a node that must not change files", details)
    return GateResult("G1", GateStatus.PASS,
                      f"{len(snap.files)} files changed (+{snap.added_lines} -{snap.removed_lines})", details)


@dataclass
class _TreeEntry:
    mode: str
    size: int | None


def _tree_entries(clone: JobClone, tree: str) -> dict[str, _TreeEntry]:
    out = clone.git("ls-tree", "-r", "-l", "-z", tree).stdout
    entries: dict[str, _TreeEntry] = {}
    for record in out.split("\0"):
        if not record:
            continue
        meta, path = record.split("\t", 1)
        mode, _kind, _sha, size = meta.split()
        entries[path] = _TreeEntry(mode, int(size) if size.isdigit() else None)
    return entries


def symlink_escapes(path: str, links: dict[str, str]) -> bool:
    """True if following `path` through the tree's symlinks leaves the repo.

    `links` maps every symlink in the tree to its target, so chains such as
    `x -> .` plus `y -> x/..` are resolved, not just read as text.
    """
    resolved: list[str] = []
    queue = list(PurePosixPath(path).parts)
    hops = 0
    while queue:
        part = queue.pop(0)
        if part in ("", "."):
            continue
        if part == "..":
            if not resolved:
                return True
            resolved.pop()
            continue
        resolved.append(part)
        current = "/".join(resolved)
        if current in links:
            hops += 1
            target = links[current]
            if hops > _MAX_SYMLINK_HOPS or target.startswith("/"):
                return True
            resolved.pop()
            queue = list(PurePosixPath(target).parts) + queue
    return False


def gate_scope(ctx: GateContext) -> GateResult:
    """G2: changes stay where the node is allowed to work, and in the repo."""
    node, snap = ctx.node, ctx.snapshot
    entries = _tree_entries(ctx.clone, snap.tree)
    links: dict[str, str] = {}
    for path, entry in entries.items():
        if entry.mode == "120000":
            target = ctx.clone.blob(snap.tree, path)
            links[path] = target.decode("utf-8", "replace") if target is not None else "/"
    protected = [*ctx.profile.protected_paths, *ctx.profile.copy_in]
    problems: list[str] = []
    for f in snap.files:
        if not matches_any(f.path, node.scope):
            problems.append(f"{f.path}: outside the node's scope")
        if matches_any(f.path, protected) and not matches_any(f.path, node.allow_protected):
            problems.append(f"{f.path}: protected path")
        if f.new_mode == "160000":
            problems.append(f"{f.path}: adds a nested repository or submodule")
        if f.new_mode == "120000" and symlink_escapes(f.path, links):
            problems.append(f"{f.path}: symlink points outside the repo")
        entry = entries.get(f.path)
        if entry and entry.size is not None and entry.size > MAX_FILE_BYTES:
            problems.append(f"{f.path}: {entry.size} bytes, the limit is {MAX_FILE_BYTES}")
    if len(snap.files) > node.max_files:
        problems.append(f"{len(snap.files)} files changed, the limit is {node.max_files}")
    lines = snap.added_lines + snap.removed_lines
    if lines > node.max_lines:
        problems.append(f"{lines} lines changed, the limit is {node.max_lines}")
    if problems:
        summary = problems[0] if len(problems) == 1 else f"{len(problems)} scope problems: {problems[0]}"
        return GateResult("G2", GateStatus.FAIL, summary, {"problems": problems})
    return GateResult("G2", GateStatus.PASS, "all changes inside scope")


_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b")),
    ("Slack token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("API key", re.compile(r"\bsk-(?:ant-|proj-|or-)?[A-Za-z0-9_-]{20,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("assigned secret", re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|passwd|password)\b[\"']?\s*[:=]\s*[\"']"
        r"(?=[^\"'\s]*\d)(?=[^\"'\s]*[A-Za-z])[^\"'\s]{20,}[\"']")),
]


def added_lines_between(old: str, new: str) -> list[tuple[int, str]]:
    """(line number, text) of lines in `new` that aren't in `old`.

    Worked out from the two texts rather than parsed from diff output, so no
    line content or file name can confuse it. Very large files are treated
    as entirely added, which can only over-report.
    """
    new_lines = new.splitlines()
    if not old:
        return list(enumerate(new_lines, 1))
    old_lines = old.splitlines()
    if len(old_lines) > _DIFF_LINE_LIMIT or len(new_lines) > _DIFF_LINE_LIMIT:
        return list(enumerate(new_lines, 1))
    out: list[tuple[int, str]] = []
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes():
        if tag in ("insert", "replace"):
            out += [(j + 1, new_lines[j]) for j in range(j1, j2)]
    return out


def _decode(blob: bytes | None) -> str:
    return blob.decode("utf-8", "replace") if blob else ""


def added_lines(clone: JobClone, snap: Snapshot) -> list[tuple[str, int, str]]:
    """(path, line number, text) for every line the snapshot adds, from the blobs."""
    out: list[tuple[str, int, str]] = []
    for f in snap.files:
        if f.status == "D" or f.new_mode in ("120000", "160000"):
            continue
        new = clone.blob(snap.tree, f.path)
        if new is None or len(new) > MAX_FILE_BYTES:
            continue
        old = clone.blob(snap.base, f.path) if f.status != "A" and f.old_mode not in ("120000", "160000") else None
        out += [(f.path, n, text) for n, text in added_lines_between(_decode(old), _decode(new))]
    return out


def scan_secrets(lines: list[tuple[str, int, str]]) -> list[dict]:
    """Regex scan of added lines. Findings never include the secret itself."""
    findings = []
    for path, line_no, text in lines:
        for kind, pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                findings.append({"file": path, "line": line_no, "kind": kind, "scanner": "mullet"})
                break
    return findings


def scan_gitleaks(lines: list[tuple[str, int, str]]) -> tuple[list[dict], str | None]:
    """Run gitleaks, if installed, on the added lines only. Returns findings and any error.

    Lines are written to flat-named temp files, so a `.gitleaks.toml` planted
    in the repo can't be picked up as gitleaks configuration.
    """
    binary = shutil.which("gitleaks")
    if not binary or not lines:
        return [], None
    by_file: dict[str, list[tuple[int, str]]] = {}
    for path, line_no, text in lines:
        by_file.setdefault(path, []).append((line_no, text))
    with tempfile.TemporaryDirectory(prefix="mullet-gitleaks-") as tmp:
        index: dict[str, tuple[str, list[int]]] = {}
        scan_dir = Path(tmp) / "scan"
        scan_dir.mkdir()
        for i, (path, entries) in enumerate(by_file.items()):
            name = f"{i:05d}.txt"
            (scan_dir / name).write_text("\n".join(t for _, t in entries) + "\n", encoding="utf-8")
            index[name] = (path, [n for n, _ in entries])
        report = Path(tmp) / "report.json"
        try:
            proc = subprocess.run(
                [binary, "dir", str(scan_dir), "--no-banner", "--redact", "--exit-code", "0",
                 "--log-level", "error", "--report-format", "json", "--report-path", str(report)],
                capture_output=True, text=True, timeout=120,
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tmp},
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return [], f"gitleaks failed: {exc}"
        if proc.returncode != 0:
            return [], f"gitleaks exited {proc.returncode}: {proc.stderr.strip()[:300]}"
        try:
            leaks = json.loads(report.read_text() or "[]")
        except (OSError, ValueError) as exc:
            return [], f"gitleaks report unreadable: {exc}"
    findings = []
    for leak in leaks or []:
        name = Path(str(leak.get("File", ""))).name
        if name not in index:
            continue
        path, numbers = index[name]
        start = int(leak.get("StartLine", 1)) - 1
        line_no = numbers[start] if 0 <= start < len(numbers) else 0
        findings.append({"file": path, "line": line_no, "kind": str(leak.get("RuleID", "secret")),
                         "scanner": "gitleaks"})
    return findings, None


# Prose has no literals or call arguments to change by accident, so the
# diff-risk checks, which are written for code, only make noise on it.
PROSE_SUFFIXES = frozenset({".md", ".markdown", ".rst", ".txt", ".adoc"})


def gate_static(ctx: GateContext) -> GateResult:
    """G3: secrets, risky edits and mass deletion, all without running code."""
    snap, node = ctx.snapshot, ctx.node
    lines = added_lines(ctx.clone, snap)
    secrets = scan_secrets(lines)
    leaks, gitleaks_error = scan_gitleaks(lines)
    seen = {(s["file"], s["line"]) for s in secrets}
    secrets += [leak for leak in leaks if (leak["file"], leak["line"]) not in seen]

    flags = []
    for f in snap.files:
        if f.status != "M" or f.new_mode == "120000" or Path(f.path).suffix.lower() in PROSE_SUFFIXES:
            continue
        old, new = ctx.clone.blob(snap.base, f.path), ctx.clone.blob(snap.tree, f.path)
        if old is None or new is None or max(len(old), len(new)) > MAX_FILE_BYTES:
            continue
        # Snippets are left out on purpose: they can contain the very literals at issue.
        flags += [{"file": r.file, "line": r.line, "category": r.category, "reason": r.reason}
                  for r in classify_diff(_decode(old), _decode(new), f.path)]

    deleted = [f.path for f in snap.files if f.status == "D"]
    details: dict = {"secrets": secrets, "risk_flags": flags, "deleted_files": deleted}
    if gitleaks_error:
        details["gitleaks_error"] = gitleaks_error
    problems = []
    if secrets:
        problems.append(f"{len(secrets)} possible secrets in added lines")
    if flags and node.kind == "mechanical":
        problems.append(f"{len(flags)} risky edits in a mechanical node")
    if len(deleted) > max(5, node.max_files // 2):
        problems.append(f"{len(deleted)} files deleted")
    if problems:
        return GateResult("G3", GateStatus.FAIL, "; ".join(problems), details)
    note = f", {len(flags)} risk flags recorded" if flags else ""
    return GateResult("G3", GateStatus.PASS, f"no secrets{note}", details)


_TEST_DIRS = {"tests", "test", "__tests__", "spec", "specs"}
_TEST_NAME_RES = [
    re.compile(p) for p in (
        r"^test_.+\.py$", r".+_test\.py$", r".+_test\.go$", r".+\.(test|spec)\.[cm]?[jt]sx?$",
        r".+Test\.php$", r".+_spec\.rb$", r".+_test\.rb$",
    )
]
_ASSERTION_RES = [
    re.compile(p) for p in (
        r"\bassert\b", r"\bself\.assert\w+\(", r"\bpytest\.raises\(", r"\bexpect\(",
        r"\bassert\.\w+\(", r"\$this->assert\w+\(", r"\bself::assert\w+\(",
        r"\bt\.(?:Error|Errorf|Fatal|Fatalf|Fail|FailNow)\(", r"\b(?:require|assert)\.\w+\(",
        r"\bassert(?:_eq|_ne)?!\(",
    )
]
_TEST_DEF_RES = [
    re.compile(p, re.MULTILINE) for p in (
        r"^\s*(?:async\s+)?def test\w*\s*\(", r"\b(?:it|test)\s*\(\s*[\"'`]",
        r"\bfunction test\w+\s*\(", r"^func Test\w+\(", r"#\[test\]", r"@Test\b",
    )
]
_SKIP_RES = [
    re.compile(p) for p in (
        r"@pytest\.mark\.(?:skip|skipif|xfail)", r"\bpytest\.(?:skip|xfail)\(", r"@unittest\.skip",
        r"\bSkipTest\b", r"\b(?:it|describe|test)\.(?:skip|todo)\(", r"\bx(?:it|describe|test)\(",
        r"markTestSkipped|markTestIncomplete", r"\bt\.Skip(?:f|Now)?\(", r"#\[ignore\]", r"@Disabled\b",
    )
]


def is_test_file(path: str) -> bool:
    """True for files named like tests, the only ones passed to a targeted run."""
    name = PurePosixPath(path).name
    return any(r.match(name) for r in _TEST_NAME_RES)


def is_test_path(path: str) -> bool:
    """True for test files and anything under a test directory (fixtures, helpers)."""
    return is_test_file(path) or any(p in _TEST_DIRS for p in PurePosixPath(path).parts[:-1])


def _count(patterns: list[re.Pattern[str]], text: str) -> int:
    return sum(len(p.findall(text)) for p in patterns)


def gate_test_weakening(ctx: GateContext) -> GateResult:
    """G6: tests may grow, but must not lose assertions, tests or files, or gain skips."""
    snap = ctx.snapshot
    problems: list[str] = []
    for f in snap.files:
        if not is_test_path(f.path):
            continue
        old = ctx.clone.blob(snap.base, f.path) if f.status != "A" else b""
        new = ctx.clone.blob(snap.tree, f.path) if f.status != "D" else b""
        if old is None or new is None or max(len(old), len(new)) > _MAX_TEST_FILE_BYTES:
            problems.append(f"{f.path}: test file can't be read to check it")
            continue
        old_text, new_text = _decode(old), _decode(new)
        if f.status == "D" and _count(_TEST_DEF_RES, old_text):
            problems.append(f"{f.path}: test file deleted")
            continue
        before = (_count(_ASSERTION_RES, old_text), _count(_TEST_DEF_RES, old_text), _count(_SKIP_RES, old_text))
        after = (_count(_ASSERTION_RES, new_text), _count(_TEST_DEF_RES, new_text), _count(_SKIP_RES, new_text))
        if after[0] < before[0]:
            problems.append(f"{f.path}: assertions {before[0]} -> {after[0]}")
        if after[1] < before[1]:
            problems.append(f"{f.path}: tests {before[1]} -> {after[1]}")
        if after[2] > before[2]:
            problems.append(f"{f.path}: skip markers {before[2]} -> {after[2]}")
    if problems and not ctx.node.allow_test_weakening:
        return GateResult("G6", GateStatus.FAIL, f"tests weakened: {problems[0]}", {"problems": problems})
    summary = "tests not weakened" if not problems else "tests weakened, allowed by the node"
    return GateResult("G6", GateStatus.PASS, summary, {"problems": problems})


def gate_build(ctx: GateContext) -> GateResult:
    """G4: every check that passed on the base commit still passes."""
    runs, failed, not_run = [], [], []
    for name in BUILD_CHECKS:
        command = getattr(ctx.profile.checks, name)
        if not command:
            continue
        base = ctx.baseline.runs.get(name)
        if base is None or not base.ok:
            not_run.append(f"{name}: already failing on the base commit")
            continue
        run = ctx.runner.run_command(name, command, ctx.runner.timeout_for(name))
        runs.append(run.to_dict())
        if not run.ok:
            why = "timed out" if run.timed_out else f"exit {run.exit_code}"
            failed.append({"check": name, "why": why, "tail": run.tail, "log_path": run.log_path})
    details = {"runs": runs, "failed": failed, "not_run": not_run}
    if failed:
        first = failed[0]
        return GateResult("G4", GateStatus.FAIL, f"{first['check']} {first['why']}", details, first["log_path"])
    if not_run:
        return GateResult("G4", GateStatus.NOT_RUN, "; ".join(not_run), details)
    if not runs:
        return GateResult("G4", GateStatus.PASS, "no lint, typecheck or build configured", details)
    return GateResult("G4", GateStatus.PASS, ", ".join(r["name"] for r in runs) + " passed", details)


def select_tests(snapshot: Snapshot, all_paths: list[str]) -> list[str]:
    """Test files for a targeted run: changed tests plus tests named after changed files."""
    existing = set(all_paths)
    selected = {f.path for f in snapshot.files
                if f.status != "D" and is_test_file(f.path) and f.path in existing}
    wanted: set[str] = set()
    for f in snapshot.files:
        if is_test_path(f.path):
            continue
        stem = PurePosixPath(f.path).stem
        wanted |= {f"test_{stem}.py", f"{stem}_test.py", f"{stem}_test.go", f"{stem}Test.php",
                   f"{stem}_spec.rb", f"{stem}_test.rb"}
        wanted |= {f"{stem}.{kind}.{ext}" for kind in ("test", "spec") for ext in ("js", "jsx", "ts", "tsx", "mjs", "cjs")}
    selected |= {p for p in all_paths if PurePosixPath(p).name in wanted and is_test_file(p)}
    return sorted(selected)


def _tracked_paths(clone: JobClone, tree: str) -> list[str]:
    out = clone.git("ls-tree", "-r", "-z", "--name-only", tree).stdout
    return [p for p in out.split("\0") if p]


Problem = set[str] | str | None


def regressions(run: CheckRun, base: CheckRun, *, full: bool, allow_lost: bool = False) -> Problem:
    """What `run` got worse than the baseline: test ids, a reason, or None.

    With junit on the base side, every test that passed on base must pass
    again if it ran (and in a full run it must run), no test may newly fail,
    a report must exist and at least one test must run. Skipping a test that
    used to pass counts as losing it, unless `allow_lost` (a node allowed to
    remove tests). Without junit, exit codes decide.
    """
    if run.timed_out:
        return "the tests timed out"
    if base.passing_tests is not None:
        if run.passing_tests is None:
            return "the test run produced no junit report"
        ran = run.passing_tests | (run.failing_tests or set()) | (run.skipped_tests or set())
        if not ran:
            return "no tests ran"
        lost = set() if allow_lost else {
            t for t in base.passing_tests if t not in run.passing_tests and (full or t in ran)}
        bad = lost | ((run.failing_tests or set()) - (base.failing_tests or set()))
        if bad:
            return bad
        if not run.ok and not run.failing_tests:
            return f"the tests exited {run.exit_code} without reporting a failing test"
        return None
    return None if run.ok else f"the tests failed (exit {run.exit_code})"


def gate_tests(ctx: GateContext) -> GateResult:
    """G5: acceptance commands pass and the tests are no worse than on the base commit."""
    profile, runner = ctx.profile, ctx.runner
    details: dict = {"acceptance": []}
    for i, command in enumerate(ctx.node.acceptance):
        run = runner.run_command(f"acceptance{i + 1}", command, profile.timeouts.check)
        details["acceptance"].append(run.to_dict())
        if not run.ok:
            why = "timed out" if run.timed_out else f"exit {run.exit_code}"
            return GateResult("G5", GateStatus.FAIL, f"acceptance check failed ({why}): {command}",
                              {**details, "tail": run.tail}, run.log_path)

    full_cmd = profile.checks.test_full
    if not full_cmd:
        return GateResult("G5", GateStatus.NOT_RUN, "no test command configured", details)
    base = ctx.baseline.runs.get("test_full")
    if base is None:
        return GateResult("G5", GateStatus.NOT_RUN, "no baseline test run", details)
    if base.timed_out:
        return GateResult("G5", GateStatus.NOT_RUN, "tests timed out on the base commit", details)
    if not base.ok and base.passing_tests is None:
        return GateResult("G5", GateStatus.NOT_RUN,
                          "tests already fail on the base commit and give no per-test report", details)

    full = ctx.full_tests or not profile.checks.test_targeted
    name, command, tests = "test_full", full_cmd, None
    if not full:
        picked = select_tests(ctx.snapshot, _tracked_paths(ctx.clone, ctx.snapshot.tree))
        if 0 < len(picked) <= MAX_TARGETED_TESTS:
            name, command, tests = "test_targeted", profile.checks.test_targeted, picked
        else:
            full = True
    details.update(mode=name, tests=tests)

    # A step that leaves the tests alone can't have lost any, so after an
    # earlier step removed some, the missing ones aren't held against it.
    # G6 still judges any test file this step does change.
    touches_tests = any(is_test_path(f.path) for f in ctx.snapshot.files)
    allow_lost = ctx.node.allow_test_weakening or (ctx.tests_retired and not touches_tests)
    # Failures are rerun to rule out flaky tests, but only a test that passed
    # in one of the runs is excused. A test that failed and then didn't pass
    # (it was skipped, or didn't run) still counts, whichever run it failed in.
    # A run with no per-test result (a timeout, no report) decides on its own.
    failed: set[str] = set()
    passed: set[str] = set()
    runs: list[CheckRun] = []
    problem: Problem = None
    for i in range(1 + FLAKY_RERUNS):
        if i and not problem:
            break
        run_name = name if not i else f"{name}-rerun"
        runs.append(runner.run_command(run_name, command, runner.timeout_for(name), tests=tests))
        found = regressions(runs[-1], base, full=full, allow_lost=allow_lost)
        passed |= runs[-1].passing_tests or set()
        if isinstance(found, str):
            problem = found
            continue
        failed |= found or set()
        problem = (failed - passed) or None
    details["runs"] = [r.to_dict() for r in runs]
    last = runs[-1]
    if problem:
        if isinstance(problem, set):
            listed = sorted(problem)
            details["regressions"] = listed
            summary = f"{len(listed)} tests regressed, e.g. {listed[0]}"
        else:
            summary = problem
        return GateResult("G5", GateStatus.FAIL, summary, {**details, "tail": last.tail}, last.log_path)
    suffix = " after rerunning flaky failures" if len(runs) > 1 else ""
    return GateResult("G5", GateStatus.PASS, f"{name} passed{suffix}", details)
