"""A job's private clone of the user's repository.

The clone's git directory lives beside its work tree, never inside it, in a
place no sandbox can write. Every git command Mullet runs names both
directories explicitly and disables hooks, so nothing in the work tree,
including a rewritten `.git` file or a tracked hooks directory, can redirect
or hijack Mullet's git. The clone gets the base commit and its history and
nothing else: no other branches, unpushed commits, stashes or tags. The
user's own repository is only read from, until
`fetch_into` copies a finished branch back on merge.

Gates judge a snapshot, not the live work tree: `snapshot()` records the
tree the executor produced, `materialize()` makes the work tree exactly
that tree (dropping ignored files the executor may have planted), and
`commit_snapshot()` commits that same tree, so what passed is what lands.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from mullet.jobs.paths import ensure_inside, jobs_home

_FALLBACK_IDENTITY = ("Mullet", "mullet@localhost")


class GitError(RuntimeError):
    """A git command failed."""


@dataclass(frozen=True)
class ChangedFile:
    """One path that differs between two trees. Renames appear as D plus A."""

    path: str
    status: str  # A, M, D or T (type change)
    old_mode: str
    new_mode: str


@dataclass
class Snapshot:
    """The executor's result as a git tree, compared with `base`."""

    tree: str
    base: str
    files: list[ChangedFile] = field(default_factory=list)
    diff: str = ""
    added_lines: int = 0
    removed_lines: int = 0

    @property
    def empty(self) -> bool:
        """True when the tree is identical to the base."""
        return not self.files


_ENV_KEEP = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR")


def _git_env(extra: dict[str, str] | None = None, *, isolated: bool = True) -> dict[str, str]:
    """A minimal environment for git: no API keys, no GIT_* from the caller.

    `isolated` also ignores the user's global and system git config, which
    could otherwise define filter or diff drivers that a `.gitattributes` in
    the work tree switches on. It is off only for commands on the user's own
    repository, where their config is theirs to apply.
    """
    env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
    env["GIT_TERMINAL_PROMPT"] = "0"
    if isolated:
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        env["GIT_CONFIG_GLOBAL"] = os.devnull
    if extra:
        env.update(extra)
    return env


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], check: bool = True,
         stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd, cwd=cwd, env=env, input=stdin, capture_output=True, text=True,
        errors="replace",
    )
    if check and proc.returncode != 0:
        raise GitError(f"{' '.join(cmd[:6])}... failed ({proc.returncode}): {proc.stderr.strip()[:500]}")
    return proc


def read_identity(repo: Path) -> tuple[str, str]:
    """The user's git name and email for `repo`, or a neutral fallback."""
    def value(key: str) -> str:
        proc = _run(["git", "-C", str(repo), "config", "--get", key], cwd=repo,
                    env=_git_env(isolated=False), check=False)
        return proc.stdout.strip()

    name, email = value("user.name"), value("user.email")
    return (name, email) if name and email else _FALLBACK_IDENTITY


def resolve_commit(repo: Path, ref: str) -> str:
    """The full SHA of `ref` in the user's repo."""
    proc = _run(["git", "-C", str(repo), "rev-parse", "--verify", f"{ref}^{{commit}}"],
                cwd=repo, env=_git_env(isolated=False))
    return proc.stdout.strip()


class JobClone:
    """A private clone at `root/work` with its git directory at `root/git`."""

    def __init__(self, root: Path):
        self.root = ensure_inside(root, jobs_home())
        self.work = self.root / "work"
        self.git_dir = self.root / "git"

    @classmethod
    def create(cls, repo: Path, base_sha: str, root: Path, branch: str,
               identity: tuple[str, str] | None = None) -> JobClone:
        """Copy `base_sha` and its history (nothing else) from `repo`, and check out a new `branch` there."""
        clone = cls(root)
        if clone.root.exists():
            raise FileExistsError(f"{clone.root} already exists")
        clone.root.mkdir(parents=True)
        repo = Path(repo).resolve()
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", base_sha):
            raise GitError(f"the base must be a full commit hash, not {base_sha!r}")
        # Only the base commit and its history come across: a fresh git dir
        # fetches that one commit, so the user's other branches, unpushed
        # commits, stashes and tags never reach the clone (or a worker). The
        # objects travel as a pack, never hard links, so any filesystem works.
        _run(["git", "init", "--quiet", "--template=", f"--separate-git-dir={clone.git_dir}", str(clone.work)],
             cwd=clone.root, env=_git_env())
        _run(["git", "-c", "core.hooksPath=/dev/null", "-c", "protocol.version=2", f"--git-dir={clone.git_dir}",
              "fetch", "--quiet", "--no-tags", "--no-write-fetch-head", "--no-recurse-submodules",
              "--", str(repo), base_sha],
             cwd=clone.root, env=_git_env())
        name, email = identity or read_identity(repo)
        for key, value in (
            ("gc.auto", "0"), ("maintenance.auto", "false"), ("core.hooksPath", "/dev/null"),
            ("core.fsmonitor", "false"), ("commit.gpgsign", "false"),
            ("user.name", name), ("user.email", email),
        ):
            clone.git("config", key, value)
        # No remote: the only way back to the user's repo is fetch_into.
        clone.git("cat-file", "-e", f"{base_sha}^{{commit}}")
        clone.git("checkout", "--quiet", "-b", branch, base_sha)
        return clone

    def git(self, *args: str, check: bool = True, stdin: str | None = None,
            env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        """Run git against this clone with explicit directories and no hooks."""
        cmd = [
            "git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
            "-c", f"core.attributesFile={os.devnull}", "-c", "core.quotePath=false",
            f"--git-dir={self.git_dir}", f"--work-tree={self.work}", *args,
        ]
        return _run(cmd, cwd=self.work, env=_git_env(env), check=check, stdin=stdin)

    def head(self) -> str:
        """The commit the job branch points at."""
        return self.git("rev-parse", "HEAD").stdout.strip()

    def branch(self) -> str:
        """The name of the job branch."""
        return self.git("symbolic-ref", "--short", "HEAD").stdout.strip()

    def snapshot(self, base: str | None = None) -> Snapshot:
        """Record the work tree as a git tree, as `git add -A` would see it.

        A throwaway index is used, so the clone's real index is untouched.
        """
        base = base or self.head()
        fd, index = tempfile.mkstemp(prefix="index-", dir=self.root)
        os.close(fd)
        os.unlink(index)  # git wants to create the index file itself
        env = {"GIT_INDEX_FILE": index}
        try:
            self.git("read-tree", base, env=env)
            self.git("add", "--all", "--", ".", env=env)
            tree = self.git("write-tree", env=env).stdout.strip()
        finally:
            if os.path.exists(index):
                os.unlink(index)
        return self._compare(base, tree)

    def compare(self, base: str, tree: str) -> Snapshot:
        """`tree` (from a snapshot already taken) seen as a change from `base`."""
        return self._compare(base, tree)

    def _compare(self, base: str, tree: str) -> Snapshot:
        raw = self.git("diff-tree", "-r", "-z", "--no-renames", "--raw", base, tree).stdout
        files: list[ChangedFile] = []
        parts = raw.split("\0")
        i = 0
        while i + 1 < len(parts):
            meta, path = parts[i], parts[i + 1]
            i += 2
            if not meta.startswith(":"):
                continue
            old_mode, new_mode, _old, _new, status = meta[1:].split(" ")
            files.append(ChangedFile(path=path, status=status[0], old_mode=old_mode, new_mode=new_mode))
        # --text: a `-diff` attribute or a NUL byte must not turn a file into
        # "Binary files differ" and hide its lines from the gates.
        added = removed = 0
        numstat = self.git("diff", "--numstat", "--text", "--no-renames", base, tree).stdout
        for line in numstat.splitlines():
            a, r, _ = line.split("\t", 2)
            if a != "-":
                added += int(a)
            if r != "-":
                removed += int(r)
        diff = self.git(
            "diff", "--no-color", "--no-ext-diff", "--no-textconv", "--no-renames", "--text", base, tree,
        ).stdout
        return Snapshot(tree=tree, base=base, files=files, diff=diff,
                        added_lines=added, removed_lines=removed)

    def blob(self, treeish: str, path: str) -> bytes | None:
        """The bytes of `path` in `treeish`, or None if it isn't there."""
        proc = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", f"--git-dir={self.git_dir}",
             "cat-file", "blob", f"{treeish}:{path}"],
            cwd=self.work, env=_git_env(), capture_output=True,
        )
        return proc.stdout if proc.returncode == 0 else None

    def materialize(self, tree: str, keep: list[str] | None = None) -> None:
        """Make the work tree exactly `tree`, keeping only the `keep` paths.

        Untracked and ignored files are removed, so anything the executor
        hid with .gitignore can't influence the gates.
        """
        self._guard()
        self.git("read-tree", "--reset", tree)
        # Write every file from the index rather than trusting stat data.
        self.git("checkout-index", "--all", "--force", "--index", "--quiet")
        self._clean(keep)

    def commit_snapshot(self, snapshot: Snapshot, message: str, keep: list[str] | None = None) -> str:
        """Commit exactly `snapshot.tree` on the job branch and return the SHA.

        The work tree is then reset to the new commit, dropping anything the
        gates left behind.
        """
        head = self.head()
        if head != snapshot.base:
            raise GitError(f"branch moved from {snapshot.base} to {head} since the snapshot")
        sha = self.git("commit-tree", snapshot.tree, "-p", head, stdin=message).stdout.strip()
        self.git("update-ref", "-m", "mullet: node passed", "HEAD", sha, head)
        self.reset_to(sha, keep)
        return sha

    def reset_to(self, sha: str, keep: list[str] | None = None) -> None:
        """Throw away everything since `sha`. Only ever touches this clone."""
        self._guard()
        self.git("reset", "--quiet", "--hard", sha)
        self._clean(keep)

    def _clean(self, keep: list[str] | None) -> None:
        args = ["clean", "-ffdxq"]
        for path in keep or []:
            args += ["-e", "/" + path.strip("/")]
        self.git(*args)

    def fetch_into(self, repo: Path, target_branch: str) -> str:
        """Copy the job branch into the user's repo as `target_branch`.

        This is the only operation that writes to the user's repository, and
        it only adds or fast-forwards that one branch ref.
        """
        branch = self.branch()
        repo = Path(repo).resolve()
        _run(
            ["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null", "fetch", "--quiet",
             "--no-tags", str(self.git_dir), f"refs/heads/{branch}:refs/heads/{target_branch}"],
            cwd=repo, env=_git_env(isolated=False),
        )
        return resolve_commit(repo, f"refs/heads/{target_branch}")

    def remove(self) -> None:
        """Delete the clone, its git directory and its logs."""
        self._guard()
        shutil.rmtree(self.root)

    def _guard(self) -> None:
        ensure_inside(self.root, jobs_home())
        ensure_inside(self.work, jobs_home())
