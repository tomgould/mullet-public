"""The context pack: what a worker should know before it starts.

A model handed a bare task spends its first turns finding its way around,
and a cheap one often never does. Mullet already knows most of what it
needs: the job and parent goals, the planner's notes, what earlier nodes
changed, how the work will be checked, why earlier attempts failed, where
the files are and what is in the ones in scope. `build_brief` writes that
down as markdown. `select_context_files` picks the files worth reading up
front (the scope, its tests and its direct imports) within a token budget
sized to the model's context window.

Both executors use the brief. Claude explores well on its own, so it gets
the brief and reads files itself; opencode also gets the picked files
attached. Nothing picked is ever a protected path, a copy_in file or
anything that looks like a .env file, and every file is read without
following symlinks out of the work tree.
"""

from __future__ import annotations

import os
import posixpath
import re
import stat
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from mullet.jobs.clone import ChangedFile, JobClone, Snapshot
from mullet.jobs.executors.base import NodeContext, Task
from mullet.jobs.gates import is_test_file, select_tests
from mullet.jobs.globs import matches_any
from mullet.jobs.profile import Profile

BUDGET_SHARE = 0.35
MAX_PRELOAD_BYTES = 256 * 1024
MAX_SEED_FILES = 40
HISTORY_CHARS = 4000
NOTES_CHARS = 3000
MAP_CHARS = 8000
OUTLINE_CHARS = 8000
MAX_OUTLINE_FILES = 40
_MAP_FILES_PER_DIR = 15

# Never pre-loaded, whatever the profile says: files that usually hold secrets.
NEVER_PRELOAD: tuple[str, ...] = (
    ".env*", "*.env", ".envrc", ".netrc", ".npmrc", ".pypirc", "*.pem", "*.key", "id_rsa*", "id_ed25519*",
)

_STOPWORDS = frozenset("""
    a an and are as at be by can do for from has have in into is it its make makes new not of on or
    so than that the then this to use uses using with without add adds added update updates change
    changes fix fixes move moves file files code test tests step should all each when where which
""".split())


@dataclass
class ContextFile:
    """A file picked for pre-loading, and why."""

    path: str
    reason: str
    tokens: int


def check_commands(profile: Profile) -> dict[str, str]:
    """The profile's configured checks, by name, as the brief lists them."""
    checks = profile.checks
    names = ("lint", "typecheck", "build", "test_full", "test_targeted")
    return {name: getattr(checks, name) for name in names if getattr(checks, name)}


def gather(clone: JobClone, base_sha: str, profile: Profile, *, parent_goal: str | None = None,
           plan_notes: str = "") -> NodeContext:
    """Collect a node's context from the clone at the commit it starts from."""
    head = clone.head()
    history = ""
    if head != base_sha:
        history = clone.git("log", "--stat", "--format=commit %h %s", f"{base_sha}..{head}").stdout
    files = [p for p in clone.git("ls-tree", "-r", "-z", "--name-only", head).stdout.split("\0") if p]
    return NodeContext(
        parent_goal=parent_goal, plan_notes=plan_notes, history=_trim(history, HISTORY_CHARS),
        files=files, checks=check_commands(profile),
        exclude=[*profile.protected_paths, *profile.copy_in],
    )


def context_budget(model: str | None, share: float = BUDGET_SHARE, context_length: int | None = None) -> int:
    """Tokens to spend on pre-loaded files: `share` of the model's context window."""
    if context_length is None:
        from mullet.core.providers.model_info import get_context_length
        context_length = get_context_length(model or "")
    return max(0, int(context_length * share))


def count_tokens(text: str) -> int:
    """Tokens in `text`, as Mullet's provider code estimates them."""
    from mullet.core.providers.tokenize import estimate_tokens
    return estimate_tokens(text) if text else 0


# Reading files


def read_text(work: Path, rel: str, max_bytes: int = MAX_PRELOAD_BYTES) -> str | None:
    """The text of `rel` under `work`, or None if it isn't a regular text file inside it.

    Symlinks are never followed, not even through a directory, so a tree
    that links somewhere outside the repo can't get a host file sent to a
    model.
    """
    root = os.path.realpath(work)
    full = os.path.normpath(os.path.join(root, rel))
    # realpath resolves every link on the way, so equality means there are none.
    if not full.startswith(root + os.sep) or os.path.realpath(full) != full:
        return None
    try:
        fd = os.open(full, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)  # a FIFO must not block
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > max_bytes:
            return None
        chunks, size = [], 0
        while size <= max_bytes:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        data = b"".join(chunks)
    finally:
        os.close(fd)
    if len(data) > max_bytes:
        return None
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _excluded(path: str, exclude: list[str]) -> bool:
    return matches_any(path, NEVER_PRELOAD) or matches_any(path, exclude)


# Imports


_PY_FROM = re.compile(r"^[ \t]*from[ \t]+(\.*)([\w.]*)[ \t]+import[ \t]+(?:\(([^)]*)\)|([^\n#]*))", re.M)
_PY_IMPORT = re.compile(r"^[ \t]*import[ \t]+([\w., \t]+)", re.M)
_JS_IMPORT = re.compile(
    r"""(?:\bfrom\s*|\bimport\s*\(\s*|\brequire\s*\(\s*|^\s*import\s+)['"]([^'"\n]+)['"]""", re.M)
_PHP_USE = re.compile(r"^\s*use\s+(?:function\s+|const\s+)?\\?([A-Za-z_][\w\\]*)(?:\s+as\s+\w+)?\s*;", re.M)
_PHP_REQUIRE = re.compile(
    r"""\b(?:require|include)(?:_once)?\s*\(?\s*(__DIR__\s*\.\s*)?['"]([^'"\n]+)['"]""")
_JS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte")
_PY_ROOTS = ("", "src/", "lib/")


def _python_imports(path: str, text: str, tracked: set[str]) -> list[str]:
    here = PurePosixPath(path).parent
    out: list[str] = []

    def module(dotted: str, base: PurePosixPath | None) -> None:
        rel = dotted.replace(".", "/")
        roots = [f"{base}/" if str(base) not in ("", ".") else ""] if base is not None else list(_PY_ROOTS)
        for root in roots:
            for candidate in (f"{root}{rel}.py", f"{root}{rel}/__init__.py"):
                if candidate in tracked:
                    out.append(candidate)
                    return

    statements = sorted([*_PY_FROM.finditer(text), *_PY_IMPORT.finditer(text)], key=lambda m: m.start())
    for match in statements:
        if match.re is _PY_IMPORT:
            for item in match.group(1).split(","):
                item = item.strip().split(" as ")[0].strip()
                if item:
                    module(item, None)
            continue
        dots, name, wrapped, plain = match.groups()
        names = wrapped if wrapped is not None else plain
        base = None
        if dots:
            base = here
            for _ in range(len(dots) - 1):
                base = base.parent
        if name:
            module(name, base)
        for item in names.split(","):
            item = item.strip().split(" as ")[0].strip()
            if re.fullmatch(r"\w+", item or ""):
                module(f"{name}.{item}" if name else item, base)
    return out


def _js_imports(path: str, text: str, tracked: set[str]) -> list[str]:
    here = posixpath.dirname(path)
    out = []
    for spec in _JS_IMPORT.findall(text):
        if not spec.startswith(("./", "../")):
            continue
        target = posixpath.normpath(posixpath.join(here, spec))
        stems = [target]
        if target.endswith((".js", ".jsx", ".mjs", ".cjs")):
            stems.append(target.rsplit(".", 1)[0])  # TypeScript imports name the compiled .js
        candidates = [*stems, *(s + ext for s in stems for ext in _JS_EXTS),
                      *(f"{target}/index{ext}" for ext in _JS_EXTS)]
        found = next((c for c in candidates if c in tracked), None)
        if found:
            out.append(found)
    return out


def _php_imports(path: str, text: str, tracked: set[str], by_name: dict[str, list[str]]) -> list[str]:
    out = []
    for name in _PHP_USE.findall(text):
        parts = [p for p in name.split("\\") if p]
        candidates = by_name.get(parts[-1] + ".php", [])
        # The longest namespace tail that matches a path wins; a bare class
        # name only counts when exactly one file has it.
        for k in range(len(parts), 0, -1):
            tail = "/".join(parts[-k:]) + ".php"
            hits = [c for c in candidates if c == tail or c.endswith("/" + tail)]
            if hits and (k > 1 or len(hits) == 1):
                out.append(sorted(hits)[0])
                break
    here = posixpath.dirname(path)
    for anchored, rel in _PHP_REQUIRE.findall(text):
        rel = rel.lstrip("/") if anchored else rel
        for base in ([here] if anchored or rel.startswith(".") else [here, ""]):
            candidate = posixpath.normpath(posixpath.join(base, rel))
            if candidate in tracked:
                out.append(candidate)
                break
    return out


def direct_imports(path: str, text: str, tracked: set[str],
                   by_name: dict[str, list[str]] | None = None) -> list[str]:
    """Repo files that `path` imports directly, best effort, in the order they appear.

    Python `import` and `from`, JavaScript and TypeScript relative imports,
    and PHP `use`, `require` and `include`. Anything that doesn't resolve to
    a tracked file (the standard library, packages) is left out.
    """
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".py":
        found = _python_imports(path, text, tracked)
    elif suffix in _JS_EXTS:
        found = _js_imports(path, text, tracked)
    elif suffix in (".php", ".module", ".inc", ".install", ".theme"):
        if by_name is None:
            by_name = _by_name(tracked)
        found = _php_imports(path, text, tracked, by_name)
    else:
        return []
    return [p for p in dict.fromkeys(found) if p != path]


def _by_name(files) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for f in files:
        index[PurePosixPath(f).name].append(f)
    return index


# Picking files


def goal_words(text: str) -> set[str]:
    """Words in a goal that might name files: identifiers, split on case and underscores."""
    words = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text):
        for part in re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", token):
            part = part.lower()
            if len(part) >= 3 and part not in _STOPWORDS:
                words.add(part)
    return words


def _path_score(path: str, words: set[str]) -> int:
    parts = set(re.split(r"[/_.\-]+|(?<=[a-z0-9])(?=[A-Z])", path.lower()))
    return sum(1 for w in words if w in parts)


def scope_seeds(task: Task) -> list[str]:
    """Tracked files the node is meant to work on, most relevant first.

    A tight scope gives its files directly. A broad one (`**`, a whole
    package) gives only the files whose paths share words with the goal.
    """
    ctx = task.context or NodeContext()
    node = task.node
    words = goal_words(f"{node.goal} {task.job_goal}")
    in_scope = [f for f in ctx.files if matches_any(f, node.scope)]
    ranked = sorted(in_scope, key=lambda f: (-_path_score(f, words), f))
    if len(ranked) > MAX_SEED_FILES:
        ranked = [f for f in ranked[:MAX_SEED_FILES] if _path_score(f, words) > 0]
    return ranked


def related_tests(seeds: list[str], files: list[str], limit: int = 6) -> list[str]:
    """Test files whose names carry a seed's name and one of its directory names.

    The gates map tests by exact name (`test_runner.py` for `runner.py`);
    many repos name them after the package too (`tests/test_jobs_runner.py`
    for `mullet/jobs/runner.py`). Those are worth reading, but not worth
    running for every node, so only the context pack uses this.
    """
    tests = [f for f in files if is_test_file(f)]
    out: list[str] = []
    for seed in seeds:
        if is_test_file(seed):
            continue
        pure = PurePosixPath(seed)
        stem = pure.stem.lower()
        dirs = {d.lower() for d in pure.parts[:-1] if len(d) >= 3}
        if len(stem) < 3:
            continue
        for test in tests:
            name = PurePosixPath(test).name.lower()
            if stem in name and (not dirs or any(d in name for d in dirs)) and test not in out:
                out.append(test)
                if len(out) >= limit:
                    return out
    return out


def select_context_files(task: Task, work: Path, budget_tokens: int) -> list[ContextFile]:
    """Files to pre-load for `task`: its scope, their tests, then their direct imports.

    Files are taken in that order while they fit in `budget_tokens`; one that
    doesn't fit is skipped and smaller ones after it can still go in.
    Protected paths, copy_in files, anything that looks like a secret file,
    binaries, symlinks and very large files are never picked.
    """
    ctx = task.context or NodeContext()
    tracked = set(ctx.files)
    seeds = scope_seeds(task)
    candidates: list[tuple[str, str]] = [(p, "in scope") for p in seeds]
    changed = [ChangedFile(p, "M", "100644", "100644") for p in seeds]
    for test in select_tests(Snapshot(tree="", base="", files=changed), ctx.files):
        candidates.append((test, "test for a file in scope"))
    for test in related_tests(seeds, ctx.files):
        candidates.append((test, "test that looks related"))
    by_name = _by_name(tracked)
    texts: dict[str, str | None] = {}
    for seed in seeds:
        if is_test_file(seed):
            continue
        texts[seed] = read_text(work, seed)
        for imported in direct_imports(seed, texts[seed] or "", tracked, by_name):
            candidates.append((imported, f"imported by {seed}"))

    picked: list[ContextFile] = []
    seen: set[str] = set()
    used = 0
    for path, reason in candidates:
        if path in seen or _excluded(path, ctx.exclude):
            continue
        seen.add(path)
        text = texts[path] if path in texts else read_text(work, path)
        if not text:
            continue
        tokens = count_tokens(text) + count_tokens(path) + 8
        if used + tokens > budget_tokens:
            continue
        used += tokens
        picked.append(ContextFile(path, reason, tokens))
    return picked


def preload_block(work: Path, files: list[ContextFile]) -> str:
    """The picked files' contents as fenced markdown, for a prompt that carries them inline."""
    parts = []
    for f in files:
        text = read_text(work, f.path)
        if text is None:
            continue
        fence = "````" if "```" in text else "```"
        parts.append(f"### {f.path} ({f.reason})\n{fence}\n{text.rstrip()}\n{fence}")
    return "\n\n".join(parts)


# The brief


def repo_map(files: list[str], highlight: list[str] | None = None, max_chars: int = MAP_CHARS) -> str:
    """Tracked files grouped by directory, cut to `max_chars`.

    Directories holding `highlight` paths list their files first; the rest
    shrink to a count when space runs out.
    """
    by_dir: dict[str, list[str]] = defaultdict(list)
    for f in files:
        by_dir[posixpath.dirname(f)].append(posixpath.basename(f))
    hot = {posixpath.dirname(h) for h in highlight or []}
    order = sorted(by_dir, key=lambda d: (d not in hot, d))

    def line(d: str, full: bool) -> str:
        names = sorted(by_dir[d])
        label = (d or ".") + "/"
        if not full:
            return f"{label} ({len(names)} files)"
        shown = names if d in hot else names[:_MAP_FILES_PER_DIR]
        more = f", +{len(names) - len(shown)} more" if len(names) > len(shown) else ""
        return f"{label} {', '.join(shown)}{more}"

    lines = [line(d, True) for d in order]
    if sum(len(x) + 1 for x in lines) > max_chars:
        lines = [line(d, d in hot) for d in order]
    out, used = [], 0
    for i, text in enumerate(lines):
        if used + len(text) + 1 > max_chars:
            out.append(f"... {len(lines) - i} more directories")
            break
        out.append(text)
        used += len(text) + 1
    return "\n".join(out)


def symbol_outline(work: Path, paths: list[str], max_chars: int = OUTLINE_CHARS,
                   max_files: int = MAX_OUTLINE_FILES) -> str:
    """Classes and functions in `paths`, with line numbers, from Mullet's symbol finder."""
    from mullet.core.knowledge.symbol_finder import detect_language, list_symbols

    blocks, used = [], 0
    for rel in paths[:max_files]:
        language = detect_language(rel)
        if language is None or read_text(work, rel) is None:  # the same checks as pre-loading
            continue
        try:
            symbols = list_symbols(str(Path(work) / rel), language)
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        if not symbols:
            continue
        lines = [rel] + [
            f"{'    ' if s.parent_class else '  '}{s.line_number}: {' '.join(s.signature.split())[:160]}"
            for s in symbols
        ]
        block = "\n".join(lines)
        if used + len(block) > max_chars:
            blocks.append(f"... outline cut at {len(blocks)} files")
            break
        blocks.append(block)
        used += len(block) + 1
    return "\n".join(blocks)


def build_brief(task: Task, work: Path, *, attached: list[ContextFile] | None = None,
                rules: str = "", attached_note: str = "") -> str:
    """The markdown brief for one attempt at `task` in `work`."""
    ctx = task.context or NodeContext()
    node = task.node
    out = [
        "# Task brief from Mullet",
        "You are doing one step of a larger coding job. Mullet checks your work itself "
        "when you stop, with the checks listed below, so report honestly.",
        "## Job goal",
        task.job_goal,
        f"## This step: {node.id} ({node.kind})",
        node.goal,
    ]
    if ctx.parent_goal:
        out.append(f"It is part of: {ctx.parent_goal}")
    out.append("Files you may change (scope): " + ", ".join(f"`{s}`" for s in node.scope))
    if node.allow_protected:
        out.append("Protected files this step may change: " + ", ".join(f"`{p}`" for p in node.allow_protected))
    if node.acceptance:
        out.append("These commands must pass when you're done:\n" + "\n".join(f"    {c}" for c in node.acceptance))
    if task.feedback:
        out += ["## Why earlier attempts failed",
                "Mullet's checks rejected earlier attempts at this step:",
                "\n\n".join(task.feedback)]
    if ctx.plan_notes.strip():
        out += ["## Planner's notes", _trim(ctx.plan_notes.strip(), NOTES_CHARS)]
    if ctx.history.strip():
        out += ["## What earlier steps of this job changed", _fenced(ctx.history.strip())]
    if ctx.checks:
        out += ["## Checks Mullet runs after you stop",
                "\n".join(f"    {name}: {command}" for name, command in ctx.checks.items()),
                "Tests that passed before must still pass, and tests may be added but not weakened."]
    if attached:
        out += ["## Files attached for you",
                attached_note or "These files are already loaded, so you don't need to read them again:",
                "\n".join(f"- `{f.path}` ({f.reason})" for f in attached)]
    seeds = scope_seeds(task)
    outline = symbol_outline(work, seeds) if seeds else ""
    if outline:
        out += ["## Outline of files in scope", _fenced(outline)]
    if ctx.files:
        out += [f"## Repository map ({len(ctx.files)} tracked files)", _fenced(repo_map(ctx.files, seeds))]
    if rules:
        out += ["## Rules", rules]
    return "\n\n".join(out) + "\n"


def _fenced(text: str) -> str:
    return f"```\n{text}\n```"


def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "\n... (cut)"
