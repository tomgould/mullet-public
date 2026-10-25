"""Turn a goal into a task tree that a person approves before anything runs.

A read-only Claude Code session (Read, Glob and Grep only) studies the clone
and answers with a flat list of nodes, each naming its parent. Leaves are
done by an executor; a node with children is an integration node: it passes
when its children have, and when gates on their combined change, with the
full test suite, pass too. Mullet checks the shape of the tree here
(ids, parents, depth, size, the order `after` needs) before anyone sees it.
A reply with no usable plan is asked for again in the same session, up to
`jobs.reask_attempts` times.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from mullet.jobs.executors.base import Budget, ExecResult
from mullet.jobs.profile import Profile
from mullet.jobs.settings import load_settings
from mullet.jobs.text import one_line
from mullet.jobs.types import NodeSpec

MAX_DEPTH = 3
MAX_NODES = 40
MAX_LISTED_FILES = 400
MAX_OPEN_QUESTIONS = 10
MAX_QUESTION_CHARS = 400

_NODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "pattern": r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"},
        "parent": {"type": ["string", "null"]},
        "goal": {"type": "string"},
        "kind": {"type": "string", "enum": ["mechanical", "design", "analysis"]},
        "door": {"type": "string", "enum": ["two-way", "one-way"]},
        "after": {"type": "array", "items": {"type": "string"}},
        "scope": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "acceptance": {"type": "array", "items": {"type": "string"}},
        "allow_protected": {"type": "array", "items": {"type": "string"}},
        "allow_test_weakening": {"type": "boolean"},
    },
    "required": ["id", "parent", "goal", "kind", "scope"],
    "additionalProperties": False,
}

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "nodes": {"type": "array", "items": _NODE_SCHEMA, "minItems": 1},
        "notes": {"type": "string"},
        "open_questions": {"type": "array", "maxItems": MAX_OPEN_QUESTIONS,
                           "items": {"type": "string", "maxLength": MAX_QUESTION_CHARS}},
    },
    "required": ["nodes", "notes"],
}

_GUIDANCE = """\
How to plan:
- Read enough of the code to plan with confidence. Change nothing.
- Each leaf is one focused change someone could make in a single sitting:
  usually under 300 changed lines in a handful of files.
- Group related leaves under a parent when they only make sense together.
  A parent's scope must cover its children's. Parents can nest, to a depth
  of {depth}. Use at most {max_nodes} nodes in all.
- List nodes so that every parent comes before its children, and siblings in
  the order they should be done.
- after: ids of nodes that must be finished first. B is after A when B needs
  code that A adds, when A and B touch the same files or modules, or when B
  depends on a decision or API shape that A settles. List A (and its children)
  before B: Mullet runs nodes in list order, children before their parent.
- door: "one-way" for a step that is hard to walk back once merged: a public
  API, a default users build on, a removal, persisted formats (files, schemas,
  stored data). Mullet treats one-way steps as risky: each gets an independent
  review that must pass, even in a job with review off. Otherwise leave it out.
- scope: path globs for the files a node may change ("src/api/**",
  "tests/test_api.py"). Keep them tight; changes outside scope are rejected.
  List any new test files a step will add in its scope.
- acceptance: shell commands that must pass when the node is done, using the
  project's own test command where possible, e.g. a targeted test run.
- kind: "mechanical" for any well-specified change a careful junior could make
  from the goal alone: renames, moves, formatting, and changes whose expected
  behaviour the acceptance commands pin down (a new function with its tests, a
  fix with a test that shows it). Mechanical steps can go to cheaper workers,
  so use it whenever the goal, scope and acceptance leave no real design
  choice. "design" when the step needs judgement about structure or behaviour
  that its checks can't settle; "analysis" only for a node that writes no code.
  A step that deletes, kills, sends data, or changes a public API or persisted
  format is never mechanical.
- Protected files (lockfiles, CI config, .env files, CLAUDE.md, test runner
  config) can only change if you list them in allow_protected, with a reason
  in the goal. Tests may be added to, never weakened; set allow_test_weakening
  only when removing tests is the point of the node, and say why.
- notes: anything the person approving the plan should know: risks,
  assumptions, what you left out.
- open_questions: only things you could not verify in the code; never guess.
  The person approving the plan answers them. Leave it out when there are none."""


class PlanError(ValueError):
    """The planner's answer isn't a usable task tree."""


# The only node fields a plan may set. Size limits are left out on purpose:
# a plan that raised them would weaken G2 and G3 without the approver seeing.
PLAN_KEYS = frozenset({
    "id", "parent", "goal", "kind", "door", "after", "scope", "acceptance", "allow_protected",
    "allow_test_weakening",
})


def validate_plan(raw: dict[str, Any], *, max_depth: int = MAX_DEPTH,
                  max_nodes: int = MAX_NODES) -> list[tuple[NodeSpec, str | None]]:
    """Check the tree's shape and turn it into ledger plan rows (spec, parent id)."""
    items = raw.get("nodes")
    if not isinstance(items, list) or not items:
        raise PlanError("the plan has no nodes")
    if len(items) > max_nodes:
        raise PlanError(f"the plan has {len(items)} nodes; the limit is {max_nodes}")
    depth: dict[str, int] = {}
    out: list[tuple[NodeSpec, str | None]] = []
    for item in items:
        if not isinstance(item, dict):
            raise PlanError("a node is not an object")
        unknown = set(item) - PLAN_KEYS
        if unknown:
            raise PlanError(f"node {item.get('id')!r} sets fields a plan can't set: {', '.join(sorted(unknown))}")
        data = dict(item)
        parent = data.pop("parent", None) or None
        if data.get("kind") == "analysis":
            data["expects_change"] = False
        try:
            spec = NodeSpec.model_validate(data)
        except ValidationError as exc:
            raise PlanError(f"node {data.get('id')!r} is invalid: {exc.errors()[0]['msg']}") from exc
        if spec.id in depth:
            raise PlanError(f"node id {spec.id!r} is used twice")
        if parent is not None and parent not in depth:
            raise PlanError(f"node {spec.id!r} names parent {parent!r}, which doesn't come before it")
        depth[spec.id] = 1 if parent is None else depth[parent] + 1
        if depth[spec.id] > max_depth:
            raise PlanError(f"node {spec.id!r} is nested {depth[spec.id]} deep; the limit is {max_depth}")
        out.append((spec, parent))
    parents = {p for _, p in out if p}
    for spec, _ in out:
        if spec.id in parents and spec.kind == "analysis":
            raise PlanError(f"node {spec.id!r} has children, so it can't be an analysis node")
    _check_after(out)
    plan_questions(raw)
    return out


def _check_after(rows: list[tuple[NodeSpec, str | None]]) -> None:
    """Every `after` names another node, has no cycle, and is finished first in the runner's order."""
    graph = {spec.id: spec.after for spec, _ in rows}
    for spec, _ in rows:
        for dep in spec.after:
            if dep == spec.id:
                raise PlanError(f"node {spec.id!r} names itself in after")
            if dep not in graph:
                raise PlanError(f"node {spec.id!r} names {dep!r} in after, which isn't a node")
    done: set[str] = set()

    def visit(node_id: str, path: list[str]) -> None:
        if node_id in path:
            cycle = [*path[path.index(node_id):], node_id]
            raise PlanError(f"the after links form a cycle: {' -> '.join(cycle)}")
        if node_id in done:
            return
        for dep in graph[node_id]:
            visit(dep, [*path, node_id])
        done.add(node_id)

    for node_id in graph:
        visit(node_id, [])
    # The runner finishes a node's children before the node itself (post-order).
    parent_of = {spec.id: parent for spec, parent in rows}
    kids: dict[str | None, list[str]] = {None: []}
    for spec, parent in rows:
        kids.setdefault(spec.id, [])
        kids[parent].append(spec.id)
    finished: dict[str, int] = {}

    def walk(node_id: str) -> None:
        for kid in kids[node_id]:
            walk(kid)
        finished[node_id] = len(finished)

    for root in kids[None]:
        walk(root)
    order = "Mullet runs nodes in list order, children before their parent"
    for spec, _ in rows:
        for dep in spec.after:
            if finished[dep] < finished[spec.id]:
                continue
            ancestor = parent_of[spec.id]
            while ancestor is not None and ancestor != dep:
                ancestor = parent_of[ancestor]
            if ancestor == dep:
                raise PlanError(f"node {spec.id!r} names {dep!r} in after, but {dep!r} contains it and so "
                                f"finishes after it ({order})")
            raise PlanError(f"node {spec.id!r} names {dep!r} in after, but {dep!r} would not be finished yet "
                            f"({order}); list {dep!r} and its children before {spec.id!r}")


def plan_questions(raw: dict[str, Any] | None) -> list[str]:
    """The plan's open questions for the approver, each one sanitised line; blank ones are dropped."""
    questions = (raw or {}).get("open_questions") or []
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        raise PlanError("open_questions must be a list of strings")
    if len(questions) > MAX_OPEN_QUESTIONS:
        raise PlanError(f"the plan has {len(questions)} open questions; at most {MAX_OPEN_QUESTIONS} are allowed")
    if any(len(q) > MAX_QUESTION_CHARS for q in questions):
        raise PlanError(f"an open question is longer than {MAX_QUESTION_CHARS} characters")
    cleaned = [one_line(q) for q in questions]
    return [q for q in cleaned if q]


def build_prompt(goal: str, profile: Profile, files: list[str]) -> str:
    """The planner's prompt: the goal, how the repo is checked, and its files."""
    checks = profile.checks
    lines = [
        "Plan a coding job on the repository in your current directory.",
        f"Goal: {goal}",
        "",
        "Mullet will check every node itself with:",
    ]
    for name in ("lint", "typecheck", "build", "test_full", "test_targeted"):
        command = getattr(checks, name)
        if command:
            lines.append(f"  {name}: {command}")
    if not checks.test_full:
        lines.append("  (no test command is configured, so nodes can't be fully verified)")
    shown = files[:MAX_LISTED_FILES]
    more = f", first {len(shown)}" if len(files) > len(shown) else ""
    lines += ["", f"Tracked files ({len(files)}{more}):"]
    lines += [f"  {f}" for f in shown]
    lines += ["", _GUIDANCE.format(depth=MAX_DEPTH, max_nodes=MAX_NODES)]
    return "\n".join(lines)


class Asker(Protocol):
    """Runs a read-only session; with `resume`, continues that session instead of starting one."""

    def __call__(self, prompt: str, cwd: Path, schema: dict[str, Any], budget: Budget, *,
                 resume: str | None = None) -> ExecResult:
        """Answer `prompt` as JSON matching `schema`; the answer is in `claim`."""
        ...


def reask_prompt(problem: PlanError) -> str:
    """What a resumed planner session is told when its reply had no usable plan."""
    return (f"Your reply could not be used as a plan: {problem}.\n"
            "Reply again with only the corrected JSON plan, matching the schema you were given. "
            "Change nothing in the repository.")


def _accept(result: ExecResult) -> list[tuple[NodeSpec, str | None]]:
    if result.claim is None:
        raise PlanError(f"the planner gave no plan: {result.error or 'no structured answer'}")
    return validate_plan(result.claim)


def plan_job(goal: str, work: Path, profile: Profile, files: list[str], ask: Asker,
             budget: Budget | None = None, *, reask_attempts: int | None = None,
             ) -> tuple[list[tuple[NodeSpec, str | None]], str, ExecResult]:
    """Ask a read-only session for a plan and validate it. Returns (nodes, notes, raw result).

    A reply with no usable plan is asked for again in the same session, at
    most `reask_attempts` times (default: the `jobs.reask_attempts` setting),
    unless the session can't be resumed or stopped on a usage limit. The
    plan's open questions come from the result: `plan_questions(result.claim)`.
    """
    budget = budget or Budget(max_turns=40, wall_clock_s=1200)
    left = load_settings().reask_attempts if reask_attempts is None else reask_attempts
    result = ask(build_prompt(goal, profile, files), work, PLAN_SCHEMA, budget)
    while True:
        try:
            nodes = _accept(result)
        except PlanError as exc:
            if left <= 0 or not result.session_id or result.paused_until or result.aborted:
                raise
            left -= 1
            result = ask(reask_prompt(exc), work, PLAN_SCHEMA, budget, resume=result.session_id)
            continue
        claim = result.claim or {}
        return nodes, str(claim.get("notes", "")).strip(), result
