"""Turn a goal into a task tree that a person approves before anything runs.

A read-only Claude Code session (Read, Glob and Grep only) studies the clone
and answers with a flat list of nodes, each naming its parent. Leaves are
done by an executor; a node with children is an integration node: it passes
when its children have, and when gates on their combined change, with the
full test suite, pass too. Mullet checks the shape of the tree here
(ids, parents, depth, size) before anyone sees it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from mullet.jobs.executors.base import Budget, ExecResult
from mullet.jobs.profile import Profile
from mullet.jobs.types import NodeSpec

MAX_DEPTH = 3
MAX_NODES = 40
MAX_LISTED_FILES = 400

_NODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "pattern": r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"},
        "parent": {"type": ["string", "null"]},
        "goal": {"type": "string"},
        "kind": {"type": "string", "enum": ["mechanical", "design", "analysis"]},
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
- scope: path globs for the files a node may change ("src/api/**",
  "tests/test_api.py"). Keep them tight; changes outside scope are rejected.
- acceptance: shell commands that must pass when the node is done, using the
  project's own test command where possible, e.g. a targeted test run.
- kind: "mechanical" for any well-specified change a careful junior could make
  from the goal alone: renames, moves, formatting, and changes whose expected
  behaviour the acceptance commands pin down (a new function with its tests, a
  fix with a test that shows it). Mechanical steps can go to cheaper workers,
  so use it whenever the goal, scope and acceptance leave no real design
  choice. "design" when the step needs judgement about structure or behaviour
  that its checks can't settle; "analysis" only for a node that writes no code.
- Protected files (lockfiles, CI config, .env files, CLAUDE.md, test runner
  config) can only change if you list them in allow_protected, with a reason
  in the goal. Tests may be added to, never weakened; set allow_test_weakening
  only when removing tests is the point of the node, and say why.
- notes: anything the person approving the plan should know: risks,
  assumptions, what you left out."""


class PlanError(ValueError):
    """The planner's answer isn't a usable task tree."""


# The only node fields a plan may set. Size limits are left out on purpose:
# a plan that raised them would weaken G2 and G3 without the approver seeing.
PLAN_KEYS = frozenset({
    "id", "parent", "goal", "kind", "scope", "acceptance", "allow_protected", "allow_test_weakening",
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
    return out


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


Asker = Callable[[str, Path, dict[str, Any], Budget], ExecResult]


def plan_job(goal: str, work: Path, profile: Profile, files: list[str], ask: Asker,
             budget: Budget | None = None) -> tuple[list[tuple[NodeSpec, str | None]], str, ExecResult]:
    """Ask a read-only session for a plan and validate it. Returns (nodes, notes, raw result)."""
    budget = budget or Budget(max_turns=40, wall_clock_s=1200)
    result = ask(build_prompt(goal, profile, files), work, PLAN_SCHEMA, budget)
    if result.claim is None:
        raise PlanError(f"the planner gave no plan: {result.error or 'no structured answer'}")
    nodes = validate_plan(result.claim)
    return nodes, str(result.claim.get("notes", "")).strip(), result
