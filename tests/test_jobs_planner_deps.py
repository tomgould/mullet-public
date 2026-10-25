"""Tests for the planner's dependencies, one-way doors, open questions and its one re-ask."""

import json

import pytest

from mullet.jobs.executors.base import ExecResult
from mullet.jobs.ledger import Ledger
from mullet.jobs.planner import (
    MAX_OPEN_QUESTIONS, MAX_QUESTION_CHARS, PLAN_SCHEMA, PlanError, build_prompt, plan_job, plan_questions,
    validate_plan,
)
from mullet.jobs.profile import Profile
from mullet.jobs.types import NodeSpec


def node(id_, parent=None, **extra):
    return {"id": id_, "parent": parent, "goal": f"do {id_}", "kind": "design", "scope": ["src/**"], **extra}


def ids(rows):
    return [(spec.id, spec.after, spec.door) for spec, _ in rows]


# Schema


def test_new_fields_default_so_old_specs_load_unchanged():
    spec = NodeSpec.model_validate_json(json.dumps({"id": "a", "goal": "g", "kind": "design", "scope": ["x"]}))
    assert spec.after == [] and spec.door == "two-way"


def test_a_plan_stored_before_the_new_fields_loads_from_the_ledger(tmp_path):
    with Ledger(tmp_path / "ledger.db") as ledger:
        job_id = ledger.create_job(tmp_path, "goal", "main", "0" * 40, {})
        ledger.set_plan(job_id, [(NodeSpec(id="a", goal="g"), None)])
        old = {"id": "a", "goal": "g", "kind": "mechanical", "scope": ["x.py"], "acceptance": [],
               "allow_protected": [], "allow_test_weakening": False, "expects_change": True,
               "max_files": 50, "max_lines": 2000}
        ledger._conn.execute("UPDATE nodes SET spec_json = ? WHERE job_id = ?", (json.dumps(old), job_id))
        spec = ledger.nodes(job_id)[0].spec
    assert (spec.kind, spec.scope, spec.after, spec.door) == ("mechanical", ["x.py"], [], "two-way")


def test_the_plan_schema_offers_after_door_and_open_questions():
    props = PLAN_SCHEMA["properties"]["nodes"]["items"]["properties"]
    assert props["after"]["type"] == "array"
    assert props["door"]["enum"] == ["two-way", "one-way"]
    questions = PLAN_SCHEMA["properties"]["open_questions"]
    assert questions["maxItems"] == MAX_OPEN_QUESTIONS
    assert questions["items"]["maxLength"] == MAX_QUESTION_CHARS
    assert "open_questions" not in PLAN_SCHEMA["required"]


def test_a_door_must_be_one_of_the_two_kinds():
    with pytest.raises(PlanError, match="is invalid"):
        validate_plan({"nodes": [node("a", door="trapdoor")]})


# validate_plan: after


def test_after_may_name_earlier_nodes_in_execution_order():
    rows = validate_plan({"nodes": [
        node("api"), node("api-model", "api"), node("api-routes", "api", after=["api-model"]),
        node("docs", after=["api"], door="one-way"), node("cli", after=["api-routes", "docs"]),
    ]})
    assert ids(rows) == [("api", [], "two-way"), ("api-model", [], "two-way"),
                         ("api-routes", ["api-model"], "two-way"), ("docs", ["api"], "one-way"),
                         ("cli", ["api-routes", "docs"], "two-way")]


def test_a_parent_may_name_its_own_child_since_children_finish_first():
    rows = validate_plan({"nodes": [node("p", after=["c"]), node("c", "p")]})
    assert rows[0][0].after == ["c"]


@pytest.mark.parametrize("nodes, message", [
    ([node("a", after=["ghost"])], "names 'ghost' in after, which isn't a node"),
    ([node("a", after=["a"])], "names itself in after"),
    ([node("a", after=["b"]), node("b", after=["a"])], "cycle"),
    ([node("a", after=["c"]), node("b", after=["a"]), node("c", after=["b"])], "cycle: a -> c -> b -> a"),
    ([node("a", after=["b"]), node("b")], "list order"),
    ([node("p"), node("c", "p", after=["p"])], "list order"),
    ([node("p"), node("c1", "p", after=["c2"]), node("c2", "p")], "list order"),
    ([node("a", after="b"), node("b")], "is invalid"),
])
def test_bad_after_references_are_refused(nodes, message):
    with pytest.raises(PlanError, match=message):
        validate_plan({"nodes": nodes})


def test_the_ordering_error_says_why_and_how_to_fix_it():
    with pytest.raises(PlanError) as err:
        validate_plan({"nodes": [node("a", after=["b"]), node("b")]})
    text = str(err.value)
    assert "runs nodes in list order" in text and "children before their parent" in text


# open questions


def test_open_questions_are_read_trimmed_and_blank_ones_dropped():
    raw = {"nodes": [node("a")], "open_questions": ["  Is v1 still served? ", "", "Who owns the cache?"]}
    validate_plan(raw)
    assert plan_questions(raw) == ["Is v1 still served?", "Who owns the cache?"]
    assert plan_questions({"nodes": []}) == []
    assert plan_questions(None) == []


@pytest.mark.parametrize("questions, message", [
    ("why?", "open_questions must be a list"),
    ([1], "open_questions must be a list"),
    (["q"] * (MAX_OPEN_QUESTIONS + 1), f"at most {MAX_OPEN_QUESTIONS}"),
    (["x" * (MAX_QUESTION_CHARS + 1)], f"{MAX_QUESTION_CHARS} characters"),
])
def test_bad_open_questions_are_refused(questions, message):
    with pytest.raises(PlanError, match=message):
        validate_plan({"nodes": [node("a")], "open_questions": questions})


# the prompt


def test_the_prompt_teaches_dependencies_doors_scope_risk_and_questions():
    prompt = " ".join(build_prompt("Goal", Profile(repo_id="x"), []).split())
    for phrase in (
        "needs code that A adds",
        "touch the same files or modules",
        "decision or API shape that A settles",
        "door: \"one-way\"",
        "persisted formats",
        "new test files",
        "never mechanical",
        "open_questions",
        "never guess",
    ):
        assert phrase in prompt, phrase


# re-asking once


def plan_claim(*nodes, **extra):
    return {"nodes": list(nodes) or [node("a")], "notes": "", **extra}


class FakeAsk:
    """An asker that answers from a script and records each call."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, prompt, cwd, schema, budget, resume=None):
        self.calls.append({"prompt": prompt, "resume": resume, "schema": schema})
        return self.results.pop(0)


def test_no_plan_asks_the_same_session_again_for_the_json(tmp_path):
    ask = FakeAsk(ExecResult(exit_code=0, session_id="s1"),
                  ExecResult(exit_code=0, claim=plan_claim(), session_id="s1"))
    rows, _, result = plan_job("Goal", tmp_path, Profile(repo_id="x"), [], ask, reask_attempts=1)
    assert [spec.id for spec, _ in rows] == ["a"]
    assert result.claim is not None
    assert [c["resume"] for c in ask.calls] == [None, "s1"]
    again = ask.calls[1]["prompt"]
    assert "only the corrected JSON" in again and "no structured answer" in again
    assert ask.calls[1]["schema"] is PLAN_SCHEMA


def test_an_invalid_plan_is_asked_again_with_the_reason(tmp_path):
    bad = plan_claim(node("a", after=["b"]), node("b"))
    ask = FakeAsk(ExecResult(exit_code=0, claim=bad, session_id="s1"),
                  ExecResult(exit_code=0, claim=plan_claim(node("b"), node("a", after=["b"])), session_id="s1"))
    rows, _, _ = plan_job("Goal", tmp_path, Profile(repo_id="x"), [], ask, reask_attempts=1)
    assert [spec.id for spec, _ in rows] == ["b", "a"]
    assert "list order" in ask.calls[1]["prompt"]


def test_reask_attempts_zero_raises_at_once(tmp_path):
    ask = FakeAsk(ExecResult(exit_code=0, session_id="s1"))
    with pytest.raises(PlanError, match="no structured answer"):
        plan_job("Goal", tmp_path, Profile(repo_id="x"), [], ask, reask_attempts=0)
    assert len(ask.calls) == 1


def test_reask_attempts_is_a_limit(tmp_path):
    ask = FakeAsk(*[ExecResult(exit_code=0, session_id="s1") for _ in range(3)])
    with pytest.raises(PlanError, match="no structured answer"):
        plan_job("Goal", tmp_path, Profile(repo_id="x"), [], ask, reask_attempts=2)
    assert [c["resume"] for c in ask.calls] == [None, "s1", "s1"]


@pytest.mark.parametrize("first", [
    ExecResult(exit_code=0),  # no session to resume
    ExecResult(exit_code=0, session_id="s1", paused_until=1e10),  # a usage limit: asking again can't help
    ExecResult(exit_code=None, session_id="s1", aborted=True),
])
def test_no_reask_without_a_session_or_after_a_limit_or_abort(tmp_path, first):
    ask = FakeAsk(first)
    with pytest.raises(PlanError, match="gave no plan"):
        plan_job("Goal", tmp_path, Profile(repo_id="x"), [], ask, reask_attempts=1)
    assert len(ask.calls) == 1


def test_reask_attempts_comes_from_the_jobs_settings_by_default(tmp_path, monkeypatch):
    from mullet.jobs import planner
    from mullet.jobs.settings import JobSettings

    monkeypatch.setattr(planner, "load_settings", lambda: JobSettings(reask_attempts=0))
    ask = FakeAsk(ExecResult(exit_code=0, session_id="s1"))
    with pytest.raises(PlanError):
        plan_job("Goal", tmp_path, Profile(repo_id="x"), [], ask)
    assert len(ask.calls) == 1
