"""Tests for mullet.jobs.planner: plan validation and the planner's prompt."""

import pytest

from mullet.jobs.executors.base import ExecResult
from mullet.jobs.planner import MAX_LISTED_FILES, PlanError, build_prompt, plan_job, validate_plan
from mullet.jobs.profile import Checks, Profile


def node(id_, parent=None, **extra):
    return {"id": id_, "parent": parent, "goal": f"do {id_}", "kind": "design", "scope": ["src/**"], **extra}


def test_a_valid_tree_becomes_plan_rows_in_order():
    rows = validate_plan({"nodes": [node("api"), node("api-routes", "api"), node("api-tests", "api"), node("docs")]})
    assert [(spec.id, parent) for spec, parent in rows] == [
        ("api", None), ("api-routes", "api"), ("api-tests", "api"), ("docs", None)]


def test_analysis_nodes_expect_no_change():
    rows = validate_plan({"nodes": [node("survey", kind="analysis")]})
    assert rows[0][0].expects_change is False


@pytest.mark.parametrize("nodes, message", [
    ([], "no nodes"),
    ([node("a"), node("a")], "used twice"),
    ([node("child", "parent"), node("parent")], "doesn't come before it"),
    ([node("a"), node("b", "a"), node("c", "b"), node("d", "c")], "nested 4 deep"),
    ([node("a"), node("b", "a", kind="analysis"), node("c", "b")], "can't be an analysis node"),
    ([{"id": "bad id!", "parent": None, "goal": "x", "kind": "design", "scope": ["**"]}], "is invalid"),
    ([{"id": "x", "parent": None, "goal": "", "kind": "design", "scope": ["**"]}], "is invalid"),
])
def test_bad_trees_are_refused(nodes, message):
    with pytest.raises(PlanError, match=message):
        validate_plan({"nodes": nodes})


def test_a_plan_cannot_raise_limits_or_set_unknown_fields():
    with pytest.raises(PlanError, match="max_files, max_lines"):
        validate_plan({"nodes": [node("a", max_files=100000, max_lines=10**9)]})
    with pytest.raises(PlanError, match="expects_change"):
        validate_plan({"nodes": [node("a", expects_change=False)]})


def test_too_many_nodes_are_refused():
    with pytest.raises(PlanError, match="limit is 3"):
        validate_plan({"nodes": [node(f"n{i}") for i in range(4)]}, max_nodes=3)


def test_prompt_lists_checks_files_and_guidance():
    profile = Profile(repo_id="x", checks=Checks(test_full="pytest", test_targeted="pytest {tests}", junit=True))
    files = [f"src/f{i}.py" for i in range(MAX_LISTED_FILES + 5)]
    prompt = build_prompt("Add caching", profile, files)
    assert "Goal: Add caching" in prompt
    assert "test_full: pytest" in prompt
    assert f"Tracked files ({MAX_LISTED_FILES + 5}, first {MAX_LISTED_FILES})" in prompt
    assert f"src/f{MAX_LISTED_FILES - 1}.py" in prompt and f"src/f{MAX_LISTED_FILES}.py" not in prompt
    assert "Change nothing" in prompt


def test_prompt_warns_when_nothing_can_be_verified():
    assert "can't be fully verified" in build_prompt("x", Profile(repo_id="x"), [])


def test_plan_job_uses_the_answer_and_its_notes(tmp_path):
    seen = {}

    def ask(prompt, cwd, schema, budget):
        seen.update(prompt=prompt, schema=schema, cwd=cwd)
        return ExecResult(exit_code=0, claim={"nodes": [node("a"), node("b", "a")], "notes": "kept it small"})

    rows, notes, _ = plan_job("Goal", tmp_path, Profile(repo_id="x"), ["a.py"], ask)
    assert [spec.id for spec, _ in rows] == ["a", "b"]
    assert notes == "kept it small"
    assert seen["cwd"] == tmp_path and "nodes" in seen["schema"]["properties"]


def test_plan_job_without_an_answer_fails(tmp_path):
    def ask(prompt, cwd, schema, budget):
        return ExecResult(exit_code=1, error="Reached maximum number of turns (40)")

    with pytest.raises(PlanError, match="maximum number of turns"):
        plan_job("Goal", tmp_path, Profile(repo_id="x"), [], ask)
