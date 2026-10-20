"""Tests for mullet.harness.skills.merger."""

from __future__ import annotations

from pathlib import Path

import pytest

from mullet.harness.skills.merger import SkillRecord, dedup_mcp_by_name, merge_skills

pytestmark = pytest.mark.harness


def _rec(name: str, scope: str, body: str = "body") -> SkillRecord:
    return SkillRecord(
        name=name,
        description=f"{name} desc",
        tools=[],
        mcp=[],
        body=body,
        source_path=Path(f"/tmp/{scope}/{name}/SKILL.md"),
        scope=scope,  # type: ignore[arg-type]
    )


def test_merge_no_overlap_returns_all() -> None:
    by_scope = {
        "project": [_rec("alpha", "project")],
        "user": [_rec("beta", "user")],
        "global": [_rec("gamma", "global")],
    }
    out = merge_skills(by_scope)
    assert [r.name for r in out] == ["alpha", "beta", "gamma"]


def test_merge_project_wins_over_user() -> None:
    by_scope = {
        "project": [_rec("dup", "project", body="P")],
        "user": [_rec("dup", "user", body="U")],
    }
    out = merge_skills(by_scope)
    assert len(out) == 1
    assert out[0].scope == "project"
    assert out[0].body == "P"


def test_merge_user_wins_over_global() -> None:
    by_scope = {
        "user": [_rec("dup", "user", body="U")],
        "global": [_rec("dup", "global", body="G")],
    }
    out = merge_skills(by_scope)
    assert len(out) == 1
    assert out[0].scope == "user"
    assert out[0].body == "U"


def test_merge_three_way_project_wins() -> None:
    by_scope = {
        "project": [_rec("dup", "project", body="P")],
        "user": [_rec("dup", "user", body="U")],
        "global": [_rec("dup", "global", body="G")],
    }
    out = merge_skills(by_scope)
    assert len(out) == 1
    assert out[0].scope == "project"


def test_merge_returns_sorted_by_name() -> None:
    by_scope = {
        "project": [_rec("zulu", "project"), _rec("alpha", "project"), _rec("mike", "project")],
    }
    out = merge_skills(by_scope)
    assert [r.name for r in out] == ["alpha", "mike", "zulu"]


def test_merge_empty_input_returns_empty() -> None:
    assert merge_skills({}) == []


def test_dedup_mcp_keeps_first() -> None:
    mcp = [
        {"name": "a", "url": "u1"},
        {"name": "b", "url": "u2"},
        {"name": "a", "url": "u3"},
    ]
    out = dedup_mcp_by_name(mcp)
    assert len(out) == 2
    assert out[0]["url"] == "u1"
    assert out[1]["name"] == "b"


def test_dedup_mcp_no_collisions_returns_all() -> None:
    mcp = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    out = dedup_mcp_by_name(mcp)
    assert out == mcp
