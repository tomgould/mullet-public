"""Tests for mullet.harness.skills.loader."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from mullet.harness.skills.builtin import BUILTIN_DIR
from mullet.harness.skills.discovery import default_scope_dirs
from mullet.harness.skills.loader import find_skill, load_skills

pytestmark = pytest.mark.harness


def _write_skill(
    base: Path,
    *,
    skill_dir: str,
    name: str | None,
    description: str = "test description",
    body: str = "body content",
) -> Path:
    """Helper to write a SKILL.md under base/skill_dir/SKILL.md."""
    target_dir = base / skill_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    fm_lines = ["---"]
    if name is not None:
        fm_lines.append(f"name: {name}")
    fm_lines.append(f"description: {description}")
    fm_lines.append("tools: [Bash]")
    fm_lines.append("---")
    fm_lines.append(body)
    path = target_dir / "SKILL.md"
    path.write_text("\n".join(fm_lines) + "\n", encoding="utf-8")
    return path


def test_load_skills_loads_8_builtin_skills_at_minimum(tmp_path: Path) -> None:
    """Using built-in scope alone we get at least the 8 shipped skills."""
    scopes = {
        "project": [tmp_path / "no_project"],
        "user": [tmp_path / "no_user"],
        "global": [BUILTIN_DIR],
    }
    skills = load_skills(scopes)
    names = {s.name for s in skills}
    expected = {"commit", "plan", "review", "simplify", "debug", "test", "security-review", "init"}
    assert expected.issubset(names), f"Missing: {expected - names}"
    assert len(skills) >= 8


def test_load_skills_project_overrides_user_overrides_global(tmp_path: Path) -> None:
    """A project-scope skill named 'commit' replaces the built-in commit."""
    project_root = tmp_path / "project"
    _write_skill(
        project_root,
        skill_dir="commit",
        name="commit",
        description="my override",
        body="OVERRIDDEN",
    )

    scopes = {
        "project": [project_root],
        "user": [tmp_path / "no_user"],
        "global": [BUILTIN_DIR],
    }
    skills = load_skills(scopes)
    by_name = {s.name: s for s in skills}
    assert "commit" in by_name
    assert by_name["commit"].scope == "project"
    assert "OVERRIDDEN" in by_name["commit"].body


def test_load_skills_skips_skill_without_name_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A SKILL.md whose frontmatter lacks 'name' is skipped with a warning."""
    project_root = tmp_path / "project"
    target = project_root / "broken"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text(
        "---\ndescription: no name here\n---\nbody\n", encoding="utf-8"
    )

    scopes = {
        "project": [project_root],
        "user": [tmp_path / "no_user"],
        "global": [tmp_path / "no_global"],
    }
    with caplog.at_level(logging.WARNING, logger="mullet.harness.skills.loader"):
        skills = load_skills(scopes)

    names = {s.name for s in skills}
    assert "broken" not in names  # would have only matched if dir-name fallback existed
    assert any("missing 'name'" in rec.message for rec in caplog.records)


def test_find_skill_returns_match() -> None:
    """find_skill returns the matching record by name."""
    record = find_skill("commit", default_scope_dirs())
    assert record is not None
    assert record.name == "commit"


def test_find_skill_returns_none_for_unknown(tmp_path: Path) -> None:
    """find_skill returns None when no skill of that name is loaded."""
    scopes = {
        "project": [tmp_path / "no_project"],
        "user": [tmp_path / "no_user"],
        "global": [BUILTIN_DIR],
    }
    assert find_skill("definitely-not-a-skill-xyz", scopes) is None


def test_load_skills_includes_source_path_and_scope_metadata(tmp_path: Path) -> None:
    """Each loaded record carries its source path and scope."""
    project_root = tmp_path / "project"
    written = _write_skill(project_root, skill_dir="myskill", name="myskill")

    scopes = {
        "project": [project_root],
        "user": [tmp_path / "no_user"],
        "global": [tmp_path / "no_global"],
    }
    skills = load_skills(scopes)
    by_name = {s.name: s for s in skills}
    assert "myskill" in by_name
    record = by_name["myskill"]
    assert record.scope == "project"
    assert Path(record.source_path).resolve() == written.resolve()
