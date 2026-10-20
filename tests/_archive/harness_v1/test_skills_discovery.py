"""Tests for mullet.harness.skills.discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from mullet.harness.skills.builtin import BUILTIN_DIR
from mullet.harness.skills.discovery import default_scope_dirs, discover_skills

pytestmark = pytest.mark.harness


def _write(path: Path, body: str = "---\nname: test\n---\nbody\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_discover_finds_skill_md_in_scope(tmp_path: Path) -> None:
    """Given project/foo/SKILL.md, discover returns it under 'project'."""
    project_root = tmp_path / "project"
    skill = _write(project_root / "foo" / "SKILL.md")

    result = discover_skills({"project": [project_root]})

    assert "project" in result
    assert skill.resolve() in result["project"]


def test_discover_recursive_depth(tmp_path: Path) -> None:
    """SKILL.md three directory levels under root is still discovered."""
    deep_path = tmp_path / "root" / "lvl1" / "lvl2" / "skill_dir" / "SKILL.md"
    _write(deep_path)

    result = discover_skills({"project": [tmp_path / "root"]})

    assert deep_path.resolve() in result["project"]


def test_discover_skips_pycache_and_git(tmp_path: Path) -> None:
    """Files under __pycache__ or .git directories are ignored."""
    root = tmp_path / "root"
    _write(root / "__pycache__" / "SKILL.md")
    _write(root / ".git" / "SKILL.md")
    _write(root / "node_modules" / "pkg" / "SKILL.md")
    real = _write(root / "real" / "SKILL.md")

    result = discover_skills({"project": [root]})

    assert result["project"] == [real.resolve()]


def test_discover_case_insensitive(tmp_path: Path) -> None:
    """Lowercase ``skill.md`` is also discovered."""
    root = tmp_path / "root"
    skill = _write(root / "foo" / "skill.md")

    result = discover_skills({"project": [root]})

    assert skill.resolve() in result["project"]


def test_discover_returns_empty_lists_for_scope_with_no_skills(tmp_path: Path) -> None:
    """Empty scope dirs produce empty lists, not missing keys."""
    empty_root = tmp_path / "empty"
    empty_root.mkdir()

    result = discover_skills({"project": [empty_root], "user": [tmp_path / "missing"]})

    assert result["project"] == []
    assert result["user"] == []


def test_default_scope_dirs_has_3_keys() -> None:
    """default_scope_dirs returns the canonical 3-scope mapping."""
    scopes = default_scope_dirs()
    assert set(scopes.keys()) == {"project", "user", "global"}
    for scope, dirs in scopes.items():
        assert isinstance(dirs, list), f"{scope} dirs should be a list"
        assert all(isinstance(p, Path) for p in dirs)
        assert len(dirs) >= 1


def test_default_scope_dirs_global_includes_builtin_dir() -> None:
    """The 'global' scope must include the bundled BUILTIN_DIR path."""
    scopes = default_scope_dirs()
    assert BUILTIN_DIR in scopes["global"]
