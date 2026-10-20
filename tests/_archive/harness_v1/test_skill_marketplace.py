"""Tests for mullet.harness.skills.marketplace and mullet.cli.skill_cli."""

from __future__ import annotations

from pathlib import Path

import pytest

from mullet.harness.skills.marketplace import (
    load_registry,
    search_registry,
    install_skill,
    remove_skill,
)
from mullet.cli.skill_cli import cmd_skill

pytestmark = pytest.mark.harness


# ---------------------------------------------------------------------------
# Marketplace tests
# ---------------------------------------------------------------------------


def test_load_registry_from_bundled_json() -> None:
    """Registry loads from the bundled JSON and contains known skills."""
    reg = load_registry()
    names = {e["name"] for e in reg}
    assert "git-master" in names
    assert "playwright" in names
    assert len(reg) >= 5


def test_search_filters_by_query() -> None:
    """search_registry returns only entries matching the query."""
    reg = load_registry()
    results = search_registry("git")
    assert len(results) >= 1
    assert all("git" in r["name"].lower() or "git" in r["description"].lower() for r in results)


def test_search_case_insensitive() -> None:
    """search_registry is case-insensitive."""
    lower = search_registry("git")
    upper = search_registry("GIT")
    assert len(lower) == len(upper)
    assert {r["name"] for r in lower} == {r["name"] for r in upper}


def test_install_writes_skill_md(monkeypatch, tmp_path: Path) -> None:
    """install_skill fetches, validates, and writes SKILL.md to the dest dir."""
    canned = "---\nname: test-skill\ndescription: A test skill\n---\n# Test Skill\nBody here.\n"

    def fake_urlopen(url, timeout=None):
        class FakeResp:
            def read(self):
                return canned.encode("utf-8")
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return FakeResp()

    monkeypatch.setattr("mullet.harness.skills.marketplace.urlopen", fake_urlopen)

    registry = [
        {"name": "test-skill", "description": "A test skill", "source_url": "http://example.com/skill.md", "version": "1.0.0"}
    ]
    dest = tmp_path / "skills"
    path = install_skill("test-skill", dest_dir=dest, registry=registry)
    assert path.exists()
    assert path.name == "SKILL.md"
    assert path.parent.name == "test-skill"
    assert "Body here." in path.read_text(encoding="utf-8")


def test_install_validates_frontmatter(monkeypatch, tmp_path: Path) -> None:
    """install_skill rejects SKILL.md text with invalid frontmatter."""
    bad_text = "---\nfoo: bar\n- item\n---\nbody\n"

    def fake_urlopen(url, timeout=None):
        class FakeResp:
            def read(self):
                return bad_text.encode("utf-8")
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return FakeResp()

    monkeypatch.setattr("mullet.harness.skills.marketplace.urlopen", fake_urlopen)

    registry = [
        {"name": "bad-skill", "description": "Bad", "source_url": "http://example.com/bad.md", "version": "1.0.0"}
    ]
    dest = tmp_path / "skills"
    with pytest.raises(ValueError):
        install_skill("bad-skill", dest_dir=dest, registry=registry)


def test_install_creates_dest_dir(monkeypatch, tmp_path: Path) -> None:
    """install_skill creates the destination directory if it does not exist."""
    canned = "---\nname: new-skill\n---\nbody\n"

    def fake_urlopen(url, timeout=None):
        class FakeResp:
            def read(self):
                return canned.encode("utf-8")
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return FakeResp()

    monkeypatch.setattr("mullet.harness.skills.marketplace.urlopen", fake_urlopen)

    registry = [
        {"name": "new-skill", "description": "New", "source_url": "http://example.com/new.md", "version": "1.0.0"}
    ]
    dest = tmp_path / "deep" / "nested" / "skills"
    assert not dest.exists()
    path = install_skill("new-skill", dest_dir=dest, registry=registry)
    assert path.exists()


def test_remove_deletes_user_scope_skill(tmp_path: Path) -> None:
    """remove_skill deletes a skill from the user scope."""
    skill_dir = tmp_path / "skills" / "remove-me"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: remove-me\n---\nbody\n", encoding="utf-8")

    assert remove_skill("remove-me", dest_dir=tmp_path / "skills") is True
    assert not skill_dir.exists()


def test_remove_refuses_builtin() -> None:
    """remove_skill raises ValueError when trying to remove a built-in skill."""
    with pytest.raises(ValueError, match="Cannot remove built-in"):
        remove_skill("commit")


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_cli_list_outputs_all_scopes(capsys) -> None:
    """cmd_skill list prints at least the built-in skills with scope tags."""
    rc = cmd_skill(["list"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "[global]" in captured.out
    assert "commit" in captured.out


def test_cli_search_outputs_matches(capsys) -> None:
    """cmd_skill search prints matching registry entries."""
    rc = cmd_skill(["search", "git"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "git-master" in captured.out


def test_cli_install_calls_install_skill(monkeypatch, tmp_path: Path) -> None:
    """cmd_skill install delegates to install_skill."""
    canned = "---\nname: demo-skill\n---\nbody\n"
    calls = []

    def fake_urlopen(url, timeout=None):
        class FakeResp:
            def read(self):
                return canned.encode("utf-8")
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return FakeResp()

    monkeypatch.setattr("mullet.harness.skills.marketplace.urlopen", fake_urlopen)

    dest = tmp_path / "skills"
    monkeypatch.setattr(
        "mullet.harness.skills.marketplace._default_dest_dir", lambda: dest
    )

    rc = cmd_skill(["install", "demo-skill"])
    # Since demo-skill isn't in the bundled registry, this will fail unless we patch load_registry.
    # Instead, patch load_registry to return our entry.
    monkeypatch.setattr(
        "mullet.harness.skills.marketplace.load_registry",
        lambda: [
            {"name": "demo-skill", "description": "Demo", "source_url": "http://example.com/demo.md", "version": "1.0.0"}
        ],
    )
    rc = cmd_skill(["install", "demo-skill"])
    assert rc == 0
    assert (dest / "demo-skill" / "SKILL.md").exists()


def test_cli_info_unknown_returns_friendly_error(capsys) -> None:
    """cmd_skill info for an unknown skill returns a friendly error and non-zero rc."""
    rc = cmd_skill(["info", "definitely-not-a-real-skill-xyz"])
    assert rc != 0
    captured = capsys.readouterr()
    assert "not found" in captured.out.lower() or "not found" in captured.err.lower()
