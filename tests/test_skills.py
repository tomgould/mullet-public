"""Tests for the skill system — loading, registry, matching, invocation."""

from pathlib import Path
from mullet.core.skills.loader import SkillLoader, parse_frontmatter
from mullet.core.skills.registry import SkillRegistry
from mullet.core.skills.types import Skill, SkillMetadata


# --- Frontmatter parsing ---

def test_parse_frontmatter():
    content = """---
name: test-skill
description: A test skill
tags:
  - testing
---

# Body content here
"""
    meta, body = parse_frontmatter(content)
    assert meta["name"] == "test-skill"
    assert meta["description"] == "A test skill"
    assert "testing" in meta["tags"]
    assert "Body content" in body


def test_parse_frontmatter_no_frontmatter():
    content = "Just plain text"
    meta, body = parse_frontmatter(content)
    assert meta == {}
    assert body == "Just plain text"


def test_parse_frontmatter_invalid_yaml():
    content = "---\n: invalid: yaml:\n---\nBody"
    meta, body = parse_frontmatter(content)
    assert meta == {}


# --- Skill loader ---

def test_load_builtin_skills():
    builtin_dir = Path(__file__).parent.parent / "mullet" / "core" / "skills" / "builtin"
    loader = SkillLoader(builtin_dir=builtin_dir)
    skills = loader.scan()
    names = {s.metadata.name for s in skills}
    assert "brainstorm" in names
    assert "mullet-job" in names
    assert "mullet-local" in names
    assert "code-review" in names
    for removed in ("autopilot", "ralph", "team", "ultrawork", "eco"):
        assert removed not in names


def test_load_skill_from_directory(tmp_path):
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("""---
name: my-skill
description: Custom skill
triggers:
  - custom
tags:
  - test
---

# Custom Skill

Do custom things.
""")

    loader = SkillLoader(builtin_dir=tmp_path, user_dir=tmp_path / "nonexistent")
    skills = loader.scan()
    assert len(skills) == 1
    assert skills[0].metadata.name == "my-skill"
    assert skills[0].metadata.triggers == ["custom"]


def test_disabled_skill(tmp_path):
    skill_dir = tmp_path / "disabled-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: disabled-skill\n---\nBody")

    loader = SkillLoader(builtin_dir=tmp_path, disabled=["disabled-skill"])
    skills = loader.scan()
    assert len(skills) == 0


def test_dedup_by_scope(tmp_path):
    """Project-scope skill overrides builtin with same name."""
    builtin = tmp_path / "builtin"
    project = tmp_path / "project"
    for d in [builtin, project]:
        skill_dir = d / "dup"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"---\nname: dup\ndescription: from {d.name}\n---\nBody")

    loader = SkillLoader(builtin_dir=builtin, project_dir=project, user_dir=tmp_path / "no")
    skills = loader.scan()
    assert len(skills) == 1
    assert "project" in skills[0].metadata.description


# --- Registry ---

def test_registry_get():
    registry = SkillRegistry()
    skill = Skill(
        metadata=SkillMetadata(name="test", description="desc", aliases=["t"]),
        body="body",
        path=Path("."),
    )
    registry.register(skill)
    assert registry.get("test") is not None
    assert registry.get("t") is not None
    assert registry.get("nonexistent") is None


def test_registry_match_slash_command():
    registry = SkillRegistry()
    skill = Skill(
        metadata=SkillMetadata(name="code-review", description="Review code"),
        body="body",
        path=Path("."),
    )
    registry.register(skill)

    match = registry.match("/code-review src/auth.py")
    assert match is not None
    assert match.match_type == "slash_command"
    assert match.skill.metadata.name == "code-review"


def test_registry_match_keyword():
    registry = SkillRegistry()
    skill = Skill(
        metadata=SkillMetadata(name="autopilot", description="Auto", triggers=["autopilot", "full auto"]),
        body="body",
        path=Path("."),
    )
    registry.register(skill)

    match = registry.match("run autopilot on this task")
    assert match is not None
    assert match.match_type == "keyword"
    assert match.trigger == "autopilot"


def test_registry_no_match():
    registry = SkillRegistry()
    skill = Skill(
        metadata=SkillMetadata(name="specific", triggers=["very-specific-trigger"]),
        body="body",
        path=Path("."),
    )
    registry.register(skill)

    assert registry.match("fix the bug") is None


def test_registry_invoke():
    registry = SkillRegistry()
    skill = Skill(
        metadata=SkillMetadata(name="test-skill", description="Test", mode="autopilot"),
        body="# Test Skill\n\nDo the thing.",
        path=Path("."),
    )
    registry.register(skill)

    invocation = registry.invoke("test-skill", "with these args")
    assert "Do the thing" in invocation.content
    assert "with these args" in invocation.content
    assert invocation.mode == "autopilot"


def test_registry_list_skills():
    registry = SkillRegistry()
    for name in ["a", "b", "c"]:
        registry.register(Skill(
            metadata=SkillMetadata(name=name, description=f"Skill {name}", tags=["test"]),
            body="", path=Path("."),
        ))
    listing = registry.list_skills()
    assert len(listing) == 3


def test_registry_build_index():
    registry = SkillRegistry()
    registry.register(Skill(
        metadata=SkillMetadata(name="autopilot", description="Full auto", tags=["orchestration"]),
        body="", path=Path("."),
    ))
    registry.register(Skill(
        metadata=SkillMetadata(name="review", description="Code review", tags=["development"]),
        body="", path=Path("."),
    ))

    index = registry.build_index_prompt()
    assert "autopilot" in index
    assert "review" in index
    assert "orchestration" in index


def test_load_all_from_builtins():
    """Integration: load built-in skills and verify they work end-to-end."""
    builtin_dir = Path(__file__).parent.parent / "mullet" / "core" / "skills" / "builtin"
    loader = SkillLoader(builtin_dir=builtin_dir)
    registry = SkillRegistry(loader=loader)
    registry.load_all()

    assert registry.get("brainstorm") is not None
    assert registry.get("mullet-job") is not None

    # Match should work
    match = registry.match("start a mullet job for this refactor")
    assert match is not None
    assert match.skill.metadata.name == "mullet-job"

    # Invoke should work
    inv = registry.invoke("mullet-job")
    assert inv.mode is None
    assert "mullet job" in inv.content
