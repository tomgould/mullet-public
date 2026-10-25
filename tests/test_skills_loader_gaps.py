"""Gap tests for mullet/core/skills/loader.py — target ≥95% coverage."""

import sys
from pathlib import Path
from unittest.mock import patch


from mullet.core.skills.loader import SkillLoader, parse_frontmatter


# ---------------------------------------------------------------------------
# Line 53 — external dirs scanned
# ---------------------------------------------------------------------------

def test_scan_includes_external_dirs(tmp_path: Path):
    ext_dir = tmp_path / "ext"
    ext_dir.mkdir()
    skill_dir = ext_dir / "ext_skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: ext_skill\n---\nbody")

    loader = SkillLoader(
        builtin_dir=tmp_path / "builtin",
        external_dirs=[ext_dir],
    )
    skills = loader.scan()
    names = [s.metadata.name for s in skills]
    assert "ext_skill" in names


# ---------------------------------------------------------------------------
# Lines 66 + 127-129 — platform mismatch skips skill
# ---------------------------------------------------------------------------

def test_platform_mismatch_skips_skill(tmp_path: Path):
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    skill_dir = builtin / "win_only"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: win_skill\nplatforms:\n  - windows\n---\nbody")

    loader = SkillLoader(builtin_dir=builtin)
    with patch.object(sys, "platform", "linux"):
        skills = loader.scan()
    assert not any(s.metadata.name == "win_skill" for s in skills)


# ---------------------------------------------------------------------------
# Lines 88-92 — security scan marks skill unsafe
# ---------------------------------------------------------------------------

def test_security_scan_unsafe_skips_skill(tmp_path: Path):
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    skill_dir = user_dir / "unsafe"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: unsafe_skill\n---\nbody")

    with patch(
        "mullet.core.skills.guard.scan_skill",
        return_value={"safe": False, "threats": [{"description": "bad"}]},
    ):
        loader = SkillLoader(user_dir=user_dir)
        skills = loader.scan()
    assert not any(s.metadata.name == "unsafe_skill" for s in skills)


# ---------------------------------------------------------------------------
# Lines 93-94 — security scan raises exception (best-effort)
# ---------------------------------------------------------------------------

def test_security_scan_exception_continues(tmp_path: Path):
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    skill_dir = user_dir / "exc"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: exc_skill\n---\nbody")

    with patch("mullet.core.skills.guard.scan_skill", side_effect=Exception("scan boom")):
        loader = SkillLoader(user_dir=user_dir)
        skills = loader.scan()
    assert any(s.metadata.name == "exc_skill" for s in skills)


# ---------------------------------------------------------------------------
# Line 99 — metadata name falls back to parent directory name
# ---------------------------------------------------------------------------

def test_metadata_name_fallback(tmp_path: Path):
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    skill_dir = builtin / "fallback_name"
    skill_dir.mkdir()
    # No 'name' key in frontmatter
    (skill_dir / "SKILL.md").write_text("---\n---\nbody")

    loader = SkillLoader(builtin_dir=builtin)
    skills = loader.scan()
    assert any(s.metadata.name == "fallback_name" for s in skills)


# ---------------------------------------------------------------------------
# Lines 119-121 — exception while loading a single skill
# ---------------------------------------------------------------------------

def test_load_skill_exception_returns_none(tmp_path: Path):
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    skill_dir = builtin / "bad"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: bad\n---\nbody")

    with patch("mullet.core.skills.loader.SkillMetadata", side_effect=Exception("meta boom")):
        loader = SkillLoader(builtin_dir=builtin)
        skills = loader.scan()
    assert not any(s.metadata.name == "bad" for s in skills)


# ---------------------------------------------------------------------------
# Line 142 — parse_frontmatter incomplete separator
# ---------------------------------------------------------------------------

def test_parse_frontmatter_incomplete_separator():
    content = "---\nname: foo\n"
    metadata, body = parse_frontmatter(content)
    assert metadata == {}
    assert body == content


def test_parse_frontmatter_no_frontmatter():
    content = "just body text"
    metadata, body = parse_frontmatter(content)
    assert metadata == {}
    assert body == content


# ---------------------------------------------------------------------------
# Lines 63-64 — disabled skill skipped with debug log
# ---------------------------------------------------------------------------

def test_disabled_skill_skipped(tmp_path: Path):
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    skill_dir = builtin / "disabled_skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: disabled_skill\n---\nbody")

    loader = SkillLoader(builtin_dir=builtin, disabled=["disabled_skill"])
    with patch("mullet.core.skills.loader.logger") as mock_logger:
        skills = loader.scan()
    assert not any(s.metadata.name == "disabled_skill" for s in skills)
    mock_logger.debug.assert_called_once()


# ---------------------------------------------------------------------------
# Lines 146-148 — malformed YAML frontmatter warning
# ---------------------------------------------------------------------------

def test_parse_frontmatter_malformed_yaml():
    content = "---\nname: [unclosed\n---\nbody"
    with patch("mullet.core.skills.loader.logger") as mock_logger:
        metadata, body = parse_frontmatter(content)
    assert metadata == {}
    assert body == "\nbody"
    mock_logger.warning.assert_called_once()
