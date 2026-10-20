"""Tests for the 8 built-in Mullet harness skills."""

from __future__ import annotations

import re
from typing import Any

import pytest
import yaml

from mullet.harness.skills.builtin import BUILTIN_DIR

EXPECTED_SKILLS: tuple[str, ...] = (
    "commit",
    "plan",
    "review",
    "simplify",
    "debug",
    "test",
    "security-review",
    "init",
)


def _regex_parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Fallback parser used when mullet.harness.skills.frontmatter is absent."""
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        return {}, text
    fm_block, body = match.group(1), match.group(2)
    parsed = yaml.safe_load(fm_block) or {}
    if not isinstance(parsed, dict):
        return {}, body
    return parsed, body


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Use the project's frontmatter parser if available, else fall back to regex."""
    try:
        from mullet.harness.skills import frontmatter as fm_mod  # type: ignore[import-not-found]
    except ImportError:
        return _regex_parse_frontmatter(text)

    parser = getattr(fm_mod, "parse_frontmatter", None)
    if parser is None:
        return _regex_parse_frontmatter(text)

    result = parser(text)
    # Accept either a (dict, body) tuple or a dict-only return.
    if isinstance(result, tuple) and len(result) == 2:
        meta, body = result
        if isinstance(meta, dict):
            return meta, body if isinstance(body, str) else ""
    if isinstance(result, dict):
        return result, ""
    return _regex_parse_frontmatter(text)


@pytest.mark.harness
def test_all_8_builtin_skills_present() -> None:
    """All 8 expected SKILL.md files must exist under BUILTIN_DIR."""
    found = sorted(p.parent.name for p in BUILTIN_DIR.glob("*/SKILL.md"))
    assert sorted(EXPECTED_SKILLS) == found, (
        f"Expected skills {sorted(EXPECTED_SKILLS)}, found {found}"
    )


@pytest.mark.harness
@pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
def test_each_has_valid_frontmatter(skill_name: str) -> None:
    """Each SKILL.md must have name, description, and tools in its frontmatter."""
    skill_path = BUILTIN_DIR / skill_name / "SKILL.md"
    assert skill_path.exists(), f"Missing SKILL.md for {skill_name}"

    text = skill_path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(text)

    assert "name" in meta and meta["name"], f"{skill_name}: missing 'name'"
    assert "description" in meta and meta["description"], f"{skill_name}: missing 'description'"
    assert "tools" in meta and meta["tools"], f"{skill_name}: missing 'tools'"
    assert isinstance(meta["tools"], list), f"{skill_name}: 'tools' must be a list"
    assert all(isinstance(t, str) and t for t in meta["tools"]), (
        f"{skill_name}: 'tools' entries must be non-empty strings"
    )
    assert body.strip(), f"{skill_name}: SKILL.md body must not be empty"


@pytest.mark.harness
@pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
def test_skill_names_match_directory_names(skill_name: str) -> None:
    """The 'name' in YAML frontmatter must equal the directory name."""
    skill_path = BUILTIN_DIR / skill_name / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8")
    meta, _ = _parse_frontmatter(text)
    assert meta.get("name") == skill_name, (
        f"name='{meta.get('name')}' does not match directory '{skill_name}'"
    )


@pytest.mark.harness
@pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
def test_no_openharness_references_remain(skill_name: str) -> None:
    """No SKILL.md may contain a residual 'openharness' reference (case-insensitive)."""
    skill_path = BUILTIN_DIR / skill_name / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8")
    matches = re.findall(r"openharness", text, flags=re.IGNORECASE)
    # Allow zero references — attribution comments with bare URLs are fine but
    # the literal token "openharness" must not appear in adapted skills.
    assert not matches, (
        f"{skill_name}: found {len(matches)} 'openharness' reference(s) — adapt to 'mullet'"
    )
