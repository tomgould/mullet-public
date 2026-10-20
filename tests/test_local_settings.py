"""Personal settings read from the environment (or .env), with neutral defaults.

MULLET_NO_AI_ATTRIBUTION: off by default, so `mullet init`, `mullet
setup-claude` and `mullet autopilot init` write no commit-trailer rule into
the projects they set up; on, they write the "no Co-Authored-By or other AI
attribution" rule. (MULLET_OPENROUTER_REFERER is covered in test_openrouter.py,
MULLET_DELEGATION in test_delegation_switch.py.)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from mullet.cli import autopilot_init, setup_claude
from mullet.cli.init_project import run_init
from mullet.core.env import no_ai_attribution, openrouter_referer


@pytest.mark.parametrize("value,expected", [
    (None, False), ("", False), ("off", False), ("0", False),
    ("on", True), ("1", True), ("true", True), ("yes", True),
])
def test_no_ai_attribution_values(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("MULLET_NO_AI_ATTRIBUTION", raising=False)
    else:
        monkeypatch.setenv("MULLET_NO_AI_ATTRIBUTION", value)
    assert no_ai_attribution() is expected


def test_openrouter_referer_values(monkeypatch):
    monkeypatch.delenv("MULLET_OPENROUTER_REFERER", raising=False)
    assert openrouter_referer() is None
    monkeypatch.setenv("MULLET_OPENROUTER_REFERER", "  ")
    assert openrouter_referer() is None
    monkeypatch.setenv("MULLET_OPENROUTER_REFERER", "https://example.com/app")
    assert openrouter_referer() == "https://example.com/app"


# ---------------------------------------------------------------------------
# mullet init: the seeded project memory
# ---------------------------------------------------------------------------

def _init(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    target = tmp_path / "project"
    target.mkdir()
    args = argparse.Namespace(
        target_dir=str(target), pool="test-pool", providers="kimi", model="m",
        speed="balanced", force=False, dry_run=False, no_claude=True,
        no_memory=False, no_pool=True, remaining=["init", str(target)],
    )
    run_init(args)
    return home / ".claude" / "projects" / ("-" + str(target).replace("/", "-")) / "memory"


def test_init_seeds_no_trailer_rule_by_default(tmp_path, monkeypatch):
    mem = _init(tmp_path, monkeypatch)
    assert (mem / "feedback_prefer_concurrent.md").exists()
    assert not (mem / "feedback_coauthored_by.md").exists()
    assert "Co-Authored-By" not in (mem / "MEMORY.md").read_text()


def test_init_seeds_the_trailer_rule_when_asked(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_NO_AI_ATTRIBUTION", "on")
    mem = _init(tmp_path, monkeypatch)
    note = mem / "feedback_coauthored_by.md"
    assert note.exists()
    assert "Co-Authored-By" in note.read_text()
    index = (mem / "MEMORY.md").read_text().splitlines()
    assert index[0] == "# Memory Index"
    assert index[2].startswith("- [No Co-Authored-By in commits](feedback_coauthored_by.md)")
    # The other notes are still listed after it.
    assert any("feedback_prefer_concurrent.md" in line for line in index[3:])


# ---------------------------------------------------------------------------
# mullet setup-claude: the /commit skill
# ---------------------------------------------------------------------------

def _commit_skill(tmp_path: Path, monkeypatch) -> str:
    monkeypatch.setattr(setup_claude, "_get_claude_dir", lambda: tmp_path / ".claude")
    setup_claude._copy_claude_skill_files(dry_run=False, force=True)
    return (tmp_path / ".claude" / "skills" / "commit.md").read_text()


def test_commit_skill_follows_repo_conventions_by_default(tmp_path, monkeypatch):
    text = _commit_skill(tmp_path, monkeypatch)
    assert setup_claude._COMMIT_STEP_NEUTRAL in text
    assert "Co-Authored-By" not in text


def test_commit_skill_forbids_attribution_when_asked(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_NO_AI_ATTRIBUTION", "on")
    text = _commit_skill(tmp_path, monkeypatch)
    assert "5. " + setup_claude._COMMIT_STEP_NO_AI_ATTRIBUTION in text
    assert setup_claude._COMMIT_STEP_NEUTRAL not in text


# ---------------------------------------------------------------------------
# mullet autopilot init: standards.md
# ---------------------------------------------------------------------------

def _standards(tmp_path: Path) -> str:
    assert autopilot_init.run_autopilot_init([], cwd=tmp_path) == 0
    return (tmp_path / ".mullet" / "autopilot" / "standards.md").read_text()


def test_autopilot_standards_have_no_trailer_rule_by_default(tmp_path):
    assert "Co-Authored-By" not in _standards(tmp_path)


def test_autopilot_standards_get_the_rule_when_asked(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_NO_AI_ATTRIBUTION", "on")
    text = _standards(tmp_path)
    git = text.split("## Git", 1)[1]
    assert autopilot_init._NO_AI_ATTRIBUTION_RULE in git


def test_rule_is_added_once_and_keeps_later_sections(monkeypatch):
    monkeypatch.setenv("MULLET_NO_AI_ATTRIBUTION", "on")
    src = "# S\n\n## Git\n- one\n\n## Other\n- two\n"
    out = autopilot_init._with_commit_rules(src)
    assert out == "# S\n\n## Git\n- one\n" + autopilot_init._NO_AI_ATTRIBUTION_RULE + "\n\n## Other\n- two\n"
    assert autopilot_init._with_commit_rules(out) == out
    assert autopilot_init._with_commit_rules("# S\n").endswith(
        "## Git\n" + autopilot_init._NO_AI_ATTRIBUTION_RULE + "\n"
    )
