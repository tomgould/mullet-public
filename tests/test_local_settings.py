"""Personal settings read from the environment (or .env), with neutral defaults.

MULLET_NO_AI_ATTRIBUTION: off by default, so `mullet init` writes no
commit-trailer rule into the projects it sets up; on, it writes the "no
Co-Authored-By or other AI attribution" rule. (MULLET_OPENROUTER_REFERER is
covered in test_openrouter.py, MULLET_DELEGATION in test_delegation_switch.py.)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

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
