"""Tests for the memory-file auto-discovery and Transform hook."""

from __future__ import annotations

from pathlib import Path

import pytest

from mullet.harness.hooks.types import TransformPayload
from mullet.harness.memory.discovery import (
    MemoryFile,
    assemble_memory_prompt,
    find_memory_files,
    make_memory_transform_hook,
)

pytestmark = [pytest.mark.harness]


# ---------------------------------------------------------------------------
# find_memory_files — surface-level existence
# ---------------------------------------------------------------------------


def _make_git_root(directory: Path) -> None:
    """Create an empty .git directory so the walker treats *directory* as root."""
    (directory / ".git").mkdir()


def test_finds_claude_md_in_cwd(tmp_path: Path) -> None:
    _make_git_root(tmp_path)
    (tmp_path / "CLAUDE.md").write_text("# Claude memory\nuse Kimi first.")
    files = find_memory_files(tmp_path)
    names = [f.path.name for f in files]
    assert "CLAUDE.md" in names
    claude = next(f for f in files if f.path.name == "CLAUDE.md")
    assert claude.kind == "CLAUDE"
    assert "Kimi" in claude.content


def test_finds_agents_md_in_cwd(tmp_path: Path) -> None:
    _make_git_root(tmp_path)
    (tmp_path / "AGENTS.md").write_text("# Agents\nlist of agents.")
    files = find_memory_files(tmp_path)
    assert any(f.path.name == "AGENTS.md" and f.kind == "AGENTS" for f in files)


def test_finds_memory_md_in_cwd(tmp_path: Path) -> None:
    _make_git_root(tmp_path)
    (tmp_path / "MEMORY.md").write_text("# Memory\nfacts.")
    files = find_memory_files(tmp_path)
    assert any(f.path.name == "MEMORY.md" and f.kind == "MEMORY" for f in files)


def test_finds_dot_mullet_memory_md_highest_priority(tmp_path: Path) -> None:
    _make_git_root(tmp_path)
    mullet_dir = tmp_path / ".mullet"
    mullet_dir.mkdir()
    (mullet_dir / "MEMORY.md").write_text("# Mullet memory")
    (tmp_path / "CLAUDE.md").write_text("# Claude")
    files = find_memory_files(tmp_path)
    # .mullet/MEMORY.md must be priority 0 (highest).
    mullet_entry = next(
        f for f in files if str(f.path).endswith(".mullet/MEMORY.md")
    )
    assert mullet_entry.priority == 0
    # All other entries have priority > 0.
    for f in files:
        if f is mullet_entry:
            continue
        assert f.priority > 0


def test_walks_up_to_git_root_and_stops(tmp_path: Path) -> None:
    # tmp_path/repo is the git root; deep subdir starts the walk.
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_git_root(repo)
    (repo / "CLAUDE.md").write_text("repo-claude")
    sub = repo / "pkg" / "module"
    sub.mkdir(parents=True)
    (sub / "AGENTS.md").write_text("sub-agents")

    # Outside the repo — should NOT be picked up.
    (tmp_path / "CLAUDE.md").write_text("outside-claude")

    files = find_memory_files(sub)
    contents = [f.content for f in files]
    assert "sub-agents" in contents
    assert "repo-claude" in contents
    assert "outside-claude" not in contents


def test_max_depth_respected(tmp_path: Path) -> None:
    # Build a 5-level nested tree; max_depth=1 should only look at sub
    # and one parent.  No git root anywhere so depth is the only limit.
    a = tmp_path / "a"
    b = a / "b"
    c = b / "c"
    d = c / "d"
    e = d / "e"
    e.mkdir(parents=True)
    (a / "CLAUDE.md").write_text("a-claude")
    (b / "CLAUDE.md").write_text("b-claude")
    (c / "CLAUDE.md").write_text("c-claude")
    (d / "CLAUDE.md").write_text("d-claude")
    (e / "CLAUDE.md").write_text("e-claude")

    files = find_memory_files(e, max_depth=1)
    contents = sorted(f.content for f in files)
    # Should contain start dir (e) and one parent (d) only.
    assert "e-claude" in contents
    assert "d-claude" in contents
    assert "c-claude" not in contents
    assert "b-claude" not in contents
    assert "a-claude" not in contents


def test_returns_empty_list_when_no_memory_files(tmp_path: Path) -> None:
    _make_git_root(tmp_path)
    files = find_memory_files(tmp_path)
    assert files == []


# ---------------------------------------------------------------------------
# assemble_memory_prompt
# ---------------------------------------------------------------------------


def test_assemble_orders_by_priority(tmp_path: Path) -> None:
    files = [
        MemoryFile(
            path=tmp_path / "CLAUDE.md", content="HIGH", priority=1, kind="CLAUDE"
        ),
        MemoryFile(
            path=tmp_path / ".mullet" / "MEMORY.md",
            content="HIGHEST",
            priority=0,
            kind="MEMORY",
        ),
        MemoryFile(
            path=tmp_path / "../parent/AGENTS.md",
            content="LOW",
            priority=5,
            kind="AGENTS",
        ),
    ]
    out = assemble_memory_prompt(files)
    # Highest priority (priority=0) must appear LAST so the LLM's
    # recency bias gives it the most weight.
    pos_high = out.find("HIGH")
    pos_highest = out.find("HIGHEST")
    pos_low = out.find("LOW")
    assert pos_low < pos_high < pos_highest


def test_assemble_truncates_at_max_chars(tmp_path: Path) -> None:
    big_content = "X" * 20000
    files = [
        MemoryFile(
            path=tmp_path / "CLAUDE.md", content=big_content, priority=1, kind="CLAUDE"
        )
    ]
    out = assemble_memory_prompt(files, max_chars=500)
    assert len(out) <= 500
    assert "[truncated]" in out


def test_assemble_returns_empty_for_empty_input() -> None:
    assert assemble_memory_prompt([]) == ""


# ---------------------------------------------------------------------------
# Transform hook
# ---------------------------------------------------------------------------


def test_hook_fires_only_once_per_session(tmp_path: Path) -> None:
    _make_git_root(tmp_path)
    (tmp_path / "CLAUDE.md").write_text("project memory")
    hook = make_memory_transform_hook(start=tmp_path)

    payload1 = TransformPayload(
        messages=[{"role": "user", "content": "hi"}],
        session_id="sess-1",
        purpose="inject",
    )
    out1 = hook(payload1)
    assert out1 is payload1
    assert any(
        m.get("role") == "system" and "project memory" in m.get("content", "")
        for m in payload1.messages
    )

    # Second fire on same session — should be a no-op.
    payload2 = TransformPayload(
        messages=[{"role": "user", "content": "hi again"}],
        session_id="sess-1",
        purpose="inject",
    )
    out2 = hook(payload2)
    assert out2 is None
    assert all(m.get("role") != "system" for m in payload2.messages)


def test_hook_prepends_system_message(tmp_path: Path) -> None:
    _make_git_root(tmp_path)
    (tmp_path / "CLAUDE.md").write_text("PRIMARY-MEM")
    hook = make_memory_transform_hook(start=tmp_path)
    payload = TransformPayload(
        messages=[{"role": "user", "content": "hi"}],
        session_id="sess-prepend",
        purpose="inject",
    )
    hook(payload)
    assert payload.messages[0]["role"] == "system"
    assert "PRIMARY-MEM" in payload.messages[0]["content"]
    assert payload.messages[1]["content"] == "hi"


def test_hook_skips_non_inject_purposes(tmp_path: Path) -> None:
    _make_git_root(tmp_path)
    (tmp_path / "CLAUDE.md").write_text("memory")
    hook = make_memory_transform_hook(start=tmp_path)
    for purpose in ("compact", "validate"):
        payload = TransformPayload(
            messages=[{"role": "user", "content": "hi"}],
            session_id=f"sess-{purpose}",
            purpose=purpose,  # type: ignore[arg-type]
        )
        result = hook(payload)
        assert result is None


def test_hook_no_op_when_no_memory_files(tmp_path: Path) -> None:
    _make_git_root(tmp_path)
    hook = make_memory_transform_hook(start=tmp_path)
    payload = TransformPayload(
        messages=[{"role": "user", "content": "hi"}],
        session_id="sess-empty",
        purpose="inject",
    )
    result = hook(payload)
    assert result is None
    assert payload.messages == [{"role": "user", "content": "hi"}]
