"""Tests for `mullet.cli.doctor_harness.check_harness_layer`."""

from __future__ import annotations

from pathlib import Path

import pytest

from mullet.cli.doctor_harness import CheckResult, check_harness_layer


pytestmark = pytest.mark.harness


def test_check_harness_layer_returns_list_of_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke test — basic shape: list of CheckResult, non-empty."""
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "cfg"))
    results = check_harness_layer(cwd=tmp_path)
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert isinstance(r, CheckResult)
        assert r.name
        assert r.status in {"ok", "warn", "info"}
        assert isinstance(r.message, str)


def test_subsystem_imports_all_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """All 14 harness subsystem packages must import cleanly."""
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "cfg"))
    results = check_harness_layer(cwd=tmp_path)
    subsystem_results = [r for r in results if r.name.startswith("Harness: subsystem")]
    assert len(subsystem_results) == 14
    for r in subsystem_results:
        assert r.status == "ok", f"{r.name}: {r.message}"


def test_profiles_config_check_with_no_yaml_returns_ok(
    isolated_profiles_dir: Path, tmp_path: Path
) -> None:
    """When profiles.yaml is empty/missing the manager seeds defaults — should report ok."""
    results = check_harness_layer(cwd=tmp_path)
    profile_results = [r for r in results if r.name == "Harness: profiles"]
    assert len(profile_results) == 1
    assert profile_results[0].status == "ok"


def test_skills_check_finds_8_builtins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The 8 lifted built-in skills should all load cleanly."""
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "cfg"))
    results = check_harness_layer(cwd=tmp_path)
    skill_results = [r for r in results if r.name == "Harness: skills"]
    assert len(skill_results) == 1
    sr = skill_results[0]
    assert sr.status == "ok"
    # message looks like "8 loaded: commit, debug, ..."
    assert "loaded" in sr.message
    # Should be at least 8 builtins; allow more if user/project skills exist.
    leading_count = int(sr.message.split(" ", 1)[0])
    assert leading_count >= 8


def test_plugins_check_handles_no_plugin_dirs_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No plugin dirs configured → ok with 0 loaded, 0 errors."""
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "cfg"))
    # Point plugin discovery at an empty dir to guarantee no plugins.
    from mullet.harness.plugins import loader as plugins_loader

    monkeypatch.setattr(
        plugins_loader,
        "default_plugin_scope_dirs",
        lambda: {"project": [], "user": [], "global": []},
    )
    results = check_harness_layer(cwd=tmp_path)
    plugin_results = [r for r in results if r.name == "Harness: plugins"]
    assert len(plugin_results) == 1
    pr = plugin_results[0]
    assert pr.status == "ok"
    assert "0 loaded" in pr.message
    assert "0 error" in pr.message


def test_sessions_dir_writable_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """sessions/ must be createable + writable under HOME."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    results = check_harness_layer(cwd=tmp_path)
    sess_results = [r for r in results if r.name == "Harness: sessions dir"]
    assert len(sess_results) == 1
    assert sess_results[0].status == "ok"
    assert (fake_home / ".config" / "mullet" / "sessions").is_dir()


def test_workflow_store_check_creates_or_validates_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workflow store check creates the SQLite db with the 3 required tables."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    results = check_harness_layer(cwd=tmp_path)
    wf_results = [r for r in results if r.name == "Harness: workflow store"]
    assert len(wf_results) == 1
    wr = wf_results[0]
    assert wr.status == "ok"
    db_file = fake_home / ".config" / "mullet" / "workflows.db"
    assert db_file.exists()
    # Inspect schema directly.
    import sqlite3

    conn = sqlite3.connect(str(db_file))
    try:
        rows = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    finally:
        conn.close()
    assert {"workflow_runs", "workflow_events", "workflow_messages"} <= rows
