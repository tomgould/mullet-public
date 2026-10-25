"""Where Mullet keeps its files: mullet.core.paths and `mullet migrate-state`.

conftest gives every test its own HOME and clears the XDG_* and MULLET_*
location variables, so Path.home() here is a throwaway directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mullet.cli.migrate_state import cmd_migrate_state
from mullet.core import paths
from mullet.jobs import paths as job_paths


def _home() -> Path:
    return Path.home()


def _old_install(*names: str) -> Path:
    """Create ~/.mullet holding the given files (a trailing / makes a directory)."""
    legacy = _home() / ".mullet"
    legacy.mkdir()
    for name in names:
        if name.endswith("/"):
            (legacy / name).mkdir()
        else:
            (legacy / name).write_text(f"{name} contents\n")
    return legacy


# ---------------------------------------------------------------------------
# XDG rules
# ---------------------------------------------------------------------------

class TestXdgDefaults:
    def test_defaults_follow_the_spec(self):
        home = _home()
        assert paths.config_dir() == home / ".config" / "mullet"
        assert paths.state_dir() == home / ".local" / "state" / "mullet"
        assert paths.data_dir() == home / ".local" / "share" / "mullet"
        assert paths.cache_dir() == home / ".cache" / "mullet"
        assert paths.pools_dir() == home / ".local" / "share" / "mullet" / "pools"

    def test_absolute_xdg_values_are_used(self, monkeypatch, tmp_path):
        for var in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
            monkeypatch.setenv(var, str(tmp_path / var.lower()))
        assert paths.config_dir() == tmp_path / "xdg_config_home" / "mullet"
        assert paths.state_dir() == tmp_path / "xdg_state_home" / "mullet"
        assert paths.data_dir() == tmp_path / "xdg_data_home" / "mullet"
        assert paths.cache_dir() == tmp_path / "xdg_cache_home" / "mullet"

    @pytest.mark.parametrize("value", ["relative/dir", "./here", "~/not-expanded", ""])
    def test_relative_or_empty_xdg_values_are_ignored(self, monkeypatch, value):
        for var in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
            monkeypatch.setenv(var, value)
        home = _home()
        assert paths.config_dir() == home / ".config" / "mullet"
        assert paths.state_dir() == home / ".local" / "state" / "mullet"
        assert paths.data_dir() == home / ".local" / "share" / "mullet"
        assert paths.cache_dir() == home / ".cache" / "mullet"

    def test_mullet_config_dir_overrides_xdg(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "mine"))
        assert paths.config_dir() == tmp_path / "mine"

    def test_mullet_config_dir_expands_the_home_marker(self, monkeypatch):
        monkeypatch.setenv("MULLET_CONFIG_DIR", "~/mullet-config")
        assert paths.config_dir() == _home() / "mullet-config"

    def test_trawl_pools_dir_is_used_verbatim(self, monkeypatch):
        monkeypatch.setenv("TRAWL_POOLS_DIR", "/srv/pools")
        assert paths.pools_dir() == Path("/srv/pools")

    def test_nothing_is_created(self):
        for fn in (paths.config_dir, paths.state_dir, paths.data_dir, paths.cache_dir, paths.pools_dir):
            assert not fn().exists()
        assert list(_home().iterdir()) == []


class TestCallTime:
    def test_paths_follow_a_changed_home(self, monkeypatch, tmp_path):
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir()
        second.mkdir()
        monkeypatch.setenv("HOME", str(first))
        assert paths.state_dir() == first / ".local" / "state" / "mullet"
        monkeypatch.setenv("HOME", str(second))
        assert paths.state_dir() == second / ".local" / "state" / "mullet"

    def test_paths_follow_a_changed_xdg_variable(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "one"))
        assert paths.data_dir() == tmp_path / "one" / "mullet"
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "two"))
        assert paths.data_dir() == tmp_path / "two" / "mullet"

    def test_consumers_follow_home_too(self):
        from mullet.core.logging.interaction_log import default_db_path
        from mullet.core.providers.cost_history import _cost_history_path
        from mullet.core.router.preflight import _cache_path

        assert default_db_path() == paths.state_dir() / "interactions.db"
        assert _cost_history_path() == paths.state_dir() / "cost_history.jsonl"
        assert _cache_path() == paths.cache_dir() / "preflight.jsonl"


# ---------------------------------------------------------------------------
# An older install in ~/.mullet
# ---------------------------------------------------------------------------

class TestLegacyInstall:
    def test_legacy_dir_is_dot_mullet(self):
        assert paths.legacy_dir() == _home() / ".mullet"

    def test_an_older_install_keeps_state_data_and_caches_in_dot_mullet(self):
        legacy = _old_install("routing.log", "pools/")
        assert paths.legacy_active()
        assert paths.state_dir() == legacy
        assert paths.data_dir() == legacy
        assert paths.cache_dir() == legacy
        assert paths.pools_dir() == legacy / "pools"

    def test_config_still_goes_to_the_config_dir(self):
        _old_install("interactions.db")
        assert paths.config_dir() == _home() / ".config" / "mullet"

    def test_xdg_variants_ignore_the_older_install(self):
        _old_install("interactions.db")
        assert paths.xdg_state_dir() == _home() / ".local" / "state" / "mullet"
        assert paths.xdg_data_dir() == _home() / ".local" / "share" / "mullet"
        assert paths.xdg_cache_dir() == _home() / ".cache" / "mullet"

    def test_no_dot_mullet_means_xdg(self):
        assert not paths.legacy_active()

    def test_an_empty_dot_mullet_is_not_an_install(self):
        _old_install()
        assert not paths.legacy_active()
        assert paths.state_dir() == _home() / ".local" / "state" / "mullet"

    def test_a_project_dir_in_home_is_not_an_install(self):
        """Running Mullet from HOME makes ~/.mullet a project's .mullet/ directory."""
        _old_install("state/", "observations/", "decisions.jsonl", "session.json", "skills/")
        assert not paths.legacy_active()

    def test_a_migrated_dot_mullet_is_not_an_install(self):
        _old_install("routing.log", paths.MOVED_NOTE)
        assert not paths.legacy_active()

    def test_a_file_named_dot_mullet_is_not_an_install(self):
        (_home() / ".mullet").write_text("not a directory")
        assert not paths.legacy_active()

    def test_jobs_never_live_in_dot_mullet(self):
        _old_install("routing.log")
        assert job_paths.jobs_home() == (_home() / ".local" / "share" / "mullet" / "jobs").resolve()


class TestConfigFile:
    def test_prefers_the_config_dir(self):
        _old_install("config.yaml")
        config = paths.config_dir()
        config.mkdir(parents=True)
        (config / "config.yaml").write_text("new: true\n")
        assert paths.config_file("config.yaml") == config / "config.yaml"

    def test_falls_back_to_an_older_install(self):
        legacy = _old_install("config.yaml")
        assert paths.config_file("config.yaml") == legacy / "config.yaml"

    def test_names_the_config_dir_when_neither_exists(self):
        assert paths.config_file("config.yaml") == paths.config_dir() / "config.yaml"

    def test_config_loader_reads_the_older_file(self, monkeypatch, tmp_path):
        from mullet.config.loader import load_from_yaml
        # The loader prefers ./mullet.yaml; a developer checkout often has one.
        monkeypatch.chdir(tmp_path)
        legacy = _old_install()
        (legacy / "config.yaml").write_text("routing:\n  max: 3\n")
        assert load_from_yaml() == {"routing": {"max": 3}}


class TestJobPaths:
    def test_overrides_are_kept(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
        monkeypatch.setenv("MULLET_PROFILES_DIR", str(tmp_path / "profiles"))
        assert job_paths.jobs_home() == (tmp_path / "jobs").resolve()
        assert job_paths.profiles_dir() == (tmp_path / "profiles").resolve()

    def test_defaults_follow_xdg(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        assert job_paths.jobs_home() == (tmp_path / "data" / "mullet" / "jobs").resolve()
        assert job_paths.profiles_dir() == (tmp_path / "config" / "mullet" / "projects").resolve()

    def test_profiles_follow_mullet_config_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "cfg"))
        assert job_paths.profiles_dir() == (tmp_path / "cfg" / "projects").resolve()


# ---------------------------------------------------------------------------
# mullet migrate-state
# ---------------------------------------------------------------------------

def _tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


class TestMigrateState:
    def test_nothing_to_migrate(self, capsys):
        assert cmd_migrate_state([]) == 0
        assert "Nothing to migrate" in capsys.readouterr().out
        assert not (_home() / ".mullet").exists()

    def test_dry_run_moves_nothing(self, capsys):
        legacy = _old_install("config.yaml", "routing.log", "pools/", "model_cache.json")
        before = _tree(legacy)
        assert cmd_migrate_state(["--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "Would move 4 entries" in out
        assert f"config.yaml -> {_home() / '.config' / 'mullet' / 'config.yaml'}" in out
        assert "Nothing was moved (dry run)." in out
        assert _tree(legacy) == before
        assert not (_home() / ".local").exists()
        assert not (_home() / ".config").exists()
        assert paths.legacy_active()

    def test_real_run_moves_each_entry_to_its_kind(self, capsys):
        legacy = _old_install(
            "config.yaml", ".env", "routing.log", "sessions/", "pools/",
            "memory.json", "model_cache.json", "cache/", "something-else.txt",
        )
        (legacy / "pools" / "mypool").mkdir()
        (legacy / "pools" / "mypool" / "config.yaml").write_text("name: mypool\n")
        (legacy / "cache" / "preflight.jsonl").write_text("{}\n")

        assert cmd_migrate_state([]) == 0
        out = capsys.readouterr().out
        assert "Moved 9 entries" in out

        home = _home()
        config = home / ".config" / "mullet"
        state = home / ".local" / "state" / "mullet"
        data = home / ".local" / "share" / "mullet"
        cache = home / ".cache" / "mullet"
        assert (config / "config.yaml").read_text() == "config.yaml contents\n"
        assert (config / ".env").is_file()
        assert (state / "routing.log").is_file()
        assert (state / "sessions").is_dir()
        assert (data / "pools" / "mypool" / "config.yaml").read_text() == "name: mypool\n"
        assert (data / "memory.json").is_file()
        assert (data / "something-else.txt").is_file()  # unknown entries count as data
        assert (cache / "model_cache.json").is_file()
        assert (cache / "cache" / "preflight.jsonl").is_file()

        assert sorted(p.name for p in legacy.iterdir()) == [paths.MOVED_NOTE]
        note = (legacy / paths.MOVED_NOTE).read_text()
        assert f"pools -> {data / 'pools'}" in note
        assert f"config {config}" in " ".join(note.split())

        assert not paths.legacy_active()
        assert paths.state_dir() == state
        assert paths.pools_dir() == data / "pools"

    def test_a_second_run_has_nothing_to_move(self, capsys):
        _old_install("routing.log")
        assert cmd_migrate_state([]) == 0
        capsys.readouterr()
        assert cmd_migrate_state([]) == 0
        assert "Nothing to migrate" in capsys.readouterr().out

    def test_refuses_when_a_target_exists_and_is_not_empty(self, capsys):
        legacy = _old_install("routing.log", "pools/")
        taken = _home() / ".local" / "share" / "mullet" / "pools"
        taken.mkdir(parents=True)
        (taken / "other").write_text("already here\n")
        before = _tree(legacy)

        assert cmd_migrate_state([]) == 1
        out = capsys.readouterr().out
        assert "nothing was moved" in out
        assert str(taken) in out
        assert _tree(legacy) == before
        assert not (_home() / ".local" / "state").exists()
        assert not (legacy / paths.MOVED_NOTE).exists()

    def test_dry_run_reports_the_refusal_too(self, capsys):
        _old_install("routing.log")
        target = _home() / ".local" / "state" / "mullet" / "routing.log"
        target.parent.mkdir(parents=True)
        target.write_text("newer\n")
        assert cmd_migrate_state(["--dry-run"]) == 1
        assert str(target) in capsys.readouterr().out

    def test_an_empty_target_directory_is_fine(self):
        legacy = _old_install("sessions/")
        (legacy / "sessions" / "s1.json").write_text("{}")
        empty = _home() / ".local" / "state" / "mullet" / "sessions"
        empty.mkdir(parents=True)
        assert cmd_migrate_state([]) == 0
        assert (empty / "s1.json").is_file()

    def test_refuses_when_trawl_pools_dir_points_inside(self, monkeypatch, capsys):
        legacy = _old_install("pools/")
        monkeypatch.setenv("TRAWL_POOLS_DIR", str(legacy / "pools"))
        assert cmd_migrate_state([]) == 1
        assert "TRAWL_POOLS_DIR" in capsys.readouterr().out
        assert (legacy / "pools").is_dir()

    def test_wired_into_the_cli(self, monkeypatch, capsys):
        from mullet.cli import main as cli_main
        _old_install("routing.log")
        monkeypatch.setattr(sys, "argv", ["mullet", "migrate-state", "--dry-run"])
        with pytest.raises(SystemExit) as exc:
            cli_main._main_inner()
        assert exc.value.code == 0
        assert "Would move 1 entry" in capsys.readouterr().out
