"""The ``jobs:`` settings: parsing, defaults, config show and validation."""

import pytest

from mullet.jobs.settings import (
    DEFAULT_GIT_TIMEOUT_SECONDS, DEFAULT_GRACE_SECONDS, DEFAULT_REASK_ATTEMPTS, JobSettings, load_settings,
    parse_settings,
)


def test_defaults_when_nothing_is_set():
    settings = parse_settings(None)
    assert settings == JobSettings()
    assert (settings.reask_attempts, settings.idle_minutes) == (DEFAULT_REASK_ATTEMPTS, None)
    assert settings.grace_seconds == DEFAULT_GRACE_SECONDS
    assert settings.git_timeout_seconds == DEFAULT_GIT_TIMEOUT_SECONDS


def test_valid_values_are_read():
    settings = parse_settings({"reask_attempts": 0, "idle_minutes": 20, "grace_seconds": 5,
                               "git_timeout_seconds": 90})
    assert (settings.reask_attempts, settings.idle_minutes, settings.grace_seconds,
            settings.git_timeout_seconds) == (0, 20.0, 5.0, 90.0)
    assert settings.problems == []


@pytest.mark.parametrize("data, key", [
    ({"reask_attempts": -1}, "reask_attempts"), ({"reask_attempts": 1.5}, "reask_attempts"),
    ({"reask_attempts": True}, "reask_attempts"), ({"reask_attempts": 9}, "reask_attempts"),
    ({"idle_minutes": 0}, "idle_minutes"), ({"idle_minutes": "ten"}, "idle_minutes"),
    ({"grace_seconds": -3}, "grace_seconds"), ({"git_timeout_seconds": 0}, "git_timeout_seconds"),
])
def test_a_bad_value_is_reported_and_the_default_kept(data, key):
    settings = parse_settings(data)
    assert any(key in p for p in settings.problems)
    assert settings.reask_attempts == DEFAULT_REASK_ATTEMPTS and settings.idle_minutes is None


def test_unknown_keys_and_non_mappings_are_reported():
    assert any("unknown setting" in p for p in parse_settings({"nope": 1}).problems)
    assert parse_settings([1, 2]).problems


def test_load_settings_reads_the_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yaml").write_text("jobs:\n  idle_minutes: 30\n  reask_attempts: 2\n")
    settings = load_settings()
    assert settings.idle_minutes == 30.0 and settings.reask_attempts == 2


def test_a_missing_or_broken_file_gives_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    assert load_settings() == JobSettings()
    (tmp_path / "config.yaml").write_text("jobs: [unclosed")
    assert load_settings().problems


def test_config_show_lists_the_jobs_settings(tmp_path, monkeypatch):
    from mullet.cli import config_show

    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yaml").write_text("jobs:\n  idle_minutes: 30\n")
    rows = {r["setting"]: r for r in config_show.collect()["sections"]["jobs"]}
    assert rows["jobs.idle_minutes"]["value"] == 30.0 and rows["jobs.idle_minutes"]["source"] != "default"
    assert rows["jobs.reask_attempts"]["source"] == "default"
    assert "jobs" not in " ".join(config_show.collect()["unrecognised"])


def test_the_validator_accepts_and_rejects_the_jobs_block():
    from mullet.config.validate import validate_config

    assert validate_config({"jobs": {"idle_minutes": 20}}).valid
    assert not validate_config({"jobs": {"reask_attempts": 99}}).valid
    assert not validate_config({"jobs": 5}).valid


@pytest.mark.parametrize("key", ["reask_attempts", "idle_minutes", "grace_seconds", "git_timeout_seconds"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")], ids=["inf", "-inf", "nan"])
def test_infinite_and_nan_values_are_reported_not_raised(key, value):
    settings = parse_settings({key: value})
    assert any(key in p for p in settings.problems)
    default = JobSettings()
    assert getattr(settings, key) == getattr(default, key)


@pytest.mark.parametrize("text", [".inf", ".nan", "-.inf"])
def test_load_settings_survives_yaml_infinity_and_nan(tmp_path, monkeypatch, text):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"jobs:\n  reask_attempts: {text}\n  idle_minutes: {text}\n  grace_seconds: {text}\n"
        f"  git_timeout_seconds: {text}\n")
    settings = load_settings()
    assert len(settings.problems) == 4
    assert (settings.reask_attempts, settings.git_timeout_seconds) == (DEFAULT_REASK_ATTEMPTS,
                                                                       DEFAULT_GIT_TIMEOUT_SECONDS)


def test_a_job_clone_git_command_still_runs_with_a_bad_setting(tmp_path, monkeypatch):
    from mullet.jobs.clone import _git_env, _run

    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yaml").write_text("jobs:\n  reask_attempts: .inf\n")
    assert _run(["git", "--version"], cwd=tmp_path, env=_git_env()).returncode == 0


def test_load_settings_never_raises(tmp_path, monkeypatch):
    from mullet.jobs import settings as mod

    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yaml").write_text("jobs:\n  reask_attempts: 1\n")

    def broken(data, source="jobs"):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "parse_settings", broken)
    settings = load_settings()
    assert settings.reask_attempts == DEFAULT_REASK_ATTEMPTS and settings.problems


def test_commit_body_is_full_by_default_and_takes_minimal():
    assert JobSettings().commit_body == "full"
    assert parse_settings({"commit_body": "minimal"}).commit_body == "minimal"
    assert parse_settings({"commit_body": "full"}).problems == []


@pytest.mark.parametrize("value", ["short", "", 1, None, True, ["minimal"]])
def test_a_bad_commit_body_is_reported_and_full_kept(value):
    settings = parse_settings({"commit_body": value})
    assert settings.commit_body == "full"
    if value is not None:
        assert any("commit_body" in p and "full" in p and "minimal" in p for p in settings.problems)


def test_config_show_and_validate_know_commit_body(tmp_path, monkeypatch):
    from mullet.cli import config_show
    from mullet.config.validate import validate_config

    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yaml").write_text("jobs:\n  commit_body: minimal\n")
    rows = {r["setting"]: r for r in config_show.collect()["sections"]["jobs"]}
    assert rows["jobs.commit_body"]["value"] == "minimal" and rows["jobs.commit_body"]["source"] != "default"
    assert "jobs.commit_body" not in " ".join(config_show.collect()["unrecognised"])
    assert validate_config({"jobs": {"commit_body": "minimal"}}).valid
    assert not validate_config({"jobs": {"commit_body": "tiny"}}).valid


@pytest.mark.parametrize("value", [0, 0.25, 4.9])
def test_grace_seconds_has_a_floor_so_the_cli_can_save_its_session(value):
    from mullet.jobs.settings import MIN_GRACE_SECONDS
    assert MIN_GRACE_SECONDS == 5
    settings = parse_settings({"grace_seconds": value})
    assert any("grace_seconds" in p and "5" in p for p in settings.problems)
    assert settings.grace_seconds == DEFAULT_GRACE_SECONDS
    assert parse_settings({"grace_seconds": 5}).grace_seconds == 5.0
