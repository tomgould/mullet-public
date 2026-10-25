"""Tests for ``mullet config show``."""
import json

import pytest

from mullet.cli.config_cli import cmd_config


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("TEST_SHOW_PROVIDER_KEY", "sk-do-not-print")
    path = tmp_path / "config.yaml"

    def write(text: str):
        path.write_text(text, encoding="utf-8")
        return path

    return write


def _rows(capsys, *args):
    assert cmd_config(["show", "--json", *args]) == 0
    return json.loads(capsys.readouterr().out)


def _by_name(data):
    return {r["setting"]: r for rows in data["sections"].values() for r in rows}


def test_defaults_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    assert cmd_config(["show"]) == 0
    out = capsys.readouterr().out
    assert "ladder.max_spend" in out and "not found" in out
    rows = _by_name(_rows(capsys))
    assert rows["ladder.max_spend"]["value"] == 2.0
    assert rows["routing.budget.daily_max_dollars"]["value"] == 10.0
    assert all(r["source"] == "default" for r in rows.values())


def test_user_file_overrides_ladder(cfg, tmp_path, capsys):
    cfg("ladder:\n  max_spend: 5.5\n  local:\n    model: lmstudio/x\n    url: http://127.0.0.1:9999\n")
    rows = _by_name(_rows(capsys))
    assert rows["ladder.max_spend"]["value"] == 5.5
    assert rows["ladder.max_spend"]["default"] == 2.0
    assert rows["ladder.max_spend"]["source"] == str(tmp_path / "config.yaml")
    assert rows["ladder.cheap_minutes"]["source"] == "default"
    assert rows["ladder.local_url"]["value"] == "http://127.0.0.1:9999"


def test_value_equal_to_default_but_set_reports_file(cfg, tmp_path, capsys):
    cfg("ladder:\n  max_spend: 2.0\n")
    assert _by_name(_rows(capsys))["ladder.max_spend"]["source"] == str(tmp_path / "config.yaml")


def test_legacy_budget_keys_labelled(cfg, capsys):
    cfg("budget:\n  per_task: 1\n  daily: 5\n  envelope:\n    weekly: 3\n")
    rows = _by_name(_rows(capsys))
    assert "legacy delegation path" in rows["budget.per_task"]["note"]
    assert "legacy delegation path" in rows["budget.daily"]["note"]
    assert rows["budget.envelope.weekly"]["note"] == ""
    assert cmd_config(["show", "budget"]) == 0
    assert "not used by gated jobs" in capsys.readouterr().out


def test_unknown_keys_flagged(cfg, capsys):
    cfg("ladder:\n  max_spnd: 3\nladdr: {}\nrouting:\n  bogus: 1\n")
    data = _rows(capsys)
    assert {"ladder.max_spnd", "laddr", "routing.bogus"} <= set(data["unrecognised"])
    assert cmd_config(["show"]) == 0
    assert "[unrecognised]" in capsys.readouterr().out


def test_problems_reported(cfg, capsys):
    cfg("ladder:\n  max_spend: lots\n")
    assert any("max_spend" in p for p in _rows(capsys)["problems"])


def test_secrets_masked_and_key_status_only(cfg, capsys):
    cfg(
        "providers:\n  p1:\n    api_key_env: TEST_SHOW_PROVIDER_KEY\n"
        "  p2:\n    api_key: sk-inline-secret\n"
        "  p3:\n    api_key_env: TEST_SHOW_UNSET_VAR\n"
        "budget:\n  api_token: tok-123\n  envelope:\n    secret_thing: abc-xyz\n"
    )
    assert cmd_config(["show"]) == 0
    text = capsys.readouterr().out
    data = _rows(capsys)
    text += json.dumps(data)
    for leaked in ("sk-do-not-print", "sk-inline-secret", "tok-123", "abc-xyz"):
        assert leaked not in text
    rows = _by_name(data)
    assert rows["providers.p1.api_key"]["value"] == "set"
    assert rows["providers.p2.api_key"]["value"] == "set"
    assert rows["providers.p3.api_key"]["value"] == "unset"
    assert rows["budget.api_token"]["value"] == "***"
    assert rows["budget.envelope.secret_thing"]["value"] == "***"


def test_json_parses_and_section_filter(cfg, capsys):
    cfg("ladder:\n  max_spend: 3\n")
    data = _rows(capsys, "ladder")
    assert set(data["sections"]) == {"ladder"}
    assert cmd_config(["show", "nonsense"]) == 1
