"""The session strategy (mullet/core/strategy.py) and `mullet strategy` (mullet/cli/strategy_cli.py)."""

from __future__ import annotations

import json
import stat

import pytest

from mullet.cli import main as cli_main
from mullet.cli.strategy_cli import cmd_strategy
from mullet.core import paths, strategy
from mullet.core.strategy import Strategy

NOW = 1_800_000_000.0


@pytest.fixture
def config(tmp_path, monkeypatch):
    """Write the user's config.yaml in a throwaway config directory."""
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "config"))
    (tmp_path / "config").mkdir()

    def write(text: str) -> None:
        (tmp_path / "config" / "config.yaml").write_text(text)

    return write


class TestStore:
    def test_missing_is_none(self):
        assert strategy.load(NOW) is None
        assert strategy.read() is None

    def test_save_then_load(self):
        saved = Strategy(5.0, "ladder", NOW, "refactor day")
        path = strategy.save(saved)
        assert path == paths.state_dir() / "strategy.json"
        assert strategy.load(NOW + 60) == saved

    def test_file_is_private(self):
        path = strategy.save(Strategy(None, "off", NOW))
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert [p.name for p in path.parent.iterdir()] == ["strategy.json"]  # no temp file left

    def test_expires_after_the_ttl(self):
        strategy.save(Strategy(1.0, "claude-only", NOW))
        assert strategy.load(NOW + 11.9 * 3600) is not None
        assert strategy.load(NOW + 12 * 3600) is None
        assert strategy.load(NOW + 2 * 3600, ttl_hours=1) is None
        assert strategy.read() is not None  # expired, not deleted

    def test_ttl_comes_from_the_config(self, config):
        config("session:\n  strategy_ttl_hours: 2\n")
        strategy.save(Strategy(1.0, "ladder", NOW))
        assert strategy.load(NOW + 3600) is not None
        assert strategy.load(NOW + 3 * 3600) is None

    @pytest.mark.parametrize("text", ["not json", "[]", '{"delegation": "maybe", "set_at": 1}',
                                      '{"delegation": "off", "set_at": "x"}',
                                      '{"delegation": "off", "set_at": 1, "budget_usd": -1}'])
    def test_unreadable_is_none(self, text):
        path = paths.state_dir() / "strategy.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        assert strategy.load(NOW) is None

    def test_clear(self):
        assert strategy.clear() is False
        strategy.save(Strategy(1.0, "ladder", NOW))
        assert strategy.clear() is True
        assert strategy.read() is None

    def test_save_refuses_bad_values(self):
        with pytest.raises(ValueError):
            strategy.save(Strategy(1.0, "sometimes", NOW))
        with pytest.raises(ValueError):
            strategy.save(Strategy(-1.0, "ladder", NOW))

    def test_delegation_off_only_when_fresh_and_off(self):
        assert strategy.delegation_off(NOW) is False
        strategy.save(Strategy(None, "off", NOW))
        assert strategy.delegation_off(NOW) is True
        assert strategy.delegation_off(NOW + 13 * 3600) is False
        strategy.save(Strategy(None, "ladder", NOW))
        assert strategy.delegation_off(NOW) is False

    def test_delegation_off_fails_open(self, monkeypatch):
        monkeypatch.setattr(strategy, "load", lambda now=None: (_ for _ in ()).throw(RuntimeError("boom")))
        assert strategy.delegation_off(NOW) is False


class TestSettings:
    def test_defaults(self):
        settings = strategy.load_settings()
        assert (settings.ask_at_start, settings.strategy_ttl_hours, settings.problems) == (True, 12.0, [])

    def test_read_from_the_config(self, config):
        config("session:\n  ask_at_start: false\n  strategy_ttl_hours: 4\n")
        settings = strategy.load_settings()
        assert (settings.ask_at_start, settings.strategy_ttl_hours, settings.problems) == (False, 4.0, [])

    @pytest.mark.parametrize("raw, problem", [
        ({"ask_at_start": "no"}, "ask_at_start"),
        ({"strategy_ttl_hours": 0}, "strategy_ttl_hours"),
        ({"strategy_ttl_hours": True}, "strategy_ttl_hours"),
        ({"strategy_ttl_hours": 10_000}, "strategy_ttl_hours"),
        ({"colour": "red"}, "unknown setting"),
        ("yes", "expected a mapping"),
    ])
    def test_bad_values_are_reported_and_ignored(self, raw, problem):
        settings = strategy.parse_settings(raw)
        assert (settings.ask_at_start, settings.strategy_ttl_hours) == (True, 12.0)
        assert any(problem in p for p in settings.problems)

    def test_unreadable_config_keeps_the_defaults(self, config):
        config("session: [unclosed\n")
        settings = strategy.load_settings()
        assert settings.ask_at_start is True and settings.problems


class TestCli:
    def _run(self, argv, now=NOW):
        out, err = [], []
        code = cmd_strategy(argv, now=now, out=out.append, err=err.append)
        return code, "\n".join(out), "\n".join(err)

    def test_show_with_none_set(self):
        code, out, _ = self._run([])
        assert code == 0 and "No session strategy is set" in out
        assert self._run(["show"])[1] == out

    def test_set_then_show(self):
        code, out, _ = self._run(["set", "--budget", "3.5", "--delegation", "claude-only", "--note", "quiet"])
        assert code == 0 and "budget $3.50, delegation claude-only" in out
        assert strategy.load(NOW) == Strategy(3.5, "claude-only", NOW, "quiet")
        code, out, _ = self._run(["show"], now=NOW + 3600)
        assert "budget $3.50, delegation claude-only" in out and "note: quiet" in out
        assert "Set at" in out and "expires at" in out

    def test_show_after_expiry(self):
        self._run(["set", "--delegation", "off"])
        _, out, _ = self._run(["show"], now=NOW + 13 * 3600)
        assert "expired" in out

    def test_set_keeps_unnamed_fields_of_a_fresh_strategy(self):
        self._run(["set", "--budget", "4", "--delegation", "off"])
        self._run(["set", "--budget", "1"], now=NOW + 60)
        assert strategy.load(NOW + 60) == Strategy(1.0, "off", NOW + 60, "")

    def test_set_without_flags_prints_usage(self):
        code, _, err = self._run(["set"])
        assert code == 2 and "usage: mullet strategy set" in err
        assert strategy.read() is None

    @pytest.mark.parametrize("amount", ["-1", "nan", "inf", "lots"])
    def test_budget_must_be_zero_or_more(self, amount):
        with pytest.raises(SystemExit):
            cmd_strategy(["set", "--budget", amount])

    def test_bad_delegation_is_refused(self):
        with pytest.raises(SystemExit):
            cmd_strategy(["set", "--delegation", "sometimes"])

    def test_clear(self):
        self._run(["set", "--delegation", "ladder"])
        assert self._run(["clear"])[1] == "Session strategy cleared."
        assert self._run(["clear"])[1] == "No session strategy was set."

    def test_wired_into_mullet(self, monkeypatch, capsys):
        assert "strategy" in cli_main.SUBCOMMANDS
        monkeypatch.setattr("sys.argv", ["mullet", "strategy", "set", "--budget", "2", "--delegation", "ladder"])
        with pytest.raises(SystemExit) as exc:
            cli_main._main_inner()
        assert exc.value.code == 0
        assert "Session strategy set" in capsys.readouterr().out
        assert json.loads((paths.state_dir() / "strategy.json").read_text())["budget_usd"] == 2.0


class TestConfigShowAndValidate:
    def test_config_show_has_a_session_section_with_defaults(self, config):
        from mullet.cli.config_show import SECTIONS, collect

        config("session:\n  ask_at_start: false\n")
        rows = {r["setting"]: r for r in collect()["sections"]["session"]}
        assert "session" in SECTIONS
        assert rows["session.ask_at_start"]["value"] is False
        assert rows["session.ask_at_start"]["default"] is True
        assert rows["session.strategy_ttl_hours"]["value"] == 12.0
        assert rows["session.strategy_ttl_hours"]["source"] == "default"

    def test_config_show_flags_unknown_session_keys(self, config):
        from mullet.cli.config_show import collect

        config("session:\n  colour: red\n")
        assert "session.colour" in collect()["unrecognised"]

    def test_validate_accepts_session(self):
        from mullet.config.validate import validate_config

        result = validate_config({"session": {"ask_at_start": False, "strategy_ttl_hours": 6}})
        assert result.valid and result.issues == []

    @pytest.mark.parametrize("raw", [{"ask_at_start": "no"}, {"strategy_ttl_hours": -1}, "on"])
    def test_validate_rejects_bad_shapes(self, raw):
        from mullet.config.validate import validate_config

        assert not validate_config({"session": raw}).valid

    def test_validate_warns_about_unknown_session_keys(self):
        from mullet.config.validate import validate_config

        result = validate_config({"session": {"colour": "red"}})
        assert result.valid and [i.path for i in result.issues] == ["session.colour"]
