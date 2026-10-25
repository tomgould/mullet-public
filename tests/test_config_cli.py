"""Tests for mullet.cli.config_cli — parse_value and config get/set/list."""
import json
from pathlib import Path
from unittest.mock import patch

from mullet.cli.config_cli import _parse_value, cmd_config


class TestParseValue:
    def test_true_string(self):
        assert _parse_value("true") is True
        assert _parse_value("True") is True
        assert _parse_value("TRUE") is True

    def test_false_string(self):
        assert _parse_value("false") is False
        assert _parse_value("False") is False

    def test_integer(self):
        result = _parse_value("42")
        assert result == 42
        assert isinstance(result, int)

    def test_float(self):
        result = _parse_value("3.14")
        assert abs(result - 3.14) < 0.001
        assert isinstance(result, float)

    def test_plain_string(self):
        assert _parse_value("hello") == "hello"

    def test_zero(self):
        assert _parse_value("0") == 0
        assert isinstance(_parse_value("0"), int)

    def test_negative_int(self):
        assert _parse_value("-5") == -5


def _patch_config(tmp_path):
    config_file = tmp_path / "config.json"
    return (
        patch("mullet.cli.config_cli._CONFIG_PATH", config_file),
        patch("mullet.cli.config_cli._CONFIG_DIR", tmp_path),
    )


class TestCmdConfigGet:
    def test_get_existing_key(self, tmp_path, capsys):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"theme": "light"}))
        p1, p2 = _patch_config(tmp_path)
        with p1, p2:
            rc = cmd_config(["get", "theme"])
        assert rc == 0
        assert "light" in capsys.readouterr().out

    def test_get_missing_key_returns_1(self, tmp_path, capsys):
        p1, p2 = _patch_config(tmp_path)
        with p1, p2:
            rc = cmd_config(["get", "nonexistent_key"])
        assert rc == 1

    def test_get_requires_argument(self, tmp_path, capsys):
        p1, p2 = _patch_config(tmp_path)
        with p1, p2:
            rc = cmd_config(["get"])
        assert rc == 1


class TestCmdConfigSet:
    def test_set_writes_value(self, tmp_path, capsys):
        p1, p2 = _patch_config(tmp_path)
        with p1, p2:
            rc = cmd_config(["set", "theme", "light"])
        assert rc == 0
        config_file = tmp_path / "config.json"
        data = json.loads(config_file.read_text())
        assert data["theme"] == "light"

    def test_set_bool_value(self, tmp_path):
        p1, p2 = _patch_config(tmp_path)
        with p1, p2:
            cmd_config(["set", "telemetry", "false"])
        data = json.loads((tmp_path / "config.json").read_text())
        assert data["telemetry"] is False

    def test_set_requires_key_and_value(self, tmp_path, capsys):
        p1, p2 = _patch_config(tmp_path)
        with p1, p2:
            rc = cmd_config(["set", "only_key"])
        assert rc == 1


class TestCmdConfigList:
    def test_list_shows_all_keys(self, tmp_path, capsys):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"theme": "dark", "telemetry": True}))
        p1, p2 = _patch_config(tmp_path)
        with p1, p2:
            rc = cmd_config(["list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "theme" in out
        assert "telemetry" in out

    def test_list_with_empty_config(self, tmp_path, capsys):
        p1, p2 = _patch_config(tmp_path)
        with p1, p2:
            rc = cmd_config(["list"])
        assert rc == 0


class TestCmdConfigReset:
    def test_reset_returns_0(self, tmp_path, capsys):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"theme": "weird"}))
        p1, p2 = _patch_config(tmp_path)
        with p1, p2:
            rc = cmd_config(["reset"])
        assert rc == 0

    def test_no_args_prints_usage(self, tmp_path, capsys):
        p1, p2 = _patch_config(tmp_path)
        with p1, p2:
            rc = cmd_config([])
        assert rc == 0


# ---- comprehensive coverage for remaining lines -----------------------------

from contextlib import contextmanager
from unittest.mock import MagicMock

from mullet.cli.config_cli import _cmd_budget, _find_mullet_dir, _write_budget_yaml
from mullet.core.providers.spend_caps import BudgetConfig, SpendCap


class TestLoadEdgeCases:
    def test_load_invalid_json_returns_defaults(self, tmp_path, capsys):
        config_file = tmp_path / "config.json"
        config_file.write_text("not valid json")
        p1, p2 = _patch_config(tmp_path)
        with p1, p2:
            rc = cmd_config(["list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "telemetry" in out


class TestCmdConfigUnknownAction:
    def test_unknown_action_prints_error(self, tmp_path, capsys):
        p1, p2 = _patch_config(tmp_path)
        with p1, p2:
            rc = cmd_config(["unknown_action"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Unknown action" in err


class TestCmdConfigEdit:
    def test_edit_no_editor_found(self, tmp_path, capsys):
        p1, p2 = _patch_config(tmp_path)
        with p1, p2, patch.dict("os.environ", {"EDITOR": ""}), \
             patch("mullet.cli.config_cli.subprocess.run", return_value=MagicMock(returncode=1)):
            rc = cmd_config(["edit"])
        assert rc == 1
        assert "No editor found" in capsys.readouterr().err

    def test_edit_success(self, tmp_path, capsys):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"theme": "dark"}))
        edit_path = tmp_path / "edit.json"
        p1, p2 = _patch_config(tmp_path)

        def mock_run(cmd, **kwargs):
            if cmd[0] == "which":
                return MagicMock(returncode=0)
            if cmd[0] == "nano":
                edit_path.write_text('{"theme": "light"}', encoding="utf-8")
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        mock_tf = MagicMock()
        mock_tf.name = str(edit_path)

        @contextmanager
        def mock_named_temp(*args, **kwargs):
            yield mock_tf

        with p1, p2, patch.dict("os.environ", {"EDITOR": "nano"}), \
             patch("mullet.cli.config_cli.subprocess.run", side_effect=mock_run), \
             patch("mullet.cli.config_cli.tempfile.NamedTemporaryFile", mock_named_temp):
            rc = cmd_config(["edit"])
        assert rc == 0
        assert "Config updated" in capsys.readouterr().out
        data = json.loads(config_file.read_text())
        assert data["theme"] == "light"

    def test_edit_success_with_fallback_editor(self, tmp_path, capsys):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"theme": "dark"}))
        edit_path = tmp_path / "edit.json"
        p1, p2 = _patch_config(tmp_path)

        def mock_run(cmd, **kwargs):
            if cmd[0] == "which":
                return MagicMock(returncode=0)
            if cmd[0] == "vim":
                edit_path.write_text('{"theme": "vimmed"}', encoding="utf-8")
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        mock_tf = MagicMock()
        mock_tf.name = str(edit_path)

        @contextmanager
        def mock_named_temp(*args, **kwargs):
            yield mock_tf

        with p1, p2, patch.dict("os.environ", {"EDITOR": ""}), \
             patch("mullet.cli.config_cli.subprocess.run", side_effect=mock_run), \
             patch("mullet.cli.config_cli.tempfile.NamedTemporaryFile", mock_named_temp):
            rc = cmd_config(["edit"])
        assert rc == 0
        data = json.loads(config_file.read_text())
        assert data["theme"] == "vimmed"

    def test_edit_editor_exits_with_error(self, tmp_path, capsys):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"theme": "dark"}))
        edit_path = tmp_path / "edit.json"
        p1, p2 = _patch_config(tmp_path)

        def mock_run(cmd, **kwargs):
            if cmd[0] == "which":
                return MagicMock(returncode=0)
            if cmd[0] == "nano":
                return MagicMock(returncode=1)
            return MagicMock(returncode=0)

        mock_tf = MagicMock()
        mock_tf.name = str(edit_path)

        @contextmanager
        def mock_named_temp(*args, **kwargs):
            yield mock_tf

        with p1, p2, patch.dict("os.environ", {"EDITOR": "nano"}), \
             patch("mullet.cli.config_cli.subprocess.run", side_effect=mock_run), \
             patch("mullet.cli.config_cli.tempfile.NamedTemporaryFile", mock_named_temp):
            rc = cmd_config(["edit"])
        assert rc == 1
        assert "exited with code 1" in capsys.readouterr().err

    def test_edit_invalid_json(self, tmp_path, capsys):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"theme": "dark"}))
        edit_path = tmp_path / "edit.json"
        edit_path.write_text("not json", encoding="utf-8")
        p1, p2 = _patch_config(tmp_path)

        def mock_run(cmd, **kwargs):
            if cmd[0] == "which":
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        mock_tf = MagicMock()
        mock_tf.name = str(edit_path)

        @contextmanager
        def mock_named_temp(*args, **kwargs):
            yield mock_tf

        with p1, p2, patch.dict("os.environ", {"EDITOR": "nano"}), \
             patch("mullet.cli.config_cli.subprocess.run", side_effect=mock_run), \
             patch("mullet.cli.config_cli.tempfile.NamedTemporaryFile", mock_named_temp):
            rc = cmd_config(["edit"])
        assert rc == 1
        assert "Invalid JSON" in capsys.readouterr().err

    def test_edit_not_a_dict(self, tmp_path, capsys):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"theme": "dark"}))
        edit_path = tmp_path / "edit.json"
        edit_path.write_text("[1, 2, 3]", encoding="utf-8")
        p1, p2 = _patch_config(tmp_path)

        def mock_run(cmd, **kwargs):
            if cmd[0] == "which":
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        mock_tf = MagicMock()
        mock_tf.name = str(edit_path)

        @contextmanager
        def mock_named_temp(*args, **kwargs):
            yield mock_tf

        with p1, p2, patch.dict("os.environ", {"EDITOR": "nano"}), \
             patch("mullet.cli.config_cli.subprocess.run", side_effect=mock_run), \
             patch("mullet.cli.config_cli.tempfile.NamedTemporaryFile", mock_named_temp):
            rc = cmd_config(["edit"])
        assert rc == 1
        assert "must be a JSON object" in capsys.readouterr().err


class TestFindMulletDir:
    def test_finds_existing_mullet_dir(self, tmp_path):
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir()
        with patch("mullet.cli.config_cli.Path.cwd", return_value=tmp_path):
            assert _find_mullet_dir() == mullet_dir

    def test_returns_cwd_path_when_not_found(self, tmp_path):
        real_is_dir = Path.is_dir

        def fake_is_dir(self):
            if str(self).endswith(".mullet"):
                return False
            return real_is_dir(self)

        with patch("mullet.cli.config_cli.Path.cwd", return_value=tmp_path), \
             patch("mullet.cli.config_cli.Path.is_dir", fake_is_dir):
            result = _find_mullet_dir()
        assert result == tmp_path / ".mullet"


class TestWriteBudgetYaml:
    def test_write_with_yaml_module(self, tmp_path):
        path = tmp_path / "budget.yaml"
        data = {"openrouter": {"cap_usd": 10.0}}
        _write_budget_yaml(path, data)
        text = path.read_text(encoding="utf-8")
        assert "openrouter" in text
        assert "cap_usd" in text

    def test_write_without_yaml_module(self, tmp_path):
        path = tmp_path / "budget.yaml"
        data = {"openrouter": {"cap_usd": 10.0}, "monthly_total_cap_usd": 50.0}
        with patch.dict("sys.modules", {"yaml": None}):
            _write_budget_yaml(path, data)
        text = path.read_text(encoding="utf-8")
        assert "openrouter:" in text
        assert "cap_usd: 10.0" in text
        assert "monthly_total_cap_usd: 50.0" in text


class TestCmdBudget:
    def test_budget_help(self, capsys):
        rc = _cmd_budget(["help"])
        assert rc == 0
        assert "Usage:" in capsys.readouterr().err

    def test_budget_show_no_caps(self, capsys):
        with patch("mullet.cli.config_cli.load_budget_config", return_value=BudgetConfig(caps={})):
            rc = _cmd_budget(["show"])
        assert rc == 0
        assert "No spend caps configured" in capsys.readouterr().out

    def test_budget_show_with_caps(self, capsys):
        cap = SpendCap(provider="openrouter", cap_usd=10.0)
        config = BudgetConfig(caps={"openrouter": cap}, monthly_total_cap_usd=50.0)
        with patch("mullet.cli.config_cli.load_budget_config", return_value=config):
            rc = _cmd_budget(["show"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "openrouter" in out

    def test_budget_show_without_rich(self, capsys):
        cap = SpendCap(provider="openrouter", cap_usd=10.0)
        config = BudgetConfig(caps={"openrouter": cap})
        real_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "rich" or name.startswith("rich."):
                raise ImportError("no rich")
            return real_import(name, *args, **kwargs)

        with patch("mullet.cli.config_cli.load_budget_config", return_value=config), \
             patch("builtins.__import__", side_effect=mock_import):
            rc = _cmd_budget(["show"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "openrouter" in out
        assert "Cap (USD)" in out

    def test_budget_show_monthly_total_no_rows(self, capsys):
        config = BudgetConfig(caps={}, monthly_total_cap_usd=25.0)
        real_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "rich" or name.startswith("rich."):
                raise ImportError("no rich")
            return real_import(name, *args, **kwargs)

        with patch("mullet.cli.config_cli.load_budget_config", return_value=config), \
             patch("builtins.__import__", side_effect=mock_import):
            rc = _cmd_budget(["show"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "monthly total" in out.lower()

    def test_budget_unknown_action(self, capsys):
        with patch("mullet.cli.config_cli.load_budget_config", return_value=BudgetConfig(caps={})):
            rc = _cmd_budget(["unknown"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Unknown budget action" in err

    def test_budget_set_missing_args(self, capsys):
        rc = _cmd_budget(["set", "openrouter"])
        assert rc == 1
        assert "Usage:" in capsys.readouterr().err

    def test_budget_set_unknown_provider(self, capsys):
        rc = _cmd_budget(["set", "unknown", "10"])
        assert rc == 1
        assert "Unknown provider" in capsys.readouterr().err

    def test_budget_set_invalid_cap(self, capsys):
        rc = _cmd_budget(["set", "openrouter", "not_a_number"])
        assert rc == 1
        assert "Invalid cap value" in capsys.readouterr().err

    def test_budget_set_valid_cap(self, tmp_path, capsys):
        budget_dir = tmp_path / ".mullet"
        budget_dir.mkdir()
        with patch("mullet.cli.config_cli._find_budget_file", return_value=None), \
             patch("mullet.cli.config_cli._find_mullet_dir", return_value=budget_dir), \
             patch("mullet.cli.config_cli._write_budget_yaml") as mock_write:
            rc = _cmd_budget(["set", "openrouter", "10.5"])
        assert rc == 0
        assert "Set openrouter cap" in capsys.readouterr().out
        mock_write.assert_called_once()

    def test_budget_set_existing_file_with_yaml(self, tmp_path, capsys):
        budget_dir = tmp_path / ".mullet"
        budget_dir.mkdir()
        budget_path = budget_dir / "budget.yaml"
        budget_path.write_text("kimi:\n  cap_usd: 5.0\n")
        with patch("mullet.cli.config_cli._find_budget_file", return_value=budget_path), \
             patch("mullet.cli.config_cli._write_budget_yaml") as mock_write:
            rc = _cmd_budget(["set", "openrouter", "10.5"])
        assert rc == 0
        mock_write.assert_called_once()
        call_data = mock_write.call_args[0][1]
        assert "openrouter" in call_data

    def test_budget_set_existing_file_without_yaml(self, tmp_path, capsys):
        budget_dir = tmp_path / ".mullet"
        budget_dir.mkdir()
        budget_path = budget_dir / "budget.yaml"
        budget_path.write_text("kimi:\n  cap_usd: 5.0\n")
        with patch("mullet.cli.config_cli._find_budget_file", return_value=budget_path), \
             patch.dict("sys.modules", {"yaml": None}), \
             patch("mullet.cli.config_cli._write_budget_yaml") as mock_write:
            rc = _cmd_budget(["set", "openrouter", "10.5"])
        assert rc == 0
        mock_write.assert_called_once()

    def test_budget_monthly_missing_args(self, capsys):
        rc = _cmd_budget(["monthly"])
        assert rc == 1
        assert "Usage:" in capsys.readouterr().err

    def test_budget_monthly_invalid_total(self, capsys):
        rc = _cmd_budget(["monthly", "bad"])
        assert rc == 1
        assert "Invalid total value" in capsys.readouterr().err

    def test_budget_monthly_valid_total(self, tmp_path, capsys):
        budget_dir = tmp_path / ".mullet"
        budget_dir.mkdir()
        with patch("mullet.cli.config_cli._find_budget_file", return_value=None), \
             patch("mullet.cli.config_cli._find_mullet_dir", return_value=budget_dir), \
             patch("mullet.cli.config_cli._write_budget_yaml") as mock_write:
            rc = _cmd_budget(["monthly", "100"])
        assert rc == 0
        assert "Set monthly total cap" in capsys.readouterr().out
        mock_write.assert_called_once()

    def test_budget_monthly_existing_file_with_yaml(self, tmp_path, capsys):
        budget_dir = tmp_path / ".mullet"
        budget_dir.mkdir()
        budget_path = budget_dir / "budget.yaml"
        budget_path.write_text("kimi:\n  cap_usd: 5.0\n")
        with patch("mullet.cli.config_cli._find_budget_file", return_value=budget_path), \
             patch("mullet.cli.config_cli._write_budget_yaml") as mock_write:
            rc = _cmd_budget(["monthly", "100"])
        assert rc == 0
        mock_write.assert_called_once()

    def test_budget_monthly_existing_file_without_yaml(self, tmp_path, capsys):
        budget_dir = tmp_path / ".mullet"
        budget_dir.mkdir()
        budget_path = budget_dir / "budget.yaml"
        budget_path.write_text("kimi:\n  cap_usd: 5.0\n")
        with patch("mullet.cli.config_cli._find_budget_file", return_value=budget_path), \
             patch.dict("sys.modules", {"yaml": None}), \
             patch("mullet.cli.config_cli._write_budget_yaml") as mock_write:
            rc = _cmd_budget(["monthly", "100"])
        assert rc == 0
        mock_write.assert_called_once()

    def test_budget_reset_existing_file(self, tmp_path, capsys):
        budget_path = tmp_path / "budget.yaml"
        budget_path.write_text("foo: bar")
        with patch("mullet.cli.config_cli._find_budget_file", return_value=budget_path):
            rc = _cmd_budget(["reset"])
        assert rc == 0
        assert "Budget configuration reset" in capsys.readouterr().out
        assert not budget_path.exists()

    def test_budget_reset_no_file(self, capsys):
        with patch("mullet.cli.config_cli._find_budget_file", return_value=None):
            rc = _cmd_budget(["reset"])
        assert rc == 0
        assert "No budget configuration to reset" in capsys.readouterr().out

    def test_budget_default_to_show(self, capsys):
        with patch("mullet.cli.config_cli.load_budget_config", return_value=BudgetConfig(caps={})):
            rc = _cmd_budget([])
        assert rc == 0
        assert "No spend caps configured" in capsys.readouterr().out

    def test_config_budget_dispatch(self, tmp_path, capsys):
        p1, p2 = _patch_config(tmp_path)
        with p1, p2, patch("mullet.cli.config_cli.load_budget_config", return_value=BudgetConfig(caps={})):
            rc = cmd_config(["budget"])
        assert rc == 0
        assert "No spend caps configured" in capsys.readouterr().out
