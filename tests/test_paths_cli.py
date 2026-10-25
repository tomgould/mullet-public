"""`mullet paths`: Mullet's directories, for scripts and for people."""
import pytest

from mullet.cli.main import _cmd_paths, main
from mullet.core import paths


def test_lists_every_directory(capsys):
    assert _cmd_paths([]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert dict(line.split("=", 1) for line in lines) == {
        "config": str(paths.config_dir()),
        "state": str(paths.state_dir()),
        "data": str(paths.data_dir()),
        "cache": str(paths.cache_dir()),
        "pools": str(paths.pools_dir()),
    }


def test_prints_one_bare_path_for_scripts(capsys):
    assert _cmd_paths(["pools"]) == 0
    assert capsys.readouterr().out == f"{paths.pools_dir()}\n"


def test_follows_xdg_variables(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    _cmd_paths(["state"])
    assert capsys.readouterr().out.strip() == str(tmp_path / "state" / "mullet")


def test_unknown_name_is_a_usage_error(capsys):
    assert _cmd_paths(["nope"]) == 2
    assert "usage: mullet paths" in capsys.readouterr().err


def test_too_many_names_is_a_usage_error(capsys):
    assert _cmd_paths(["state", "data"]) == 2


def test_mullet_paths_dispatches(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["mullet", "paths", "data"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert capsys.readouterr().out == f"{paths.data_dir()}\n"
