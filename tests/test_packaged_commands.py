"""`mullet reindex` and `mullet evaluate` run Mullet's own modules safely."""
import pytest

import mullet.cli.main as cli_main


def test_a_module_planted_in_the_working_directory_is_not_run(tmp_path, monkeypatch):
    marker = tmp_path / "planted-ran"
    (tmp_path / "argparse.py").write_text(
        f"open({str(marker)!r}, 'w').close()\nraise SystemExit(99)\n"
    )
    monkeypatch.chdir(tmp_path)
    # json.tool imports argparse, as the reindexer and evaluator do.
    assert cli_main._run_packaged("json.tool", ["--help"]) == 0
    assert not marker.exists()


def test_arguments_reach_the_module(tmp_path, monkeypatch):
    data = tmp_path / "in.json"
    data.write_text('{"b": 1, "a": 2}')
    out = tmp_path / "out.json"
    monkeypatch.chdir(tmp_path)
    assert cli_main._run_packaged("json.tool", ["--sort-keys", str(data), str(out)]) == 0
    assert out.read_text().index('"a"') < out.read_text().index('"b"')


@pytest.fixture
def captured(monkeypatch):
    calls = []
    monkeypatch.setattr(cli_main, "_run_packaged", lambda module, args: calls.append((module, args)) or 0)
    return calls


def _run(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["mullet", *argv])
    with pytest.raises(SystemExit) as exc:
        cli_main.main()
    return exc.value.code


@pytest.mark.parametrize("argv, expected", [
    (["reindex", "mypool"], ["--pool", "mypool"]),
    (["reindex", "mypool", "--graph"], ["--pool", "mypool", "--graph"]),
    (["reindex", "--pool", "mypool", "--workers", "4"], ["--pool", "mypool", "--workers", "4"]),
    (["reindex", "--list"], ["--list"]),
    (["reindex", "--all", "--sync-sources"], ["--all", "--sync-sources"]),
    (["reindex", "mypool", "--no-wait"], ["--pool", "mypool", "--no-wait"]),
    (["reindex", "--all", "--processes", "2", "--max-graph-chunks", "50"],
     ["--all", "--processes", "2", "--max-graph-chunks", "50"]),
])
def test_reindex_passes_its_flags_through(monkeypatch, captured, argv, expected):
    assert _run(monkeypatch, *argv) == 0
    assert captured == [("mullet.knowledge.reindex", expected)]


@pytest.mark.parametrize("argv, expected", [
    (["evaluate", "trag"], ["--client", "trag"]),
    (["evaluate", "trag", "--top-k", "5", "--json"], ["--client", "trag", "--top-k", "5", "--json"]),
    (["evaluate", "--list"], ["--list"]),
])
def test_evaluate_passes_its_flags_through(monkeypatch, captured, argv, expected):
    assert _run(monkeypatch, *argv) == 0
    assert captured == [("mullet.knowledge.evaluate", expected)]
