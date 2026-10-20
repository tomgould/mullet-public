"""Regression tests for run_pool_prune hardening."""
import sqlite3
from pathlib import Path


from mullet.cli.pool_audit import _force_rmtree, _purge_pool_sidecars, run_pool_prune


def _make_pool(pools_dir: Path, name: str, with_yaml: bool = False) -> Path:
    p = pools_dir / name
    p.mkdir(parents=True)
    (p / "index").mkdir()
    (p / "documents").mkdir()
    if with_yaml:
        (p / "config.yaml").write_text("description: x\nmachine_name: " + name.lower().replace(' ', '_') + "\n")
    return p


def test_force_rmtree_deletes_dir(tmp_path):
    d = tmp_path / "pool"
    d.mkdir()
    (d / "f.txt").write_text("x")
    ok, err = _force_rmtree(d)
    assert ok
    assert err is None
    assert not d.exists()


def test_force_rmtree_removes_broken_symlink_before_delete(tmp_path):
    d = tmp_path / "pool"
    d.mkdir()
    (d / "documents").symlink_to(tmp_path / "nonexistent")
    ok, err = _force_rmtree(d)
    assert ok, err
    assert not d.exists()


def test_force_rmtree_missing_path_ok(tmp_path):
    ok, err = _force_rmtree(tmp_path / "nope")
    assert ok
    assert err is None


def test_purge_sidecars_removes_db_rows(tmp_path):
    home = tmp_path / ".mullet"
    home.mkdir()
    db = home / "mullet.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE projects (id TEXT PRIMARY KEY, pool_name TEXT)")
    conn.execute("INSERT INTO projects VALUES ('1', 'Mullet - Dev')")
    conn.execute("INSERT INTO projects VALUES ('2', 'kept')")
    conn.commit()
    conn.close()

    rows, warnings = _purge_pool_sidecars("Mullet - Dev", "mullet___dev", home)
    assert rows == 1
    assert warnings == []

    conn = sqlite3.connect(str(db))
    remaining = [r[0] for r in conn.execute("SELECT pool_name FROM projects").fetchall()]
    conn.close()
    assert remaining == ["kept"]


def test_purge_sidecars_missing_db_silent(tmp_path):
    home = tmp_path / ".mullet"
    home.mkdir()
    rows, warnings = _purge_pool_sidecars("pool", "pool", home)
    assert rows == 0
    assert warnings == []


def test_purge_sidecars_removes_analytics_files(tmp_path):
    home = tmp_path / ".mullet"
    (home / "analytics").mkdir(parents=True)
    (home / "analytics" / "pool.json").write_text("{}")
    (home / "state").mkdir()
    (home / "state" / "pool.json").write_text("{}")

    rows, warnings = _purge_pool_sidecars("pool", "pool", home)
    assert warnings == []
    assert not (home / "analytics" / "pool.json").exists()
    assert not (home / "state" / "pool.json").exists()


def test_run_pool_prune_deletes_pool_and_sidecars(tmp_path, monkeypatch, capsys):
    pools = tmp_path / "pools"
    pools.mkdir()
    bad = _make_pool(pools, "bad_pool", with_yaml=False)
    kept = _make_pool(pools, "good_pool", with_yaml=True)

    home = tmp_path / ".mullet"
    home.mkdir()
    db = home / "mullet.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE projects (id TEXT PRIMARY KEY, pool_name TEXT)")
    conn.execute("INSERT INTO projects VALUES ('1', 'bad_pool')")
    conn.commit()
    conn.close()

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    run_pool_prune(pools, yes=True)

    assert not bad.exists()
    assert kept.exists()

    conn = sqlite3.connect(str(db))
    remaining = conn.execute("SELECT count(*) FROM projects").fetchone()[0]
    conn.close()
    assert remaining == 0

    out = capsys.readouterr().out
    assert "deleted bad_pool" in out
    assert "1 db row" in out


def test_run_pool_prune_dry_run_no_op(tmp_path, capsys):
    pools = tmp_path / "pools"
    pools.mkdir()
    bad = _make_pool(pools, "bad", with_yaml=False)

    run_pool_prune(pools, yes=False)
    assert bad.exists()
    out = capsys.readouterr().out
    assert "Dry run" in out


def test_run_pool_prune_no_candidates(tmp_path, capsys):
    pools = tmp_path / "pools"
    pools.mkdir()
    _make_pool(pools, "good", with_yaml=True)
    run_pool_prune(pools, yes=True)
    out = capsys.readouterr().out
    assert "Nothing to prune" in out
