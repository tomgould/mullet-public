"""Tests for mullet.cli.pool_tune — hardware-adaptive pool settings tuner."""

from unittest.mock import MagicMock, patch

import yaml

from mullet.cli.pool_tune import (
    _add_tuned_comment,
    _load_yaml,
    _recommend_inter_batch_sleep,
    _save_yaml,
)


# --- _recommend_inter_batch_sleep -----------------------------------------


def test_sleep_cpu_only_fallback():
    """free_mb=None → 2.0s (CPU-only, conservative)."""
    assert _recommend_inter_batch_sleep(None) == 2.0


def test_sleep_high_vram():
    assert _recommend_inter_batch_sleep(16000) == 0.0
    assert _recommend_inter_batch_sleep(10000) == 0.0


def test_sleep_mid_vram():
    assert _recommend_inter_batch_sleep(6000) == 0.5
    assert _recommend_inter_batch_sleep(5000) == 0.5


def test_sleep_low_vram():
    assert _recommend_inter_batch_sleep(3000) == 1.0
    assert _recommend_inter_batch_sleep(1024) == 1.0


def test_sleep_boundary_conditions():
    """Exact thresholds match the > (strict greater than) in the rules."""
    assert _recommend_inter_batch_sleep(8192) == 0.5  # not > 8192 → fails high bucket
    assert _recommend_inter_batch_sleep(8193) == 0.0
    assert _recommend_inter_batch_sleep(4096) == 1.0  # not > 4096 → fails mid bucket


# --- _load_yaml / _save_yaml round-trip ------------------------------------


def test_load_yaml_round_trip_with_pyyaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("foo: 1\nbar:\n  baz: 2\n")
    engine, data = _load_yaml(path)
    # engine may be a ruamel YAML() or None depending on install; data must parse
    assert data == {"foo": 1, "bar": {"baz": 2}}


def test_save_yaml_with_pyyaml_fallback(tmp_path):
    path = tmp_path / "out.yaml"
    _save_yaml(path, None, {"foo": "bar", "nested": {"a": 1}})
    assert path.exists()
    loaded = yaml.safe_load(path.read_text())
    assert loaded == {"foo": "bar", "nested": {"a": 1}}


def test_save_yaml_with_ruamel_engine(tmp_path):
    """When engine is provided, it's used to dump (preserves comments)."""
    path = tmp_path / "out.yaml"
    engine = MagicMock()
    _save_yaml(path, engine, {"a": 1})
    engine.dump.assert_called_once()


# --- _add_tuned_comment ----------------------------------------------------


def test_add_tuned_comment_swallows_error():
    """Ruamel comment API varies — failure shouldn't crash."""
    broken_section = MagicMock()
    broken_section.yaml_set_start_comment.side_effect = AttributeError("no such method")
    # Must not raise
    _add_tuned_comment(None, broken_section, "2026-04-22T00:00:00Z")


def test_add_tuned_comment_calls_ruamel_api():
    """When ruamel API is available, it's called with timestamp."""
    section = MagicMock()
    _add_tuned_comment(None, section, "2026-04-22T12:00:00Z")
    section.yaml_set_start_comment.assert_called_once()
    call_args = section.yaml_set_start_comment.call_args
    assert "2026-04-22T12:00:00Z" in call_args.args[0]


# --- run_pool_tune ---------------------------------------------------------


def test_run_pool_tune_skips_non_directories(tmp_path):
    """Files in pools_dir are ignored."""
    (tmp_path / "not_a_pool.txt").write_text("noise")
    from mullet.cli.pool_tune import run_pool_tune
    with patch("mullet.knowledge.engine.resource_manager.ResourceManager") as mock_rm:
        mock_rm.return_value.probe_gpu.return_value = None
        mock_rm.return_value.recommend_embed_batch.return_value = 32
        # Must not crash on a dir with no pool subdirs
        run_pool_tune(tmp_path, dry_run=False)  # dry_run=False so else branch runs


def test_run_pool_tune_skips_pools_without_config(tmp_path):
    """Pool dir without config.yaml or client.yaml is skipped."""
    pool = tmp_path / "empty_pool"
    pool.mkdir()
    from mullet.cli.pool_tune import run_pool_tune
    with patch("mullet.knowledge.engine.resource_manager.ResourceManager") as mock_rm:
        mock_rm.return_value.probe_gpu.return_value = None
        mock_rm.return_value.recommend_embed_batch.return_value = 32
        run_pool_tune(tmp_path, dry_run=True)  # no crash


def test_run_pool_tune_applies_to_pool_with_graph_config(tmp_path):
    """Pool with config.yaml and 'graph' section gets tuned settings."""
    pool = tmp_path / "test_pool"
    pool.mkdir()
    config = pool / "config.yaml"
    config.write_text(yaml.safe_dump({
        "graph": {
            "extract_workers": 8,
            "inter_batch_sleep": 0.0,
            "embed_batch_size": 128,
        },
    }))

    from mullet.cli.pool_tune import run_pool_tune
    with patch("mullet.knowledge.engine.resource_manager.ResourceManager") as mock_rm:
        mock_rm.return_value.probe_gpu.return_value = None  # CPU-only
        mock_rm.return_value.recommend_embed_batch.return_value = 32
        run_pool_tune(tmp_path, dry_run=False)

    updated = yaml.safe_load(config.read_text())
    assert updated["graph"]["extract_workers"] == 1  # always 1
    assert updated["graph"]["inter_batch_sleep"] == 2.0  # CPU-only → 2.0
    assert updated["graph"]["embed_batch_size"] == min(64, 32)


def test_run_pool_tune_pool_filter_skips_non_matching(tmp_path, capsys):
    """Line 96: pool_filter set → pool_dir.name != pool_filter → continue."""
    import yaml
    from unittest.mock import patch
    from mullet.cli.pool_tune import run_pool_tune

    # Create two pools: "pool_a" and "pool_b"
    for name in ("pool_a", "pool_b"):
        p = tmp_path / name
        p.mkdir()
        (p / "config.yaml").write_text(yaml.safe_dump({"graph": {"extract_workers": 8}}))

    with patch("mullet.knowledge.engine.resource_manager.ResourceManager") as mock_rm:
        mock_rm.return_value.probe_gpu.return_value = None
        mock_rm.return_value.recommend_embed_batch.return_value = 32
        run_pool_tune(tmp_path, pool_filter="pool_a", dry_run=True)

    out = capsys.readouterr().out
    assert "pool_a" in out
    assert "pool_b" not in out


def test_run_pool_tune_skips_config_without_graph_section(tmp_path, capsys):
    """Line 106: data does not have 'graph' key → continue."""
    import yaml
    from unittest.mock import patch
    from mullet.cli.pool_tune import run_pool_tune

    p = tmp_path / "no_graph_pool"
    p.mkdir()
    (p / "config.yaml").write_text(yaml.safe_dump({"settings": {"x": 1}}))

    with patch("mullet.knowledge.engine.resource_manager.ResourceManager") as mock_rm:
        mock_rm.return_value.probe_gpu.return_value = None
        mock_rm.return_value.recommend_embed_batch.return_value = 32
        run_pool_tune(tmp_path, dry_run=False)  # dry_run=False so else branch runs

    out = capsys.readouterr().out
    assert "No pools with graph: section found" in out


def test_run_pool_tune_no_pools_with_graph_shows_message(tmp_path, capsys):
    """Lines 145-146: no pools → 'No pools with graph: section found{pool_msg}.'"""
    from unittest.mock import patch
    from mullet.cli.pool_tune import run_pool_tune

    # Create a pool directory with no config.yaml at all
    (tmp_path / "empty_pool").mkdir()

    with patch("mullet.knowledge.engine.resource_manager.ResourceManager") as mock_rm:
        mock_rm.return_value.probe_gpu.return_value = None
        mock_rm.return_value.recommend_embed_batch.return_value = 32
        run_pool_tune(tmp_path, pool_filter="empty_pool", dry_run=False)

    out = capsys.readouterr().out
    assert "No pools with graph: section found" in out
    assert "'empty_pool'" in out
