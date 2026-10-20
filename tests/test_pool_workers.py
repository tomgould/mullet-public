"""Tests for --workers flag on pool batch commands."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mullet.cli.pool_audit import PoolHealth, run_pool_health


@pytest.fixture
def mock_sys_argv():
    """Fixture to temporarily replace sys.argv for CLI argument parsing."""
    original_argv = sys.argv
    sys.argv = ["mullet"]
    yield sys.argv
    sys.argv = original_argv


def test_workers_flag_accepted(mock_sys_argv, capsys):
    """CLI accepts --workers 4 on 'pool health --all' without error."""
    from mullet.cli import main

    received = {}

    def fake_run_pool_health(pools_dir, pool_name=None, workers=1):
        received["workers"] = workers
        return 0

    original_argv = sys.argv
    sys.argv = ["mullet", "pool", "health", "--all", "--workers", "4"]
    try:
        with patch("mullet.cli.pool_audit.run_pool_health", side_effect=fake_run_pool_health):
            main.main()
    except SystemExit as e:
        assert e.code == 0
    finally:
        sys.argv = original_argv

    assert received.get("workers") == 4


def test_workers_gt1_uses_threadpool(tmp_path: Path):
    """When workers > 1, run_pool_health uses ThreadPoolExecutor."""
    (tmp_path / "pool_a").mkdir()
    (tmp_path / "pool_b").mkdir()

    with patch("concurrent.futures.ThreadPoolExecutor") as mock_tpe, \
         patch("mullet.cli.pool_audit.check_pool_health") as mock_check:

        mock_check.side_effect = lambda d: PoolHealth(
            name=d.name,
            pool_dir=d,
            ok=True,
            chunk_count=0,
            file_count=0,
            last_indexed_human=None,
            issues=[],
        )

        mock_instance = MagicMock()

        def mock_submit(fn, *args, **kwargs):
            result = MagicMock()
            result.result.return_value = fn(*args, **kwargs)
            return result

        mock_instance.submit = mock_submit
        mock_tpe.return_value.__enter__.return_value = mock_instance

        rc = run_pool_health(tmp_path, workers=2)

        mock_tpe.assert_called_once_with(max_workers=2)
        assert rc == 0
