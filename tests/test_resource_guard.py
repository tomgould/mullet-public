"""Tests for ResourceGuard — pre-dispatch resource gate for local LLM."""
from unittest.mock import MagicMock, patch

import psutil
import pytest

from mullet.core.router.resource_guard import ResourceGuard


@pytest.fixture
def guard():
    """Fresh ResourceGuard with mocked ResourceManager."""
    with patch("mullet.core.router.resource_guard.ResourceManager") as MockRM:
        rm = MockRM.return_value
        rm.probe_cpu.return_value = {"total_cores": 8, "available_cores": 7}
        rm.probe_ram.return_value = {"total_gb": 64.0, "available_gb": 32.0, "usable_gb": 30.0}
        rm.probe_gpu.return_value = {"total_mb": 12288, "used_mb": 4096, "free_mb": 8192, "utilization_pct": 30}
        rm._gpu_probed = False
        g = ResourceGuard()
        yield g


@patch("psutil.process_iter", return_value=[])
@patch("os.getloadavg", return_value=(2.0, 2.0, 2.0))
@patch("psutil.cpu_count", return_value=8)
def test_can_dispatch_ok(mock_cpu, mock_load, mock_procs, guard):
    ok, reason = guard.can_dispatch_local()
    assert ok is True
    assert reason == "ok"


@patch("os.getloadavg", return_value=(20.0, 15.0, 10.0))
@patch("psutil.cpu_count", return_value=8)
def test_blocks_high_load(mock_cpu, mock_load, guard):
    ok, reason = guard.can_dispatch_local()
    assert ok is False
    assert "load" in reason.lower()


@patch("os.getloadavg", return_value=(1.0, 1.0, 1.0))
@patch("psutil.cpu_count", return_value=8)
@patch("psutil.virtual_memory")
def test_blocks_low_ram(mock_vmem, mock_cpu, mock_load, guard):
    mock_vmem.return_value.available = int(1.5 * 1024**3)  # 1.5 GB
    ok, reason = guard.can_dispatch_local()
    assert ok is False
    assert "ram" in reason.lower()


@patch("os.getloadavg", return_value=(1.0, 1.0, 1.0))
@patch("psutil.cpu_count", return_value=8)
def test_blocks_low_vram(mock_cpu, mock_load, guard):
    guard.resource_manager.probe_gpu.return_value = {"total_mb": 12288, "used_mb": 11500, "free_mb": 788, "utilization_pct": 50}
    ok, reason = guard.can_dispatch_local()
    assert ok is False
    assert "vram" in reason.lower()


@patch("os.getloadavg", return_value=(1.0, 1.0, 1.0))
@patch("psutil.cpu_count", return_value=8)
def test_blocks_gpu_saturated(mock_cpu, mock_load, guard):
    guard.resource_manager.probe_gpu.return_value = {"total_mb": 12288, "used_mb": 8000, "free_mb": 4288, "utilization_pct": 97}
    ok, reason = guard.can_dispatch_local()
    assert ok is False
    assert "saturated" in reason.lower() or "util" in reason.lower()


@patch("psutil.process_iter", return_value=[])
@patch("os.getloadavg", return_value=(1.0, 1.0, 1.0))
@patch("psutil.cpu_count", return_value=8)
def test_blocks_semaphore_busy(mock_cpu, mock_load, mock_procs, guard):
    guard.semaphore.acquire(blocking=False)
    try:
        ok, reason = guard.can_dispatch_local()
        assert ok is False
        assert "slot" in reason.lower() or "semaphore" in reason.lower()
    finally:
        guard.semaphore.release()


def test_acquire_release_slot(guard):
    assert guard.acquire_local_slot() is True
    assert guard.acquire_local_slot() is False
    guard.release_local_slot()
    assert guard.acquire_local_slot() is True
    guard.release_local_slot()


@patch("psutil.getloadavg", return_value=(3.5, 2.0, 1.0))
def test_get_system_snapshot(mock_load, guard):
    snapshot = guard.get_system_snapshot()
    assert "load_avg" in snapshot
    assert "cpu_cores" in snapshot
    assert "ram_available_gb" in snapshot
    assert "ram_total_gb" in snapshot
    assert "gpu" in snapshot


def test_singleton():
    import mullet.core.router.resource_guard as mod
    mod._guard = None
    from mullet.core.router.resource_guard import get_resource_guard
    g1 = get_resource_guard()
    g2 = get_resource_guard()
    assert g1 is g2
    mod._guard = None


@patch("psutil.getloadavg", return_value=(1.0, 1.0, 1.0))
def test_flush_resources(mock_load, guard):
    result = guard.flush_resources()
    assert "before" in result
    assert "after" in result
    assert "freed_ram_gb" in result


@patch("psutil.process_iter", return_value=[])
@patch("os.getloadavg", return_value=(1.0, 1.0, 1.0))
@patch("psutil.cpu_count", return_value=8)
def test_no_gpu_still_dispatches(mock_cpu, mock_load, mock_procs, guard):
    guard.resource_manager.probe_gpu.return_value = None
    ok, reason = guard.can_dispatch_local()
    assert ok is True
    assert reason == "ok"


@patch("os.getloadavg", return_value=(1.0, 1.0, 1.0))
@patch("psutil.cpu_count", return_value=8)
def test_blocks_when_trag_reindex_running(mock_cpu, mock_load, guard):
    """Lines 68-72: TRAG reindex process found → dispatch blocked."""
    mock_proc = MagicMock()
    mock_proc.info = {"name": "python", "cmdline": ["python", "-m", "trove", "reindex"]}
    with patch("psutil.process_iter", return_value=[mock_proc]):
        ok, reason = guard.can_dispatch_local()
    assert ok is False
    assert "trag" in reason.lower() or "trove" in reason.lower() or "reindex" in reason.lower()


@patch("os.getloadavg", return_value=(1.0, 1.0, 1.0))
@patch("psutil.cpu_count", return_value=8)
def test_process_iter_no_such_process_is_caught(mock_cpu, mock_load, guard):
    """Lines 73-74: psutil.NoSuchProcess during iteration is caught and logged."""
    with patch("psutil.process_iter", side_effect=psutil.NoSuchProcess(pid=999)):
        ok, reason = guard.can_dispatch_local()
    # Should not raise; proceeds to semaphore check and return ok
    assert ok is True


@patch("os.getloadavg", return_value=(1.0, 1.0, 1.0))
@patch("psutil.cpu_count", return_value=8)
def test_process_iter_access_denied_is_caught(mock_cpu, mock_load, guard):
    """Lines 73-74: psutil.AccessDenied during iteration is caught and logged."""
    with patch("psutil.process_iter", side_effect=psutil.AccessDenied(pid=999)):
        ok, reason = guard.can_dispatch_local()
    assert ok is True


def test_flush_resources_terminates_old_children(guard):
    """Lines 123-127: old child process with create_time < now-60 is terminated."""
    import time
    mock_child = MagicMock()
    mock_child.create_time.return_value = time.time() - 120  # 2 min old
    mock_child.pid = 12345
    mock_child.name.return_value = "stuck-worker"
    mock_child.terminate.return_value = None

    mock_parent = MagicMock()
    mock_parent.children.return_value = [mock_child]

    with patch("psutil.Process", return_value=mock_parent):
        result = guard.flush_resources()

    mock_child.terminate.assert_called_once()
    assert "before" in result


def test_flush_resources_child_no_such_process_is_caught(guard):
    """Lines 128-129: NoSuchProcess accessing child is caught per-child."""
    mock_child = MagicMock()
    mock_child.create_time.side_effect = psutil.NoSuchProcess(pid=9)

    mock_parent = MagicMock()
    mock_parent.children.return_value = [mock_child]

    with patch("psutil.Process", return_value=mock_parent):
        result = guard.flush_resources()  # must not raise

    assert "before" in result


def test_flush_resources_parent_no_such_process_is_caught(guard):
    """Lines 130-131: NoSuchProcess fetching the parent process is caught."""
    with patch("psutil.Process", side_effect=psutil.NoSuchProcess(pid=1)):
        result = guard.flush_resources()  # must not raise

    assert "before" in result


def test_flush_resources_torch_exception_is_caught(guard):
    """Lines 145-146: torch.cuda exception during cache clearing is caught."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.empty_cache.side_effect = RuntimeError("CUDA OOM")

    with patch("psutil.Process", side_effect=psutil.NoSuchProcess(pid=1)):
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = guard.flush_resources()

    assert "before" in result  # must not raise
