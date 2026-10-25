"""Tests for mullet.core.router.resource_guard."""

import logging
import threading
from unittest.mock import MagicMock, patch

import psutil
import pytest

from mullet.core.router.resource_guard import ResourceGuard, get_resource_guard


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level singleton before every test."""
    import mullet.core.router.resource_guard as _mod
    _mod._guard = None
    yield
    _mod._guard = None


# --- __init__ ----------------------------------------------------------------

def test_init():
    rg = ResourceGuard()
    assert rg.resource_manager is not None
    assert isinstance(rg.semaphore, threading.BoundedSemaphore)


# --- can_dispatch_local ------------------------------------------------------

@patch("mullet.core.router.resource_guard.psutil.cpu_count", return_value=4)
@patch("mullet.core.router.resource_guard.os.getloadavg", return_value=(1.0, 2.0, 3.0))
@patch("mullet.core.router.resource_guard.psutil.virtual_memory")
@patch("mullet.core.router.resource_guard.psutil.process_iter")
def test_can_dispatch_local_all_good(_mock_iter, mock_vm, _mock_loadavg, _mock_cpu_count):
    mock_vm.return_value.available = 16 * (1024 ** 3)
    _mock_iter.return_value = []

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.resource_manager = MagicMock()
        rg.resource_manager.probe_gpu.return_value = None
        rg.semaphore = threading.BoundedSemaphore(1)

    ok, reason = rg.can_dispatch_local()
    assert ok is True
    assert reason == "ok"


@patch("mullet.core.router.resource_guard.psutil.cpu_count", return_value=4)
@patch("mullet.core.router.resource_guard.os.getloadavg", return_value=(10.0, 2.0, 3.0))
@patch("mullet.core.router.resource_guard.psutil.virtual_memory")
def test_can_dispatch_local_high_load(mock_vm, _mock_loadavg, _mock_cpu_count):
    mock_vm.return_value.available = 16 * (1024 ** 3)

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.resource_manager = MagicMock()
        rg.resource_manager.probe_gpu.return_value = None
        rg.semaphore = threading.BoundedSemaphore(1)

    ok, reason = rg.can_dispatch_local()
    assert ok is False
    assert "High load average" in reason


@patch("mullet.core.router.resource_guard.psutil.cpu_count", return_value=4)
@patch("mullet.core.router.resource_guard.os.getloadavg", return_value=(1.0, 2.0, 3.0))
@patch("mullet.core.router.resource_guard.psutil.virtual_memory")
def test_can_dispatch_local_low_ram(mock_vm, _mock_loadavg, _mock_cpu_count):
    mock_vm.return_value.available = 1 * (1024 ** 3)

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.resource_manager = MagicMock()
        rg.resource_manager.probe_gpu.return_value = None
        rg.semaphore = threading.BoundedSemaphore(1)

    ok, reason = rg.can_dispatch_local()
    assert ok is False
    assert "Insufficient RAM" in reason


@patch("mullet.core.router.resource_guard.psutil.cpu_count", return_value=4)
@patch("mullet.core.router.resource_guard.os.getloadavg", return_value=(1.0, 2.0, 3.0))
@patch("mullet.core.router.resource_guard.psutil.virtual_memory")
def test_can_dispatch_local_gpu_low_vram(mock_vm, _mock_loadavg, _mock_cpu_count):
    mock_vm.return_value.available = 16 * (1024 ** 3)

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.resource_manager = MagicMock()
        rg.resource_manager.probe_gpu.return_value = {"free_mb": 500, "utilization_pct": 50}
        rg.semaphore = threading.BoundedSemaphore(1)

    ok, reason = rg.can_dispatch_local()
    assert ok is False
    assert "GPU VRAM low" in reason


@patch("mullet.core.router.resource_guard.psutil.cpu_count", return_value=4)
@patch("mullet.core.router.resource_guard.os.getloadavg", return_value=(1.0, 2.0, 3.0))
@patch("mullet.core.router.resource_guard.psutil.virtual_memory")
def test_can_dispatch_local_gpu_saturated(mock_vm, _mock_loadavg, _mock_cpu_count):
    mock_vm.return_value.available = 16 * (1024 ** 3)

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.resource_manager = MagicMock()
        rg.resource_manager.probe_gpu.return_value = {"free_mb": 2000, "utilization_pct": 96}
        rg.semaphore = threading.BoundedSemaphore(1)

    ok, reason = rg.can_dispatch_local()
    assert ok is False
    assert "GPU saturated" in reason


@patch("mullet.core.router.resource_guard.psutil.cpu_count", return_value=4)
@patch("mullet.core.router.resource_guard.os.getloadavg", return_value=(1.0, 2.0, 3.0))
@patch("mullet.core.router.resource_guard.psutil.virtual_memory")
@patch("mullet.core.router.resource_guard.psutil.process_iter")
def test_can_dispatch_local_trag_running(mock_iter, mock_vm, _mock_loadavg, _mock_cpu_count):
    mock_vm.return_value.available = 16 * (1024 ** 3)
    proc = MagicMock()
    proc.info = {"cmdline": ["/usr/bin/python", "trove", "reindex"]}
    mock_iter.return_value = [proc]

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.resource_manager = MagicMock()
        rg.resource_manager.probe_gpu.return_value = None
        rg.semaphore = threading.BoundedSemaphore(1)

    ok, reason = rg.can_dispatch_local()
    assert ok is False
    assert "TRAG reindex running" in reason


@patch("mullet.core.router.resource_guard.psutil.cpu_count", return_value=4)
@patch("mullet.core.router.resource_guard.os.getloadavg", return_value=(1.0, 2.0, 3.0))
@patch("mullet.core.router.resource_guard.psutil.virtual_memory")
@patch("mullet.core.router.resource_guard.psutil.process_iter")
def test_can_dispatch_local_trag_ingest(mock_iter, mock_vm, _mock_loadavg, _mock_cpu_count):
    mock_vm.return_value.available = 16 * (1024 ** 3)
    proc = MagicMock()
    proc.info = {"cmdline": ["python", "trove", "ingest"]}
    mock_iter.return_value = [proc]

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.resource_manager = MagicMock()
        rg.resource_manager.probe_gpu.return_value = None
        rg.semaphore = threading.BoundedSemaphore(1)

    ok, reason = rg.can_dispatch_local()
    assert ok is False
    assert "TRAG reindex running" in reason


@patch("mullet.core.router.resource_guard.psutil.cpu_count", return_value=4)
@patch("mullet.core.router.resource_guard.os.getloadavg", return_value=(1.0, 2.0, 3.0))
@patch("mullet.core.router.resource_guard.psutil.virtual_memory")
@patch("mullet.core.router.resource_guard.psutil.process_iter")
def test_can_dispatch_local_trag_embed(mock_iter, mock_vm, _mock_loadavg, _mock_cpu_count):
    mock_vm.return_value.available = 16 * (1024 ** 3)
    proc = MagicMock()
    proc.info = {"cmdline": ["python", "trove", "embed"]}
    mock_iter.return_value = [proc]

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.resource_manager = MagicMock()
        rg.resource_manager.probe_gpu.return_value = None
        rg.semaphore = threading.BoundedSemaphore(1)

    ok, reason = rg.can_dispatch_local()
    assert ok is False
    assert "TRAG reindex running" in reason


@patch("mullet.core.router.resource_guard.psutil.cpu_count", return_value=4)
@patch("mullet.core.router.resource_guard.os.getloadavg", return_value=(1.0, 2.0, 3.0))
@patch("mullet.core.router.resource_guard.psutil.virtual_memory")
@patch("mullet.core.router.resource_guard.psutil.process_iter")
def test_can_dispatch_local_process_iter_exception(mock_iter, mock_vm, _mock_loadavg, _mock_cpu_count):
    mock_vm.return_value.available = 16 * (1024 ** 3)
    mock_iter.side_effect = psutil.AccessDenied("nope")

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.resource_manager = MagicMock()
        rg.resource_manager.probe_gpu.return_value = None
        rg.semaphore = threading.BoundedSemaphore(1)

    ok, reason = rg.can_dispatch_local()
    assert ok is True
    assert reason == "ok"


@patch("mullet.core.router.resource_guard.psutil.cpu_count", return_value=4)
@patch("mullet.core.router.resource_guard.os.getloadavg", return_value=(1.0, 2.0, 3.0))
@patch("mullet.core.router.resource_guard.psutil.virtual_memory")
@patch("mullet.core.router.resource_guard.psutil.process_iter")
def test_can_dispatch_local_semaphore_taken(mock_iter, mock_vm, _mock_loadavg, _mock_cpu_count):
    mock_vm.return_value.available = 16 * (1024 ** 3)
    mock_iter.return_value = []

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.resource_manager = MagicMock()
        rg.resource_manager.probe_gpu.return_value = None
        rg.semaphore = threading.BoundedSemaphore(1)

    rg.semaphore.acquire()  # take the only slot
    ok, reason = rg.can_dispatch_local()
    assert ok is False
    assert "semaphore acquired" in reason
    rg.semaphore.release()


@patch("mullet.core.router.resource_guard.psutil.cpu_count", return_value=4)
@patch("mullet.core.router.resource_guard.os.getloadavg", return_value=(1.0, 2.0, 3.0))
@patch("mullet.core.router.resource_guard.psutil.virtual_memory")
@patch("mullet.core.router.resource_guard.psutil.process_iter")
def test_can_dispatch_local_no_cmdline(mock_iter, mock_vm, _mock_loadavg, _mock_cpu_count):
    mock_vm.return_value.available = 16 * (1024 ** 3)
    proc = MagicMock()
    proc.info = {"cmdline": None}
    mock_iter.return_value = [proc]

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.resource_manager = MagicMock()
        rg.resource_manager.probe_gpu.return_value = None
        rg.semaphore = threading.BoundedSemaphore(1)

    ok, reason = rg.can_dispatch_local()
    assert ok is True
    assert reason == "ok"


@patch("mullet.core.router.resource_guard.psutil.cpu_count", return_value=4)
@patch("mullet.core.router.resource_guard.os.getloadavg", return_value=(1.0, 2.0, 3.0))
@patch("mullet.core.router.resource_guard.psutil.virtual_memory")
@patch("mullet.core.router.resource_guard.psutil.process_iter")
def test_can_dispatch_local_no_trag_in_cmdline(mock_iter, mock_vm, _mock_loadavg, _mock_cpu_count):
    mock_vm.return_value.available = 16 * (1024 ** 3)
    proc = MagicMock()
    proc.info = {"cmdline": ["python", "other_tool", "reindex"]}
    mock_iter.return_value = [proc]

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.resource_manager = MagicMock()
        rg.resource_manager.probe_gpu.return_value = None
        rg.semaphore = threading.BoundedSemaphore(1)

    ok, reason = rg.can_dispatch_local()
    assert ok is True
    assert reason == "ok"


# --- acquire_local_slot / release_local_slot ---------------------------------

def test_acquire_local_slot_success():
    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.semaphore = threading.BoundedSemaphore(1)
    assert rg.acquire_local_slot() is True


def test_acquire_local_slot_failure():
    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.semaphore = threading.BoundedSemaphore(1)
    rg.semaphore.acquire()
    assert rg.acquire_local_slot() is False
    rg.semaphore.release()


def test_release_local_slot():
    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.semaphore = threading.BoundedSemaphore(1)
    rg.semaphore.acquire()
    rg.release_local_slot()
    assert rg.semaphore.acquire(blocking=False) is True
    rg.semaphore.release()


# --- flush_resources ---------------------------------------------------------

@patch("mullet.core.router.resource_guard.os.getpid", return_value=1234)
@patch("mullet.core.router.resource_guard.psutil.Process")
@patch("mullet.core.router.resource_guard.gc.collect")
@patch("mullet.core.router.resource_guard.time.time", return_value=1000)
def test_flush_resources_no_children(mock_time, _mock_gc, mock_Process, _mock_getpid):
    parent = MagicMock()
    parent.children.return_value = []
    mock_Process.return_value = parent

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.semaphore = threading.BoundedSemaphore(1)
    rg.resource_manager = MagicMock()
    rg.resource_manager.probe_ram.return_value = {"available_gb": 8.0, "total_gb": 16.0}
    rg.resource_manager.probe_cpu.return_value = {"total_cores": 4}
    rg.resource_manager.probe_gpu.return_value = None
    rg.resource_manager._gpu_probed = False

    with patch.object(rg, "get_system_snapshot") as mock_snapshot:
        mock_snapshot.return_value = {"ram_available_gb": 8.0}
        result = rg.flush_resources()

    assert "before" in result
    assert "after" in result
    assert "freed_ram_gb" in result
    parent.children.assert_called_once_with(recursive=True)


@patch("mullet.core.router.resource_guard.os.getpid", return_value=1234)
@patch("mullet.core.router.resource_guard.psutil.Process")
@patch("mullet.core.router.resource_guard.gc.collect")
@patch("mullet.core.router.resource_guard.time.time", return_value=1000)
def test_flush_resources_terminates_old_children(mock_time, mock_gc, mock_Process, _mock_getpid):
    child = MagicMock()
    child.create_time.return_value = 900  # older than 60s
    child.pid = 5678
    child.name.return_value = "orphan"
    child.terminate.return_value = None

    parent = MagicMock()
    parent.children.return_value = [child]
    mock_Process.return_value = parent

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.semaphore = threading.BoundedSemaphore(1)
    rg.resource_manager = MagicMock()
    rg.resource_manager.probe_ram.return_value = {"available_gb": 8.0, "total_gb": 16.0}
    rg.resource_manager.probe_cpu.return_value = {"total_cores": 4}
    rg.resource_manager.probe_gpu.return_value = None
    rg.resource_manager._gpu_probed = False

    with patch.object(rg, "get_system_snapshot") as mock_snapshot:
        mock_snapshot.return_value = {"ram_available_gb": 8.0}
        rg.flush_resources()

    child.terminate.assert_called_once()
    assert mock_gc.call_count >= 3


@patch("mullet.core.router.resource_guard.os.getpid", return_value=1234)
@patch("mullet.core.router.resource_guard.psutil.Process")
@patch("mullet.core.router.resource_guard.gc.collect")
@patch("mullet.core.router.resource_guard.time.time", return_value=1000)
def test_flush_resources_skips_young_children(mock_time, _mock_gc, mock_Process, _mock_getpid):
    child = MagicMock()
    child.create_time.return_value = 990  # only 10s old
    child.pid = 5678
    child.name.return_value = "fresh"

    parent = MagicMock()
    parent.children.return_value = [child]
    mock_Process.return_value = parent

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.semaphore = threading.BoundedSemaphore(1)
    rg.resource_manager = MagicMock()
    rg.resource_manager.probe_ram.return_value = {"available_gb": 8.0, "total_gb": 16.0}
    rg.resource_manager.probe_cpu.return_value = {"total_cores": 4}
    rg.resource_manager.probe_gpu.return_value = None
    rg.resource_manager._gpu_probed = False

    with patch.object(rg, "get_system_snapshot") as mock_snapshot:
        mock_snapshot.return_value = {"ram_available_gb": 8.0}
        rg.flush_resources()

    child.terminate.assert_not_called()


@patch("mullet.core.router.resource_guard.os.getpid", return_value=1234)
@patch("mullet.core.router.resource_guard.psutil.Process")
@patch("mullet.core.router.resource_guard.gc.collect")
@patch("mullet.core.router.resource_guard.time.time", return_value=1000)
def test_flush_resources_child_access_denied(mock_time, _mock_gc, mock_Process, _mock_getpid):
    child = MagicMock()
    child.create_time.side_effect = psutil.AccessDenied("nope")

    parent = MagicMock()
    parent.children.return_value = [child]
    mock_Process.return_value = parent

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.semaphore = threading.BoundedSemaphore(1)
    rg.resource_manager = MagicMock()
    rg.resource_manager.probe_ram.return_value = {"available_gb": 8.0, "total_gb": 16.0}
    rg.resource_manager.probe_cpu.return_value = {"total_cores": 4}
    rg.resource_manager.probe_gpu.return_value = None
    rg.resource_manager._gpu_probed = False

    with patch.object(rg, "get_system_snapshot") as mock_snapshot:
        mock_snapshot.return_value = {"ram_available_gb": 8.0}
        rg.flush_resources()

    child.terminate.assert_not_called()


@patch("mullet.core.router.resource_guard.os.getpid", return_value=1234)
@patch("mullet.core.router.resource_guard.psutil.Process")
@patch("mullet.core.router.resource_guard.gc.collect")
@patch("mullet.core.router.resource_guard.time.time", return_value=1000)
def test_flush_resources_parent_not_found(mock_time, _mock_gc, mock_Process, _mock_getpid):
    mock_Process.side_effect = psutil.NoSuchProcess(1234)

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.semaphore = threading.BoundedSemaphore(1)
    rg.resource_manager = MagicMock()
    rg.resource_manager.probe_ram.return_value = {"available_gb": 8.0, "total_gb": 16.0}
    rg.resource_manager.probe_cpu.return_value = {"total_cores": 4}
    rg.resource_manager.probe_gpu.return_value = None
    rg.resource_manager._gpu_probed = False

    with patch.object(rg, "get_system_snapshot") as mock_snapshot:
        mock_snapshot.return_value = {"ram_available_gb": 8.0}
        rg.flush_resources()

    # Should not raise; just logs warning


@patch("mullet.core.router.resource_guard.os.getpid", return_value=1234)
@patch("mullet.core.router.resource_guard.psutil.Process")
@patch("mullet.core.router.resource_guard.gc.collect")
@patch("mullet.core.router.resource_guard.time.time", return_value=1000)
def test_flush_resources_torch_available(mock_time, _mock_gc, mock_Process, _mock_getpid):
    parent = MagicMock()
    parent.children.return_value = []
    mock_Process.return_value = parent

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.semaphore = threading.BoundedSemaphore(1)
    rg.resource_manager = MagicMock()
    rg.resource_manager.probe_ram.return_value = {"available_gb": 8.0, "total_gb": 16.0}
    rg.resource_manager.probe_cpu.return_value = {"total_cores": 4}
    rg.resource_manager.probe_gpu.return_value = None
    rg.resource_manager._gpu_probed = False

    with patch.dict("sys.modules", {"torch": mock_torch}):
        with patch.object(rg, "get_system_snapshot") as mock_snapshot:
            mock_snapshot.return_value = {"ram_available_gb": 8.0}
            rg.flush_resources()

    mock_torch.cuda.empty_cache.assert_called_once()
    mock_torch.cuda.synchronize.assert_called_once()


@patch("mullet.core.router.resource_guard.os.getpid", return_value=1234)
@patch("mullet.core.router.resource_guard.psutil.Process")
@patch("mullet.core.router.resource_guard.gc.collect")
@patch("mullet.core.router.resource_guard.time.time", return_value=1000)
def test_flush_resources_torch_import_error(mock_time, _mock_gc, mock_Process, _mock_getpid):
    parent = MagicMock()
    parent.children.return_value = []
    mock_Process.return_value = parent

    # torch not available → import fails
    modules = dict(__import__("sys").modules)
    modules.pop("torch", None)

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.semaphore = threading.BoundedSemaphore(1)
    rg.resource_manager = MagicMock()
    rg.resource_manager.probe_ram.return_value = {"available_gb": 8.0, "total_gb": 16.0}
    rg.resource_manager.probe_cpu.return_value = {"total_cores": 4}
    rg.resource_manager.probe_gpu.return_value = None
    rg.resource_manager._gpu_probed = False

    with patch.object(rg, "get_system_snapshot") as mock_snapshot:
        mock_snapshot.return_value = {"ram_available_gb": 8.0}
        rg.flush_resources()

    # Should not raise


@patch("mullet.core.router.resource_guard.os.getpid", return_value=1234)
@patch("mullet.core.router.resource_guard.psutil.Process")
@patch("mullet.core.router.resource_guard.gc.collect")
@patch("mullet.core.router.resource_guard.time.time", return_value=1000)
def test_flush_resources_torch_cuda_raises(mock_time, _mock_gc, mock_Process, _mock_getpid):
    parent = MagicMock()
    parent.children.return_value = []
    mock_Process.return_value = parent

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.side_effect = RuntimeError("cuda exploded")

    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.semaphore = threading.BoundedSemaphore(1)
    rg.resource_manager = MagicMock()
    rg.resource_manager.probe_ram.return_value = {"available_gb": 8.0, "total_gb": 16.0}
    rg.resource_manager.probe_cpu.return_value = {"total_cores": 4}
    rg.resource_manager.probe_gpu.return_value = None
    rg.resource_manager._gpu_probed = False

    with patch.dict("sys.modules", {"torch": mock_torch}):
        with patch.object(rg, "get_system_snapshot") as mock_snapshot:
            mock_snapshot.return_value = {"ram_available_gb": 8.0}
            rg.flush_resources()

    # Should not raise


# --- get_system_snapshot -----------------------------------------------------

def test_get_system_snapshot():
    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
    rg.resource_manager = MagicMock()
    rg.resource_manager.probe_ram.return_value = {"available_gb": 8.0, "total_gb": 16.0}
    rg.resource_manager.probe_cpu.return_value = {"total_cores": 4}
    rg.resource_manager.probe_gpu.return_value = {"free_mb": 2000}

    with patch("mullet.core.router.resource_guard.psutil.getloadavg", return_value=(1.5, 2.0, 3.0)):
        snapshot = rg.get_system_snapshot()

    assert snapshot["load_avg"] == 1.5
    assert snapshot["cpu_cores"] == 4
    assert snapshot["ram_available_gb"] == 8.0
    assert snapshot["ram_total_gb"] == 16.0
    assert snapshot["gpu"] == {"free_mb": 2000}


def test_get_system_snapshot_no_gpu():
    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
    rg.resource_manager = MagicMock()
    rg.resource_manager.probe_ram.return_value = {"available_gb": 4.0, "total_gb": 8.0}
    rg.resource_manager.probe_cpu.return_value = {"total_cores": 2}
    rg.resource_manager.probe_gpu.return_value = None

    with patch("mullet.core.router.resource_guard.psutil.getloadavg", return_value=(0.5, 1.0, 1.5)):
        snapshot = rg.get_system_snapshot()

    assert snapshot["gpu"] is None


# --- get_resource_guard (singleton) ------------------------------------------

def test_get_resource_guard_singleton():
    g1 = get_resource_guard()
    g2 = get_resource_guard()
    assert g1 is g2


def test_get_resource_guard_returns_resource_guard():
    g = get_resource_guard()
    assert isinstance(g, ResourceGuard)


# --- logging checks ----------------------------------------------------------

def test_acquire_local_slot_logs_info(caplog):
    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.semaphore = threading.BoundedSemaphore(1)

    with caplog.at_level(logging.INFO, logger="mullet"):
        rg.acquire_local_slot()
    assert "Local dispatch slot acquired" in caplog.text


def test_release_local_slot_logs_info(caplog):
    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.semaphore = threading.BoundedSemaphore(1)
    rg.semaphore.acquire()

    with caplog.at_level(logging.INFO, logger="mullet"):
        rg.release_local_slot()
    assert "Local dispatch slot released" in caplog.text


def test_acquire_local_slot_logs_warning(caplog):
    with patch.object(ResourceGuard, "__init__", lambda self: None):
        rg = ResourceGuard.__new__(ResourceGuard)
        rg.semaphore = threading.BoundedSemaphore(1)
    rg.semaphore.acquire()

    with caplog.at_level(logging.WARNING):
        rg.acquire_local_slot()
    assert "Failed to acquire local dispatch slot" in caplog.text
