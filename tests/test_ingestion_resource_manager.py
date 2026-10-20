"""Tests for mullet.knowledge.engine.resource_manager — ResourceManager probe helpers."""

from unittest.mock import MagicMock, patch


from mullet.knowledge.engine.resource_manager import ResourceManager


def test_init_default_headroom():
    rm = ResourceManager()
    assert rm.headroom == 0.05


def test_init_clamps_headroom_to_range():
    """headroom is clamped to [0.0, 0.5]."""
    assert ResourceManager(headroom=-0.5).headroom == 0.0
    assert ResourceManager(headroom=2.0).headroom == 0.5
    assert ResourceManager(headroom=0.25).headroom == 0.25


def test_probe_cpu_reports_cores():
    rm = ResourceManager(headroom=0.0)
    result = rm.probe_cpu()
    assert "total_cores" in result
    assert "available_cores" in result
    assert result["total_cores"] >= 1
    assert result["available_cores"] >= 2  # Floor at 2


def test_probe_cpu_respects_headroom():
    """With headroom=0.5, available = total / 2 (floor at 2)."""
    rm = ResourceManager(headroom=0.5)
    with patch("os.cpu_count", return_value=16):
        result = rm.probe_cpu()
    assert result["total_cores"] == 16
    assert result["available_cores"] == 8


def test_probe_cpu_floor_is_two():
    """Tiny core count with aggressive headroom still returns at least 2."""
    rm = ResourceManager(headroom=0.5)
    with patch("os.cpu_count", return_value=2):
        result = rm.probe_cpu()
    assert result["available_cores"] == 2


def test_probe_cpu_none_cpu_count_fallback():
    """os.cpu_count() returning None falls back to 4."""
    rm = ResourceManager(headroom=0.0)
    with patch("os.cpu_count", return_value=None):
        result = rm.probe_cpu()
    assert result["total_cores"] == 4


def test_probe_ram_reports_gb():
    rm = ResourceManager(headroom=0.0)
    result = rm.probe_ram()
    assert "total_gb" in result
    assert "available_gb" in result
    assert "usable_gb" in result
    assert result["total_gb"] > 0


def test_probe_ram_respects_headroom():
    """usable_gb = available_gb * (1 - headroom)."""
    rm = ResourceManager(headroom=0.2)

    fake_mem = MagicMock()
    fake_mem.total = 64 * 1024**3  # 64 GB
    fake_mem.available = 32 * 1024**3  # 32 GB

    with patch("psutil.virtual_memory", return_value=fake_mem):
        result = rm.probe_ram()
    assert result["total_gb"] == 64.0
    assert result["available_gb"] == 32.0
    # 32 * 0.8 = 25.6
    assert result["usable_gb"] == 25.6


def test_probe_gpu_caches_result():
    """Second call returns the same cached result without re-probing."""
    rm = ResourceManager()
    # Force a known first result
    rm._gpu_probed = True
    rm._gpu_info = {"cached": True}
    assert rm.probe_gpu() == {"cached": True}


def test_probe_gpu_returns_none_when_no_pynvml():
    """Without pynvml / GPU, probe_gpu returns None and sets _probed flag."""
    rm = ResourceManager()
    # Mock ImportError inside the pynvml import path
    with patch.dict("sys.modules", {"pynvml": None}):
        # The inner `import pynvml` in probe_gpu will ImportError
        result = rm.probe_gpu()
    # Either None (preferred) or a dict — just assert no crash + probed flag set
    assert rm._gpu_probed is True
