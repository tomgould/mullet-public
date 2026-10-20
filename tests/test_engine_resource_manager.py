from unittest.mock import MagicMock, patch

from mullet.knowledge.engine.resource_manager import ResourceManager


def test_probe_cpu_respects_headroom():
    rm = ResourceManager(headroom=0.5)
    with patch("os.cpu_count", return_value=16):
        result = rm.probe_cpu()
    assert result["total_cores"] == 16
    assert result["available_cores"] == 8


def test_probe_cpu_floor_is_two():
    rm = ResourceManager(headroom=0.5)
    with patch("os.cpu_count", return_value=2):
        result = rm.probe_cpu()
    assert result["available_cores"] == 2


def test_probe_cpu_none_fallback():
    rm = ResourceManager(headroom=0.0)
    with patch("os.cpu_count", return_value=None):
        result = rm.probe_cpu()
    assert result["total_cores"] == 4
    assert result["available_cores"] == 4


def test_probe_ram_respects_headroom():
    rm = ResourceManager(headroom=0.2)
    fake = MagicMock()
    fake.total = 64 * 1024 ** 3
    fake.available = 32 * 1024 ** 3
    with patch("psutil.virtual_memory", return_value=fake):
        result = rm.probe_ram()
    assert result["total_gb"] == 64.0
    assert result["available_gb"] == 32.0
    assert result["usable_gb"] == 25.6


def test_recommend_parse_workers():
    rm = ResourceManager(headroom=0.0)
    with patch("os.cpu_count", return_value=32):
        assert rm.recommend_parse_workers(100) == 4
        assert rm.recommend_parse_workers(5000) == 16
        assert rm.recommend_parse_workers(20000) == 32


def test_classify_client():
    rm = ResourceManager()
    assert rm.classify_client(0, 100) == "small"
    assert rm.classify_client(0, 10000) == "medium"
    assert rm.classify_client(0, 100000) == "large"


def test_recommend_embed_batch_no_gpu():
    rm = ResourceManager()
    with patch.object(rm, "probe_gpu", return_value=None):
        assert rm.recommend_embed_batch(384) == 16


def test_recommend_embed_batch_with_gpu():
    rm = ResourceManager(headroom=0.0)
    with patch.object(rm, "probe_gpu", return_value={"free_mb": 10000}):
        assert rm.recommend_embed_batch(384) == 64


def test_recommend_embed_batch_zero_dim_returns_256():
    """Line 106: model_dim=0 makes mb_per_batch_of_512<=0 → returns 256."""
    rm = ResourceManager()
    with patch.object(rm, "probe_gpu", return_value={"free_mb": 5000}):
        assert rm.recommend_embed_batch(0) == 256


def _fake_pynvml(total_mb=8192, used_mb=2048, free_mb=6144, util_pct=30):
    """Build a MagicMock that replaces pynvml."""
    fake = MagicMock()
    mem_info = MagicMock()
    mem_info.total = total_mb * (1024 ** 2)
    mem_info.used = used_mb * (1024 ** 2)
    mem_info.free = free_mb * (1024 ** 2)
    util_info = MagicMock()
    util_info.gpu = util_pct
    fake.nvmlDeviceGetMemoryInfo.return_value = mem_info
    fake.nvmlDeviceGetUtilizationRates.return_value = util_info
    return fake


def test_probe_gpu_success():
    """Lines 46-80: pynvml available → returns GPU info dict."""
    rm = ResourceManager()
    fake_pynvml = _fake_pynvml(total_mb=8192, used_mb=2048, free_mb=6144, util_pct=30)
    import sys
    with patch.dict(sys.modules, {"pynvml": fake_pynvml}):
        result = rm.probe_gpu()
    assert result is not None
    assert result["total_mb"] == 8192
    assert result["free_mb"] == 6144
    assert result["utilization_pct"] == 30


def test_probe_gpu_cached_on_second_call():
    """Line 46-47: second call returns cached result without re-probing."""
    rm = ResourceManager()
    rm._gpu_probed = True
    rm._gpu_info = {"total_mb": 4096}
    result = rm.probe_gpu()
    assert result == {"total_mb": 4096}


def test_probe_gpu_nvml_shutdown_failure_silenced():
    """Lines 71-75: nvmlShutdown raising is silenced."""
    rm = ResourceManager()
    fake_pynvml = _fake_pynvml()
    fake_pynvml.nvmlShutdown.side_effect = RuntimeError("shutdown boom")
    import sys
    with patch.dict(sys.modules, {"pynvml": fake_pynvml}):
        result = rm.probe_gpu()
    assert result is not None  # still returns info despite shutdown failure


def test_probe_gpu_import_failure_returns_none():
    """Lines 77-80: ImportError (pynvml absent) → returns None."""
    rm = ResourceManager()
    import sys
    with patch.dict(sys.modules, {"pynvml": None}):
        result = rm.probe_gpu()
    assert result is None


def test_summary_no_gpu():
    """Lines 126-140: summary when GPU not available."""
    rm = ResourceManager(headroom=0.0)
    with patch("os.cpu_count", return_value=8), \
         patch("psutil.virtual_memory", return_value=MagicMock(
             total=16 * 1024 ** 3, available=8 * 1024 ** 3
         )), \
         patch.object(rm, "probe_gpu", return_value=None):
        s = rm.summary()
    assert "GPU: not available" in s
    assert "CPU:" in s
    assert "RAM:" in s


def test_summary_with_gpu():
    """Lines 133-137: summary with GPU info."""
    rm = ResourceManager(headroom=0.0)
    gpu_info = {"free_mb": 6144, "total_mb": 8192, "utilization_pct": 25}
    with patch("os.cpu_count", return_value=8), \
         patch("psutil.virtual_memory", return_value=MagicMock(
             total=16 * 1024 ** 3, available=8 * 1024 ** 3
         )), \
         patch.object(rm, "probe_gpu", return_value=gpu_info):
        s = rm.summary()
    assert "GPU:" in s
    assert "6144MB" in s
    assert "25%" in s
