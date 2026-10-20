
import pytest
from unittest.mock import patch, MagicMock
from mullet.core.knowledge.resource_manager import ResourceManager
from mullet.core.knowledge.job_manager import JobManager

@pytest.fixture
def resource_manager():
    return ResourceManager()

@pytest.fixture
def job_manager():
    return JobManager(max_concurrent_jobs=2)

class MockCudaDeviceProperties:
    def __init__(self, total_memory):
        self.total_memory = total_memory

# ResourceManager Tests
def test_resource_manager_probe_keys(resource_manager):
    with patch('psutil.cpu_count', return_value=4), \
         patch('psutil.virtual_memory', return_value=MagicMock(total=8 * (1024**3))), \
         patch('torch.cuda.is_available', return_value=False):
        probe_data = resource_manager.probe()
        assert isinstance(probe_data, dict)
        assert "cpu_count" in probe_data
        assert "ram_gb" in probe_data
        assert "gpu_available" in probe_data
        assert "gpu_vram_gb" in probe_data

def test_resource_manager_probe_gpu_available(resource_manager):
    with patch('psutil.cpu_count', return_value=4), \
         patch('psutil.virtual_memory', return_value=MagicMock(total=8 * (1024**3))), \
         patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.get_device_properties', return_value=MockCudaDeviceProperties(16 * (1024**3))):
        probe_data = resource_manager.probe()
        assert probe_data["gpu_available"] is True
        assert probe_data["gpu_vram_gb"] == 16.0

def test_resource_manager_probe_gpu_not_available(resource_manager):
    with patch('psutil.cpu_count', return_value=4), \
         patch('psutil.virtual_memory', return_value=MagicMock(total=8 * (1024**3))), \
         patch('torch.cuda.is_available', return_value=False):
        probe_data = resource_manager.probe()
        assert probe_data["gpu_available"] is False
        assert probe_data["gpu_vram_gb"] == 0.0

def test_resource_manager_recommend_batch_size_gpu(resource_manager):
    assert resource_manager.recommend_batch_size(gpu_available=True) == 32

def test_resource_manager_recommend_batch_size_no_gpu(resource_manager):
    assert resource_manager.recommend_batch_size(gpu_available=False) == 8

# JobManager Tests
def test_job_manager_start_get_job(job_manager):
    job = job_manager.start_job("job1", "Test Job 1")
    assert job.id == "job1"
    assert job.description == "Test Job 1"
    assert job.status == "running"
    assert job_manager.get_job("job1") == job

def test_job_manager_start_job_already_exists(job_manager):
    job_manager.start_job("job1", "Test Job 1")
    with pytest.raises(ValueError):
        job_manager.start_job("job1", "Another Job")

def test_job_manager_max_concurrent_enforcement(job_manager):
    job_manager.start_job("job1", "Job 1")
    job_manager.start_job("job2", "Job 2")
    with pytest.raises(RuntimeError):
        job_manager.start_job("job3", "Job 3")

def test_job_manager_list_jobs(job_manager):
    job1 = job_manager.start_job("job1", "Job 1")
    job2 = job_manager.start_job("job2", "Job 2")
    jobs = job_manager.list_jobs()
    assert len(jobs) == 2
    assert job1 in jobs
    assert job2 in jobs

def test_job_manager_cancel_job(job_manager):
    job = job_manager.start_job("job1", "Job to cancel")
    job_manager.cancel_job("job1")
    cancelled_job = job_manager.get_job("job1")
    assert cancelled_job.status == "cancelled"
    assert cancelled_job.completed_at is not None
    assert job_manager.running_jobs_count == 0

def test_job_manager_cancel_nonexistent_job(job_manager):
    with pytest.raises(ValueError):
        job_manager.cancel_job("nonexistent_job")

def test_job_manager_complete_job(job_manager):
    job = job_manager.start_job("job1", "Job to complete")
    job_manager.complete_job("job1", status="completed", progress=100)
    completed_job = job_manager.get_job("job1")
    assert completed_job.status == "completed"
    assert completed_job.progress == 100
    assert completed_job.completed_at is not None
    assert job_manager.running_jobs_count == 0

def test_job_manager_complete_job_failed(job_manager):
    job = job_manager.start_job("job1", "Job to fail")
    job_manager.complete_job("job1", status="failed", progress=50)
    failed_job = job_manager.get_job("job1")
    assert failed_job.status == "failed"
    assert failed_job.progress == 50
    assert failed_job.completed_at is not None
    assert job_manager.running_jobs_count == 0

def test_job_manager_update_job_progress(job_manager):
    job = job_manager.start_job("job1", "Job to update")
    job_manager.update_job_progress("job1", 50)
    updated_job = job_manager.get_job("job1")
    assert updated_job.progress == 50

def test_job_manager_update_nonexistent_job_progress(job_manager):
    with pytest.raises(ValueError):
        job_manager.update_job_progress("nonexistent_job", 50)

def test_job_status_transitions(job_manager):
    job = job_manager.start_job("job1", "Test transitions")
    assert job.status == "running"

    job_manager.update_job_progress("job1", 25)
    assert job_manager.get_job("job1").progress == 25

    job_manager.complete_job("job1", status="completed")
    assert job_manager.get_job("job1").status == "completed"

    # Start a new job and cancel it
    job2 = job_manager.start_job("job2", "Test cancel")
    assert job2.status == "running"
    job_manager.cancel_job("job2")
    assert job_manager.get_job("job2").status == "cancelled"
