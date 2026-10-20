"""Thread-safe Job and JobManager primitives for knowledge-ingestion runs."""
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Dict, List

@dataclass
class Job:
    """Represents a job with its execution status and progress."""
    id: str
    description: str
    status: str = "pending"  # pending/running/completed/failed/cancelled
    progress: int = 0      # 0-100
    created_at: datetime = None
    started_at: datetime = None
    completed_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class JobManager:
    """Manages job execution with thread-safe operations and concurrency limits."""
    def __init__(self, max_concurrent_jobs: int = 5):
        self.jobs: Dict[str, Job] = {}
        self.lock = Lock()
        self.max_concurrent_jobs = max_concurrent_jobs
        self.running_jobs_count = 0

    def _can_start_new_job(self) -> bool:
        """Check if a new job can be started based on concurrency limits."""
        return self.running_jobs_count < self.max_concurrent_jobs

    def start_job(self, job_id: str, description: str) -> Job:
        """Start a new job with the given ID and description."""
        with self.lock:
            if job_id in self.jobs:
                raise ValueError(f"Job with ID {job_id} already exists.")
            
            if not self._can_start_new_job():
                raise RuntimeError("Max concurrent jobs reached. Cannot start new job.")

            job = Job(id=job_id, description=description, status="running", started_at=datetime.now())
            self.jobs[job_id] = job
            self.running_jobs_count += 1
            return job

    def get_job(self, job_id: str) -> Job:
        """Retrieve a job by its ID."""
        with self.lock:
            return self.jobs.get(job_id)

    def list_jobs(self) -> List[Job]:
        """List all jobs managed by this JobManager."""
        with self.lock:
            return list(self.jobs.values())

    def cancel_job(self, job_id: str):
        """Cancel a job by its ID."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job:
                if job.status == "running":
                    self.running_jobs_count -= 1
                job.status = "cancelled"
                job.completed_at = datetime.now()
            else:
                raise ValueError(f"Job with ID {job_id} not found.")

    def complete_job(self, job_id: str, status: str = "completed", progress: int = 100):
        """Mark a job as completed with optional status and progress."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job:
                if job.status == "running":
                    self.running_jobs_count -= 1
                job.status = status
                job.progress = progress
                job.completed_at = datetime.now()
            else:
                raise ValueError(f"Job with ID {job_id} not found.")

    def update_job_progress(self, job_id: str, progress: int):
        """Update the progress percentage of a running job."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job:
                job.progress = progress
            else:
                raise ValueError(f"Job with ID {job_id} not found.")
