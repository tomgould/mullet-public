"""Tests for mullet.harness.cron scheduler and CLI."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mullet.harness.cron.scheduler import CronJob, CronScheduler, CronStore
from mullet.cli.cron_cli import cmd_cron


# ------------------------------------------------------------------
# CronJob dataclass
# ------------------------------------------------------------------
def test_cronjob_dataclass():
    job = CronJob(
        id="abc123",
        name="nightly-docs",
        schedule="@daily",
        task="update docs",
        provider="kimi",
        model="kimi-latest",
        enabled=True,
        last_run=None,
        next_run=None,
        run_count=0,
        output_dir="/tmp/cron-out",
    )
    assert job.id == "abc123"
    assert job.name == "nightly-docs"
    assert job.schedule == "@daily"
    assert job.task == "update docs"
    assert job.provider == "kimi"
    assert job.model == "kimi-latest"
    assert job.enabled is True
    assert job.output_dir == "/tmp/cron-out"


# ------------------------------------------------------------------
# Store operations
# ------------------------------------------------------------------
def test_store_add_list_remove(tmp_path: Path):
    store = CronStore(path=tmp_path / "jobs.json")
    job = CronJob(id="j1", name="test", schedule="@hourly", task="hello")
    store.add(job)
    assert len(store.list()) == 1
    assert store.get("j1") == job
    assert store.remove("j1") is True
    assert store.list() == []
    assert store.remove("j1") is False


def test_store_persists_jsonl_atomic(tmp_path: Path):
    store = CronStore(path=tmp_path / "jobs.json")
    job = CronJob(id="j2", name="persist", schedule="every 5 minutes", task="run tests")
    store.add(job)

    # Verify no temp file left behind
    assert not (tmp_path / "jobs.json.tmp").exists()

    # Verify JSON is valid and round-trips
    raw = json.loads((tmp_path / "jobs.json").read_text())
    assert "j2" in raw
    assert raw["j2"]["name"] == "persist"

    # Fresh load
    store2 = CronStore(path=tmp_path / "jobs.json")
    assert len(store2.list()) == 1
    loaded = store2.get("j2")
    assert loaded is not None
    assert loaded.name == "persist"
    assert loaded.schedule == "every 5 minutes"


# ------------------------------------------------------------------
# Next-run computation
# ------------------------------------------------------------------
def test_next_run_time_hourly():
    store = CronStore(path=Path("/dev/null"))
    sched = CronScheduler(engine_factory=MagicMock, store=store)
    now = time.time()
    nxt = sched._next_run_time("@hourly", after=now)
    assert nxt > now
    dt = datetime.fromtimestamp(nxt)
    assert dt.minute == 0
    assert dt.second == 0


def test_next_run_time_daily():
    store = CronStore(path=Path("/dev/null"))
    sched = CronScheduler(engine_factory=MagicMock, store=store)
    now = time.time()
    nxt = sched._next_run_time("@daily", after=now)
    assert nxt > now
    dt = datetime.fromtimestamp(nxt)
    assert dt.hour == 0
    assert dt.minute == 0
    assert dt.second == 0


def test_next_run_time_every_n_minutes():
    store = CronStore(path=Path("/dev/null"))
    sched = CronScheduler(engine_factory=MagicMock, store=store)
    now = time.time()
    nxt = sched._next_run_time("every 5 minutes", after=now)
    assert nxt == pytest.approx(now + 300, abs=1)


# ------------------------------------------------------------------
# Scheduler run_now
# ------------------------------------------------------------------
def test_run_now_invokes_engine(tmp_path: Path):
    store = CronStore(path=tmp_path / "jobs.json")
    job = CronJob(id="j3", name="run-now", schedule="@hourly", task="say hi", provider="kimi")
    store.add(job)

    mock_engine = MagicMock()
    mock_engine.run.return_value = {"response": "hi"}

    scheduler = CronScheduler(engine_factory=lambda: mock_engine, store=store)
    result = scheduler.run_now("j3")

    mock_engine.run.assert_called_once_with("say hi", provider="kimi", model=None)
    assert "path" in result
    assert Path(result["path"]).exists()

    # Job metadata updated
    updated = store.get("j3")
    assert updated is not None
    assert updated.run_count == 1
    assert updated.last_run is not None
    assert updated.next_run is not None


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def test_cli_list_outputs_jobs(capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store_path = tmp_path / "cron-jobs.json"
    monkeypatch.setattr(
        "mullet.cli.cron_cli.CronStore",
        lambda *args, **kwargs: CronStore(path=store_path),
    )
    job = CronJob(id="j4", name="list-me", schedule="@daily", task="task")
    CronStore(path=store_path).add(job)

    rc = cmd_cron(["list"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "list-me" in captured.out
    assert "j4" in captured.out


def test_cli_add_creates_job(capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store_path = tmp_path / "cron-jobs.json"
    monkeypatch.setattr(
        "mullet.cli.cron_cli.CronStore",
        lambda *args, **kwargs: CronStore(path=store_path),
    )

    rc = cmd_cron([
        "add", "my-job",
        "--schedule", "every 10 minutes",
        "--task", "do something",
        "--provider", "gemini",
        "--model", "gemini-2.5-flash",
    ])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Added job" in captured.out

    store = CronStore(path=store_path)
    jobs = store.list()
    assert len(jobs) == 1
    assert jobs[0].name == "my-job"
    assert jobs[0].schedule == "every 10 minutes"
    assert jobs[0].task == "do something"
    assert jobs[0].provider == "gemini"
    assert jobs[0].model == "gemini-2.5-flash"


def test_cli_crontab_outputs_lines(capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store_path = tmp_path / "cron-jobs.json"
    monkeypatch.setattr(
        "mullet.cli.cron_cli.CronStore",
        lambda *args, **kwargs: CronStore(path=store_path),
    )
    store = CronStore(path=store_path)
    store.add(CronJob(id="j5", name="hourly", schedule="@hourly", task="t1"))
    store.add(CronJob(id="j6", name="every5", schedule="every 5 minutes", task="t2"))

    rc = cmd_cron(["crontab"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "@hourly" in captured.out
    assert "*/5 * * * *" in captured.out
    assert "j5" in captured.out
    assert "j6" in captured.out
    assert "hourly" in captured.out
    assert "every5" in captured.out
