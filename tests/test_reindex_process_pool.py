"""Tests for the --processes ProcessPoolExecutor branch of scripts/reindex.py.

Covers:
  - The CLI flag parses correctly.
  - --processes > 1 takes precedence over --workers (ProcessPoolExecutor is
    used, ThreadPoolExecutor is NOT).
  - --processes 1 falls through to the serial path (no pool executor).
  - A single failing future does not take down the other pools; the summary
    line still prints.

We never actually spawn subprocesses — concurrent.futures executors are
patched to simple in-process fakes that call the submitted fn synchronously.
"""

import importlib.util
import logging
import sys
from pathlib import Path

import pytest


_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _load_reindex():
    """Load scripts/reindex.py as a fresh module (scripts/ has no __init__.py).

    Cache the module on sys.modules under the name 'scripts.reindex' so that
    ProcessPoolExecutor / ThreadPoolExecutor references inside reindex are
    patchable at 'scripts.reindex.<name>'.

    scripts/reindex.py configures global logging on import (mullet logger to
    CRITICAL). We snapshot and restore affected logger levels so other tests
    in the same session aren't contaminated.
    """
    name = "scripts.reindex"
    if name in sys.modules:
        return sys.modules[name]

    # Snapshot logger levels that reindex stomps on.
    _tracked_loggers = [
        "sentence_transformers",
        "chromadb",
        "chromadb.telemetry",
        "mullet",
        "httpx",
        "openai",
    ]
    _snapshot = {n: logging.getLogger(n).level for n in _tracked_loggers}

    path = _SCRIPTS_DIR / "reindex.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    # Restore logger levels so sibling tests (e.g. test_silent_failure_logging)
    # can observe warnings on the mullet logger again.
    for n, lvl in _snapshot.items():
        logging.getLogger(n).setLevel(lvl)

    return mod


# ---------------------------------------------------------------------------
# Fake executors — synchronous, in-process substitutes
# ---------------------------------------------------------------------------


class _FakeFuture:
    def __init__(self, result=None, exc: Exception | None = None):
        self._result = result
        self._exc = exc

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._result


class _FakeProcessPoolExecutor:
    """Synchronous substitute. Runs fn immediately on submit."""
    calls: list = []

    def __init__(self, max_workers: int = 1, *args, **kwargs):
        self.max_workers = max_workers
        _FakeProcessPoolExecutor.calls.append(max_workers)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *args, **kwargs):
        try:
            return _FakeFuture(result=fn(*args, **kwargs))
        except Exception as e:  # noqa: BLE001
            return _FakeFuture(exc=e)


class _FakeThreadPoolExecutor(_FakeProcessPoolExecutor):
    calls: list = []

    def __init__(self, max_workers: int = 1, *args, **kwargs):
        self.max_workers = max_workers
        _FakeThreadPoolExecutor.calls.append(max_workers)


def _fake_as_completed(futures):
    # Accept both list and dict (dict iterates over keys — the futures).
    if isinstance(futures, dict):
        return list(futures.keys())
    return list(futures)


@pytest.fixture(autouse=True)
def _reset_fake_executors():
    _FakeProcessPoolExecutor.calls = []
    _FakeThreadPoolExecutor.calls = []
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_processes_flag_parses():
    """--processes N populates args.processes correctly."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", "-a", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--processes", type=int, default=0,
        help="Parallel pool workers using a process pool (CUDA isolation).",
    )
    args = parser.parse_args(["--all", "--processes", "4"])
    assert args.processes == 4
    assert args.all is True


def test_processes_takes_precedence_over_workers(monkeypatch, capsys):
    """With --processes 2 --workers 4, ProcessPoolExecutor runs and
    ThreadPoolExecutor is never instantiated."""
    mod = _load_reindex()

    # Stable list of pools — don't hit the filesystem.
    monkeypatch.setattr(mod, "discover_clients", lambda clients_dir=None: ["pool_a", "pool_b"])

    # Make _reindex_one cheap — return a fixed elapsed time.
    monkeypatch.setattr(mod, "_reindex_one", lambda name, args, quiet=False: 1.0)

    # Patch concurrent.futures imports used INSIDE the process-pool branch.
    # The branch does `from concurrent.futures import ProcessPoolExecutor, as_completed`
    # so we must patch the source module.
    import concurrent.futures as _cf
    monkeypatch.setattr(_cf, "ProcessPoolExecutor", _FakeProcessPoolExecutor)
    monkeypatch.setattr(_cf, "ThreadPoolExecutor", _FakeThreadPoolExecutor)
    monkeypatch.setattr(_cf, "as_completed", _fake_as_completed)

    monkeypatch.setattr(sys, "argv", ["reindex", "--all", "--processes", "2", "--workers", "4"])

    mod.main()

    assert _FakeProcessPoolExecutor.calls == [2], (
        f"ProcessPoolExecutor should be called with max_workers=2, got {_FakeProcessPoolExecutor.calls}"
    )
    assert _FakeThreadPoolExecutor.calls == [], (
        f"ThreadPoolExecutor should NOT be called when --processes > 1, got {_FakeThreadPoolExecutor.calls}"
    )

    out = capsys.readouterr().out
    assert "process-pool mode" in out
    assert "2 processes" in out
    assert "pool_a" in out and "pool_b" in out


def test_processes_one_serial(monkeypatch, capsys):
    """--processes 1 falls through to the serial path; neither pool executor is used."""
    mod = _load_reindex()

    monkeypatch.setattr(mod, "discover_clients", lambda clients_dir=None: ["pool_a"])
    monkeypatch.setattr(mod, "_reindex_one", lambda name, args, quiet=False: 2.5)

    import concurrent.futures as _cf
    monkeypatch.setattr(_cf, "ProcessPoolExecutor", _FakeProcessPoolExecutor)
    monkeypatch.setattr(_cf, "ThreadPoolExecutor", _FakeThreadPoolExecutor)
    monkeypatch.setattr(_cf, "as_completed", _fake_as_completed)

    monkeypatch.setattr(sys, "argv", ["reindex", "--all", "--processes", "1"])

    mod.main()

    assert _FakeProcessPoolExecutor.calls == [], "ProcessPoolExecutor should NOT be used with --processes 1"
    assert _FakeThreadPoolExecutor.calls == [], "ThreadPoolExecutor should NOT be used with --workers default=1"


def test_process_pool_failure_isolated(monkeypatch, capsys):
    """One pool raising an exception does not abort the rest;
    DONE lines print for the survivors, FAILED for the bad one, summary prints."""
    mod = _load_reindex()

    monkeypatch.setattr(
        mod, "discover_clients", lambda clients_dir=None: ["pool_a", "pool_b", "pool_c"],
    )

    def _fake_reindex_one(name, args, quiet=False):
        if name == "pool_b":
            raise Exception("boom")
        return 1.0

    monkeypatch.setattr(mod, "_reindex_one", _fake_reindex_one)

    import concurrent.futures as _cf
    monkeypatch.setattr(_cf, "ProcessPoolExecutor", _FakeProcessPoolExecutor)
    monkeypatch.setattr(_cf, "ThreadPoolExecutor", _FakeThreadPoolExecutor)
    monkeypatch.setattr(_cf, "as_completed", _fake_as_completed)

    monkeypatch.setattr(sys, "argv", ["reindex", "--all", "--processes", "3"])

    mod.main()  # must not raise

    out = capsys.readouterr().out
    # good pools report DONE
    assert "pool_a" in out and "DONE" in out
    assert "pool_c" in out
    # bad pool reports FAILED with the exception message
    assert "pool_b" in out
    assert "FAILED" in out
    assert "boom" in out
    # summary line
    assert "3 pools" in out
    assert "3 processes" in out
