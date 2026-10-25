import importlib
import ipaddress
import logging
import os
import pkgutil
import re
import site
import socket
import subprocess

import pytest
import mullet.core.providers as _providers_pkg
import mullet.core.providers.constraints as _constraints

# Import every provider module before any test runs. Several tests import a
# provider inside patch.dict(sys.modules, ...), and on exit patch.dict drops
# every module first imported inside the block. A later
# patch("mullet.core.providers.<name>.<Class>") then lands on the dropped
# module while the code under test imports a fresh copy, so whether those
# tests pass depended on which tests had run before them in the same worker.
for _mod in pkgutil.iter_modules(_providers_pkg.__path__):
    try:
        importlib.import_module(f"{_providers_pkg.__name__}.{_mod.name}")
    except Exception:
        pass  # optional SDKs load lazily; real import errors surface in the tests


# Environment variables that hold credentials. Mullet loads .env on import,
# so without this every test would run with the user's real provider keys.
_SECRET_ENV = re.compile(r"(API_KEY|_TOKEN|_SECRET|_PASSWORD)$|^OPENAI_|^ANTHROPIC_|^MOONSHOT_|^KIMI_")

# Settings that move Mullet's directories or its local LLM endpoints, or pick
# the local model. Each test starts without them, so every path follows the
# throwaway HOME (see mullet.core.paths) and every URL and model is the default.
_LOCATION_ENV = (
    "XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
    "MULLET_CONFIG_DIR", "TRAWL_POOLS_DIR", "MULLET_JOBS_DIR", "MULLET_PROFILES_DIR",
    "MULLET_INTERACTIONS_DB", "LOCAL_LLM_URL", "OLLAMA_URL",
    "MULLET_LOCAL_PREFERRED_MODELS",
)


class NetworkBlocked(ConnectionError):
    """A test tried to reach a real host. Mock it, or mark the test live."""


def _is_local(address) -> bool:
    if not isinstance(address, tuple) or not address:
        return True  # AF_UNIX paths and the like
    host = address[0]
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex

# Local model servers on this machine: LM Studio (1234) and Ollama (11434).
# LM Studio loads a model onto the GPU for any request, so a test that
# reaches one hangs for minutes and fills the GPU.
_LOCAL_MODEL_PORTS = frozenset({1234, 11434})


def _check(address) -> None:
    if not _is_local(address):
        raise NetworkBlocked(f"tests may not connect to {address!r}")
    if isinstance(address, tuple) and len(address) > 1 and address[1] in _LOCAL_MODEL_PORTS:
        raise NetworkBlocked(f"tests may not reach a local model server at {address!r}")


def _guarded_connect(self, address):
    _check(address)
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    _check(address)
    return _real_connect_ex(self, address)


_NO_GPU = """#!/bin/sh
# A stand-in for nvidia-smi: tests see no GPU unless they fake one themselves.
echo "NVIDIA-SMI has failed (test stand-in)" >&2
exit 9
"""


@pytest.fixture(scope="session")
def _no_gpu_bin(tmp_path_factory):
    """A directory holding the failing nvidia-smi stand-in."""
    bindir = tmp_path_factory.mktemp("no-gpu")
    script = bindir / "nvidia-smi"
    script.write_text(_NO_GPU)
    script.chmod(0o755)
    return bindir


def _block_lms_for_the_session() -> None:
    """Point LM Studio's lms CLI at nothing for the whole run, not just per test.

    lms loads models without touching the network, so the socket guard can't
    stop it, and its path is resolved at import from the real home. Per-test
    patching alone missed loads from threads that outlive their test.
    """
    import mullet.core.providers.lmstudio_manager as _lms
    _lms.LMStudioManager.LMS_CLI = "/nonexistent/lms-blocked-in-tests"


_block_lms_for_the_session()

# Child processes started by tests (mullet doctor, the CLI end-to-end tests)
# are outside the socket guard, so they get local model URLs on a closed port.
# In-process code keeps the real defaults, which several tests check.
_DEAD_LOCAL_URLS = {"LOCAL_LLM_URL": "http://127.0.0.1:9/v1", "OLLAMA_URL": "http://127.0.0.1:9"}
_real_popen_init = subprocess.Popen.__init__


def _popen_without_local_models(self, *args, **kwargs):
    env = kwargs.get("env")
    base = dict(os.environ if env is None else env)
    for key, value in _DEAD_LOCAL_URLS.items():
        if env is None or key not in env:
            base[key] = value
    kwargs["env"] = base
    return _real_popen_init(self, *args, **kwargs)


@pytest.fixture(autouse=True)
def _isolated_from_the_real_world(request, monkeypatch, tmp_path_factory, _no_gpu_bin):
    """Every test runs with no real network, no provider keys, no GPU and a throwaway home.

    Mullet's config, state, data and cache directories all follow that home.
    nvidia-smi is a stand-in that fails, so `mullet doctor` and the machine
    guard (mullet/jobs/machine.py) see no GPU, in subprocesses too.

    Tests marked `live` (excluded by default) keep the real environment.
    """
    if request.node.get_closest_marker("live"):
        yield
        return
    for key in list(os.environ):
        if _SECRET_ENV.search(key):
            monkeypatch.delenv(key, raising=False)
    for key in _LOCATION_ENV:
        monkeypatch.delenv(key, raising=False)
    # Subprocesses still need the packages installed under the real home
    # (pip --user), which Python finds through HOME unless told otherwise.
    monkeypatch.setenv("PYTHONUSERBASE", site.getuserbase())
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    # Subprocesses get closed-port local model URLs (see _popen_without_local_models).
    monkeypatch.setattr(subprocess.Popen, "__init__", _popen_without_local_models)
    monkeypatch.setenv("PATH", f"{_no_gpu_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    # These would point Claude Code's and Mullet's config back at the real
    # home: `mullet setup-claude` writes to both.
    for name in ("CLAUDE_CONFIG_DIR", "MULLET_CONFIG_DIR", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(name, raising=False)
    # A job runner notifies when a job blocks, pauses or finishes; no test
    # should pop up a real desktop notification (tests pass their own env).
    monkeypatch.setenv("MULLET_NOTIFY", "off")
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    yield


_FAKE_CLAUDE = """#!/bin/sh
# A stand-in for the claude CLI: logs its arguments and keeps a pretend
# user-scope MCP registration, so no test runs the real Claude Code.
echo "$@" >> "@LOG@"
state="@STATE@"
case "$1" in
  --version) echo "9.9.9 (Claude Code)"; exit 0;;
  mcp)
    case "$2" in
      add) touch "$state"; echo "Added stdio MCP server $5"; exit 0;;
      get) [ -f "$state" ] && { echo "mullet:"; exit 0; }; echo "No MCP server found with name: $3" >&2; exit 1;;
      remove) [ -f "$state" ] && { rm -f "$state"; exit 0; }; echo "No MCP server found with name" >&2; exit 1;;
    esac;;
esac
exit 0
"""


@pytest.fixture
def fake_claude(tmp_path_factory, monkeypatch):
    """Put a fake `claude` first on PATH; returns the file it logs its calls to.

    `mullet doctor` runs `claude --version` and `mullet setup-claude` runs
    `claude mcp ...`; tests that reach either use this instead of the real CLI.
    The fake's pretend MCP registration is the file at `<log>.registered`.
    """
    bindir = tmp_path_factory.mktemp("fake-claude")
    log = bindir / "calls.log"
    script = bindir / "claude"
    script.write_text(_FAKE_CLAUDE.replace("@LOG@", str(log))
                      .replace("@STATE@", str(bindir / "calls.log.registered")))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    return log

try:
    from mullet.server import middleware as _mw_mod
    from mullet.server.middleware import RateLimiter as _RateLimiter
    _HAS_SERVER = True
except ImportError:
    _HAS_SERVER = False


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the global rate limiter before every test to prevent 429 bleed."""
    if _HAS_SERVER:
        _mw_mod._rate_limiter = _RateLimiter(max_requests=1000, window_seconds=60)
    yield
    if _HAS_SERVER:
        _mw_mod._rate_limiter = _RateLimiter(max_requests=60, window_seconds=60)

@pytest.fixture(scope="session", autouse=True)
def _isolate_learned_constraints(tmp_path_factory):
    # Point the registry at a temporary file for the session. The real
    # ~/.mullet/learned_constraints.json belongs to the user's Mullet, so the
    # tests must not delete it.
    tmp = tmp_path_factory.mktemp("mullet_test_constraints") / "learned_constraints.json"
    
    # Store original value
    original_path = _constraints.LEARNED_CONSTRAINTS_PATH
    
    # Apply monkeypatch and reset singleton
    _constraints.LEARNED_CONSTRAINTS_PATH = tmp
    _constraints._registry = None  # force re-init with patched path
    
    yield
    
    # Restore original values
    _constraints.LEARNED_CONSTRAINTS_PATH = original_path
    _constraints._registry = None


@pytest.fixture(autouse=True)
def _neutral_personal_settings(monkeypatch):
    """Run every test with delegation on and the personal settings unset.

    The CLI and the MCP server load the checkout's .env into os.environ on
    import, so without this a local MULLET_DELEGATION=off (or a personal
    setting such as MULLET_NO_AI_ATTRIBUTION) would change what the tests see.
    Tests that need another value set it themselves.
    """
    monkeypatch.setenv("MULLET_DELEGATION", "on")
    for name in ("MULLET_NO_AI_ATTRIBUTION", "MULLET_OPENROUTER_REFERER", "MULLET_GPU_MIN_FREE_MB",
                 "MULLET_RAM_MIN_FREE_MB", "MULLET_GPU_MAX_TEMP_C", "MULLET_MAX_LOAD",
                 "MULLET_GPU_WAIT_MAX_S"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session", autouse=True)
def _isolate_unavailable_models(tmp_path_factory):
    """Keep the record of unavailable models out of the real ~/.mullet."""
    import mullet.core.providers.model_availability as _availability

    original = _availability.STORE_PATH
    _availability.STORE_PATH = tmp_path_factory.mktemp("mullet_test_availability") / "unavailable_models.json"
    yield
    _availability.STORE_PATH = original


@pytest.fixture(autouse=True)
def _restore_mullet_logger():
    """Undo any logging set-up a test does to the "mullet" logger.

    The CLI's setup_logging() sets that logger to WARNING and replaces its
    handlers. Left in place, it hides INFO records from every later test in
    the same worker, so caplog assertions failed or passed depending on test
    order.
    """
    logger = logging.getLogger("mullet")
    saved = (logger.level, list(logger.handlers), logger.propagate, logger.disabled)
    yield
    logger.setLevel(saved[0])
    logger.handlers[:] = saved[1]
    logger.propagate = saved[2]
    logger.disabled = saved[3]


# Tests must never leave files in the repo itself (they have produced
# ./.mullet/memory, ./nano/config.json and a file called "None" before).
_REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
_ROOT_BEFORE: set[str] = set()


def pytest_sessionstart(session):
    if not hasattr(session.config, "workerinput"):
        _ROOT_BEFORE.update(p.name for p in _REPO_ROOT.iterdir())


def pytest_terminal_summary(terminalreporter):
    if hasattr(terminalreporter.config, "workerinput") or not _ROOT_BEFORE:
        return
    new = sorted({p.name for p in _REPO_ROOT.iterdir()} - _ROOT_BEFORE)
    if new:
        terminalreporter.write_line(
            f"WARNING: tests left new files in the repo root: {', '.join(new)}", red=True, bold=True
        )
