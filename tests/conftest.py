import importlib
import logging
import pkgutil

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

try:
    from mullet.server import middleware as _mw_mod
    from mullet.server.middleware import RateLimiter as _RateLimiter
    _HAS_SERVER = True
except ImportError:
    _HAS_SERVER = False


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


@pytest.fixture(scope="session", autouse=True)
def _isolate_routing_history(tmp_path_factory):
    import mullet.core.routing_history as _rh
    
    # Create temporary path for the test session
    tmp = tmp_path_factory.mktemp("mullet_test_routing_history") / ".mullet" / "routing_history.jsonl"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    
    # Store original function
    original_history_path = _rh.history_path
    
    def _patched_history_path(base=None):
        if base is None:
            return tmp
        return original_history_path(base)
    
    # Apply monkeypatch
    _rh.history_path = _patched_history_path
    
    yield
    
    # Restore original function
    _rh.history_path = original_history_path
