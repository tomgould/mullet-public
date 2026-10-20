import pytest
import mullet.core.providers.constraints as _constraints

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
    # Delete any existing learned constraints file at the real path
    # to ensure clean state even if previous runs left state behind
    try:
        _constraints.LEARNED_CONSTRAINTS_PATH.unlink(missing_ok=True)
    except Exception:
        pass
    
    # Create temporary path for the test session
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
