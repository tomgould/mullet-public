"""G4 — escalation helper must respect the user's provider allowlist."""
import pytest
from mullet.core.providers import openai_compat as oc


@pytest.fixture(autouse=True)
def _reset():
    with oc._consecutive_failures_lock:
        oc._consecutive_failures.clear()
    yield
    with oc._consecutive_failures_lock:
        oc._consecutive_failures.clear()


def _set(name, count):
    with oc._consecutive_failures_lock:
        oc._consecutive_failures[name] = count


def test_legacy_true_when_over_threshold():
    _set("local", oc._ESCALATION_THRESHOLD)
    assert oc.is_escalation_recommended("local") is True


def test_false_when_allowlist_local_only():
    _set("local", oc._ESCALATION_THRESHOLD)
    assert oc.is_escalation_recommended("local", allowed={"local"}) is False


def test_true_when_cloud_in_allowlist():
    _set("local", oc._ESCALATION_THRESHOLD)
    assert oc.is_escalation_recommended("local", allowed={"kimi", "local"}) is True


def test_false_when_not_over_threshold():
    _set("local", 0)
    assert oc.is_escalation_recommended("local", allowed={"kimi"}) is False


def test_false_when_allowlist_empty():
    _set("local", oc._ESCALATION_THRESHOLD)
    assert oc.is_escalation_recommended("local", allowed=frozenset()) is False
