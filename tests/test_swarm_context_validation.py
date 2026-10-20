"""
Tests for SwarmContext validation in __post_init__.
"""
import pytest
from mullet.core.orchestration.swarm_types import SwarmContext


def test_defaults_valid():
    """SwarmContext() with no args should not raise."""
    try:
        SwarmContext()
    except Exception as e:
        pytest.fail(f"SwarmContext() raised {type(e).__name__}: {e}")


def test_valid_files_list():
    """SwarmContext(files=["a.py", "b.py"]) should not raise."""
    try:
        SwarmContext(files=["a.py", "b.py"])
    except Exception as e:
        pytest.fail(f"SwarmContext(files=[...]) raised {type(e).__name__}: {e}")


def test_valid_items_list():
    """SwarmContext(items=["x", "y"]) should not raise."""
    try:
        SwarmContext(items=["x", "y"])
    except Exception as e:
        pytest.fail(f"SwarmContext(items=[...]) raised {type(e).__name__}: {e}")


def test_files_not_a_list():
    """SwarmContext(files="a.py") raises TypeError with message containing 'must be list[str]'."""
    with pytest.raises(TypeError, match=r"must be list\[str\]"):
        SwarmContext(files="a.py")


def test_files_contains_non_str():
    """SwarmContext(files=["ok.py", 42]) raises TypeError with message containing 'files[1]' and 'int'."""
    with pytest.raises(TypeError, match="files\\[1\\].*int"):
        SwarmContext(files=["ok.py", 42])


def test_items_not_a_list():
    """SwarmContext(items=123) raises TypeError with message containing 'must be list[str]'."""
    with pytest.raises(TypeError, match=r"must be list\[str\]"):
        SwarmContext(items=123)


def test_items_contains_non_str():
    """SwarmContext(items=[None]) raises TypeError with message containing 'items[0]'."""
    with pytest.raises(TypeError, match="items\\[0\\]"):
        SwarmContext(items=[None])