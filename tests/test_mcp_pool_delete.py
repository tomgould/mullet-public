"""Tests for handle_delete_pool validation logic."""


def test_handle_delete_pool_missing_pool_name():
    from mullet.server.mcp_knowledge import handle_delete_pool
    result = handle_delete_pool({})
    assert "Error" in result and "pool_name" in result.lower()


def test_handle_delete_pool_nonexistent_pool():
    from mullet.server.mcp_knowledge import handle_delete_pool
    result = handle_delete_pool({"pool_name": "pool_that_does_not_exist_xyz"})
    assert "Error" in result and "not found" in result.lower()


def test_handle_delete_pool_requires_confirm():
    from mullet.server.mcp_knowledge import handle_delete_pool
    # Pool doesn't need to exist — the confirm check fires even for existing pools
    # but first the existence check fires. Test with confirm=False and a non-existent pool
    # — we get "not found" before the confirm check.
    # Test the confirm guard by using mock to make pool exist.
    from unittest.mock import patch
    from pathlib import Path
    with patch("mullet.server.mcp_knowledge._get_pools_dir", return_value=Path("/fake")):
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                result = handle_delete_pool({"pool_name": "mypool", "confirm": False})
    assert "Refusing" in result or "confirm" in result.lower()
