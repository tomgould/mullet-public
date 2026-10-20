"""Test the __main__ and __init__ modules."""

import runpy
import subprocess
import sys
from unittest.mock import patch


def test_python_m_mullet():
    """python -m mullet --version should work."""
    result = subprocess.run(
        [sys.executable, "-m", "mullet", "--version"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "mullet" in result.stdout


def test_mullet_version():
    """mullet.__version__ should be set."""
    import mullet
    assert mullet.__version__
    assert "." in mullet.__version__


def test_main_block_calls_main():
    """Line 6: `if __name__ == '__main__': main()` executes via runpy."""
    with patch("mullet.cli.main.main") as mock_main:
        runpy.run_module("mullet", run_name="__main__", alter_sys=True)
    mock_main.assert_called_once()
