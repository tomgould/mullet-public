
import logging
import os
from logging.handlers import RotatingFileHandler
import pytest

# Temporarily modify the sys.path to import the module under test
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mullet.core.agent.error_log import setup_error_log, get_error_log_path

@pytest.fixture(autouse=True)
def reset_logging_handlers():
    """Fixture to clean up logging handlers after each test."""
    yield
    # Remove all handlers added by the tests
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
    logging.getLogger().setLevel(logging.NOTSET) # Reset level

def test_get_error_log_path_with_tmp_path(tmp_path):
    """Test that get_error_log_path returns the correct path and creates directories."""
    # Override os.path.expanduser for this test
    original_expanduser = os.path.expanduser
    os.path.expanduser = lambda path: str(tmp_path)

    try:
        log_path = get_error_log_path()
        expected_dir = tmp_path / ".mullet" / "logs"
        expected_path = expected_dir / "errors.log"

        assert log_path == str(expected_path)
        assert os.path.isdir(expected_dir)
        # The file itself is not created by get_error_log_path, only the directory
        assert not os.path.exists(expected_path)
    finally:
        os.path.expanduser = original_expanduser # Restore original

def test_setup_error_log_creates_handler(tmp_path):
    """Test that setup_error_log adds a RotatingFileHandler to the root logger."""
    original_expanduser = os.path.expanduser
    os.path.expanduser = lambda path: str(tmp_path)

    try:
        setup_error_log()
        root_logger = logging.getLogger()
        
        # Check if a RotatingFileHandler is present
        assert any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers)
        
        # Check handler's level and formatter
        handler = next(h for h in root_logger.handlers if isinstance(h, RotatingFileHandler))
        assert handler.level == logging.WARNING
        assert handler.formatter is not None
        assert "errors.log" in handler.baseFilename
    finally:
        os.path.expanduser = original_expanduser

def test_log_file_created_on_warning(tmp_path):
    """Test that a log file is created and contains WARNING messages."""
    original_expanduser = os.path.expanduser
    os.path.expanduser = lambda path: str(tmp_path)

    try:
        setup_error_log()
        log_path = get_error_log_path()
        
        test_message = "This is a test warning."
        logging.warning(test_message)
        
        # Ensure the log is flushed
        for handler in logging.getLogger().handlers:
            handler.flush()

        assert os.path.exists(log_path)
        with open(log_path, 'r') as f:
            content = f.read()
            assert test_message in content
            assert "WARNING" in content
            assert "INFO" not in content # Info messages should not be logged
        
        logging.info("This is an info message.")
        for handler in logging.getLogger().handlers:
            handler.flush()
        with open(log_path, 'r') as f:
            content = f.read()
            assert "This is an info message." not in content
    finally:
        os.path.expanduser = original_expanduser

def test_error_log_main_block_runs(tmp_path):
    """The if __name__ == '__main__' block in error_log.py executes without error."""
    import runpy
    original_expanduser = os.path.expanduser
    os.path.expanduser = lambda path: str(tmp_path)
    try:
        runpy.run_path(
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../mullet/core/agent/error_log.py")
            ),
            run_name="__main__",
        )
    finally:
        os.path.expanduser = original_expanduser


def test_duplicate_setup_is_safe(tmp_path):
    """Test that calling setup_error_log multiple times does not add duplicate handlers."""
    original_expanduser = os.path.expanduser
    os.path.expanduser = lambda path: str(tmp_path)

    try:
        setup_error_log()
        root_logger = logging.getLogger()
        initial_handlers_count = len(root_logger.handlers)
        
        setup_error_log() # Call again
        
        # Ensure no new handlers are added if a RotatingFileHandler for errors.log already exists
        # This checks for the specific behavior of the original setup_error_log which might add a new handler
        # if not careful. The current implementation does add a new handler each time.
        # Let's refine the test to check if it's safe in the sense of not breaking,
        # and acknowledge it might add multiple handlers if not explicitly prevented in the source.
        # For this test, we'll assert that the number of RotatingFileHandlers for errors.log
        # does not increase if the file handler is already present.
        
        # A more robust setup_error_log would check for existing handlers.
        # Let's assume the current implementation adds a new one.
        # The prompt says "duplicate setup is safe", which implies no adverse effects.
        # Adding multiple handlers is generally safe but inefficient.
        
        # After modifying setup_error_log to be idempotent, there should only be one handler.
        error_log_handlers = [h for h in root_logger.handlers if isinstance(h, RotatingFileHandler) and "errors.log" in h.baseFilename]
        assert len(error_log_handlers) == 1
        
        # Check that no errors occur
        try:
            logging.warning("Another warning after duplicate setup.")
        except Exception as e:
            pytest.fail(f"Logging failed after duplicate setup: {e}")
            
    finally:
        os.path.expanduser = original_expanduser
