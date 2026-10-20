import sys
from unittest.mock import MagicMock
from mullet.core.agent.safety import sanitize_surrogates, install_safe_stdio, _SafeWriter

def test_sanitize_surrogates_strips_null_bytes():
    assert sanitize_surrogates("hello\x00world") == "helloworld"

def test_sanitize_surrogates_strips_surrogates():
    # Invalid UTF-16 surrogate characters
    assert sanitize_surrogates("hello\ud800world") == "helloworld"
    assert sanitize_surrogates("hello\udfffworld") == "helloworld"
    assert sanitize_surrogates("hello\ud800\udfffworld") == "helloworld"

def test_sanitize_surrogates_preserves_normal_text():
    assert sanitize_surrogates("hello world") == "hello world"
    assert sanitize_surrogates("123!@#$") == "123!@#$"

def test_sanitize_surrogates_handles_empty_string():
    assert sanitize_surrogates("") == ""

def test_install_safe_stdio_does_not_crash():
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Use MagicMock to prevent actual replacement of sys.stdout/stderr
    sys.stdout = MagicMock()
    sys.stderr = MagicMock()

    try:
        install_safe_stdio()
        # Verify that sys.stdout and sys.stderr are now _SafeWriter instances
        assert isinstance(sys.stdout, _SafeWriter)
        assert isinstance(sys.stderr, _SafeWriter)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def test_safe_writer_write_normal():
    mock_stream = MagicMock()
    writer = _SafeWriter(mock_stream)
    writer.write("hello")
    mock_stream.write.assert_called_once_with("hello")


def test_safe_writer_write_swallows_broken_pipe():
    mock_stream = MagicMock()
    mock_stream.write.side_effect = BrokenPipeError("pipe broken")
    writer = _SafeWriter(mock_stream)
    writer.write("data")  # must not raise


def test_safe_writer_write_swallows_oserror():
    mock_stream = MagicMock()
    mock_stream.write.side_effect = OSError("os error")
    writer = _SafeWriter(mock_stream)
    writer.write("data")  # must not raise


def test_safe_writer_flush_normal():
    mock_stream = MagicMock()
    writer = _SafeWriter(mock_stream)
    writer.flush()
    mock_stream.flush.assert_called_once()


def test_safe_writer_flush_swallows_broken_pipe():
    mock_stream = MagicMock()
    mock_stream.flush.side_effect = BrokenPipeError("pipe broken")
    writer = _SafeWriter(mock_stream)
    writer.flush()  # must not raise


def test_safe_writer_flush_swallows_oserror():
    mock_stream = MagicMock()
    mock_stream.flush.side_effect = OSError("os error")
    writer = _SafeWriter(mock_stream)
    writer.flush()  # must not raise


def test_safe_writer_getattr_delegates():
    mock_stream = MagicMock()
    mock_stream.encoding = "utf-8"
    writer = _SafeWriter(mock_stream)
    assert writer.encoding == "utf-8"
