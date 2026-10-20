"""Safety utilities to guard against broken pipes and invalid UTF-16 surrogates in output streams."""

import sys
import logging
from typing import TextIO

logger = logging.getLogger(__name__)

class _SafeWriter:
    """
    A wrapper for sys.stdout/stderr that catches BrokenPipeError and OSError.
    This prevents crashes when running in daemon/Docker/systemd mode and the
    output pipe is closed unexpectedly.
    """
    def __init__(self, stream: TextIO):
        self._stream = stream

    def write(self, data: str):
        """Write ``data`` to the wrapped stream, swallowing pipe errors."""
        try:
            self._stream.write(data)
        except (BrokenPipeError, OSError) as e:
            logger.warning("safety: broken pipe during write: %s", e)

    def flush(self):
        """Flush the wrapped stream, swallowing pipe errors."""
        try:
            self._stream.flush()
        except (BrokenPipeError, OSError) as e:
            logger.warning("safety: broken pipe during flush: %s", e)

    def __getattr__(self, name):
        return getattr(self._stream, name)

def install_safe_stdio():
    """Replace sys.stdout/sys.stderr with SafeWriter wrappers that swallow
    BrokenPipeError and OSError.

    Useful for daemon / Docker / systemd modes where the output pipes
    might be closed unexpectedly mid-write.
    """
    if not isinstance(sys.stdout, _SafeWriter):
        sys.stdout = _SafeWriter(sys.stdout)
    if not isinstance(sys.stderr, _SafeWriter):
        sys.stderr = _SafeWriter(sys.stderr)

def sanitize_surrogates(text: str) -> str:
    """
    Strip invalid UTF-16 surrogate characters (U+D800..U+DFFF) and null bytes
    from text. This prevents crashes, especially when dealing with text copied
    from sources like Google Docs.
    """
    return (
        text.encode("utf-8", errors="ignore")
        .decode("utf-8")
        .replace("\x00", "")
    )
