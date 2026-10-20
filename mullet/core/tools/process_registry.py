"""Thread-safe registry for tracking active named processes."""
import threading

class ProcessRegistry:
    """Track and manage active named processes."""

    def __init__(self):
        self._processes = {}
        self._lock = threading.Lock()

    def register(self, pid, name):
        """Associate a process ID with a name."""
        with self._lock:
            self._processes[pid] = name

    def unregister(self, pid):
        """Remove a process from tracking."""
        with self._lock:
            if pid in self._processes:
                del self._processes[pid]

    def list_active(self):
        """Return a list of currently tracked processes."""
        with self._lock:
            return list(self._processes.items())

    def cleanup_all(self):
        """Clear all tracked processes."""
        with self._lock:
            self._processes.clear()

