"""Provides timing and performance profiling for agent operations."""
import time
from typing import Dict

class Profiler:
    """Simple timer-based profiler for agent operations."""
    def __init__(self):
        """Initialize empty timer and duration stores."""
        self._timers: Dict[str, float] = {}
        self._durations: Dict[str, float] = {}

    def start_timer(self, name: str):
        """Start a named timer."""
        if name in self._timers:
            raise ValueError(f'Timer \'{name}\' already started.')
        self._timers[name] = time.monotonic()

    def stop_timer(self, name: str):
        """Stop a named timer and accumulate its duration."""
        if name not in self._timers:
            raise ValueError(f'Timer \'{name}\' not started.')
        
        end_time = time.monotonic()
        duration = end_time - self._timers[name]
        self._durations[name] = self._durations.get(name, 0.0) + duration
        del self._timers[name] # Allow restarting the same timer

    def report(self) -> Dict[str, float]:
        """Return accumulated durations, finalizing any running timers."""
        # If any timers are still running, report their current duration up to the point of report call
        current_time = time.monotonic()
        for name, start_time in self._timers.items():
            self._durations[name] = self._durations.get(name, 0.0) + (current_time - start_time)
            # Note: This will effectively stop and report running timers. 
            # If a timer needs to continue after report, it should be re-started.
            
        # Clear running timers after reporting them as 'stopped' for this report
        self._timers.clear()
        
        return self._durations

    def reset(self):
        """Clear all timers and accumulated durations."""
        self._timers.clear()
        self._durations.clear()

