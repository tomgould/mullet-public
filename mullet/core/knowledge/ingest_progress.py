"""Tracks progress and calculates metrics for knowledge ingestion processes."""

import time

class IngestProgress:
    """Track progress and calculate metrics for knowledge ingestion."""

    def __init__(self):
        self._done = 0
        self._total = 0
        self._start_time = time.time()
        self._last_update_time = time.time()
        self._rate = 0.0 # items per second

    def update(self, done, total):
        """Update progress with current done and total counts."""
        self._done = done
        self._total = total
        current_time = time.time()
        
        total_elapsed = current_time - self._start_time
        if total_elapsed > 0:
            self._rate = self._done / total_elapsed
        else:
            self._rate = 0.0

        self._last_update_time = current_time


    @property
    def percent(self):
        """Return completion percentage."""
        if self._total == 0:
            return 0.0
        return (self._done / self._total) * 100

    @property
    def eta_seconds(self):
        """Return estimated time remaining in seconds."""
        if self._rate == 0.0 or self._done == self._total:
            return 0.0
        remaining_items = self._total - self._done
        return remaining_items / self._rate

