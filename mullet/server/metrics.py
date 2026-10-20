"""Prometheus-compatible metrics for Mullet server.

Exposes /metrics endpoint with counters and histograms for:
- Request counts by endpoint and status
- Token usage (input, output, total)
- LLM call latency
- Tool call counts and durations
- Active tasks gauge

No external dependencies — generates Prometheus text format directly.
"""

import time
import threading
from collections import defaultdict


class Metrics:
    """Thread-safe metrics collector."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = defaultdict(float)
        self._start_time = time.time()

    def inc(self, name: str, value: float = 1.0, labels: dict | None = None):
        """Increment a counter."""
        key = _label_key(name, labels)
        with self._lock:
            self._counters[key] += value

    def observe(self, name: str, value: float, labels: dict | None = None):
        """Record an observation for a histogram."""
        key = _label_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)
            # Keep last 1000 observations per metric
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]

    def set_gauge(self, name: str, value: float, labels: dict | None = None):
        """Set a gauge value."""
        key = _label_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def format_prometheus(self) -> str:
        """Format all metrics in Prometheus text exposition format."""
        lines = []
        lines.append("# Mullet metrics")
        lines.append(f"mullet_uptime_seconds {time.time() - self._start_time:.0f}")
        lines.append("")

        with self._lock:
            # Counters
            for key, value in sorted(self._counters.items()):
                lines.append(f"{key} {value}")

            # Gauges
            for key, value in sorted(self._gauges.items()):
                lines.append(f"{key} {value}")

            # Histograms (emit count, sum, and quantiles)
            for key, values in sorted(self._histograms.items()):
                if not values:
                    continue
                count = len(values)
                total = sum(values)
                avg = total / count if count else 0
                sorted_vals = sorted(values)
                p50 = sorted_vals[int(count * 0.5)] if count else 0
                p95 = sorted_vals[int(count * 0.95)] if count else 0
                p99 = sorted_vals[int(count * 0.99)] if count else 0

                lines.append(f"{key}_count {count}")
                lines.append(f"{key}_sum {total:.3f}")
                lines.append(f"{key}_avg {avg:.3f}")
                lines.append(f'{key}{{quantile="0.5"}} {p50:.3f}')
                lines.append(f'{key}{{quantile="0.95"}} {p95:.3f}')
                lines.append(f'{key}{{quantile="0.99"}} {p99:.3f}')

        lines.append("")
        return "\n".join(lines)


# Global metrics instance
_metrics = Metrics()


def get_metrics() -> Metrics:
    """Get the global metrics instance."""
    return _metrics


def _label_key(name: str, labels: dict | None) -> str:
    """Build a Prometheus metric key with labels."""
    if not labels:
        return name
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{label_str}}}"
