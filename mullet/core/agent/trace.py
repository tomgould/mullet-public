"""Lightweight in-process tracer (spans + events) for agent execution flow."""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class TraceEvent:
    """A single timed event within a trace timeline."""
    event_type: str
    timestamp: float
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SpanContext:
    """Context for an active trace span."""
    name: str
    start_time: float
    events: List[TraceEvent]

class AgentTracer:
    """Collects and summarizes execution events for an agent run."""
    def __init__(self):
        self._timeline: List[TraceEvent] = []
        self._spans: Dict[str, SpanContext] = {} # Not used in this implementation, but good for future expansion

    def start_span(self, name: str) -> SpanContext:
        """Begin a new named trace span."""
        start_time = time.time()
        span = SpanContext(name=name, start_time=start_time, events=[])
        self._spans[name] = span # Store span, though not directly used for timeline in this version
        return span

    def _log_event(self, event_type: str, duration_ms: Optional[float] = None, **kwargs):
        self._timeline.append(TraceEvent(
            event_type=event_type,
            timestamp=time.time(),
            duration_ms=duration_ms,
            metadata=kwargs
        ))

    def log_tool_call(self, name: str, args: Dict[str, Any], result: Any, duration: float):
        """Record a tool invocation and its result."""
        self._log_event(
            "tool_call",
            duration_ms=duration,
            tool_name=name,
            args=args,
            result=result
        )

    def log_llm_call(self, model: str, tokens_in: int, tokens_out: int, duration: float):
        """Record an LLM call with token usage and latency."""
        self._log_event(
            "llm_call",
            duration_ms=duration,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out
        )

    def log_error(self, message: str, details: Dict[str, Any] = None):
        """Record an error event with optional details."""
        self._log_event(
            "error",
            message=message,
            details=details if details is not None else {}
        )

    def get_timeline(self) -> List[TraceEvent]:
        """Return the ordered list of recorded trace events."""
        return self._timeline

    def summary(self) -> Dict[str, Any]:
        """Compute aggregate statistics from the recorded events."""
        total_duration = 0.0
        tool_calls_count = 0
        llm_calls_count = 0
        total_tokens = 0
        error_count = 0

        for event in self._timeline:
            if event.duration_ms is not None:
                total_duration += event.duration_ms

            if event.event_type == "tool_call":
                tool_calls_count += 1
            elif event.event_type == "llm_call":
                llm_calls_count += 1
                total_tokens += event.metadata.get("tokens_in", 0) + event.metadata.get("tokens_out", 0)
            elif event.event_type == "error":
                error_count += 1

        return {
            "total_duration_ms": total_duration,
            "tool_calls_count": tool_calls_count,
            "llm_calls_count": llm_calls_count,
            "total_tokens": total_tokens,
            "error_count": error_count
        }
