import time
from mullet.core.agent.trace import AgentTracer, TraceEvent, SpanContext

def test_trace_event_creation():
    event = TraceEvent("test_event", time.time(), 100.0, {"key": "value"})
    assert event.event_type == "test_event"
    assert isinstance(event.timestamp, float)
    assert event.duration_ms == 100.0
    assert event.metadata == {"key": "value"}

def test_agent_tracer_initialization():
    tracer = AgentTracer()
    assert isinstance(tracer.get_timeline(), list)
    assert len(tracer.get_timeline()) == 0

def test_log_tool_call():
    tracer = AgentTracer()
    start_time = time.time()
    tracer.log_tool_call("test_tool", {"arg1": 1}, "success", 50.0)
    timeline = tracer.get_timeline()
    assert len(timeline) == 1
    event = timeline[0]
    assert event.event_type == "tool_call"
    assert event.duration_ms == 50.0
    assert event.metadata["tool_name"] == "test_tool"
    assert event.metadata["args"] == {"arg1": 1}
    assert event.metadata["result"] == "success"
    assert event.timestamp >= start_time

def test_log_llm_call():
    tracer = AgentTracer()
    start_time = time.time()
    tracer.log_llm_call("gpt-4", 100, 200, 150.0)
    timeline = tracer.get_timeline()
    assert len(timeline) == 1
    event = timeline[0]
    assert event.event_type == "llm_call"
    assert event.duration_ms == 150.0
    assert event.metadata["model"] == "gpt-4"
    assert event.metadata["tokens_in"] == 100
    assert event.metadata["tokens_out"] == 200
    assert event.timestamp >= start_time

def test_get_timeline_order():
    tracer = AgentTracer()
    tracer.log_tool_call("tool1", {}, "", 10.0)
    time.sleep(0.01) # Ensure timestamps are different
    tracer.log_llm_call("llm1", 10, 10, 20.0)
    timeline = tracer.get_timeline()
    assert len(timeline) == 2
    assert timeline[0].event_type == "tool_call"
    assert timeline[1].event_type == "llm_call"
    assert timeline[0].timestamp < timeline[1].timestamp

def test_summary_empty_tracer():
    tracer = AgentTracer()
    summary = tracer.summary()
    assert summary["total_duration_ms"] == 0.0
    assert summary["tool_calls_count"] == 0
    assert summary["llm_calls_count"] == 0
    assert summary["total_tokens"] == 0
    assert summary["error_count"] == 0

def test_summary_with_data():
    tracer = AgentTracer()
    tracer.log_tool_call("tool1", {}, "", 10.0)
    tracer.log_llm_call("llm1", 10, 20, 30.0)
    tracer.log_tool_call("tool2", {}, "", 5.0)
    tracer.log_llm_call("llm2", 5, 10, 15.0)
    summary = tracer.summary()
    assert summary["total_duration_ms"] == 60.0 # 10 + 30 + 5 + 15
    assert summary["tool_calls_count"] == 2
    assert summary["llm_calls_count"] == 2
    assert summary["total_tokens"] == 45 # (10+20) + (5+10)
    assert summary["error_count"] == 0

def test_start_span():
    tracer = AgentTracer()
    span = tracer.start_span("test_span")
    assert isinstance(span, SpanContext)
    assert span.name == "test_span"
    assert isinstance(span.start_time, float)
    assert len(span.events) == 0

def test_log_error():
    tracer = AgentTracer()
    tracer.log_error("Something went wrong", {"code": 500})
    timeline = tracer.get_timeline()
    assert len(timeline) == 1
    event = timeline[0]
    assert event.event_type == "error"
    assert event.metadata["message"] == "Something went wrong"
    assert event.metadata["details"] == {"code": 500}
    assert event.duration_ms is None

def test_summary_with_errors():
    tracer = AgentTracer()
    tracer.log_tool_call("tool1", {}, "", 10.0)
    tracer.log_error("Error 1")
    tracer.log_llm_call("llm1", 10, 20, 30.0)
    tracer.log_error("Error 2", {"severity": "high"})
    summary = tracer.summary()
    assert summary["total_duration_ms"] == 40.0
    assert summary["tool_calls_count"] == 1
    assert summary["llm_calls_count"] == 1
    assert summary["total_tokens"] == 30
    assert summary["error_count"] == 2

def test_log_tool_call_no_args_result():
    tracer = AgentTracer()
    tracer.log_tool_call("simple_tool", {}, None, 20.0)
    timeline = tracer.get_timeline()
    assert len(timeline) == 1
    event = timeline[0]
    assert event.metadata["args"] == {}
    assert event.metadata["result"] is None

def test_log_llm_call_zero_tokens():
    tracer = AgentTracer()
    tracer.log_llm_call("zero_model", 0, 0, 5.0)
    timeline = tracer.get_timeline()
    assert len(timeline) == 1
    event = timeline[0]
    assert event.metadata["tokens_in"] == 0
    assert event.metadata["tokens_out"] == 0
    summary = tracer.summary()
    assert summary["total_tokens"] == 0

def test_multiple_spans_not_affecting_timeline_directly():
    tracer = AgentTracer()
    span1 = tracer.start_span("span1")
    tracer.log_tool_call("tool_in_span1", {}, "", 10.0)
    span2 = tracer.start_span("span2")
    tracer.log_llm_call("llm_in_span2", 1, 1, 5.0)
    timeline = tracer.get_timeline()
    assert len(timeline) == 2
    assert timeline[0].event_type == "tool_call"
    assert timeline[1].event_type == "llm_call"
    # Spans themselves don't add events to the main timeline in this implementation

def test_summary_with_only_errors():
    tracer = AgentTracer()
    tracer.log_error("Critical Error")
    summary = tracer.summary()
    assert summary["total_duration_ms"] == 0.0
    assert summary["tool_calls_count"] == 0
    assert summary["llm_calls_count"] == 0
    assert summary["total_tokens"] == 0
    assert summary["error_count"] == 1
