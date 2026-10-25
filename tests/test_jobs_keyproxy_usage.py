"""The key proxy reads the usage a provider reports and enforces an attempt's allowance on it."""

import http.client
import json
import shutil
import socket
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mullet.jobs.executors.keyproxy import KeyProxy
from mullet.jobs.executors.usage_meter import MAX_JSON_BYTES, ResponseMeter, parse_usage

OPENROUTER_USAGE = {
    "prompt_tokens": 1000, "completion_tokens": 200, "cost": 0.0125,
    "prompt_tokens_details": {"cached_tokens": 800, "cache_write_tokens": 50},
    "completion_tokens_details": {"reasoning_tokens": 120},
}
MOONSHOT_USAGE = {"prompt_tokens": 2000, "completion_tokens": 100, "cached_tokens": 1500}


class _Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: list[dict] = []
    reply: dict = {}

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        type(self).seen.append({"path": self.path, "headers": dict(self.headers)})
        mode = type(self).reply.get("mode", "json")
        usage = type(self).reply.get("usage")
        if mode == "error":
            payload = json.dumps({"error": {"message": "nope"}, "usage": usage}).encode()
            self.send_response(500)
        elif mode == "sse":
            payload = (b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                       + b"data: " + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\n"
                       b"data: [DONE]\n\n")
            self.send_response(200)
        else:
            payload = json.dumps({"choices": [], "usage": usage}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def upstream():
    _Upstream.seen = []
    _Upstream.reply = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/api/v1"
    server.shutdown()


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: Path):
        super().__init__("sandbox")
        self.unix_path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.unix_path)


@pytest.fixture
def make_proxy(upstream):
    dirs, started = [], []

    def make(**kwargs):
        d = Path(tempfile.mkdtemp(prefix="kp", dir="/tmp"))
        dirs.append(d)
        p = KeyProxy(d / "s", upstream, "sk-test-key", **kwargs).start()
        started.append(p)
        return p

    yield make
    for p in started:
        p.stop()
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


def post(proxy):
    """POST once and return (status, body), after the proxy has finished metering the response.

    The proxy meters in the handler's `finally`, which can run just after the client has read the last byte.
    """
    before = proxy.metered
    conn = _UnixConnection(proxy.socket_path)
    conn.request("POST", "/api/v1/chat/completions", body=b"{}", headers={"Authorization": "Bearer x"})
    resp = conn.getresponse()
    result = resp.status, resp.read()
    for _ in range(200):
        if proxy.metered > before or resp.status != 200:
            break
        time.sleep(0.01)
    return result


def test_a_stated_cost_is_taken_as_it_is_and_tokens_do_not_overlap(make_proxy):
    _Upstream.reply = {"usage": OPENROUTER_USAGE}
    p = make_proxy(pricer=lambda tokens: 99.0)
    assert post(p)[0] == 200
    summary = p.usage_summary()
    assert summary["proxy_cost_usd"] == pytest.approx(0.0125)
    assert summary["proxy_provider_cost_usd"] == pytest.approx(0.0125)
    assert summary["proxy_tokens"] == {"input": 150, "output": 80, "reasoning": 120,
                                       "cache_read": 800, "cache_write": 50}


def test_a_stream_with_a_final_usage_event_is_read(make_proxy):
    _Upstream.reply = {"mode": "sse", "usage": OPENROUTER_USAGE}
    p = make_proxy()
    post(p)
    assert p.usage_summary()["proxy_cost_usd"] == pytest.approx(0.0125)


def test_tokens_without_a_stated_cost_are_priced_by_the_pricer(make_proxy):
    _Upstream.reply = {"usage": MOONSHOT_USAGE}
    seen = []

    def pricer(tokens):
        seen.append(tokens)
        return tokens["input"] * 1e-6 + tokens["cache_read"] * 1e-7 + tokens["output"] * 4e-6

    p = make_proxy(pricer=pricer)
    post(p)
    assert seen[0] == {"input": 500, "output": 100, "reasoning": 0, "cache_read": 1500, "cache_write": 0}
    summary = p.usage_summary()
    assert summary["proxy_cost_usd"] == pytest.approx(0.00105)
    assert summary["proxy_provider_cost_usd"] == 0


def test_tokens_with_no_cost_and_no_price_are_counted_as_unpriced(make_proxy):
    _Upstream.reply = {"usage": MOONSHOT_USAGE}
    p = make_proxy()
    post(p)
    summary = p.usage_summary()
    assert summary["proxy_cost_usd"] == 0 and summary["proxy_unpriced"] == 1


def test_an_error_response_is_not_metered(make_proxy):
    _Upstream.reply = {"mode": "error", "usage": OPENROUTER_USAGE}
    p = make_proxy()
    assert post(p)[0] == 500
    assert p.usage_summary()["proxy_metered"] == 0


def test_the_attempt_is_stopped_once_its_allowance_is_spent(make_proxy):
    _Upstream.reply = {"usage": OPENROUTER_USAGE}
    p = make_proxy(allowance_usd=0.01)
    assert post(p)[0] == 200  # one request may overshoot, the next may not
    status, body = post(p)
    assert status == 402 and b"allowance" in body
    assert p.out_of_allowance and "allowance" in (p.capped or "")
    assert len(_Upstream.seen) == 1


def test_no_allowance_means_no_limit(make_proxy):
    _Upstream.reply = {"usage": OPENROUTER_USAGE}
    p = make_proxy()
    assert [post(p)[0] for _ in range(3)] == [200, 200, 200]
    assert not p.out_of_allowance


def test_upstream_is_asked_for_plain_bodies(make_proxy):
    _Upstream.reply = {"usage": MOONSHOT_USAGE}
    conn = _UnixConnection(make_proxy().socket_path)
    conn.request("POST", "/api/v1/chat/completions", body=b"{}",
                 headers={"Authorization": "Bearer x", "Accept-Encoding": "gzip, br"})
    conn.getresponse().read()
    sent = {k.lower(): v for k, v in _Upstream.seen[-1]["headers"].items()}
    assert sent["accept-encoding"] == "identity"


def test_parse_usage_ignores_junk_and_negative_numbers():
    usage = parse_usage({"prompt_tokens": -5, "completion_tokens": True, "cost": -1})
    assert usage.tokens() == {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0}
    assert usage.cost is None


def test_the_meter_finds_usage_in_the_tail_of_a_huge_stream():
    meter = ResponseMeter()
    meter.feed(b"data: " + b"x" * (MAX_JSON_BYTES + 10) + b"\n\n")
    meter.feed(b"data: " + json.dumps({"usage": {"prompt_tokens": 7, "completion_tokens": 3}}).encode() + b"\n\n")
    meter.feed(b"data: [DONE]\n\n")
    assert meter.usage().tokens()["input"] == 7


def test_a_body_with_no_usage_gives_none():
    meter = ResponseMeter()
    meter.feed(b'{"error": "nope"}')
    assert meter.usage() is None


class _FakeProxy:
    def __init__(self, cost, provider_cost):
        self._summary = {"proxy_cost_usd": cost, "proxy_provider_cost_usd": provider_cost,
                         "proxy_metered": 3, "proxy_unpriced": 0, "proxy_tokens": {}}

    def usage_summary(self):
        return dict(self._summary)


def test_a_cost_opencode_under_reports_is_raised_to_the_providers_figure():
    from mullet.jobs.executors.opencode import OpencodeExecutor

    usage = {"total_cost_usd": 3.66, "cost_reported": True}  # GLM 5.3's cache reads were priced near zero
    OpencodeExecutor._take_provider_cost(usage, _FakeProxy(11.8, 11.8))
    assert usage["total_cost_usd"] == 11.8 and usage["cost_source"] == "provider"


def test_a_higher_opencode_cost_is_kept():
    from mullet.jobs.executors.opencode import OpencodeExecutor

    usage = {"total_cost_usd": 2.0, "cost_reported": True}
    OpencodeExecutor._take_provider_cost(usage, _FakeProxy(1.5, 0.0))
    assert usage["total_cost_usd"] == 2.0 and "cost_source" not in usage


def test_a_configured_price_is_named_as_the_source_when_the_provider_states_no_cost():
    from mullet.jobs.executors.opencode import OpencodeExecutor

    usage = {"total_cost_usd": 0.0, "cost_reported": False}
    OpencodeExecutor._take_provider_cost(usage, _FakeProxy(4.2, 0.0))
    assert usage["total_cost_usd"] == 4.2 and usage["cost_source"] == "configured price"
    assert usage["cost_reported"] is True


def test_a_refused_request_still_gets_its_answer_when_it_carries_a_body(make_proxy):
    """The proxy reads a small body before closing, so the client sees the 400 and not a reset."""
    p = make_proxy()
    body = b"x" * 100_000
    for _ in range(20):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(str(p.socket_path))
        sock.sendall(b"POST /api/v1/../../admin HTTP/1.1\r\nHost: s\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        reply = b""
        while chunk := sock.recv(65536):
            reply += chunk
        sock.close()
        assert reply.split(b"\r\n", 1)[0].split()[1] == b"400"
    assert _Upstream.seen == []


def test_progress_records_carry_the_proxys_running_cost(tmp_path):
    """A runner killed mid-attempt keeps what the provider has charged: the ledger gets the proxy's figure."""
    from mullet.jobs.executors.opencode import OpencodeExecutor, _Stream

    executor = OpencodeExecutor(tmp_path / "logs", "fakeai/m1")
    state = _Stream(cost=3.66, cost_reported=True)
    assert executor._usage(state, 1.0)["total_cost_usd"] == 3.66
    executor._proxy = type("P", (), {"spent_usd": 11.8})()
    usage = executor._usage(state, 1.0)
    assert usage["total_cost_usd"] == 11.8 and usage["cost_reported"] is True
    zero = executor._usage(_Stream(), 1.0)
    assert zero["total_cost_usd"] == 11.8 and zero["cost_reported"] is True
