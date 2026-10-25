"""Tests for the key proxy: a fake upstream on loopback, clients on the proxy's Unix socket."""

import http.client
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mullet.jobs.executors.keyproxy import KeyProxy, ProxyError

KEY = "sk-or-real-key-0123456789abcdef"


class _Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: list[dict] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        type(self).seen.append({"path": self.path, "headers": dict(self.headers), "body": body.decode()})
        if self.path.endswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i, pause in ((1, 0), (2, 1.5)):
                time.sleep(pause)
                data = f"data: {i}\n\n".encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            return
        payload = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST


@pytest.fixture
def upstream():
    _Upstream.seen = []
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
def proxy(tmp_path, upstream):
    def start(key=KEY, header="bearer", **kwargs):
        p = KeyProxy(tmp_path / "s", upstream, key, header=header, log_path=tmp_path / "proxy.log", **kwargs)
        return p.start()
    started = []
    yield lambda **kw: started.append(start(**kw)) or started[-1]
    for p in started:
        p.stop()


def call(proxy, method="POST", path="/api/v1/chat/completions", body=b"{}", headers=None):
    conn = _UnixConnection(proxy.socket_path)
    conn.request(method, path, body=body, headers={"Authorization": "Bearer mullet-dummy", **(headers or {})})
    resp = conn.getresponse()
    return resp.status, resp.read()


def test_the_real_key_replaces_whatever_the_sandbox_sent(proxy):
    p = proxy()
    status, body = call(p, headers={"x-api-key": "planted", "X-Custom": "kept"})
    assert status == 200 and json.loads(body) == {"ok": True}
    seen = _Upstream.seen[-1]
    assert seen["headers"]["Authorization"] == f"Bearer {KEY}"
    assert "x-api-key" not in {k.lower() for k in seen["headers"]}
    assert seen["headers"]["X-Custom"] == "kept" and seen["path"] == "/api/v1/chat/completions"


@pytest.mark.parametrize("style, header", [("x-api-key", "x-api-key"), ("x-goog-api-key", "x-goog-api-key")])
def test_provider_specific_key_headers(proxy, style, header):
    call(proxy(header=style))
    headers = {k.lower(): v for k, v in _Upstream.seen[-1]["headers"].items()}
    assert headers[header] == KEY and "authorization" not in headers


def test_without_a_key_the_requests_own_credentials_pass(proxy):
    call(proxy(key=None))
    assert _Upstream.seen[-1]["headers"]["Authorization"] == "Bearer mullet-dummy"


@pytest.mark.parametrize("method, path, status", [
    ("CONNECT", "evil.example:443", 405),
    ("GET", "http://evil.example/x", 400),
    ("GET", "//evil.example/x", 403),  # read as the path /evil.example/x
    ("POST", "/other/v1/chat/completions", 403),
    ("POST", "/api/v1x/chat", 403),
])
def test_anything_but_the_providers_api_is_refused(proxy, method, path, status):
    p = proxy()
    assert call(p, method=method, path=path)[0] == status
    assert _Upstream.seen == []


def test_oversized_requests_are_refused(proxy):
    p = proxy(max_body=1000)
    assert call(p, body=b"x" * 2000)[0] == 413
    assert _Upstream.seen == []


def test_streamed_responses_are_passed_on_as_they_arrive(proxy):
    p = proxy()
    conn = _UnixConnection(p.socket_path)
    conn.request("POST", "/api/v1/stream", body=b"{}")
    resp = conn.getresponse()
    started = time.monotonic()
    first = resp.read1(100)
    first_at = time.monotonic() - started
    rest = resp.read()
    assert first.startswith(b"data: 1") and first_at < 1.0  # not held until the upstream finished
    assert b"data: 2" in first + rest


def test_the_log_has_no_key_and_no_headers(proxy, tmp_path):
    p = proxy()
    call(p, path="/api/v1/chat/completions?key=planted")
    deadline = time.monotonic() + 5
    while p.requests == 0 and time.monotonic() < deadline:  # logged once the response is out
        time.sleep(0.02)
    log = (tmp_path / "proxy.log").read_text()
    assert "POST /api/v1/chat/completions 200" in log
    assert KEY not in log and "planted" not in log and "Bearer" not in log


def test_chunked_request_bodies_are_forwarded(proxy):
    p = proxy()
    conn = _UnixConnection(p.socket_path)
    conn.putrequest("POST", "/api/v1/chat/completions")
    conn.putheader("Transfer-Encoding", "chunked")
    conn.endheaders()
    for part in (b'{"a":', b'"b"}'):
        conn.send(f"{len(part):x}\r\n".encode() + part + b"\r\n")
    conn.send(b"0\r\n\r\n")
    assert conn.getresponse().status == 200
    assert _Upstream.seen[-1]["body"] == '{"a":"b"}'


@pytest.mark.parametrize("url", ["http://api.example.com/v1", "ftp://x/v1", "https://user:pw@x/v1",
                                 "https://x/v1?key=1", "not a url"])
def test_unsafe_upstreams_are_refused(tmp_path, url):
    with pytest.raises(ProxyError):
        KeyProxy(tmp_path / "s", url, KEY)


def test_a_dead_upstream_is_a_502_not_a_hang(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with KeyProxy(tmp_path / "s", f"http://127.0.0.1:{port}/v1", KEY) as p:
        assert call(p, path="/v1/x")[0] == 502


# Hardening: sizes, paths, caps, the pinned model, connections, timeouts, leaks


def raw(proxy, data: bytes, timeout: float = 10.0) -> bytes:
    """Send `data` as-is on the proxy's socket and return the first line of the answer."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(proxy.socket_path))
        s.sendall(data)
        out = b""
        while b"\r\n" not in out:
            piece = s.recv(4096)
            if not piece:
                break
            out += piece
    return out.split(b"\r\n", 1)[0]


@pytest.mark.parametrize("length", ["-5", "+5", "1_0", "0x10", "abc", "99999999999999999999"])
def test_bad_content_lengths_are_refused(proxy, length):
    p = proxy()
    line = raw(p, f"POST /api/v1/x HTTP/1.1\r\nHost: s\r\nContent-Length: {length}\r\n\r\nhello".encode())
    assert line.split()[1] in (b"400", b"413")
    assert _Upstream.seen == []


def test_conflicting_content_lengths_are_refused(proxy):
    p = proxy()
    line = raw(p, b"POST /api/v1/x HTTP/1.1\r\nHost: s\r\nContent-Length: 2\r\nContent-Length: 5\r\n\r\nhello")
    assert line.split()[1] == b"400" and _Upstream.seen == []


@pytest.mark.parametrize("chunk", [b"-1\r\nx\r\n0\r\n\r\n", b"0x2\r\nab\r\n0\r\n\r\n", b"zz\r\n", b"3\r\nab",
                                   b"2\r\nabXX0\r\n\r\n", b"1" * 2000 + b"\r\n"])
def test_bad_chunk_sizes_and_truncated_chunks_are_refused(proxy, chunk, monkeypatch):
    from mullet.jobs.executors import keyproxy

    monkeypatch.setattr(keyproxy._Handler, "timeout", 1.0)  # a truncated body waits for the read timeout
    p = proxy()
    head = b"POST /api/v1/x HTTP/1.1\r\nHost: s\r\nTransfer-Encoding: chunked\r\n\r\n"
    line = raw(p, head + chunk, timeout=10)
    assert line == b"" or line.split()[1] == b"400"
    assert _Upstream.seen == []


@pytest.mark.parametrize("path", [
    "/api/v1/../../admin", "/api/v1/%2e%2e/admin", "/api/v1/%2E%2e/x", "/api/v1/a%2fb", "/api/v1/a%2Fb",
    "/api/v1/a\\b", "/api/v1/./x", "/api/v1/a%5cb", "/api/v1/a%00b",
])
def test_paths_that_climb_or_hide_separators_are_refused(proxy, path):
    p = proxy()
    assert call(p, path=path)[0] == 400
    assert _Upstream.seen == []


def test_a_pinned_proxy_refuses_another_model(proxy):
    p = proxy(models=["moonshotai/kimi-k2.7-code"])
    assert call(p, body=json.dumps({"model": "moonshotai/kimi-k2.7-code", "stream": True}).encode())[0] == 200
    assert call(p, body=json.dumps({"model": "anthropic/claude-opus-5"}).encode())[0] == 403
    # No model, or a non-string one: OpenRouter would pick the account's default.
    assert call(p, body=b"{}")[0] == 403
    assert call(p, body=b'{"model": null}')[0] == 403
    assert call(p, body=b'{"model": ["moonshotai/kimi-k2.7-code"]}')[0] == 403
    assert len(_Upstream.seen) == 1 and p.refused_models == 4


KIMI_BODY = '{"model": "moonshotai/kimi-k2.7-code"'
KIMI_JSON = (KIMI_BODY + "}").encode()


@pytest.mark.parametrize("body, status", [
    (b"not json", 400),
    (b"[1, 2]", 400),
    ((KIMI_BODY + ', "model": "anthropic/claude-opus-5"}').encode(), 400),  # duplicate keys
    ((KIMI_BODY + ', "models": ["anthropic/claude-opus-5"]}').encode(), 403),  # OpenRouter fallbacks
    ((KIMI_BODY + ', "fallback_model": "x/y"}').encode(), 403),
])
def test_a_pinned_proxy_takes_only_plain_json_naming_its_own_model(proxy, body, status):
    p = proxy(models=["moonshotai/kimi-k2.7-code"])
    assert call(p, body=body)[0] == status
    assert _Upstream.seen == []


def test_encoded_request_bodies_are_refused_and_the_header_never_forwarded(proxy):
    p = proxy(models=["moonshotai/kimi-k2.7-code"])
    assert call(p, body=b"\x1f\x8b", headers={"Content-Encoding": "gzip"})[0] == 415
    assert call(p, body=(KIMI_BODY + "}").encode(), headers={"Content-Encoding": "identity"})[0] == 200
    assert "content-encoding" not in {k.lower() for k in _Upstream.seen[-1]["headers"]}


@pytest.mark.parametrize("method, path, status", [
    ("POST", "/api/v1/chat/completions", 200), ("POST", "/api/v1/completions", 200),
    ("GET", "/api/v1/models", 200), ("GET", "/api/v1/models/moonshotai/kimi-k2.7-code", 200),
    ("POST", "/api/v1/batches", 403), ("POST", "/api/v1/files", 403), ("POST", "/api/v1/uploads", 403),
    ("POST", "/api/v1/models/x", 403), ("POST", "/api/v1/chat/completions/extra", 403),
])
def test_an_allowlisted_proxy_serves_only_generation_and_the_model_list(proxy, method, path, status):
    p = proxy(models=["moonshotai/kimi-k2.7-code"], paths=("chat/completions", "completions", "models"))
    assert call(p, method=method, path=path, body=KIMI_JSON if method == "POST" else None)[0] == status


def test_the_socket_is_private_from_the_start(proxy):
    import stat as stat_mod

    p = proxy()
    assert stat_mod.S_IMODE(p.socket_path.stat().st_mode) == 0o600


def test_the_request_cap_stops_an_attempt(proxy):
    p = proxy(max_requests=2)
    assert [call(p)[0] for _ in range(3)] == [200, 200, 429]
    assert len(_Upstream.seen) == 2 and "request cap (2)" in p.capped


def test_the_byte_cap_stops_an_attempt(proxy):
    p = proxy(max_bytes=100)
    assert call(p, body=b"x" * 60)[0] == 200
    assert call(p, body=b"x" * 60)[0] == 429
    assert "byte cap" in p.capped and len(_Upstream.seen) == 1


def test_connections_beyond_the_cap_are_refused_not_queued(proxy):
    p = proxy(max_connections=1)
    held = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    held.connect(str(p.socket_path))
    held.sendall(b"POST /api/v1/x HTTP/1.1\r\nHost: s\r\nContent-Length: 10\r\n\r\n")  # never finished
    time.sleep(0.3)
    try:
        assert raw(p, b"GET /api/v1/x HTTP/1.1\r\nHost: s\r\n\r\n").split()[1] == b"503"
    finally:
        held.close()


def test_a_stalled_sender_times_out(proxy, monkeypatch):
    from mullet.jobs.executors import keyproxy

    monkeypatch.setattr(keyproxy._Handler, "timeout", 0.5)
    p = proxy()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(10)
        s.connect(str(p.socket_path))
        s.sendall(b"POST /api/v1/x HTTP/1.1\r\nHost: s\r\nContent-Length: 10\r\n\r\nabc")
        started = time.monotonic()
        while s.recv(4096):
            pass
        assert time.monotonic() - started < 8  # closed by the proxy, not left hanging
    assert _Upstream.seen == []


def test_a_failed_start_leaves_no_socket_or_open_server(tmp_path, monkeypatch, upstream):
    from mullet.jobs.executors import keyproxy

    def refuse(*args):
        raise OSError("chmod refused")

    monkeypatch.setattr(keyproxy.os, "chmod", refuse)
    p = KeyProxy(tmp_path / "s", upstream, KEY)
    with pytest.raises(OSError):
        p.start()
    assert not (tmp_path / "s").exists() and p._server is None


def test_body_fields_replace_the_workers_own_routing(proxy):
    routing = {"provider": {"data_collection": "deny", "quantizations": ["fp8"]}}
    p = proxy(models=["moonshotai/kimi-k2.7-code"], body_fields=routing)
    sent = {"model": "moonshotai/kimi-k2.7-code", "stream": True, "provider": {"data_collection": "allow"}}
    assert call(p, body=json.dumps(sent).encode())[0] == 200
    seen = _Upstream.seen[-1]
    assert json.loads(seen["body"]) == {"model": "moonshotai/kimi-k2.7-code", "stream": True, **routing}
    assert seen["headers"]["Content-Length"] == str(len(seen["body"].encode()))


def test_body_fields_need_a_json_object_body(proxy):
    p = proxy(body_fields={"provider": {"data_collection": "deny"}})
    assert call(p, body=b"not json")[0] == 400
    assert call(p, body=b"[1]")[0] == 400
    assert call(p, method="GET", path="/api/v1/models", body=b"")[0] == 200
    assert [s["path"] for s in _Upstream.seen] == ["/api/v1/models"]
