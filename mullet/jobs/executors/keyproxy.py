"""A one-provider forwarding proxy that adds the API key outside the sandbox.

The opencode worker runs with no network at all. Its provider's base URL
points at a loopback port inside the sandbox, where socat passes each
connection to this proxy's Unix socket on the host side. The proxy:

- forwards only to one upstream (scheme, host and port fixed, path kept
  under the upstream's own prefix), so the worker can't reach anything else;
- refuses CONNECT and absolute-form request targets, paths that climb or
  hide separators (`..`, `%2e`, `%2f`, backslashes), and caps request size;
- reads request bodies in bounded pieces with a read timeout, refuses
  negative or malformed sizes, and serves a few connections at a time;
- caps each attempt's requests and bytes, and, given a pinned model, takes
  only plain JSON bodies (no content encoding, no duplicate keys) that name
  that model and no other (no OpenRouter `models` list), on an allowlist of
  API paths, so a worker can't run up a bill on a model nobody chose or
  reach batch and file endpoints;
- drops whatever credentials the request carries and adds the real key in
  the provider's own header, so the key never enters the sandbox;
- forwards only an allowlist of the worker's other headers (`FORWARD_HEADERS`,
  or the `forward_headers` parameter), so it can't steer routing, billing or
  account selection with headers such as `openai-organization`;
- clamps `n` in the body to `max_n` (default 1) and, when `max_output_tokens`
  is set, `max_tokens`, `max_completion_tokens` and `max_output_tokens` to it
  (the default, None, leaves them alone);
- streams the response back as it arrives, so server-sent events aren't
  held up;
- logs the method, path (without query), status and size, never headers.

It lives in a thread for one attempt; `stop()` ends it.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socketserver
import ssl
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit

from mullet.jobs.executors.usage_meter import ReportedUsage, ResponseMeter

MAX_BODY_BYTES = 32 * 1024 * 1024
UPSTREAM_TIMEOUT_S = 600.0
READ_TIMEOUT_S = 60.0  # for the sandbox side to send a request
MAX_CONNECTIONS = 16
MAX_REQUESTS = 400  # per attempt, unless the executor sets its own
MAX_BYTES = 512 * 1024 * 1024  # request and response bytes per attempt
_PIECE = 64 * 1024
_LINE_LIMIT = 1024
_MAX_TRAILERS = 64
_DIGITS = re.compile(r"^[0-9]{1,15}$")
_HEX = re.compile(r"^[0-9A-Fa-f]{1,15}$")
_HIDDEN = ("%2e", "%2f", "%5c", "%00")

# Request headers never forwarded: hop-by-hop ones. When the proxy has a key,
# every header a request can carry credentials in is dropped too, and the
# real one added; without a key (a local server, a free tier) they pass.
_HOP_BY_HOP = frozenset({
    "host", "connection", "keep-alive", "proxy-connection", "proxy-authorization", "proxy-authenticate",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length", "content-encoding",
})
_CREDENTIALS = frozenset({"authorization", "x-api-key", "x-goog-api-key", "api-key"})
_DROP_RESPONSE = frozenset({
    "connection", "keep-alive", "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade",
})

# Request headers a worker may send on, lower case. `anthropic-version` and
# `anthropic-beta` are the Anthropic API's protocol headers, which its route
# can't work without. Everything else is dropped, `openai-organization`
# included. A KeyProxy takes its own set through `forward_headers`.
FORWARD_HEADERS = frozenset({
    "content-type", "accept", "user-agent", "accept-language", "x-title", "http-referer",
    "anthropic-version", "anthropic-beta",
})
MAX_N = 1  # choices per request; a larger `n` multiplies the bill
_TOKEN_FIELDS = ("max_tokens", "max_completion_tokens", "max_output_tokens")

HEADER_STYLES = ("bearer", "x-api-key", "x-goog-api-key")


class ProxyError(ValueError):
    """The upstream can't be proxied safely."""


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    proxy: KeyProxy
    slots: threading.BoundedSemaphore

    def process_request(self, request, client_address) -> None:
        """Serve the connection in a thread, or refuse it when every slot is taken."""
        if not self.slots.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n"
                                b"Connection: close\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class _DuplicateKey(ValueError):
    """A JSON object with the same key twice."""


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise _DuplicateKey(repr(key))
        out[key] = value
    return out


class _BodyError(ValueError):
    """A request body that is malformed or truncated."""


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server
    timeout = READ_TIMEOUT_S  # the socket's timeout, so a stalled sender can't hold a slot

    def address_string(self) -> str:  # a Unix socket has no client address
        return "sandbox"

    def log_message(self, format: str, *args) -> None:  # the proxy keeps its own log
        return

    def do_CONNECT(self) -> None:
        self._refuse(405, "CONNECT is not allowed")

    def do_GET(self) -> None:
        self._forward()

    do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = do_GET

    def _refuse(self, status: int, reason: str) -> None:
        self.server.proxy._log(self.command, self.path, status, 0)
        body = (reason + "\n").encode()
        self.close_connection = True
        try:
            self.send_response_only(status)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except OSError:
            pass  # the sandbox side went away

    def _read_exact(self, size: int, out: bytearray) -> None:
        """Append exactly `size` bytes to `out`, in bounded pieces."""
        while size > 0:
            piece = self.rfile.read(min(size, _PIECE))
            if not piece:
                raise _BodyError("the request body ended early")
            out += piece
            size -= len(piece)

    def _line(self) -> bytes:
        line = self.rfile.readline(_LINE_LIMIT + 1)
        if len(line) > _LINE_LIMIT:
            raise _BodyError("a chunk line is too long")
        return line

    def _read_body(self) -> bytes | None:
        """The request body, or None if it is over the limit. Raises _BodyError when malformed."""
        limit = self.server.proxy.max_body
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            out = bytearray()
            while True:
                text = self._line().split(b";", 1)[0].strip().decode("ascii", "replace")
                if not _HEX.match(text):
                    raise _BodyError("bad chunk size")
                size = int(text, 16)
                if size == 0:
                    for _ in range(_MAX_TRAILERS):
                        if self._line() in (b"\r\n", b"\n", b""):
                            return bytes(out)
                    raise _BodyError("too many trailers")
                if len(out) + size > limit:
                    return None
                self._read_exact(size, out)
                if self._line() not in (b"\r\n", b"\n"):
                    raise _BodyError("a chunk didn't end where its size said")
        values = self.headers.get_all("Content-Length") or []
        if len({v.strip() for v in values}) > 1:
            raise _BodyError("conflicting Content-Length headers")
        text = values[0].strip() if values else "0"
        if not _DIGITS.match(text):
            raise _BodyError("bad Content-Length")
        length = int(text)
        if length > limit:
            return None
        out = bytearray()
        self._read_exact(length, out)
        return bytes(out)

    @staticmethod
    def _bad_path(path_only: str) -> bool:
        """True for a path that climbs, or hides a dot, separator or NUL behind an escape."""
        low = path_only.lower()
        if "\\" in path_only or any(h in low for h in _HIDDEN):
            return True
        return any(part in (".", "..") for part in path_only.split("/"))

    def _forward(self) -> None:
        proxy = self.server.proxy
        if not self.path.startswith("/") or self.path.startswith("//"):
            self._refuse(400, "only origin-form request targets are allowed")
            return
        path_only = self.path.split("?", 1)[0]
        if self._bad_path(path_only):
            self._refuse(400, "that path is not allowed")
            return
        if proxy.prefix and not (path_only == proxy.prefix or path_only.startswith(proxy.prefix + "/")):
            self._refuse(403, "that path is outside the provider's API")
            return
        if not proxy.path_allowed(self.command, path_only):
            self._refuse(403, "Mullet: that endpoint isn't one a worker may use")
            return
        encoding = self.headers.get("Content-Encoding", "").strip().lower()
        if encoding not in ("", "identity"):
            self._refuse(415, "request bodies must not be encoded")
            return
        try:
            body = self._read_body()
        except (ValueError, OSError):
            self._refuse(400, "malformed request body")
            return
        if body is None:
            self._refuse(413, "request too large")
            return
        refusal = proxy.body_refusal(body)
        if refusal:
            self._refuse(403 if "own model" in refusal else 400, refusal)
            return
        body = proxy.with_fields(body)
        if body is None:
            self._refuse(400, "the request body must be a JSON object")
            return
        refusal = proxy.take_request(len(body))
        if refusal:
            self._refuse(402 if proxy.out_of_allowance else 429, refusal)
            return
        headers = {k: v for k, v in self.headers.items() if proxy.forwards_header(k)}
        headers.update(proxy.auth_headers())
        # Plain bodies only, so the usage in a response can be read.
        headers = {k: v for k, v in headers.items() if k.lower() != "accept-encoding"}
        headers["Accept-Encoding"] = "identity"
        if body or self.command in ("POST", "PUT", "PATCH"):
            headers["Content-Length"] = str(len(body))
        conn = proxy.connect()
        try:
            conn.request(self.command, self.path, body=body or None, headers=headers)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            conn.close()
            self._refuse(502, f"upstream unreachable: {type(exc).__name__}")
            return
        sent = 0
        meter = ResponseMeter() if resp.status == 200 else None
        try:
            self.send_response_only(resp.status, resp.reason)
            for key, value in resp.getheaders():
                low = key.lower()
                if low in _DROP_RESPONSE or (low == "content-length" and resp.chunked):
                    continue
                self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)  # unbuffered: each piece goes out as it arrives
                    sent += len(chunk)
                    if meter is not None:
                        meter.feed(chunk)
                    if not proxy.take_bytes(len(chunk)):
                        break  # the attempt's byte cap: cut the response off
        except OSError:
            pass  # the sandbox side went away
        finally:
            if meter is not None:
                proxy.record_usage(meter.usage())
            conn.close()
            self.close_connection = True
            proxy._log(self.command, self.path, resp.status, sent)


class KeyProxy:
    """Serve `socket_path`, forwarding to `upstream` (a base URL) with `key` added."""

    def __init__(self, socket_path: Path, upstream: str, key: str | None, *, header: str = "bearer",
                 log_path: Path | None = None, max_body: int = MAX_BODY_BYTES,
                 max_requests: int = MAX_REQUESTS, max_bytes: int = MAX_BYTES,
                 models: list[str] | None = None, max_connections: int = MAX_CONNECTIONS,
                 paths: tuple[str, ...] | list[str] | None = None,
                 body_fields: dict[str, object] | None = None,
                 allowance_usd: float | None = None,
                 pricer: Callable[[dict[str, int]], float] | None = None,
                 forward_headers: frozenset[str] | set[str] | None = None,
                 max_n: int = MAX_N, max_output_tokens: int | None = None):
        parts = urlsplit(upstream)
        if parts.scheme not in ("https", "http") or not parts.hostname:
            raise ProxyError(f"not an http(s) base URL: {upstream!r}")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ProxyError("the upstream URL must be a plain base URL")
        if parts.scheme == "http" and parts.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ProxyError("plain http is only allowed to a local server; the key must not travel unencrypted")
        if header not in HEADER_STYLES:
            raise ProxyError(f"unknown key header style {header!r}")
        self.socket_path = Path(socket_path)
        self.scheme = parts.scheme
        self.host = parts.hostname
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        self.prefix = parts.path.rstrip("/")
        self.key = key
        self.header = header
        self.max_body = max_body
        self.max_requests = max_requests
        self.max_bytes = max_bytes
        self.max_connections = max_connections
        # The model ids a request body may name; None allows any.
        self.models = frozenset(models) if models else None
        # API paths (under the prefix) requests may use; None allows any.
        self.paths = frozenset(p.strip("/") for p in paths) if paths else None
        # Top-level fields set on every JSON request body, replacing the worker's
        # own (OpenRouter's `provider` routing, say); None leaves bodies alone.
        self.body_fields = dict(body_fields) if body_fields else None
        # Client headers passed on (lower case); every other one is dropped.
        self.forward_headers = (frozenset(h.lower() for h in forward_headers)
                                if forward_headers is not None else FORWARD_HEADERS)
        self.max_n = max_n
        # Cap on the output-token fields of a body; None leaves them as sent.
        self.max_output_tokens = max_output_tokens
        self.log_path = log_path
        self.requests = 0  # every request seen, refused ones included
        self.forwarded = 0
        self.bytes = 0
        self.capped: str | None = None  # why the attempt's caps stopped a request
        self.refused_models = 0
        # Dollars this attempt may still spend; None means no limit. Spend is the
        # provider's own stated cost, else the tokens it reported at `pricer`'s
        # prices ($ per million tokens), else nothing.
        self.allowance_usd = allowance_usd
        self.pricer = pricer
        self.out_of_allowance = False
        self.metered = 0  # responses whose usage was read
        self.spent_usd = 0.0
        self.provider_cost_usd = 0.0  # the part of spent_usd the provider stated itself
        self.unpriced = 0  # responses with tokens but neither a stated cost nor a price
        self.tokens = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0}
        self._log_lock = threading.Lock()
        self._count_lock = threading.Lock()
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None
        self._ssl = ssl.create_default_context() if parts.scheme == "https" else None

    def auth_headers(self) -> dict[str, str]:
        """The provider's own credential header, carrying the real key. None without a key."""
        if not self.key:
            return {}
        if self.header == "x-api-key":
            return {"x-api-key": self.key}
        if self.header == "x-goog-api-key":
            return {"x-goog-api-key": self.key}
        return {"Authorization": f"Bearer {self.key}"}

    def body_refusal(self, body: bytes) -> str | None:
        """Why a pinned proxy refuses this body, or None.

        It must be plain JSON (no duplicate keys, which upstream parsers may
        read differently), name the pinned model, and carry no other
        field that names a model (OpenRouter's `models` fallback list, say).
        """
        if self.models is None or not body:
            return None
        try:
            data = json.loads(body, object_pairs_hook=_no_duplicates)
        except _DuplicateKey as exc:
            return f"duplicate key {exc} in the request body"
        except ValueError:
            return "the request body must be JSON"
        if not isinstance(data, dict):
            return "the request body must be a JSON object"
        # A missing model is refused too: OpenRouter would use the account's
        # default model, which nobody chose for this attempt.
        model = data.get("model")
        others = [k for k in data if k != "model" and "model" in k.lower()]
        if not isinstance(model, str) or model not in self.models or others:
            with self._count_lock:
                self.refused_models += 1
            return "Mullet: this attempt may only use its own model"
        return None

    def forwards_header(self, name: str) -> bool:
        """True when a client header may go upstream.

        Hop-by-hop headers never do. Credential headers pass only when the proxy
        has no key of its own (a local server, a free tier); every other header
        must be on the allowlist.
        """
        low = name.lower()
        if low in _HOP_BY_HOP:
            return False
        if low in _CREDENTIALS:
            return not self.key
        return low in self.forward_headers

    def with_fields(self, body: bytes) -> bytes | None:
        """`body` with `body_fields` set and `n` and the token fields clamped.

        None when `body_fields` is set and a non-empty body isn't a JSON object,
        or when a body repeats a key (upstream parsers may read the other one).
        A body that isn't JSON at all is passed on untouched when nothing is set.
        """
        if not body:
            return body
        try:
            data = json.loads(body, object_pairs_hook=_no_duplicates)
        except _DuplicateKey:
            return None
        except ValueError:
            return None if self.body_fields else body
        if not isinstance(data, dict):
            return None if self.body_fields else body
        changed = False
        if self.body_fields:
            data.update(self.body_fields)
            changed = True
        if "n" in data and data["n"] != self.max_n and not (
            isinstance(data["n"], int) and not isinstance(data["n"], bool) and data["n"] <= self.max_n
        ):
            data["n"] = self.max_n
            changed = True
        cap = self.max_output_tokens
        if cap is not None:
            for field in _TOKEN_FIELDS:
                value = data.get(field)
                if field in data and value is not None and not (
                    isinstance(value, int) and not isinstance(value, bool) and value <= cap
                ):
                    data[field] = cap
                    changed = True
        return json.dumps(data, separators=(",", ":")).encode() if changed else body

    def path_allowed(self, method: str, path_only: str) -> bool:
        """True for one of the allowlisted API paths under the prefix (any path when none is set)."""
        if self.paths is None:
            return True
        rest = path_only[len(self.prefix):].strip("/")
        if rest in self.paths:
            return True
        return method in ("GET", "HEAD") and "models" in self.paths and rest.startswith("models/")

    def take_request(self, size: int) -> str | None:
        """Count a request against the attempt's caps; the reason it's refused, or None."""
        with self._count_lock:
            if self.allowance_usd is not None and self.spent_usd >= self.allowance_usd:
                self.out_of_allowance = True
                self.capped = (f"Mullet: this attempt's spending allowance (${self.allowance_usd:.2f}) "
                               "is used up")
            elif self.forwarded >= self.max_requests:
                self.capped = f"Mullet: this attempt's request cap ({self.max_requests}) is used up"
            elif self.bytes + size > self.max_bytes:
                self.capped = f"Mullet: this attempt's byte cap ({self.max_bytes} bytes) is used up"
            else:
                self.forwarded += 1
                self.bytes += size
                return None
            return self.capped

    def record_usage(self, usage: ReportedUsage | None) -> None:
        """Add one response's usage to the attempt's spend."""
        if usage is None:
            return
        tokens = usage.tokens()
        if usage.cost is not None:
            cost, stated = usage.cost, True
        elif self.pricer is not None and any(tokens.values()):
            cost, stated = self.pricer(tokens), False
        else:
            cost, stated = 0.0, False
        with self._count_lock:
            self.metered += 1
            self.spent_usd += cost
            if stated:
                self.provider_cost_usd += cost
            elif cost <= 0 and any(tokens.values()):
                self.unpriced += 1
            for name, count in tokens.items():
                self.tokens[name] += count

    def usage_summary(self) -> dict[str, object]:
        """What the provider's own responses said this attempt used and cost."""
        with self._count_lock:
            return {"proxy_cost_usd": round(self.spent_usd, 6),
                    "proxy_provider_cost_usd": round(self.provider_cost_usd, 6),
                    "proxy_metered": self.metered, "proxy_unpriced": self.unpriced,
                    "proxy_tokens": dict(self.tokens)}

    def take_bytes(self, size: int) -> bool:
        """Count response bytes; False once the attempt's byte cap is passed."""
        with self._count_lock:
            self.bytes += size
            if self.bytes > self.max_bytes:
                self.capped = f"Mullet: this attempt's byte cap ({self.max_bytes} bytes) is used up"
                return False
            return True

    def connect(self) -> http.client.HTTPConnection:
        """A new connection to the one upstream this proxy may reach."""
        if self.scheme == "https":
            return http.client.HTTPSConnection(self.host, self.port, timeout=UPSTREAM_TIMEOUT_S,
                                               context=self._ssl)
        return http.client.HTTPConnection(self.host, self.port, timeout=UPSTREAM_TIMEOUT_S)

    def _log(self, method: str, path: str, status: int, size: int) -> None:
        with self._log_lock:
            if self.log_path is not None:
                line = f"{time.strftime('%H:%M:%S')} {method} {path.split('?', 1)[0]} {status} {size}\n"
                with open(self.log_path, "a", encoding="utf-8") as fh:
                    fh.write(line)
            self.requests += 1  # after the line is written, so a reader that sees the count sees the line

    def start(self) -> KeyProxy:
        """Listen on the socket, in a background thread."""
        if self.socket_path.exists() or self.socket_path.is_symlink():
            self.socket_path.unlink()
        # Created under a 0077 umask, so there is no moment the socket is open to others.
        old_umask = os.umask(0o077)
        try:
            server = _Server(str(self.socket_path), _Handler)
        finally:
            os.umask(old_umask)
        try:
            server.proxy = self
            server.slots = threading.BoundedSemaphore(self.max_connections)
            os.chmod(self.socket_path, 0o600)
        except BaseException:
            server.server_close()
            self.socket_path.unlink(missing_ok=True)
            raise
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2},
                                        name="mullet-keyproxy", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and remove the socket."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        try:
            self.socket_path.unlink()
        except OSError:
            pass

    def __enter__(self) -> KeyProxy:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
