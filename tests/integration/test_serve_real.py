"""Integration test: mullet serve as subprocess.

Starts the server, hits endpoints, verifies responses.
"""

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from mullet.server.guard import get_or_create_token


def _auth():
    """The server's token: same throwaway HOME as the subprocess, so the same file."""
    return {"Authorization": f"Bearer {get_or_create_token()}"}


def _wait_for_server(port, timeout=30):
    """Wait until server responds."""
    # First wait for TCP port to be accepting connections
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection(("localhost", port), timeout=1)
            break
        except Exception:
            time.sleep(0.2)
    else:
        return False
    
    # Then wait for HTTP 200 with longer timeout for slow handlers
    try:
        req = urllib.request.Request(f"http://localhost:{port}/health")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except Exception:
        return False


def test_serve_health():
    """Server should start and respond to health check."""
    port = 19876
    proc = subprocess.Popen(
        [sys.executable, "-m", "mullet.cli.main", "serve", "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_for_server(port), "Server did not start"

        req = urllib.request.Request(f"http://localhost:{port}/health")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            assert data["status"] == "ok"
            assert "version" in data
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_serve_info():
    """Server should return system info."""
    port = 19877
    proc = subprocess.Popen(
        [sys.executable, "-m", "mullet.cli.main", "serve", "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_for_server(port), "Server did not start"

        req = urllib.request.Request(f"http://localhost:{port}/api/v1/info", headers=_auth())
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            assert data["tool_count"] >= 13
            assert data["skill_count"] >= 6
            assert "providers" in data
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_serve_models():
    """Server should list models."""
    port = 19878
    proc = subprocess.Popen(
        [sys.executable, "-m", "mullet.cli.main", "serve", "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_for_server(port), "Server did not start"

        req = urllib.request.Request(f"http://localhost:{port}/v1/models", headers=_auth())
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            assert data["object"] == "list"
            assert len(data["data"]) >= 1
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_serve_refuses_a_request_without_the_token():
    """The real server needs this install's token for everything but /health."""
    port = 19879
    proc = subprocess.Popen(
        [sys.executable, "-m", "mullet.cli.main", "serve", "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_for_server(port), "Server did not start"
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=15)
        except urllib.error.HTTPError as exc:
            assert exc.code == 401
        else:
            raise AssertionError("an unauthenticated request was served")
    finally:
        proc.terminate()
        proc.wait(timeout=5)
