"""Mullet MCP server — exposes Mullet tools to Claude Code.

Run with: mullet mcp
Or add to Claude Code settings as an MCP server.

Uses stdio transport (stdin/stdout JSON-RPC) for Claude Code compatibility.
"""

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

from mullet.core.env import load_dotenv

load_dotenv()


def _code_fingerprint() -> float:
    """Return the newest modification time of any .py file in the mullet package."""
    package_dir = Path(__import__("mullet").__file__).parent
    newest = 0.0
    for path in package_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > newest:
            newest = mtime
    return newest


def _drop_mullet_modules() -> None:
    """Remove cached mullet modules so the next import re-reads source files."""
    protected = {"mullet", "mullet.server", "mullet.server.mcp"}
    to_remove = [name for name in sys.modules if name.startswith("mullet.") and name not in protected]
    for name in to_remove:
        del sys.modules[name]


def _load_registries() -> tuple[list, dict]:
    """Import the MCP tool/handler registries (fresh copies after a module drop)."""
    from mullet.server.mcp_handlers import MCP_TOOLS, MCP_HANDLERS
    from mullet.server.mcp_knowledge import MCP_KNOWLEDGE_TOOLS, MCP_KNOWLEDGE_HANDLERS
    return MCP_TOOLS + MCP_KNOWLEDGE_TOOLS, {**MCP_HANDLERS, **MCP_KNOWLEDGE_HANDLERS}


MCP_TOOLS, MCP_HANDLERS = _load_registries()

logger = logging.getLogger(__name__)

_RELOAD_INTERVAL = 5.0
_RELOAD_LOCK = threading.Lock()
_reload_last_check: float = 0.0
_reload_last_fingerprint: float = _code_fingerprint()
_reload_pending_warning: str | None = None


def _maybe_reload() -> str | None:
    """Check whether Mullet source changed and, if so, reload tool registries.

    Called at the start of tools/list and tools/call. Throttled to once every
    five seconds. On success it sends an MCP notification so the client
    refetches the tool list. On failure the existing registries are kept and a
    short message is returned for surfacing to the user.
    """
    global _reload_last_check, _reload_last_fingerprint, _reload_pending_warning
    global MCP_TOOLS, MCP_HANDLERS

    if os.environ.get("MULLET_MCP_RELOAD", "").lower() == "off":
        return None

    now = time.monotonic()
    if now - _reload_last_check < _RELOAD_INTERVAL:
        return None
    _reload_last_check = now

    fingerprint = _code_fingerprint()
    if fingerprint == _reload_last_fingerprint:
        return None

    with _RELOAD_LOCK:
        # Re-check under lock in case another thread already reloaded.
        if fingerprint == _reload_last_fingerprint:
            return None

        old_tools, old_handlers = MCP_TOOLS, MCP_HANDLERS
        # Kept so a failed reload can put the old modules back: the old
        # handlers' lazy imports must keep resolving to the code they came from.
        old_modules = {k: v for k, v in sys.modules.items() if k.startswith("mullet.")}
        try:
            # Only a real reload drops the cached modules; at import the
            # registries load normally, so nothing already imported is orphaned.
            _drop_mullet_modules()
            MCP_TOOLS, MCP_HANDLERS = _load_registries()
            _reload_last_fingerprint = fingerprint
            _reload_pending_warning = None
            logger.info("Mullet code changed; reloaded MCP tool registries")
            _write_response({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
            return None
        except Exception as exc:
            for name in [k for k in sys.modules if k.startswith("mullet.")]:
                if name not in old_modules:
                    del sys.modules[name]
            sys.modules.update(old_modules)
            MCP_TOOLS, MCP_HANDLERS = old_tools, old_handlers
            _reload_last_fingerprint = fingerprint
            message = (
                f"Mullet's code changed but the new version failed to load "
                f"({type(exc).__name__}: {str(exc).splitlines()[0]}); "
                f"run /mcp in Claude Code to restart the server."
            )
            _reload_pending_warning = message
            logger.warning("MCP reload failed: %s", exc, exc_info=True)
            return message


def _take_reload_warning() -> str | None:
    global _reload_pending_warning
    warning = _reload_pending_warning
    _reload_pending_warning = None
    return warning


def handle_request(request: dict) -> dict:
    """Handle a single JSON-RPC request."""
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": True}},
                "serverInfo": {
                    "name": "mullet",
                    "version": __import__("mullet").__version__,
                },
            },
        }

    elif method == "notifications/initialized":
        return None  # No response for notifications

    elif method == "tools/list":
        _maybe_reload()
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": MCP_TOOLS,
            },
        }

    elif method == "tools/call":
        _maybe_reload()
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        handler = MCP_HANDLERS.get(tool_name)
        if not handler:
            content = [{"type": "text", "text": json.dumps({"error": f"Unknown tool: {tool_name}"})}]
            warning = _take_reload_warning()
            if warning:
                content.append({"type": "text", "text": warning})
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": content,
                    "isError": True,
                },
            }

        try:
            result_text = handler(arguments)
            # D7 (Cycle 17): handle_mullet_run now returns a dict per the
            # in-process contract; serialise here for the MCP protocol.
            # Other handlers still return strings (markdown / JSON-string)
            # for human display — pass those through unchanged.
            if isinstance(result_text, dict):
                result_text = json.dumps(result_text)
            content = [{"type": "text", "text": result_text}]
            warning = _take_reload_warning()
            if warning:
                content.append({"type": "text", "text": warning})
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": content,
                },
            }
        except Exception as e:
            logger.exception("Tool %s failed", tool_name)
            content = [{"type": "text", "text": json.dumps({"error": str(e)})}]
            warning = _take_reload_warning()
            if warning:
                content.append({"type": "text", "text": warning})
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": content,
                    "isError": True,
                },
            }

    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


_STDOUT_LOCK = __import__("threading").Lock()


def _write_response(response: dict) -> None:
    """Serialize a single JSON-RPC response to stdout under a lock.

    Concurrent tool handlers run on worker threads; the lock prevents
    interleaved bytes when two threads try to flush a response at the same time.
    """
    try:
        with _STDOUT_LOCK:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
    except (BrokenPipeError, OSError) as exc:
        sys.stderr.write(f"mullet mcp: stdout write failed ({exc}), client disconnected\n")
        sys.stderr.flush()


def _dispatch_concurrent(request: dict) -> None:
    """Run a single ``tools/call`` handler on a worker thread and emit the reply."""
    try:
        response = handle_request(request)
    except Exception as exc:
        logger.exception("MCP dispatch error")
        response = {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "content": [{"type": "text", "text": json.dumps({"error": str(exc)})}],
                "isError": True,
            },
        }
    if response is not None:
        try:
            _write_response(response)
        except (BrokenPipeError, OSError) as exc:
            sys.stderr.write(f"mullet mcp: worker write failed ({exc})\n")
            sys.stderr.flush()


def _parent_is_alive(ppid: int) -> bool:
    """Return True if a process with this PID exists and is reachable.

    Uses signal 0 (no-op probe, sends nothing) — the standard portable way
    to check PID liveness without side effects.
    """
    if ppid <= 1:
        # PID 1 (init) or 0 means we've already been reparented — the
        # process that actually spawned us is gone.
        return False
    try:
        os.kill(ppid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but belongs to another user — still alive.
        return True
    except OSError:
        return False


def _watch_for_orphaning(original_ppid: int, poll_interval: float = 10.0) -> None:
    """Daemon-thread body: force-exit if our original parent process disappears.

    Independent safety net on top of the stdin-EOF shutdown path below —
    covers cases where the parent dies abruptly (crash, SIGKILL, or a
    grandchild subprocess holding a duplicate stdin fd open) without ever
    closing our stdin pipe, which would otherwise leave this server running
    forever as an orphan (reparented to init) waiting for input that will
    never arrive. Skips the graceful drain-active-threads shutdown path
    entirely — if the parent is already gone, nothing is listening for a
    response anyway.
    """
    while True:
        time.sleep(poll_interval)
        if not _parent_is_alive(original_ppid):
            sys.stderr.write(
                f"mullet mcp: parent pid {original_ppid} no longer alive — orphaned, exiting\n"
            )
            sys.stderr.flush()
            os._exit(1)


def run_mcp_server():
    """Run the MCP server on stdio (JSON-RPC over stdin/stdout).

    ``tools/call`` requests are dispatched on worker threads so multiple
    concurrent provider invocations (Kimi + Gemini + local + Anthropic) can run
    in parallel. Writes to stdout are serialized by a single lock. Metadata
    calls (``initialize`` / ``tools/list`` / ``notifications/*``) are handled
    inline to preserve strict protocol ordering.
    """
    import threading

    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,  # Log to stderr, protocol on stdout
        format="%(name)s: %(message)s",
    )

    _active_threads: list = []
    MCP_TOOL_TIMEOUT = int(os.environ.get("MCP_TOOL_TIMEOUT", "600"))

    # Orphan-prevention: stdin EOF (below) is the normal clean-shutdown path,
    # but a parent that dies abruptly (crash/SIGKILL) or a grandchild
    # subprocess holding a duplicate stdin fd open can prevent EOF from ever
    # arriving. This watchdog is an independent safety net for that case.
    # (Deliberately NOT using prctl(PR_SET_PDEATHSIG) here — it fires when
    # the specific THREAD that spawned us exits, not necessarily when the
    # whole parent process does, which caused spurious "Connection closed"
    # kills against Claude Code's actual spawning model. Polling is slower
    # (up to 10s) but only acts on a real, confirmed-dead parent PID.)
    _watchdog_thread = threading.Thread(
        target=_watch_for_orphaning, args=(os.getppid(),), daemon=True,
    )
    _watchdog_thread.start()

    sys.stderr.write("mullet mcp: server ready (stdio transport)\n")
    sys.stderr.flush()

    while True:
        try:
            line = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            break
        except OSError as exc:
            # Broken pipe or closed stdin — parent process terminated
            sys.stderr.write(f"mullet mcp: stdin closed ({exc}), exiting\n")
            break

        if not line:
            break  # EOF

        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            sys.stderr.write(f"mullet mcp: invalid JSON ignored: {line[:100]}\n")
            continue

        try:
            method = request.get("method", "")
        except AttributeError:
            sys.stderr.write("mullet mcp: non-object JSON ignored\n")
            continue

        # tools/call is the only method that blocks on I/O for tens of seconds.
        # Dispatch it on a worker so the next request can start immediately.
        if method == "tools/call":
            t = threading.Thread(target=_dispatch_concurrent, args=(request,), daemon=True)
            t.start()
            _active_threads.append(t)
            # Prune completed threads
            _active_threads[:] = [x for x in _active_threads if x.is_alive()]
            continue

        try:
            response = handle_request(request)
        except Exception as exc:
            logger.exception("MCP main loop error for method %s", method)
            sys.stderr.write(f"mullet mcp: unhandled error on {method}: {exc}\n")
            continue

        if response is not None:
            try:
                _write_response(response)
            except (BrokenPipeError, OSError):
                break  # Client disconnected — clean exit

    alive = [t for t in _active_threads if t.is_alive()]
    if alive:
        sys.stderr.write(f"mullet mcp: waiting for {len(alive)} active tool call(s)...\n")
        sys.stderr.flush()
        for t in alive:
            t.join(timeout=MCP_TOOL_TIMEOUT)
        still_alive = [t for t in alive if t.is_alive()]
        if still_alive:
            sys.stderr.write(f"mullet mcp: {len(still_alive)} tool call(s) timed out at shutdown\n")
            sys.stderr.flush()

    sys.stderr.write("mullet mcp: server stopped\n")


if __name__ == "__main__":
    run_mcp_server()
