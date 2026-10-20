"""Parallel tool execution — run independent tool calls concurrently.

Ported from Hermes _should_parallelize_tool_batch() and
_execute_tool_calls_concurrent(). Analyses a batch of tool calls for
safety (path conflicts, resource contention) and dispatches safe
batches via ThreadPoolExecutor.

Sequential fallback when tools have dependencies or are unsafe to
parallelize (interactive tools, shared state mutations).
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Maximum concurrent tool workers
MAX_TOOL_WORKERS = 8

# Tools that must never run concurrently (interactive, user-facing, or stateful)
_NEVER_PARALLEL = frozenset({
    "clarify",          # Needs user input
    "delegate",         # Spawns child agents
    "session_search",   # Reads shared session DB
    "session_list",     # Reads shared session DB
})

# Read-only tools safe to run in parallel without path checks
_PARALLEL_SAFE = frozenset({
    "web_search",
    "fetch_page",
    "browse",
    "git",
    "run_tests",
    "lsp",
    "docker",
    "analyze_image",
    "trove_search",
    "trove_search_multi",
    "trove_recall",
})

# File tools that can run concurrently only when targeting independent paths
_PATH_SCOPED = frozenset({
    "read_file",
    "write_file",
    "patch",
    "list_directory",
    "search_files",
})


def should_parallelize(tool_calls: list[dict]) -> bool:
    """Return True when a tool-call batch is safe to run concurrently.

    Checks:
    1. More than 1 tool call in the batch
    2. No tools in the _NEVER_PARALLEL set
    3. Path-scoped tools don't target overlapping paths
    4. All remaining tools are in _PARALLEL_SAFE or _PATH_SCOPED
    """
    if len(tool_calls) <= 1:
        return False

    tool_names = []
    for tc in tool_calls:
        func = tc.get("function", tc) if isinstance(tc, dict) else {}
        name = func.get("name", "")
        tool_names.append(name)

    # Check for never-parallel tools
    if any(name in _NEVER_PARALLEL for name in tool_names):
        return False

    # Check path-scoped tools for conflicts
    reserved_paths: list[Path] = []
    for tc in tool_calls:
        func = tc.get("function", tc) if isinstance(tc, dict) else {}
        name = func.get("name", "")
        args_raw = func.get("arguments", "{}")

        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except (json.JSONDecodeError, TypeError):
            logger.debug("Cannot parse args for %s — falling back to sequential", name)
            return False

        if not isinstance(args, dict):
            return False

        if name in _PATH_SCOPED:
            scoped_path = _extract_scope_path(args)
            if scoped_path is None:
                return False
            if any(_paths_overlap(scoped_path, existing) for existing in reserved_paths):
                logger.debug("Path conflict detected: %s overlaps with reserved path", scoped_path)
                return False
            reserved_paths.append(scoped_path)
            continue

        if name not in _PARALLEL_SAFE:
            return False

    return True


def execute_parallel(
    tool_calls: list[dict],
    dispatch_fn: Callable[[str, str], str],
    on_tool_call: Callable[[str, str], None] | None = None,
    on_tool_result: Callable[[str, str], None] | None = None,
    on_tool_start: Callable[[str, str], None] | None = None,
    on_tool_complete: Callable[[str, str], None] | None = None,
) -> list[dict]:
    """Execute tool calls concurrently using a thread pool.

    Returns results in the original tool-call order as a list of
    dicts with keys: tool_call_id, name, result, duration_ms, success.

    Args:
        tool_calls: List of tool call dicts (OpenAI format)
        dispatch_fn: Function(tool_name, tool_args_json) -> result_str
        on_tool_call: Optional callback when a tool starts
        on_tool_result: Optional callback when a tool completes
    """
    num_tools = len(tool_calls)
    results = [None] * num_tools

    def _run_tool(index: int, tc: dict):
        func = tc.get("function", tc) if isinstance(tc, dict) else {}
        name = func.get("name", "unknown")
        args = func.get("arguments", "{}")
        tool_id = tc.get("id", f"call_{index:06d}")

        if on_tool_call:
            try:
                on_tool_call(name, args)
            except Exception as exc:
                logger.warning("parallel: on_tool_call callback failed: %s", exc)

        if on_tool_start:
            try:
                on_tool_start(name, args)
            except Exception as exc:
                logger.warning("parallel: on_tool_start callback failed: %s", exc)

        start = time.time()
        try:
            result = dispatch_fn(name, args)
            success = True
        except Exception as e:
            result = json.dumps({"error": str(e), "tool": name})
            success = False
            logger.warning("Parallel tool %s failed: %s", name, e)

        duration_ms = (time.time() - start) * 1000

        if on_tool_complete:
            try:
                preview = result[:150].replace("\n", " ") if result else ""
                on_tool_complete(name, preview)
            except Exception as exc:
                logger.warning("parallel: on_tool_complete callback failed: %s", exc)

        if on_tool_result:
            try:
                preview = result[:150].replace("\n", " ") if result else ""
                on_tool_result(name, preview)
            except Exception as exc:
                logger.warning("parallel: on_tool_result callback failed: %s", exc)

        results[index] = {
            "tool_call_id": tool_id,
            "name": name,
            "result": result,
            "duration_ms": duration_ms,
            "success": success,
        }

    max_workers = min(num_tools, MAX_TOOL_WORKERS)
    logger.info("Executing %d tools in parallel (max %d workers)", num_tools, max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, tc in enumerate(tool_calls):
            f = executor.submit(_run_tool, i, tc)
            futures.append(f)

        # Wait for all to complete
        for f in futures:
            f.result()  # Raises if thread raised (shouldn't — _run_tool catches)

    # Replace any None slots (failed tool executions) with error placeholders
    for i, r in enumerate(results):
        if r is None:
            results[i] = {
                "tool_call_id": f"call_{i:06d}",
                "name": "unknown",
                "result": '{"error": "Tool execution failed", "tool": "unknown"}',
                "duration_ms": 0.0,
                "success": False,
            }

    return results


def _extract_scope_path(args: dict) -> Path | None:
    """Extract the file path target from tool arguments."""
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    expanded = Path(raw_path).expanduser()
    if expanded.is_absolute():
        return Path(os.path.abspath(str(expanded)))

    # Avoid resolve() — file may not exist yet
    return Path(os.path.abspath(str(Path.cwd() / expanded)))


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return True when two paths may refer to the same subtree."""
    left_parts = left.parts
    right_parts = right.parts
    if not left_parts or not right_parts:
        return False
    common_len = min(len(left_parts), len(right_parts))
    return left_parts[:common_len] == right_parts[:common_len]
