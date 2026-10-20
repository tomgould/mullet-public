"""Delegation safety D7 — handle_mullet_run dict-not-string return contract.

Cycle 17 (this commit). Locks the contract introduced by D7:

    ``handle_mullet_run(arguments) -> dict``  on every code path.

Pre-D7 the handler returned ``json.dumps(...)`` (a string) on every
return — early-exit error paths included. The Cycle 1 Python adapter
crashed on ``result.get("error")`` because it expected a dict. D6 test
5 pinned the *parseability* contract via ``json.loads(raw)``; D7
tightens it to a native dict (no JSON round-trip required for
in-process callers).

The MCP protocol wrapper at ``mullet.server.mcp.handle_request``
serialises the dict via ``json.dumps`` before sending it over stdio,
so MCP-protocol behaviour is preserved end-to-end.

Each test asserts ``isinstance(result, dict)`` directly, then walks
the failure-shape invariants the bridge layer guarantees:

  - ``error``: present on every non-success path.
  - ``filtered_providers``: present on constraint-conflict paths
    (P1-1, D2 surface).
  - ``tried_providers`` / ``errors_by_provider``: present on
    all-providers-failed paths (P1-1).

Validation: ``python3 -m pytest tests/test_delegation_safety_d7.py
-q --tb=short`` — all 6 tests must pass on first run.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock



# ---------------------------------------------------------------------------
# Helpers (mirror D6 patterns).
# ---------------------------------------------------------------------------


def _mock_provider(name: str = "kimi", model: str = "kimi-k2.5") -> MagicMock:
    p = MagicMock()
    p.name = name
    p.model = model
    return p


def _mock_response(content: str = "ok") -> MagicMock:
    r = MagicMock()
    r.content = content
    r.usage = MagicMock(total_tokens=10, input_tokens=5, output_tokens=5)
    r.tool_calls = []
    return r


def _clear_env(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# 1. Happy path — success returns dict (not json-encoded string).
# ---------------------------------------------------------------------------


def test_handle_mullet_run_returns_dict_on_success(monkeypatch, tmp_path):
    """Red (pre-D7, Cycle 17): success path returned ``json.dumps(out)`` —
    callers had to ``json.loads()`` before reading ``.get("response")``.

    Failure mode: the Cycle 1 Python adapter expected a dict and crashed
    with ``'str' object has no attribute 'get'`` on success too — only
    the error path was visible because that was the canonical failure
    pattern.

    D7 contract: every successful agent-mode dispatch yields a dict
    with at least ``completed``, ``response``, ``actual_provider`` keys.
    """
    from mullet.server import mcp_handlers

    monkeypatch.setenv("KIMI_API_KEY", "fake")
    _clear_env(monkeypatch)

    # Throwaway file for path-sanitisation coverage.
    f = tmp_path / "hello.py"
    f.write_text("print('hi')\n")

    kimi = _mock_provider("kimi", "kimi-k2.5")
    kimi.create.return_value = _mock_response("done")

    monkeypatch.setattr(mcp_handlers, "_resolve_provider", lambda **kw: kimi)
    monkeypatch.setattr(mcp_handlers, "_build_fallback_chain", lambda **kw: [])
    monkeypatch.setattr(mcp_handlers, "_write_agent_state_fn", lambda *a, **kw: None)
    monkeypatch.setattr(mcp_handlers, "_clear_agent_state", lambda: None)

    result = mcp_handlers.handle_mullet_run({
        "task": "do thing",
        "files": [str(f)],
        "target_dir": str(tmp_path),
        "providers": "kimi",
    })

    assert isinstance(result, dict), (
        f"D7 success path must return dict, got {type(result).__name__!r}"
    )
    assert result.get("completed") is True
    assert result.get("actual_provider") == "kimi"
    assert "response" in result


# ---------------------------------------------------------------------------
# 2. No provider available -> dict with error.
# ---------------------------------------------------------------------------


def test_handle_mullet_run_returns_dict_on_no_provider_available(monkeypatch):
    """Red (pre-D7, Cycle 17): when every provider failed init,
    ``handle_mullet_run`` returned a string ``'{"error":"No LLM
    provider available"}'`` — agent-mode path also string-returned.

    Failure mode: Cycle 1 adapter; ``result["error"]`` raised TypeError.

    D7 contract: agent-mode early-exit (``provider is None``) returns a
    dict with ``error`` set; no ``json.loads`` step required.
    """
    from mullet.server import mcp_handlers

    # Force agent-mode path (no files = no bridge).
    monkeypatch.setattr(mcp_handlers, "_resolve_provider", lambda **kw: None)
    monkeypatch.setattr(mcp_handlers, "_write_agent_state_fn", lambda *a, **kw: None)
    monkeypatch.setattr(mcp_handlers, "_clear_agent_state", lambda: None)

    result = mcp_handlers.handle_mullet_run({"task": "hi"})

    assert isinstance(result, dict), (
        f"D7 no-provider path must return dict, got {type(result).__name__!r}"
    )
    assert "error" in result
    assert "No LLM provider" in result["error"]


# ---------------------------------------------------------------------------
# 3. Constraint conflict -> dict with filtered_providers.
# ---------------------------------------------------------------------------


def test_handle_mullet_run_returns_dict_on_constraint_conflict(monkeypatch):
    """Red (pre-D7, Cycle 17): the constraint-conflict early-exit
    returned ``json.dumps({"error": ..., "filtered_providers": ...})`` —
    Python adapter crashed.

    Failure mode (Cycle 1 origin, 2026-04-23 23:24 UTC): the canonical
    agent-mode failure path. ``providers={"kimi"}`` + every kimi
    candidate filtered out by constraint registry.

    D7 contract: returns native dict with ``error`` AND
    ``filtered_providers`` keys; both consumable without ``json.loads``.
    """
    from mullet.server import mcp_handlers

    monkeypatch.setenv("KIMI_API_KEY", "fake")
    _clear_env(monkeypatch)

    kimi = _mock_provider("kimi", "kimi-k2.5")
    monkeypatch.setattr(mcp_handlers, "_resolve_provider", lambda **kw: kimi)
    monkeypatch.setattr(mcp_handlers, "_write_agent_state_fn", lambda *a, **kw: None)
    monkeypatch.setattr(mcp_handlers, "_clear_agent_state", lambda: None)

    def boom_chain(**kwargs):
        err = ValueError("All providers in allowlist {'kimi'} filtered out")
        err.filtered_providers = {  # type: ignore[attr-defined]
            "kimi": "context_tokens=14485 exceeds moonshot-v1-8k max_context=8192",
        }
        raise err

    monkeypatch.setattr(mcp_handlers, "_build_fallback_chain", boom_chain)

    result = mcp_handlers.handle_mullet_run({
        "task": "do something",
        "tier": "ADVANCED",
        "providers": "kimi",
    })

    assert isinstance(result, dict), (
        f"D7 constraint-conflict path must return dict, got "
        f"{type(result).__name__!r}"
    )
    assert "error" in result
    assert "allowlist/constraint conflict" in result["error"]
    assert isinstance(result.get("filtered_providers"), dict), (
        f"filtered_providers must be a dict, got {type(result.get('filtered_providers'))!r}"
    )
    assert "kimi" in result["filtered_providers"]
    assert "moonshot-v1-8k" in result["filtered_providers"]["kimi"]


# ---------------------------------------------------------------------------
# 4. All providers raised -> dict with tried_providers + errors_by_provider.
# ---------------------------------------------------------------------------


def test_handle_mullet_run_returns_dict_on_all_providers_failed(monkeypatch, tmp_path):
    """Red (pre-D7, Cycle 17): the ``response is None`` exhaustion path
    returned ``json.dumps(err_out)`` — string.

    Failure mode: the agent inner loop tried every fallback provider,
    each raised, and the final ``err_out`` was string-encoded; the
    adapter crashed reading ``result["tried_providers"]``.

    D7 contract: native dict carrying ``error``, ``tried_providers``
    (list[str]), ``errors_by_provider`` (dict[str,str]).
    """
    from mullet.server import mcp_handlers

    monkeypatch.setenv("KIMI_API_KEY", "fake")
    _clear_env(monkeypatch)

    f = tmp_path / "x.py"
    f.write_text("pass\n")

    kimi = _mock_provider("kimi", "kimi-k2.5")
    kimi.create.side_effect = TimeoutError("boom kimi")
    gemini = _mock_provider("gemini", "gemini-2.5-flash")
    gemini.create.side_effect = RuntimeError("boom gemini")

    monkeypatch.setattr(mcp_handlers, "_resolve_provider", lambda **kw: kimi)
    monkeypatch.setattr(mcp_handlers, "_build_fallback_chain", lambda **kw: [gemini])
    monkeypatch.setattr(mcp_handlers, "_write_agent_state_fn", lambda *a, **kw: None)
    monkeypatch.setattr(mcp_handlers, "_clear_agent_state", lambda: None)

    result = mcp_handlers.handle_mullet_run({
        "task": "do thing",
        "tier": "ADVANCED",
    })

    assert isinstance(result, dict), (
        f"D7 all-providers-failed path must return dict, got "
        f"{type(result).__name__!r}"
    )
    assert "error" in result
    assert "Agent execution failed" in result["error"]
    assert isinstance(result.get("tried_providers"), list)
    assert "kimi" in result["tried_providers"]
    assert "gemini" in result["tried_providers"]


# ---------------------------------------------------------------------------
# 5. Empty-task validation -> dict (not string-of-JSON).
# ---------------------------------------------------------------------------


def test_handle_mullet_run_returns_dict_on_empty_task():
    """Red (pre-D7, Cycle 17): the very first input-validation early-out
    returned ``json.dumps({"error": "task is required"})``.

    Failure mode: the simplest possible programming-error adapter
    crashed at the very first defensive check.

    D7 contract: input-validation early-exits return native dict.
    """
    from mullet.server.mcp_handlers import handle_mullet_run

    for arg in ({}, {"task": ""}, {"task": None}):
        result = handle_mullet_run(arg)
        assert isinstance(result, dict), (
            f"D7 input-validation must return dict for {arg!r}, got "
            f"{type(result).__name__!r}"
        )
        assert result.get("error") == "task is required"


# ---------------------------------------------------------------------------
# 6. MCP protocol wrapper json-encodes the dict for stdio.
# ---------------------------------------------------------------------------


def test_mcp_wrapper_jsonifies_dict_for_protocol(monkeypatch):
    """Red (pre-D7, Cycle 17): converting ``handle_mullet_run`` to return
    a dict would have broken the MCP stdio transport — the wrapper
    pasted ``result_text`` straight into the ``"text"`` field, which
    must be a string per JSON-RPC.

    Failure mode (post-D7 hypothetical): MCP clients receive a dict in
    a string-typed field and fail to deserialise.

    This test pins the wrapper invariant: the protocol layer detects
    dict returns and ``json.dumps`` them before sending. String returns
    from other handlers (e.g. ``handle_mullet_status``) pass through
    unchanged.
    """
    from mullet.server import mcp as mcp_protocol

    fake_dict_handler = lambda args: {"completed": True, "answer": 42}  # noqa: E731
    fake_str_handler = lambda args: "raw markdown text"  # noqa: E731

    monkeypatch.setitem(
        mcp_protocol.MCP_HANDLERS, "fake_dict_tool", fake_dict_handler
    )
    monkeypatch.setitem(
        mcp_protocol.MCP_HANDLERS, "fake_str_tool", fake_str_handler
    )

    # Dict handler: wrapper must JSON-encode before wrapping in the
    # text content field.
    resp_dict = mcp_protocol.handle_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "fake_dict_tool", "arguments": {}},
    })
    text_dict = resp_dict["result"]["content"][0]["text"]
    assert isinstance(text_dict, str), (
        "MCP protocol field must be a string per JSON-RPC; "
        f"got {type(text_dict).__name__!r}"
    )
    assert json.loads(text_dict) == {"completed": True, "answer": 42}

    # String handler: passes through unchanged (other handlers like
    # handle_mullet_status still return JSON-string for human display).
    resp_str = mcp_protocol.handle_request({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "fake_str_tool", "arguments": {}},
    })
    assert resp_str["result"]["content"][0]["text"] == "raw markdown text"
