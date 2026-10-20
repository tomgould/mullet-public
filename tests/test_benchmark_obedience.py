"""Obedience tests for `mullet benchmark` (G26).

`mullet benchmark` historically hardcoded `detect_local_provider()` and a
fixed toolset list, ignoring `--provider` / `--providers` envelopes the
user passed on the CLI. These tests pin the new contract:

- `run_benchmark(provider_name=...)` resolves through the central
  `resolve_provider` helper instead of always calling
  `detect_local_provider`.
- `run_benchmark(allowed_providers=...)` is forwarded to `AgentRunner`
  via the `allowed_providers` constructor kwarg.
- A resolved provider that is not in the allowlist raises ValueError
  (loud trust violation, never silent).
- When the resolver returns None and an allowlist is set, agent tasks
  are marked failed with an "No allowed provider" error string.
- Default path (no provider_name, no allowlist) still uses
  `detect_local_provider` — backward compatible.
- Complexity-only tasks (router scoring) still execute even when no
  provider is available, since they don't need an LLM.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _stub_provider(name: str) -> MagicMock:
    """MagicMock provider with real `.name` and `.model` attrs."""
    p = MagicMock()
    p.name = name
    p.model = f"{name}-model"
    return p


def _stub_agent_result(text: str = "ok") -> MagicMock:
    """Build a stub AgentResult-like object that satisfies the benchmark loop."""
    r = MagicMock()
    r.completed = True
    r.final_response = text
    r.tool_calls_made = 1
    tu = MagicMock()
    tu.total_tokens = 100
    r.tokens_used = tu
    return r


def test_benchmark_resolves_named_provider():
    """Expected Red failure: TypeError — run_benchmark() got an unexpected keyword argument 'provider_name'."""
    from mullet.cli import benchmark as bench_mod

    kimi = _stub_provider("kimi")

    with patch.object(bench_mod, "_resolve_named_provider", return_value=kimi, create=True) as resolver, \
         patch("mullet.core.providers.local_detection.detect_local_provider") as detect_local, \
         patch("mullet.core.agent.runner.AgentRunner") as MockRunner:
        instance = MockRunner.return_value
        instance.run.return_value = _stub_agent_result("BENCHMARK_OK 56 1024")
        bench_mod.run_benchmark(verbose=False, provider_name="kimi")

    assert resolver.called, "named-provider resolver was not called"
    detect_local.assert_not_called()
    # AgentRunner must have received the resolved kimi stub
    kwargs_used = MockRunner.call_args_list[0].kwargs
    assert kwargs_used["provider"] is kimi


def test_benchmark_passes_allowlist_to_agent_runner():
    """Expected Red failure: TypeError — run_benchmark() got an unexpected keyword argument 'allowed_providers'."""
    from mullet.cli import benchmark as bench_mod

    kimi = _stub_provider("kimi")

    with patch.object(bench_mod, "_resolve_named_provider", return_value=kimi, create=True), \
         patch("mullet.core.providers.local_detection.detect_local_provider", return_value=kimi), \
         patch("mullet.core.agent.runner.AgentRunner") as MockRunner:
        instance = MockRunner.return_value
        instance.run.return_value = _stub_agent_result("BENCHMARK_OK 56 1024")
        bench_mod.run_benchmark(
            verbose=False,
            provider_name="kimi",
            allowed_providers=frozenset({"kimi"}),
        )

    assert MockRunner.called, "AgentRunner was never instantiated"
    kwargs_used = MockRunner.call_args_list[0].kwargs
    assert kwargs_used.get("allowed_providers") == frozenset({"kimi"}), (
        f"allowed_providers not forwarded; saw kwargs: {kwargs_used}"
    )


def test_benchmark_raises_when_resolved_provider_not_in_allowlist():
    """Expected Red failure: TypeError — run_benchmark() got an unexpected keyword argument 'provider_name'."""
    from mullet.cli import benchmark as bench_mod

    opus = _stub_provider("opus")

    with patch.object(bench_mod, "_resolve_named_provider", return_value=opus, create=True):
        with pytest.raises(ValueError) as excinfo:
            bench_mod.run_benchmark(
                verbose=False,
                provider_name="opus",
                allowed_providers=frozenset({"kimi"}),
            )

    msg = str(excinfo.value)
    assert "opus" in msg, f"error message does not mention 'opus': {msg}"
    assert "kimi" in msg, f"error message does not mention 'kimi': {msg}"


def test_benchmark_marks_results_failed_when_no_allowed_provider():
    """Expected Red failure: TypeError — run_benchmark() got an unexpected keyword argument 'allowed_providers'."""
    from mullet.cli import benchmark as bench_mod

    # Resolver returns None (e.g. no API key for kimi)
    with patch.object(bench_mod, "_resolve_named_provider", return_value=None, create=True), \
         patch("mullet.core.providers.local_detection.detect_local_provider", return_value=None):
        results = bench_mod.run_benchmark(
            verbose=False,
            provider_name=None,
            allowed_providers=frozenset({"kimi"}),
        )

    agent_results = [r for r in results if not _is_complexity(r.name)]
    assert agent_results, "no agent results produced — expected non-empty list"
    for r in agent_results:
        assert not r.passed, f"agent task {r.name} unexpectedly passed without provider"
        assert "No allowed provider" in r.error, (
            f"agent task {r.name}: expected 'No allowed provider' in error, got {r.error!r}"
        )


def test_benchmark_default_path_unchanged():
    """Expected Red failure: TypeError — run_benchmark() got an unexpected keyword argument 'provider_name'.

    With no provider_name and no allowed_providers, behaviour must match
    the legacy code path: call detect_local_provider, run all tasks.
    """
    from mullet.cli import benchmark as bench_mod

    local = _stub_provider("local")

    with patch("mullet.core.providers.local_detection.detect_local_provider", return_value=local) as detect_local, \
         patch("mullet.core.agent.runner.AgentRunner") as MockRunner:
        instance = MockRunner.return_value
        instance.run.return_value = _stub_agent_result("BENCHMARK_OK 56 1024")
        results = bench_mod.run_benchmark(verbose=False)

    assert detect_local.called, "detect_local_provider was not used in default path"
    # Provider passed to AgentRunner must be the local stub
    kwargs_used = MockRunner.call_args_list[0].kwargs
    assert kwargs_used["provider"] is local
    # And we got results back
    assert results, "no benchmark results returned"


def test_complexity_only_benchmarks_run_without_provider():
    """Expected Red failure: TypeError — run_benchmark() got an unexpected keyword argument 'allowed_providers'.

    Complexity-check tasks score complexity locally; they don't need an
    LLM. They must produce results even when no allowed provider exists.
    """
    from mullet.cli import benchmark as bench_mod

    with patch.object(bench_mod, "_resolve_named_provider", return_value=None, create=True), \
         patch("mullet.core.providers.local_detection.detect_local_provider", return_value=None):
        results = bench_mod.run_benchmark(
            verbose=False,
            provider_name=None,
            allowed_providers=frozenset({"kimi"}),
        )

    complexity_results = [r for r in results if _is_complexity(r.name)]
    assert complexity_results, "complexity-check results missing"
    for r in complexity_results:
        # complexity tasks don't carry provider errors, just pass/fail on tier
        assert "Score:" in (r.details or ""), (
            f"complexity task {r.name} missing details: {r.details!r}"
        )


# ---------- helpers ----------


_COMPLEXITY_NAMES = {
    "routing_simple",
    "routing_critical",
    "routing_advanced",
    "routing_standard",
}


def _is_complexity(name: str) -> bool:
    return name in _COMPLEXITY_NAMES
