"""Mullet benchmark — test agent quality against known tasks.

Runs a set of predefined tasks and measures success rate,
tool usage, and quality. Useful for comparing providers
and validating that changes don't regress agent behavior.
"""

import argparse
import time
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

BENCHMARK_TASKS = [
    {
        "name": "simple_question",
        "prompt": "What is 7 * 8? Answer with just the number.",
        "expected_contains": "56",
        "max_iterations": 3,
        "should_use_tools": False,
    },
    {
        "name": "file_listing",
        "prompt": "Use the terminal tool to run 'echo BENCHMARK_OK' and tell me the output.",
        "expected_contains": "BENCHMARK_OK",
        "max_iterations": 5,
        "should_use_tools": True,
    },
    {
        "name": "code_execution",
        "prompt": "Use the code_exec tool to run: print(2**10)",
        "expected_contains": "1024",
        "max_iterations": 5,
        "should_use_tools": True,
    },
    {
        "name": "web_search",
        "prompt": "Search the web for 'Python programming language' and tell me one fact.",
        "expected_contains": None,  # Just needs to complete
        "max_iterations": 5,
        "should_use_tools": True,
    },
    {
        "name": "git_status",
        "prompt": "Use the git tool to check the repository status and tell me the current branch.",
        "expected_contains": None,
        "max_iterations": 5,
        "should_use_tools": True,
    },
    {
        "name": "routing_simple",
        "prompt": "list files",
        "expected_tier": "LOW",
        "complexity_check": True,
    },
    {
        "name": "routing_critical",
        "prompt": "redesign the entire authentication architecture to support OAuth2, SAML, and OpenID Connect across three microservices with zero-downtime deployment, comprehensive security audit, and database migration with rollback strategy",
        "expected_tier": "CRITICAL",
        "complexity_check": True,
    },
    {
        "name": "routing_advanced",
        "prompt": "refactor the payment processing module to support Stripe and PayPal, including database migration and error handling for all payment failure modes",
        "expected_tier": "ADVANCED",
        "complexity_check": True,
    },
    {
        "name": "routing_standard",
        "prompt": "refactor the user profile component to use React hooks instead of class components",
        "expected_tier": "STANDARD",
        "complexity_check": True,
    },
]


@dataclass
class BenchmarkResult:
    """Single result from a benchmark task."""
    name: str
    passed: bool
    duration: float = 0.0
    tool_calls: int = 0
    tokens: int = 0
    error: str = ""
    details: str = ""


def _resolve_named_provider(provider_name: str, allowed_providers: frozenset[str] | None):
    """Resolve a named provider via the central CLI resolver.

    Builds a minimal argparse.Namespace with the same shape `resolve_provider`
    expects (provider, providers, base_url, model, speed). Returns None if
    the resolver can't materialise the provider (e.g. missing API key).
    """
    from mullet.cli.main import resolve_provider

    providers_csv = ",".join(sorted(allowed_providers)) if allowed_providers else None
    args = argparse.Namespace(
        provider=provider_name,
        providers=providers_csv,
        base_url=None,
        model=None,
        speed=None,
    )
    try:
        return resolve_provider(args, allowed_providers=allowed_providers)
    except ValueError:
        # Conflict between --provider and --providers — let caller surface it.
        raise
    except Exception as exc:
        logger.warning("benchmark: resolve_provider failed for %r: %s", provider_name, exc)
        return None


def run_benchmark(
    verbose: bool = False,
    provider_name: str | None = None,
    allowed_providers: frozenset[str] | None = None,
) -> list[BenchmarkResult]:
    """Run all benchmark tasks and return results.

    Args:
        verbose: Print per-task PASS/FAIL lines.
        provider_name: Force a named provider via the central resolver
            (e.g. "kimi", "haiku"). When None, falls back to
            `detect_local_provider` for backward compatibility.
        allowed_providers: Optional `--providers` allowlist. If set, the
            resolved provider must be a member; otherwise ValueError is
            raised. Forwarded to AgentRunner so any downstream fallback
            also stays inside the envelope.

    Raises:
        ValueError: when the resolved provider is not in
            `allowed_providers`.
    """
    from mullet.cli.main import build_tool_registry
    from mullet.core.providers.local_detection import detect_local_provider
    from mullet.core.agent.runner import AgentRunner
    from mullet.core.agent.prompts import build_base_prompt
    from mullet.core.router.complexity import extract_signals, score_complexity, score_to_tier

    results = []

    # Check for complexity-only benchmarks first (no provider needed).
    for task in BENCHMARK_TASKS:
        if task.get("complexity_check"):
            signals = extract_signals(task["prompt"])
            score = score_complexity(signals)
            tier = score_to_tier(score)
            passed = tier.value == task.get("expected_tier", "")
            results.append(BenchmarkResult(
                name=task["name"],
                passed=passed,
                details=f"Score: {score:.1f}, Tier: {tier.value} (expected: {task.get('expected_tier')})",
            ))
            if verbose:
                status = "PASS" if passed else "FAIL"
                print(f"  {status} {task['name']}: {tier.value} (score {score:.1f})")
            continue

    # Agent benchmarks require a provider.
    if provider_name:
        provider = _resolve_named_provider(provider_name, allowed_providers)
    else:
        provider = detect_local_provider()

    # Allowlist enforcement: a resolved provider must be a member.
    if provider is not None and allowed_providers:
        resolved_name = getattr(provider, "name", None) or provider_name or ""
        if resolved_name not in allowed_providers:
            raise ValueError(
                "benchmark: resolved provider %r is not in allowlist %s"
                % (resolved_name, sorted(allowed_providers))
            )

    if not provider:
        error_msg = (
            "No allowed provider available"
            if allowed_providers
            else "No LLM provider available"
        )
        for task in BENCHMARK_TASKS:
            if not task.get("complexity_check"):
                results.append(BenchmarkResult(
                    name=task["name"], passed=False, error=error_msg,
                ))
        return results

    registry = build_tool_registry()
    system_prompt = build_base_prompt(model=getattr(provider, "model", ""))

    for task in BENCHMARK_TASKS:
        if task.get("complexity_check"):
            continue  # Already handled above

        start = time.time()
        try:
            agent = AgentRunner(
                provider=provider,
                tool_registry=registry,
                toolsets=["terminal", "file", "web", "code", "browser", "git", "testing", "vision", "docker"],
                system_prompt=system_prompt,
                max_iterations=task.get("max_iterations", 5),
                allowed_providers=allowed_providers,
            )
            result = agent.run(task["prompt"])
            duration = time.time() - start

            passed = result.completed
            if task.get("expected_contains") and passed:
                passed = task["expected_contains"] in result.final_response
            if task.get("should_use_tools") and result.tool_calls_made == 0:
                passed = False

            details = f"{result.tool_calls_made} tools, {result.tokens_used.total_tokens} tokens, {duration:.1f}s"
            results.append(BenchmarkResult(
                name=task["name"],
                passed=passed,
                duration=duration,
                tool_calls=result.tool_calls_made,
                tokens=result.tokens_used.total_tokens,
                details=details,
            ))

            if verbose:
                status = "PASS" if passed else "FAIL"
                print(f"  {status} {task['name']}: {details}")

        except Exception as e:
            results.append(BenchmarkResult(
                name=task["name"], passed=False, error=str(e),
                duration=time.time() - start,
            ))
            if verbose:
                print(f"  FAIL {task['name']}: {e}")

    return results


def print_benchmark_summary(results: list[BenchmarkResult]):
    """Print a formatted benchmark summary."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    total_time = sum(r.duration for r in results)
    total_tokens = sum(r.tokens for r in results)

    print(f"\nBenchmark Results: {passed}/{total} passed")
    print(f"Total time: {total_time:.1f}s")
    print(f"Total tokens: {total_tokens:,}")
    print()

    for r in results:
        icon = "✓" if r.passed else "✗"
        print(f"  {icon} {r.name:25s} {r.details or r.error}")
