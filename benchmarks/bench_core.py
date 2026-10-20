"""Performance benchmarks for Mullet core operations."""
import time
import statistics

from mullet.core.agent.runner import AgentRunner
from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.tools.registry import ToolRegistry
from mullet.core.router.resource_guard import ResourceGuard


class BenchProvider(Provider):
    """Instant mock provider for benchmarking."""
    name = "bench"
    model = "bench-instant"

    def create(self, messages, tools=None, **kwargs):
        return LLMResponse(
            content="Done.", tool_calls=None, finish_reason="stop",
            usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            model=self.model, provider=self.name,
        )


def bench(fn, n=10, label=""):
    """Run fn() n times, print stats."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    mean = statistics.mean(times) * 1000
    mn = min(times) * 1000
    mx = max(times) * 1000
    print(f"  {label:<40} mean={mean:>8.2f}ms  min={mn:>8.2f}ms  max={mx:>8.2f}ms")
    return result


def main():
    print("Mullet Core Benchmarks")
    print("=" * 70)

    # 1. Agent.run() with mock provider
    print("\n1. AgentRunner.run() — single turn, mock provider")
    agent = AgentRunner(
        provider=BenchProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=5,
    )
    bench(lambda: agent.run("Say hello"), n=10, label="agent.run()")

    # 2. ResourceGuard.can_dispatch_local()
    print("\n2. ResourceGuard.can_dispatch_local()")
    guard = ResourceGuard()
    bench(lambda: guard.can_dispatch_local(), n=50, label="can_dispatch_local()")

    # 3. ResourceGuard.get_system_snapshot()
    print("\n3. ResourceGuard.get_system_snapshot()")
    bench(lambda: guard.get_system_snapshot(), n=50, label="get_system_snapshot()")

    # 4. ResourceGuard.flush_resources()
    print("\n4. ResourceGuard.flush_resources()")
    bench(lambda: guard.flush_resources(), n=5, label="flush_resources()")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
