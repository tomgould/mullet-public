"""E2E smoke test for the swarm orchestrator.

Requires:
  - KIMI_API_KEY or MOONSHOT_API_KEY env var set
  - Real Kimi API access

Run with:  pytest tests/test_swarm_e2e.py -m live -s
"""

import asyncio
import os
import pathlib

import pytest
import yaml

pytestmark = pytest.mark.live


@pytest.fixture
def kimi_key():
    # Check environment variables first
    key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
    
    # Check config file as fallback
    if not key:
        config_path = pathlib.Path.home() / ".mullet" / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    key = (
                        config.get("api_keys", {}).get("kimi") or
                        config.get("api_keys", {}).get("moonshot")
                    )
            except Exception:
                pass
    
    if not key:
        pytest.skip(
            "No Kimi API key found. Set KIMI_API_KEY or MOONSHOT_API_KEY env vars, "
            "or add 'api_keys.kimi' or 'api_keys.moonshot' to ~/.mullet/config.yaml"
        )
    os.environ.setdefault("KIMI_API_KEY", key)
    return key


def test_swarm_dry_run_produces_tasks(kimi_key):
    """Dry run: LLMDecomposer splits goal into tasks without dispatching."""
    from mullet.core.orchestration.swarm import run_swarm
    from mullet.core.orchestration.swarm_types import SwarmContext

    ctx = SwarmContext(
        items=["Python", "TypeScript", "Go"],
        output_shape="bullet list",
        concurrency=3,
    )
    result = asyncio.run(run_swarm("Compare these languages for backend API dev", ctx, dry_run=True))
    assert result.tasks, "dry_run should produce at least one task"
    assert result.synthesis == "", "dry_run must not synthesize"
    print(f"\n  Planned {len(result.tasks)} tasks")
    for t in result.tasks:
        print(f"    [{t.id}] {t.instruction[:80]}")


def test_swarm_static_items_full_run(kimi_key):
    """Full run with 3 items — dispatches to Kimi, synthesizes."""
    from mullet.core.orchestration.swarm import run_swarm
    from mullet.core.orchestration.swarm_types import SwarmContext
    from mullet.core.orchestration.static_decomposer import StaticDecomposer

    ctx = SwarmContext(
        items=["asyncio", "threading", "multiprocessing"],
        output_shape="markdown table",
        concurrency=3,
    )
    result = asyncio.run(
        run_swarm(
            "Summarise when to use each Python concurrency primitive in one sentence",
            ctx,
            decomposer=StaticDecomposer(),
        )
    )
    assert len(result.tasks) == 3
    assert result.synthesis, "synthesis should not be empty"
    assert result.wall_clock_seconds > 0
    print(f"\n  {len(result.tasks)} tasks, {result.wall_clock_seconds:.1f}s")
    print(f"  Tokens: {result.total_tokens.input_tokens} in / {result.total_tokens.output_tokens} out")
    print(f"  Synthesis ({len(result.synthesis)} chars):\n{result.synthesis[:300]}")
