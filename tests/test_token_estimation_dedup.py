"""P2 cycle 21 — verify char-to-token heuristics now use the canonical helper.

The canonical helper is `mullet.core.providers.tokenize.estimate_tokens`
(P1-2, 95f9043). Cycle 11 (D1, e7bc1fc) replaced the dispatch-time
heuristic in `swarm_dispatcher.py`. This module asserts the remaining sites
either use the helper or are documented intentional fallbacks.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


from mullet.core.orchestration import swarm
from mullet.core.orchestration.swarm_types import SwarmTask


REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# swarm.py — _estimate_cost must call estimate_tokens, not char // 4         #
# --------------------------------------------------------------------------- #
def test_swarm_estimate_cost_uses_estimate_tokens():
    """Failure mode: `swarm._estimate_cost` still uses raw `len(...) / 4`,
    so it never invokes `estimate_tokens` and therefore over-estimates JSON
    payloads by 20-40%."""
    tasks = [
        SwarmTask(id="t1", instruction="abcdefgh" * 100),  # 800 chars
        SwarmTask(id="t2", instruction="ijklmnop" * 50),   # 400 chars
    ]
    sentinel_value = 12345
    with patch(
        "mullet.core.providers.tokenize.estimate_tokens",
        return_value=sentinel_value,
    ) as mock_est:
        total_tokens, _cost = swarm._estimate_cost(tasks)

    assert mock_est.call_count == len(tasks), (
        f"estimate_tokens should be called once per task; got {mock_est.call_count}"
    )
    # Each task contributes sentinel + 200 overhead → 2 * (12345 + 200)
    assert total_tokens == 2 * (sentinel_value + 200), (
        f"total_tokens={total_tokens} — _estimate_cost did not use the mocked helper"
    )


def test_swarm_estimate_cost_no_legacy_char_div_four():
    """Failure mode: `swarm.py` still contains the literal `/ 4 * 1.3`
    expression, which is the old char-to-token heuristic."""
    src = (REPO_ROOT / "mullet/core/orchestration/swarm.py").read_text()
    assert "/ 4 * 1.3" not in src, (
        "swarm.py still contains the legacy '/ 4 * 1.3' char-to-token heuristic"
    )
    assert "prompt_chars / 4" not in src, (
        "swarm.py still contains the legacy 'prompt_chars / 4' heuristic"
    )


# --------------------------------------------------------------------------- #
# swarm_dispatcher.py — D1 already done; assert the legacy form is gone     #
# --------------------------------------------------------------------------- #
def test_swarm_dispatcher_no_legacy_div_four_times_two():
    """Failure mode: D1 (cycle 11) replaced `prompt_len // 4 * 2` with
    `estimate_tokens(prompt)`. Regression check: the old expression must
    not reappear."""
    src = (REPO_ROOT / "mullet/core/orchestration/swarm_dispatcher.py").read_text()
    assert "// 4 * 2" not in src, (
        "swarm_dispatcher.py reintroduced the legacy '// 4 * 2' heuristic"
    )
    # Confirm the canonical import is still present.
    assert "from mullet.core.providers.tokenize import estimate_tokens" in src, (
        "swarm_dispatcher.py no longer imports the canonical estimate_tokens helper"
    )


def test_swarm_dispatcher_fallback_div_six_is_documented():
    """The `prompt_len // 6` expression is a documented fallback for when
    `estimate_tokens` itself raises. It must remain inside the except branch
    or the `prompt is None` branch — not as the primary path.

    Failure mode: the `// 6` heuristic moved out of the except/else branches
    and back into the main happy path.
    """
    src = (REPO_ROOT / "mullet/core/orchestration/swarm_dispatcher.py").read_text()
    # The fallback comment must reference estimate_tokens
    assert "estimate_tokens" in src
    # Every executable `// 6` site (i.e., not in a docstring/comment line) must
    # live inside the documented fallback path that runs when estimate_tokens
    # raises or when prompt is None.
    lines = src.splitlines()
    code_div_six_lines = [
        i for i, ln in enumerate(lines)
        if "// 6" in ln and "max(1, prompt_len // 6)" in ln
    ]
    assert code_div_six_lines, (
        "expected at least one executable `// 6` fallback line in swarm_dispatcher"
    )
    for idx in code_div_six_lines:
        window = "\n".join(lines[max(0, idx - 6): idx + 1]).lower()
        assert "fallback" in window or "except" in window or "else" in window, (
            f"line {idx + 1} `// 6` is not in a documented fallback context"
        )


# --------------------------------------------------------------------------- #
# prompt_cache.py — semantic divergence: assert no char-to-token heuristic   #
# --------------------------------------------------------------------------- #
def test_prompt_cache_has_no_char_to_token_heuristic():
    """`prompt_cache.py` does NOT do token estimation — it manages
    cache_control breakpoints (Anthropic max=4). The TODO mentioned it as a
    candidate but inspection shows it's a different concern.

    Failure mode: this test fails if someone later adds a char-to-token
    heuristic into prompt_cache.py without routing it through the canonical
    helper.
    """
    src = (REPO_ROOT / "mullet/core/providers/prompt_cache.py").read_text()
    # No `len(...) // N` token estimation patterns
    import re
    matches = re.findall(r"len\([^)]+\)\s*//\s*[3-6]\b", src)
    assert not matches, (
        f"prompt_cache.py introduced char-to-token heuristic(s): {matches} — "
        f"use mullet.core.providers.tokenize.estimate_tokens instead"
    )


# --------------------------------------------------------------------------- #
# Cross-cutting: every dispatch-affecting path now goes through the helper   #
# --------------------------------------------------------------------------- #
def test_swarm_uses_estimate_tokens_consistently_in_all_paths():
    """Failure mode: a dispatch/cost path bypasses estimate_tokens.

    Walks the two known callers (`swarm._estimate_cost`,
    `SwarmDispatcher._make_provider`) and confirms each invokes the helper
    when given non-empty input.
    """
    # 1. swarm._estimate_cost
    with patch(
        "mullet.core.providers.tokenize.estimate_tokens", return_value=42
    ) as mock_est:
        swarm._estimate_cost([SwarmTask(id="x", instruction="hello world")])
        assert mock_est.called, "_estimate_cost must call estimate_tokens"

    # 2. SwarmDispatcher._make_provider — when prompt is provided
    from mullet.core.orchestration.swarm_dispatcher import SwarmDispatcher

    with patch(
        "mullet.core.providers.tokenize.estimate_tokens", return_value=99
    ) as mock_est:
        try:
            SwarmDispatcher()._make_provider("STANDARD", 100, prompt="hello world")
        except Exception:
            # Provider construction may fail without API keys/registry — that's
            # fine. We only care that estimate_tokens was invoked.
            pass
        assert mock_est.called, "_make_provider must call estimate_tokens when prompt is given"
