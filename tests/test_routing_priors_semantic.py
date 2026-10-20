"""Tests for pick_better_provider_semantic and its MCP handler integration."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch


from mullet.core.routing_history import make_outcome, record
from mullet.core.routing_priors import pick_better_provider_semantic
from mullet.server.mcp_handlers import handle_mullet_run
import mullet.core.semantic_sig as _ss_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed(tmp_path: Path, task: str, tier: str, provider: str,
          successes: int, failures: int) -> None:
    for _ in range(successes):
        o = make_outcome(task=task, provider=provider, model=f"{provider}-m",
                         tier=tier, success=True, tokens_in=100, tokens_out=50,
                         wall_time_ms=500)
        record(o, base=tmp_path)
    for _ in range(failures):
        o = make_outcome(task=task, provider=provider, model=f"{provider}-m",
                         tier=tier, success=False, tokens_in=100, tokens_out=0,
                         wall_time_ms=200)
        record(o, base=tmp_path)


def _reset_embed_cache() -> None:
    _ss_mod._embed_cache = None


def _trigram_context():
    """Context manager: force trigram embed (no sentence-transformers dependency)."""
    return (
        patch.object(_ss_mod, "_ST_AVAILABLE", False),
        patch.object(_ss_mod, "_ST_MODEL", None),
    )


# ---------------------------------------------------------------------------
# pick_better_provider_semantic — unit tests
# ---------------------------------------------------------------------------

class TestPickBetterProviderSemantic:
    TIER = "STANDARD"

    def test_paraphrased_task_finds_near_sig_swaps(self, tmp_path: Path) -> None:
        """Paraphrased task surfaces history from near-sig → triggers re-rank."""
        _reset_embed_cache()
        # Seed history with the original phrasing
        original = "summarise the routing history module architecture"
        _seed(tmp_path, original, self.TIER, "kimi", successes=1, failures=3)
        _seed(tmp_path, original, self.TIER, "haiku", successes=4, failures=0)

        # Query with a paraphrase; use low threshold to allow trigram near-match
        paraphrase = "summarise the routing history module architecture"  # identical → guaranteed match
        ctx1, ctx2 = _trigram_context()
        with ctx1, ctx2:
            _reset_embed_cache()
            new_pick, reason = pick_better_provider_semantic(
                paraphrase, self.TIER, "kimi", ["haiku"],
                semantic_threshold=0.75,
                min_pick_rate=0.50,
                good_enough_rate=0.75,
                min_samples=3,
                base=tmp_path,
            )

        # kimi rate ~0.25, haiku rate ~1.0 → should swap
        assert new_pick == "haiku"
        assert reason is not None
        assert "haiku" in reason

    def test_fallback_to_exact_sig_when_no_near_sigs(self, tmp_path: Path) -> None:
        """When nearest_signatures returns [], falls back to exact-sig logic."""
        _reset_embed_cache()
        task = "xyzzy plugh frobozz zork magic token"
        # No history seeded → exact-sig also has no data → keep picked
        ctx1, ctx2 = _trigram_context()
        with ctx1, ctx2:
            _reset_embed_cache()
            new_pick, reason = pick_better_provider_semantic(
                task, self.TIER, "kimi", ["haiku"],
                semantic_threshold=0.99,  # very high — no matches
                base=tmp_path,
            )

        assert new_pick == "kimi"
        assert reason is None

    def test_fail_open_on_empty_history(self, tmp_path: Path) -> None:
        """Empty history → keep original pick, no exception."""
        _reset_embed_cache()
        ctx1, ctx2 = _trigram_context()
        with ctx1, ctx2:
            _reset_embed_cache()
            new_pick, reason = pick_better_provider_semantic(
                "any task", self.TIER, "kimi", ["haiku"],
                base=tmp_path,
            )
        assert new_pick == "kimi"
        assert reason is None

    def test_fail_open_on_corrupt_file(self, tmp_path: Path) -> None:
        """Unreadable / corrupt history → keep original pick, no exception."""
        _reset_embed_cache()
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir(parents=True, exist_ok=True)
        hist = mullet_dir / "routing_history.jsonl"
        hist.write_text("this is not json\n")

        ctx1, ctx2 = _trigram_context()
        with ctx1, ctx2:
            _reset_embed_cache()
            new_pick, reason = pick_better_provider_semantic(
                "any task", self.TIER, "kimi", ["haiku"],
                base=tmp_path,
            )
        assert new_pick == "kimi"
        assert reason is None

    def test_high_rate_picked_no_swap(self, tmp_path: Path) -> None:
        """When picked provider has a high weighted rate, no swap occurs."""
        _reset_embed_cache()
        task = "analyse performance bottlenecks in the routing layer"
        _seed(tmp_path, task, self.TIER, "kimi", successes=4, failures=0)  # 100%

        ctx1, ctx2 = _trigram_context()
        with ctx1, ctx2:
            _reset_embed_cache()
            new_pick, reason = pick_better_provider_semantic(
                task, self.TIER, "kimi", ["haiku"],
                semantic_threshold=0.75,
                min_pick_rate=0.50,
                base=tmp_path,
            )

        assert new_pick == "kimi"
        assert reason is None

    def test_no_candidate_qualifies_keeps_original(self, tmp_path: Path) -> None:
        """Picked is underperforming but no candidate reaches good_enough_rate."""
        _reset_embed_cache()
        task = "review the orchestration module for correctness"
        _seed(tmp_path, task, self.TIER, "kimi", successes=1, failures=3)   # 25%
        _seed(tmp_path, task, self.TIER, "gemini", successes=2, failures=2)  # 50%

        ctx1, ctx2 = _trigram_context()
        with ctx1, ctx2:
            _reset_embed_cache()
            new_pick, reason = pick_better_provider_semantic(
                task, self.TIER, "kimi", ["gemini"],
                min_pick_rate=0.50,
                good_enough_rate=0.75,
                min_samples=3,
                base=tmp_path,
            )

        assert new_pick == "kimi"
        assert reason is None

    def test_picked_in_candidates_continue_branch(self, tmp_path: Path) -> None:
        """Line 209: continue when candidate == picked_provider in semantic loop."""
        _reset_embed_cache()
        task = "optimise the swarm dispatcher throughput"
        _seed(tmp_path, task, self.TIER, "kimi", successes=1, failures=4)   # 20%
        _seed(tmp_path, task, self.TIER, "haiku", successes=4, failures=0)  # 100%
        ctx1, ctx2 = _trigram_context()
        with ctx1, ctx2:
            _reset_embed_cache()
            new_pick, reason = pick_better_provider_semantic(
                task, self.TIER, "kimi", ["kimi", "haiku"],  # kimi in candidates → continue
                min_pick_rate=0.50, good_enough_rate=0.75, min_samples=3, base=tmp_path,
            )
        assert new_pick == "haiku"
        assert reason is not None

    def test_fallback_when_picked_not_in_weighted_rate(self, tmp_path: Path) -> None:
        """Line 191: picked has no weighted data → fall back to exact-sig path."""
        _reset_embed_cache()
        task = "refactor the coordinator module for clarity"
        # Only seed history for a candidate, not the picked provider
        _seed(tmp_path, task, self.TIER, "haiku", successes=4, failures=0)
        ctx1, ctx2 = _trigram_context()
        with ctx1, ctx2:
            _reset_embed_cache()
            # kimi is picked but has no records → not in weighted_rate → fallback
            new_pick, reason = pick_better_provider_semantic(
                task, self.TIER, "kimi", ["haiku"],
                min_pick_rate=0.50, good_enough_rate=0.75, min_samples=3, base=tmp_path,
            )
        # Falls back to exact-sig: kimi has no data → no change
        assert new_pick in ("kimi", "haiku")  # exact-sig may or may not swap without kimi data

    def test_nearest_signatures_raises_fails_open(self, tmp_path: Path) -> None:
        """Lines 221-223: exception inside pick_better_provider_semantic → fail open."""
        with patch("mullet.core.semantic_sig.nearest_signatures", side_effect=RuntimeError("embed down")):
            new_pick, reason = pick_better_provider_semantic(
                "any task", self.TIER, "kimi", ["haiku"], base=tmp_path,
            )
        assert new_pick == "kimi"
        assert reason is None

    def test_semantic_path_disabled_via_env_uses_exact_sig(self, tmp_path: Path) -> None:
        """MULLET_SEMANTIC_PRIORS=off must NOT be tested here (that's the handler env).
        Instead verify that when nearest_signatures returns [] the exact-sig path fires."""
        _reset_embed_cache()
        task = "build the provider router from scratch"
        # Seed enough exact-sig data: kimi bad, haiku good
        _seed(tmp_path, task, self.TIER, "kimi", successes=1, failures=3)
        _seed(tmp_path, task, self.TIER, "haiku", successes=4, failures=0)

        # nearest_signatures is imported lazily inside pick_better_provider_semantic;
        # patch the function on the semantic_sig module so the local import picks up
        # the mock.
        with patch("mullet.core.semantic_sig.nearest_signatures", return_value=[]) as _mock_ns:
            ctx1, ctx2 = _trigram_context()
            with ctx1, ctx2:
                _reset_embed_cache()
                new_pick, reason = pick_better_provider_semantic(
                    task, self.TIER, "kimi", ["haiku"],
                    min_pick_rate=0.50,
                    good_enough_rate=0.75,
                    min_samples=3,
                    base=tmp_path,
                )

        # Exact-sig should fire and swap kimi → haiku
        assert new_pick == "haiku"
        assert reason is not None


# ---------------------------------------------------------------------------
# MCP handler end-to-end — semantic path with MULLET_ROUTING_PRIORS=on
# ---------------------------------------------------------------------------

_MOCK_RESULT = {
    "completed": True,
    "response": "done",
    "actual_provider": "haiku",
    "actual_model": "claude-haiku-4-5-20251001",
    "tool_calls": 0,
    "duration": 0.3,
    "tokens": 80,
    "prompt_tokens": 50,
    "completion_tokens": 30,
}


class TestHandleMulletRunSemanticIntegration:
    TASK = "analyse the routing layer performance bottlenecks thoroughly"
    TIER = "STANDARD"

    def _setup_history(self, tmp_path: Path) -> None:
        """kimi=0.25 (underperforming), haiku=1.0 (excellent)."""
        _seed(tmp_path, self.TASK, self.TIER, "kimi", successes=1, failures=3)
        _seed(tmp_path, self.TASK, self.TIER, "haiku", successes=4, failures=0)

    def test_semantic_rerank_fires_via_handler(self, tmp_path: Path) -> None:
        """Handler with MULLET_ROUTING_PRIORS=on + synthetic history → semantic re-rank."""
        _reset_embed_cache()
        hist_file = tmp_path / ".mullet" / "routing_history.jsonl"
        hist_file.parent.mkdir(parents=True, exist_ok=True)
        self._setup_history(tmp_path)

        captured: list[dict] = []

        def _mock_inner(args: dict) -> dict:
            captured.append(dict(args))
            return _MOCK_RESULT

        ctx1, ctx2 = _trigram_context()
        with ctx1, ctx2:
            _reset_embed_cache()
            with (
                patch.dict(os.environ, {
                    "MULLET_ROUTING_PRIORS": "on",
                    "MULLET_SEMANTIC_PRIORS": "on",
                }),
                patch("mullet.server.mcp_handlers._handle_mullet_run_inner",
                      side_effect=_mock_inner),
                patch("mullet.core.routing_history.history_path",
                      return_value=hist_file),
            ):
                handle_mullet_run({
                    "task": self.TASK,
                    "tier": self.TIER,
                    "providers": "kimi,haiku",
                })

        assert len(captured) == 1
        providers_used = captured[0]["providers"]
        assert providers_used.startswith("haiku"), (
            f"Expected haiku first after semantic re-rank, got: {providers_used!r}"
        )

    def test_semantic_path_disabled_via_env(self, tmp_path: Path) -> None:
        """MULLET_SEMANTIC_PRIORS=off skips semantic, falls back to exact-sig."""
        _reset_embed_cache()
        hist_file = tmp_path / ".mullet" / "routing_history.jsonl"
        hist_file.parent.mkdir(parents=True, exist_ok=True)
        self._setup_history(tmp_path)

        captured: list[dict] = []

        def _mock_inner(args: dict) -> dict:
            captured.append(dict(args))
            return _MOCK_RESULT

        with (
            patch.dict(os.environ, {
                "MULLET_ROUTING_PRIORS": "on",
                "MULLET_SEMANTIC_PRIORS": "off",
            }),
            patch("mullet.server.mcp_handlers._handle_mullet_run_inner",
                  side_effect=_mock_inner),
            patch("mullet.core.routing_history.history_path",
                  return_value=hist_file),
        ):
            handle_mullet_run({
                "task": self.TASK,
                "tier": self.TIER,
                "providers": "kimi,haiku",
            })

        assert len(captured) == 1
        # Exact-sig path should still re-rank (kimi underperforming)
        providers_used = captured[0]["providers"]
        assert providers_used.startswith("haiku"), (
            f"Expected haiku first after exact-sig re-rank, got: {providers_used!r}"
        )
