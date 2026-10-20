"""Tests for mullet.core.routing_priors."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mullet.core.routing_history import make_outcome, record
from mullet.core.routing_priors import pick_better_provider, summarize_priors
from mullet.server.mcp_handlers import handle_mullet_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_history(tmp_path: Path, task: str, tier: str, provider: str, successes: int, failures: int) -> None:
    """Write engineered outcome rows to a tmp history file."""
    for _ in range(successes):
        o = make_outcome(task=task, provider=provider, model=f"{provider}-model",
                         tier=tier, success=True, tokens_in=100, tokens_out=50, wall_time_ms=500)
        record(o, base=tmp_path)
    for _ in range(failures):
        o = make_outcome(task=task, provider=provider, model=f"{provider}-model",
                         tier=tier, success=False, tokens_in=100, tokens_out=0, wall_time_ms=200)
        record(o, base=tmp_path)


# ---------------------------------------------------------------------------
# pick_better_provider unit tests
# ---------------------------------------------------------------------------

class TestPickBetterProvider:
    TASK = "summarise the codebase architecture in detail"
    TIER = "STANDARD"

    def test_high_rate_no_change(self, tmp_path: Path) -> None:
        """When picked provider has a high success rate, keep it."""
        _seed_history(tmp_path, self.TASK, self.TIER, "kimi", successes=4, failures=0)
        new_pick, reason = pick_better_provider(
            self.TASK, self.TIER, "kimi", ["gemini", "haiku"],
            min_pick_rate=0.50, good_enough_rate=0.75, min_samples=3, base=tmp_path,
        )
        assert new_pick == "kimi"
        assert reason is None

    def test_low_rate_candidate_qualifies_swaps(self, tmp_path: Path) -> None:
        """When picked has low rate and a candidate qualifies, swap to candidate."""
        # kimi: 1/4 = 0.25 (below min_pick_rate=0.50)
        _seed_history(tmp_path, self.TASK, self.TIER, "kimi", successes=1, failures=3)
        # haiku: 4/4 = 1.0 (above good_enough_rate=0.75)
        _seed_history(tmp_path, self.TASK, self.TIER, "haiku", successes=4, failures=0)

        new_pick, reason = pick_better_provider(
            self.TASK, self.TIER, "kimi", ["gemini", "haiku"],
            min_pick_rate=0.50, good_enough_rate=0.75, min_samples=3, base=tmp_path,
        )
        assert new_pick == "haiku"
        assert reason is not None
        assert "kimi" in reason
        assert "haiku" in reason

    def test_low_rate_no_candidate_qualifies_keeps_original(self, tmp_path: Path) -> None:
        """When picked is low-rate but no candidate reaches good_enough_rate, keep original."""
        # kimi: 1/4 = 0.25 (underperforming)
        _seed_history(tmp_path, self.TASK, self.TIER, "kimi", successes=1, failures=3)
        # gemini: 2/4 = 0.50 (below good_enough_rate=0.75)
        _seed_history(tmp_path, self.TASK, self.TIER, "gemini", successes=2, failures=2)

        new_pick, reason = pick_better_provider(
            self.TASK, self.TIER, "kimi", ["gemini"],
            min_pick_rate=0.50, good_enough_rate=0.75, min_samples=3, base=tmp_path,
        )
        assert new_pick == "kimi"
        assert reason is None

    def test_not_enough_samples_no_change(self, tmp_path: Path) -> None:
        """When picked has fewer than min_samples records, don't re-rank."""
        # Only 2 records — below min_samples=3
        _seed_history(tmp_path, self.TASK, self.TIER, "kimi", successes=0, failures=2)
        # haiku has plenty
        _seed_history(tmp_path, self.TASK, self.TIER, "haiku", successes=4, failures=0)

        new_pick, reason = pick_better_provider(
            self.TASK, self.TIER, "kimi", ["haiku"],
            min_pick_rate=0.50, good_enough_rate=0.75, min_samples=3, base=tmp_path,
        )
        assert new_pick == "kimi"
        assert reason is None

    def test_no_history_file_no_change(self, tmp_path: Path) -> None:
        """When the history file doesn't exist, keep the original pick."""
        new_pick, reason = pick_better_provider(
            self.TASK, self.TIER, "kimi", ["haiku"],
            min_pick_rate=0.50, good_enough_rate=0.75, min_samples=3, base=tmp_path,
        )
        assert new_pick == "kimi"
        assert reason is None

    def test_fail_open_on_error(self, tmp_path: Path) -> None:
        """Errors inside the function return (picked, None) without raising."""
        with patch("mullet.core.routing_priors.success_rate", side_effect=RuntimeError("boom")):
            new_pick, reason = pick_better_provider(
                self.TASK, self.TIER, "kimi", ["haiku"], base=tmp_path,
            )
        assert new_pick == "kimi"
        assert reason is None


# ---------------------------------------------------------------------------
# summarize_priors unit tests
# ---------------------------------------------------------------------------

class TestSummarizePriors:
    TASK = "refactor the entire authentication module"
    TIER = "ADVANCED"

    def test_returns_sorted_by_sample_count(self, tmp_path: Path) -> None:
        _seed_history(tmp_path, self.TASK, self.TIER, "kimi", successes=5, failures=1)
        _seed_history(tmp_path, self.TASK, self.TIER, "haiku", successes=2, failures=0)
        _seed_history(tmp_path, self.TASK, self.TIER, "gemini", successes=3, failures=0)

        result = summarize_priors(self.TASK, self.TIER, base=tmp_path)
        providers = list(result.keys())
        sample_counts = [result[p][1] for p in providers]
        assert sample_counts == sorted(sample_counts, reverse=True)

    def test_rate_and_count_correct(self, tmp_path: Path) -> None:
        _seed_history(tmp_path, self.TASK, self.TIER, "kimi", successes=3, failures=1)
        result = summarize_priors(self.TASK, self.TIER, base=tmp_path)
        assert "kimi" in result
        rate, count = result["kimi"]
        assert count == 4
        assert rate == pytest.approx(0.75)

    def test_empty_history_returns_empty_dict(self, tmp_path: Path) -> None:
        result = summarize_priors(self.TASK, self.TIER, base=tmp_path)
        assert result == {}

    def test_fail_open_on_error(self, tmp_path: Path) -> None:
        with patch("mullet.core.routing_priors.iter_recent", side_effect=RuntimeError("boom")):
            result = summarize_priors(self.TASK, self.TIER, base=tmp_path)
        assert result == {}


# ---------------------------------------------------------------------------
# End-to-end integration via handle_mullet_run
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


class TestHandleMulletRunPriorsIntegration:
    TASK = "analyse performance bottlenecks in the routing layer"
    TIER = "STANDARD"

    def _setup_history(self, tmp_path: Path) -> None:
        """kimi=0.25 (underperforming), haiku=1.0 (excellent)."""
        _seed_history(tmp_path, self.TASK, self.TIER, "kimi", successes=1, failures=3)
        _seed_history(tmp_path, self.TASK, self.TIER, "haiku", successes=4, failures=0)

    def test_rerank_fires_when_flag_on(self, tmp_path: Path) -> None:
        """With MULLET_ROUTING_PRIORS=on and bad kimi history, haiku is picked first."""
        hist_file = tmp_path / ".mullet" / "routing_history.jsonl"
        hist_file.parent.mkdir(parents=True, exist_ok=True)
        self._setup_history(tmp_path)
        captured_args: list[dict] = []

        def _mock_inner(args: dict) -> dict:
            captured_args.append(dict(args))
            return _MOCK_RESULT

        with (
            patch.dict(os.environ, {"MULLET_ROUTING_PRIORS": "on"}),
            patch("mullet.server.mcp_handlers._handle_mullet_run_inner", side_effect=_mock_inner),
            patch("mullet.core.routing_history.history_path", return_value=hist_file),
        ):
            handle_mullet_run({
                "task": self.TASK,
                "tier": self.TIER,
                "providers": "kimi,haiku",
            })

        assert len(captured_args) == 1
        providers_used = captured_args[0]["providers"]
        assert providers_used.startswith("haiku"), (
            f"Expected haiku first after re-rank, got: {providers_used!r}"
        )

    def test_rerank_tag_recorded(self, tmp_path: Path) -> None:
        """rerank tag appears in the history row when a re-rank fires."""
        hist_file = tmp_path / ".mullet" / "routing_history.jsonl"
        hist_file.parent.mkdir(parents=True, exist_ok=True)
        self._setup_history(tmp_path)

        with (
            patch.dict(os.environ, {"MULLET_ROUTING_PRIORS": "on"}),
            patch("mullet.server.mcp_handlers._handle_mullet_run_inner", return_value=_MOCK_RESULT),
            patch("mullet.core.routing_history.history_path", return_value=hist_file),
        ):
            handle_mullet_run({
                "task": self.TASK,
                "tier": self.TIER,
                "providers": "kimi,haiku",
            })

        rows = [json.loads(l) for l in hist_file.read_text().splitlines() if l.strip()]
        # The last row is the one written by handle_mullet_run (after the call)
        last = rows[-1]
        rerank_tags = [t for t in last.get("tags", []) if t.startswith("rerank:")]
        assert rerank_tags, f"Expected a rerank: tag, got tags={last.get('tags')}"
        assert rerank_tags[0] == "rerank:kimi→haiku"

    def test_no_rerank_when_flag_off(self, tmp_path: Path) -> None:
        """With MULLET_ROUTING_PRIORS unset, providers list is unchanged."""
        hist_file = tmp_path / ".mullet" / "routing_history.jsonl"
        hist_file.parent.mkdir(parents=True, exist_ok=True)
        self._setup_history(tmp_path)
        captured_args: list[dict] = []

        def _mock_inner(args: dict) -> dict:
            captured_args.append(dict(args))
            return _MOCK_RESULT

        env = {k: v for k, v in os.environ.items() if k != "MULLET_ROUTING_PRIORS"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch("mullet.server.mcp_handlers._handle_mullet_run_inner", side_effect=_mock_inner),
            patch("mullet.core.routing_history.history_path", return_value=hist_file),
        ):
            handle_mullet_run({
                "task": self.TASK,
                "tier": self.TIER,
                "providers": "kimi,haiku",
            })

        assert len(captured_args) == 1
        assert captured_args[0]["providers"] == "kimi,haiku"


class TestPickBetterProviderContinue:
    """Line 61: continue when candidate == picked_provider."""
    TASK = "check routing priors continue branch"
    TIER = "STANDARD"

    def test_picked_in_candidates_list_is_skipped(self, tmp_path: Path) -> None:
        """Line 61: when picked_provider appears in candidates, it is skipped via continue."""
        _seed_history(tmp_path, self.TASK, self.TIER, "kimi", successes=1, failures=3)   # 25%
        _seed_history(tmp_path, self.TASK, self.TIER, "haiku", successes=4, failures=0)  # 100%
        # Include picked_provider in candidates list — the continue branch fires first
        new_pick, reason = pick_better_provider(
            self.TASK, self.TIER, "kimi", ["kimi", "haiku"],
            min_pick_rate=0.50, good_enough_rate=0.75, min_samples=3, base=tmp_path,
        )
        assert new_pick == "haiku"
        assert reason is not None
