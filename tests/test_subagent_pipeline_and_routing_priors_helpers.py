"""Tests for routing_priors fail-open behavior."""


from mullet.core.routing_priors import (
    pick_better_provider,
    summarize_priors,
)


# ---------------------------------------------------------------------------
# routing_priors — fail-open behavior (no history data → returns picked unchanged)
# ---------------------------------------------------------------------------

class TestPickBetterProviderFailOpen:
    def test_no_history_returns_picked(self, tmp_path):
        # tmp_path has no routing history → min_samples not met → returns picked
        result, reason = pick_better_provider(
            task="do something",
            tier="fast",
            picked_provider="kimi",
            candidates=["gemini", "local"],
            base=tmp_path,
        )
        assert result == "kimi"
        assert reason is None

    def test_returns_tuple(self, tmp_path):
        result = pick_better_provider("task", "fast", "kimi", [], base=tmp_path)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_empty_candidates_returns_picked(self, tmp_path):
        result, reason = pick_better_provider("task", "fast", "kimi", [], base=tmp_path)
        assert result == "kimi"

    def test_returns_string_provider(self, tmp_path):
        provider, _ = pick_better_provider("task", "fast", "kimi", ["gemini"], base=tmp_path)
        assert isinstance(provider, str)


class TestSummarizePriorsFailOpen:
    def test_no_history_returns_empty_dict(self, tmp_path):
        result = summarize_priors("task", "fast", base=tmp_path)
        assert isinstance(result, dict)

    def test_returns_dict(self, tmp_path):
        result = summarize_priors("some task", "balanced", base=tmp_path)
        assert isinstance(result, dict)
