"""Tests for mullet.cli.live_test pure helpers."""


from mullet.cli.live_test import (
    _parse_args,
    _DEFAULT_SECONDS,
    _DEFAULT_SWARM_WORKERS,
    _TRIVIAL_PROMPT,
    _TRIVIAL_MAX_TOKENS,
    _KIMI_MODELS,
    _GEMINI_MODELS,
    _ANTHROPIC_MODELS,
)


# ---------------------------------------------------------------------------
# live_test._parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_no_args_uses_default_seconds(self):
        ns = _parse_args([])
        assert ns.seconds == _DEFAULT_SECONDS

    def test_explicit_seconds(self):
        ns = _parse_args(["30"])
        assert ns.seconds == 30

    def test_no_swarm_default_false(self):
        ns = _parse_args([])
        assert ns.no_swarm is False

    def test_no_swarm_flag(self):
        ns = _parse_args(["--no-swarm"])
        assert ns.no_swarm is True

    def test_swarm_size_default(self):
        ns = _parse_args([])
        assert ns.swarm_size == _DEFAULT_SWARM_WORKERS

    def test_custom_swarm_size(self):
        ns = _parse_args(["--swarm-size", "5"])
        assert ns.swarm_size == 5

    def test_no_local_default_false(self):
        ns = _parse_args([])
        assert ns.no_local is False

    def test_no_local_flag(self):
        ns = _parse_args(["--no-local"])
        assert ns.no_local is True

    def test_combined_flags(self):
        ns = _parse_args(["60", "--no-swarm", "--no-local"])
        assert ns.seconds == 60
        assert ns.no_swarm is True
        assert ns.no_local is True


# ---------------------------------------------------------------------------
# live_test constants
# ---------------------------------------------------------------------------

class TestLiveTestConstants:
    def test_trivial_prompt_is_string(self):
        assert isinstance(_TRIVIAL_PROMPT, str)
        assert len(_TRIVIAL_PROMPT) > 0

    def test_trivial_max_tokens_positive(self):
        assert _TRIVIAL_MAX_TOKENS > 0

    def test_kimi_models_non_empty(self):
        assert len(_KIMI_MODELS) > 0

    def test_gemini_models_non_empty(self):
        assert len(_GEMINI_MODELS) > 0

    def test_anthropic_models_non_empty(self):
        assert len(_ANTHROPIC_MODELS) > 0

    def test_default_seconds_positive(self):
        assert _DEFAULT_SECONDS > 0

    def test_default_swarm_workers_positive(self):
        assert _DEFAULT_SWARM_WORKERS > 0
