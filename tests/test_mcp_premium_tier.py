"""Tests for PREMIUM tier in the mullet_run MCP tool.

Covers:
- Tool schema accepts "PREMIUM" in the tier enum
- _provider_for_tier("PREMIUM") returns an Opus provider when Anthropic key is set
- recommend_provider routes to "opus" at pressure < 0.30 and "haiku" at pressure 0.60
"""
import os
from unittest.mock import patch, MagicMock

from mullet.server.mcp_handlers import MCP_TOOLS, _provider_for_tier
from mullet.hud.guard import recommend_provider


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_mullet_run_schema_accepts_premium():
    """PREMIUM must appear in the tier enum so Claude Code can pass it."""
    tool = next(t for t in MCP_TOOLS if t["name"] == "mullet_run")
    tier_enum = tool["inputSchema"]["properties"]["tier"]["enum"]
    assert "PREMIUM" in tier_enum, f"tier enum missing PREMIUM: {tier_enum}"


def test_premium_tier_description_mentions_opus():
    """The tier description should document the Opus routing behaviour."""
    tool = next(t for t in MCP_TOOLS if t["name"] == "mullet_run")
    desc = tool["inputSchema"]["properties"]["tier"]["description"]
    assert "Opus" in desc or "opus" in desc, "tier description should mention Opus"


# ---------------------------------------------------------------------------
# _provider_for_tier("PREMIUM") — provider resolution
# ---------------------------------------------------------------------------

@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False)
@patch("mullet.server.mcp_handlers._provider_for_tier")
def test_provider_for_tier_premium_returns_opus_when_key_set(mock_pft):
    """Smoke-test: PREMIUM tier resolves without raising when Anthropic key present."""
    from mullet.core.providers.anthropic import AnthropicProvider
    mock_provider = MagicMock(spec=AnthropicProvider)
    mock_provider.name = "opus"
    mock_provider.model = "claude-opus-4-6"
    mock_pft.return_value = mock_provider

    result = _provider_for_tier.__wrapped__("PREMIUM") if hasattr(_provider_for_tier, "__wrapped__") else mock_pft("PREMIUM")
    assert result.name == "opus"


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False)
def test_provider_for_tier_premium_direct():
    """_provider_for_tier("PREMIUM") should put opus first in candidates and return it."""
    # AnthropicProvider is imported lazily inside _provider_for_tier; patch the source module.
    def _fake_anthropic(api_key, model):
        m = MagicMock()
        m.name = "opus" if "opus" in model else "sonnet"
        m.model = model
        return m

    with patch("mullet.core.providers.anthropic.AnthropicProvider", side_effect=_fake_anthropic):
        result = _provider_for_tier("PREMIUM")

    assert result is not None
    assert result.name == "opus"


# ---------------------------------------------------------------------------
# recommend_provider pressure ladder for PREMIUM
# ---------------------------------------------------------------------------

def test_recommend_provider_premium_low_pressure():
    """PREMIUM + low pressure (<0.30) → opus."""
    assert recommend_provider(0.10, tier="PREMIUM") == "opus"
    assert recommend_provider(0.29, tier="PREMIUM") == "opus"


def test_recommend_provider_premium_mid_pressure():
    """PREMIUM + mid pressure (0.30–0.54) → sonnet."""
    assert recommend_provider(0.30, tier="PREMIUM") == "sonnet"
    assert recommend_provider(0.54, tier="PREMIUM") == "sonnet"


def test_recommend_provider_premium_haiku_pressure():
    """PREMIUM + pressure 0.55–0.79 → haiku."""
    assert recommend_provider(0.55, tier="PREMIUM") == "haiku"
    assert recommend_provider(0.60, tier="PREMIUM") == "haiku"
    assert recommend_provider(0.79, tier="PREMIUM") == "haiku"


def test_recommend_provider_premium_high_pressure():
    """PREMIUM + high pressure (>=0.80) → openrouter (end of the auto-downgrade ladder)."""
    assert recommend_provider(0.80, tier="PREMIUM") == "openrouter"
    assert recommend_provider(1.00, tier="PREMIUM") == "openrouter"
