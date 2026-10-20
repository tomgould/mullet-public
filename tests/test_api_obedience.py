"""G14 / HTTP API obedience — server/api.py must honour 'providers' in request body.

These tests exercise `_build_server_agent` directly with opts dicts, asserting
that the parsed allowlist is forwarded to the router and onto the resulting
AgentRunner so children inherit the same constraint envelope.
"""
from unittest.mock import MagicMock, patch



def _fake_resolver(captured):
    """Return a resolver whose route_and_resolve records its kwargs then returns a stub provider."""
    resolver = MagicMock()
    stub_provider = MagicMock()
    stub_provider.name = "kimi"
    stub_provider.model = "kimi-k2.5"

    def _route_and_resolve(**kwargs):
        captured.update(kwargs)
        return stub_provider, MagicMock(allowed_providers=kwargs.get("allowed_providers"))

    resolver.route_and_resolve.side_effect = _route_and_resolve
    return resolver, stub_provider


def _patched_build_agent(opts, captured_router_kwargs):
    """Call _build_server_agent with all external deps mocked; return (agent, provider)."""
    from mullet.server import api as api_mod

    resolver, stub_provider = _fake_resolver(captured_router_kwargs)

    with patch("mullet.cli.main.build_tool_registry", return_value=MagicMock()), \
         patch("mullet.cli.main.detect_local_provider", return_value=stub_provider), \
         patch("mullet.core.router.resolver.build_default_router",
               return_value=(MagicMock(), resolver)), \
         patch("mullet.core.agent.prompts.build_base_prompt", return_value="sys"):
        agent, provider = api_mod._build_server_agent("hello world", opts)
    return agent, provider


def test_build_server_agent_without_providers_omits_allowlist():
    """Baseline: no 'providers' in opts -> allowed_providers=None on both router + agent."""
    captured = {}
    agent, provider = _patched_build_agent({}, captured)
    assert captured.get("allowed_providers") is None
    assert agent is not None
    assert agent.allowed_providers is None


def test_build_server_agent_with_csv_providers_parses_frozenset():
    """opts['providers']='kimi,haiku' -> allowed_providers == frozenset({'kimi','haiku'})."""
    captured = {}
    agent, provider = _patched_build_agent({"providers": "kimi,haiku"}, captured)
    assert captured.get("allowed_providers") == frozenset({"kimi", "haiku"})
    assert agent.allowed_providers == frozenset({"kimi", "haiku"})


def test_build_server_agent_with_single_provider_string():
    """opts['providers']='kimi' -> frozenset({'kimi'})."""
    captured = {}
    agent, provider = _patched_build_agent({"providers": "kimi"}, captured)
    assert captured.get("allowed_providers") == frozenset({"kimi"})
    assert agent.allowed_providers == frozenset({"kimi"})


def test_build_server_agent_accepts_list_form():
    """opts['providers']=['kimi','haiku'] (list form, not CSV) is accepted."""
    captured = {}
    agent, provider = _patched_build_agent({"providers": ["kimi", "haiku"]}, captured)
    assert captured.get("allowed_providers") == frozenset({"kimi", "haiku"})


def test_build_server_agent_accepts_allowed_providers_alias():
    """opts['allowed_providers'] is an accepted alias for opts['providers']."""
    captured = {}
    agent, provider = _patched_build_agent({"allowed_providers": "kimi"}, captured)
    assert captured.get("allowed_providers") == frozenset({"kimi"})


def test_build_server_agent_strips_whitespace_and_empty_entries():
    """'kimi, , haiku ,' -> frozenset({'kimi','haiku'}); empty CSV -> None."""
    captured = {}
    agent, _ = _patched_build_agent({"providers": "kimi, , haiku ,"}, captured)
    assert captured.get("allowed_providers") == frozenset({"kimi", "haiku"})

    captured2 = {}
    agent2, _ = _patched_build_agent({"providers": "   "}, captured2)
    assert captured2.get("allowed_providers") is None


def test_build_server_agent_agent_runner_inherits_allowlist_attr():
    """The constructed AgentRunner carries allowed_providers so children inherit it."""
    captured = {}
    agent, _ = _patched_build_agent({"providers": "kimi"}, captured)
    assert hasattr(agent, "allowed_providers")
    assert agent.allowed_providers == frozenset({"kimi"})
