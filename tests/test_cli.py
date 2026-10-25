"""Tests for the CLI entry point and argument parsing."""

from mullet.cli.main import (
    build_parser, build_tool_registry, SUBCOMMANDS,
)


def test_parser_basic():
    parser = build_parser()
    # --version calls sys.exit, so test with a normal parse
    args = parser.parse_args(["hello"])
    assert args.remaining == ["hello"]


def test_parser_verbose():
    parser = build_parser()
    args = parser.parse_args(["--verbose", "hello", "world"])
    assert args.verbose is True
    assert args.remaining == ["hello", "world"]


def test_parser_client():
    parser = build_parser()
    args = parser.parse_args(["--client", "backend", "fix", "bug"])
    assert args.client == "backend"
    assert args.remaining == ["fix", "bug"]


def test_parser_local_only():
    parser = build_parser()
    args = parser.parse_args(["--local-only", "task"])
    assert args.local_only is True


def test_parser_tier():
    parser = build_parser()
    args = parser.parse_args(["--tier", "ADVANCED", "task"])
    assert args.tier == "ADVANCED"


def test_parser_json():
    parser = build_parser()
    args = parser.parse_args(["--json", "task"])
    assert args.json_output is True


def test_parser_tools_flag():
    parser = build_parser()
    args = parser.parse_args(["--tools"])
    assert args.tools is True


def test_parser_skills_flag():
    parser = build_parser()
    args = parser.parse_args(["--skills"])
    assert args.list_skills is True


def test_subcommands_defined():
    assert "serve" in SUBCOMMANDS
    assert "mcp" in SUBCOMMANDS
    assert "doctor" in SUBCOMMANDS
    assert "login" in SUBCOMMANDS
    for removed in ("autopilot", "ralph", "ralphthon", "ultrawork", "team", "eco"):
        assert removed not in SUBCOMMANDS


def test_tool_registry_has_all_tools():
    registry = build_tool_registry()
    tools = registry.list_tools()
    assert "terminal" in tools
    assert "web_search" in tools
    assert "fetch_page" in tools
    assert "browse" in tools
    assert "code_exec" in tools
    assert "delegate" in tools
    assert "session_search" in tools
    assert "session_list" in tools
    assert "git" in tools
    assert "run_tests" in tools
    assert "analyze_image" in tools


def test_tool_registry_toolsets():
    registry = build_tool_registry()
    toolsets = registry.list_toolsets()
    assert "terminal" in toolsets
    assert "web" in toolsets
    assert "delegation" in toolsets
    assert "code" in toolsets
    assert "browser" in toolsets
    assert "session" in toolsets
    assert "git" in toolsets
    assert "testing" in toolsets


def test_tool_registry_profiles():
    registry = build_tool_registry()
    profiles = registry.list_profiles()
    assert "default" in profiles
    assert "research" in profiles
    assert "coding" in profiles
    assert "minimal" in profiles


def test_parser_speed_flag():
    parser = build_parser()
    args = parser.parse_args(["--speed", "fast", "task"])
    assert args.speed == "fast"


def test_parser_speed_choices():
    parser = build_parser()
    for mode in ("economy", "balanced", "fast", "max"):
        args = parser.parse_args(["--speed", mode, "task"])
        assert args.speed == mode


def test_parser_speed_default_none():
    parser = build_parser()
    args = parser.parse_args(["task"])
    assert args.speed is None


# ---------------------------------------------------------------------------
# Cluster 4 — obedience plumbing for --providers + --speed
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    """Build an argparse.Namespace mimicking parsed CLI args with overrides."""
    parser = build_parser()
    args = parser.parse_args(["task"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_parse_providers_allowlist_none():
    from mullet.cli.main import _parse_providers_allowlist
    assert _parse_providers_allowlist(_make_args(providers=None)) is None
    assert _parse_providers_allowlist(_make_args(providers="")) is None


def test_parse_providers_allowlist_normalises_aliases():
    from mullet.cli.main import _parse_providers_allowlist
    allowed = _parse_providers_allowlist(_make_args(providers="anthropic,moonshot"))
    assert allowed == frozenset({"sonnet", "kimi"})


def test_parse_providers_allowlist_trims_and_lowercases():
    from mullet.cli.main import _parse_providers_allowlist
    allowed = _parse_providers_allowlist(_make_args(providers=" Kimi , GEMINI "))
    assert allowed == frozenset({"kimi", "gemini"})


def test_pick_speed_provider_no_allowlist():
    from mullet.cli.main import _pick_speed_provider
    # OpenRouter is default cloud; max/premium are Anthropic-first
    assert _pick_speed_provider("fast", None) == "openrouter"
    assert _pick_speed_provider("max", None) == "sonnet"
    assert _pick_speed_provider("economy", None) == "local"
    assert _pick_speed_provider("balanced", None) == "openrouter"


def test_pick_speed_provider_within_allowlist():
    from mullet.cli.main import _pick_speed_provider
    # fast prefers openrouter; allowlist restricts to haiku — falls back to haiku
    assert _pick_speed_provider("fast", frozenset({"haiku"})) == "haiku"
    # max prefers sonnet/opus/haiku; allowlist restricts to kimi — falls back to kimi
    assert _pick_speed_provider("max", frozenset({"kimi"})) == "kimi"


def test_pick_speed_provider_raises_when_allowlist_excludes_all():
    import pytest
    from mullet.cli.main import _pick_speed_provider
    # max needs sonnet/opus/kimi/haiku; local is none of those
    with pytest.raises(ValueError) as exc:
        _pick_speed_provider("max", frozenset({"local"}))
    msg = str(exc.value)
    assert "max" in msg
    assert "local" in msg


def test_resolve_provider_conflict_raises(monkeypatch):
    import pytest
    # Clear env keys so we don't accidentally resolve a real provider
    for var in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
                "KIMI_API_KEY", "MOONSHOT_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    from mullet.cli.main import resolve_provider
    args = _make_args(provider="haiku", providers="kimi,gemini")
    with pytest.raises(ValueError) as exc:
        resolve_provider(args)
    msg = str(exc.value)
    assert "--provider" in msg
    assert "'haiku'" in msg
    assert "--providers" in msg


def test_resolve_provider_fast_with_kimi_allowlist(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "sk-test-kimi")
    for var in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    from mullet.cli.main import resolve_provider
    args = _make_args(speed="fast", providers="kimi")
    provider = resolve_provider(args)
    assert provider is not None
    # KimiProvider is the expected class
    assert type(provider).__name__ == "KimiProvider"


def test_resolve_provider_fast_with_haiku_allowlist_falls_back(monkeypatch):
    # fast prefers openrouter; allowlist restricts to haiku — must pick haiku
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
    for var in ("KIMI_API_KEY", "MOONSHOT_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    from mullet.cli.main import resolve_provider
    args = _make_args(speed="fast", providers="haiku")
    provider = resolve_provider(args)
    assert provider is not None
    assert type(provider).__name__ == "AnthropicProvider"


def test_resolve_provider_max_with_kimi_allowlist_falls_back(monkeypatch):
    # max prefers sonnet/opus/haiku; allowlist restricts to kimi — must fall back to kimi
    monkeypatch.setenv("KIMI_API_KEY", "sk-test-kimi")
    for var in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    from mullet.cli.main import resolve_provider
    args = _make_args(speed="max", providers="kimi")
    provider = resolve_provider(args)
    assert provider is not None
    assert type(provider).__name__ == "KimiProvider"


def test_resolve_provider_providers_only_passes_through(monkeypatch):
    # --providers kimi with no --speed and no --provider — auto-detect path
    # should be gated, but with KIMI_API_KEY set, the speed-mode default
    # "economy" picks local first, then kimi. Without local available, kimi wins.
    monkeypatch.setenv("KIMI_API_KEY", "sk-test-kimi")
    for var in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    from mullet.cli.main import resolve_provider
    args = _make_args(providers="kimi")
    # economy default: picks "local" first from options, but local in allowlist? No.
    # So fallback to next in economy priority = kimi → should resolve KimiProvider
    provider = resolve_provider(args)
    assert provider is not None
    assert type(provider).__name__ == "KimiProvider"


# ---------------------------------------------------------------------------
# G24: build_agent propagates --providers allowlist onto AgentRunner
# ---------------------------------------------------------------------------


def test_build_agent_attaches_allowed_providers(monkeypatch):
    """build_agent must set AgentRunner.allowed_providers from --providers."""
    monkeypatch.setenv("KIMI_API_KEY", "sk-test-kimi")
    for var in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    from mullet.cli.main import build_agent
    args = _make_args(provider="kimi", providers="kimi,haiku")
    agent, _prompt, _toolsets = build_agent(args, task="hello")
    assert agent.allowed_providers == frozenset({"kimi", "haiku"})


def test_build_agent_allowed_providers_none_when_flag_absent(monkeypatch):
    """Without --providers, allowed_providers on AgentRunner should be None."""
    monkeypatch.setenv("KIMI_API_KEY", "sk-test-kimi")
    for var in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    from mullet.cli.main import build_agent
    args = _make_args(provider="kimi")  # no --providers
    agent, _prompt, _toolsets = build_agent(args, task="hello")
    assert agent.allowed_providers is None


def test_run_repl_accepts_allowed_providers_kwarg():
    """run_repl signature must accept allowed_providers kwarg (G24)."""
    import inspect
    from mullet.cli.repl import run_repl
    sig = inspect.signature(run_repl)
    assert "allowed_providers" in sig.parameters
    param = sig.parameters["allowed_providers"]
    assert param.default is None
