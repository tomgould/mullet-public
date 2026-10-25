"""Mullet CLI entry point — Cloud in the front, Local GPU in the back."""

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path


from mullet.core.env import load_dotenv

load_dotenv()

from mullet.core.agent.runner import AgentRunner
from mullet.core.agent.types import AgentResult
from mullet.core.providers.base import Provider
from mullet.core.providers.local_detection import (  # noqa: F401  back-compat re-export — see Cycle 7 C1
    LOCAL_PORTS,
    clear_local_provider_cache,
    detect_local_provider,
)
from mullet.core.providers.openai_compat import OpenAICompatibleProvider
from mullet.core.tools.registry import ToolRegistry
from mullet.hud.guard import check_budget, BudgetStatus
from mullet.core.tools import terminal, file_ops, web, delegate, code_exec, browser, session as session_tool, git as git_tool, test_runner, vision, docker, lsp as lsp_tool, todo as todo_tool, clarify as clarify_tool, memory as memory_tool, patch as patch_tool, image_gen as image_gen_tool, skill_mgr as skill_mgr_tool, notepad as notepad_tool, python_repl as python_repl_tool, shared_mem as shared_mem_tool, skill_invoke as skill_invoke_tool

logger = logging.getLogger("mullet")


SUBCOMMANDS = {"serve", "mcp", "doctor", "init", "update", "history", "config", "stats", "completion", "explain", "login", "watch", "cost", "budget", "review", "plugins", "schedule", "research", "ingest", "search", "graph", "source", "reindex", "hud", "hud-claude", "pool", "docs", "evaluate", "setup-claude", "logs", "models", "clean", "override", "observations", "decisions", "routing", "review-session", "memory-sync", "live-test", "preset", "reap", "ladder", "strategy"}


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser used by `mullet [...]`."""
    parser = argparse.ArgumentParser(
        prog="mullet",
        description="Cloud in the front, Local GPU in the back.",
        usage="mullet [OPTIONS] [COMMAND|TASK]",
        epilog="""commands:
  (task)        Run a task (default — just type your prompt)
  serve         Start the API server (REST + WebSocket)
  mcp           Start MCP server for Claude Code integration
  doctor        Check system health and provider status
  init          Generate mullet.yaml config for current project
  update        Update existing project scaffolding (LOOP_AGREEMENT, hooks, providers)
  setup-claude  Register Mullet with Claude Code (MCP server, skills, opt-in hooks)
  hook          Run one of Mullet's Claude Code hooks (mullet hook list)
  job           Plan and run a gated job in a sandboxed clone (mullet job --help)
  ladder        Report on how the delegation ladder has done across jobs (mullet ladder report --help)
  migrate-state Move an older ~/.mullet into the XDG directories (--dry-run first)
  paths         Print Mullet's config, state, data, cache and pools directories
  reap          Stop idle or orphaned Mullet work (use --dry-run first)
  strategy      Show or set this session's budget and delegation choice
  override      Toggle budget override (mullet override on/off)

  hud           Show/configure the terminal status HUD
  hud-claude    Show Claude Code plugin-style HUD
  models        List available models and pricing by provider
  cost          Show model pricing across tiers
  explain       Show complexity analysis for a task (no LLM call)
  config        Show or set configuration values
  stats         Show usage statistics from session history

  pool          Manage knowledge pools (list, create, delete, prune)
  docs          Manage library pool documentation (list, sync)
  ingest        Ingest files/URLs into a knowledge pool
  search        Semantic search across indexed pools
  graph         Knowledge graph operations (build, stats, export)
  source        Manage knowledge sources
  reindex       Rebuild pool index

  observations  View tool-call history and LLM digests
  decisions     Scan git+docs for decision records
  routing       Show provider routing outcomes and reliability
  review-session 6-section session retrospective

  login         Authenticate with a Personal Access Token
  history       Show recent sessions or search past conversations
  logs          View LLM interaction history (tail/stats/suggest)
  completion    Generate shell completion scripts (bash/zsh)
  watch         Watch files and re-run task on changes
  review        Review recent git changes with AI analysis
  research      Deep research mode with web + knowledge sources
  plugins       List installed plugins
  schedule      Schedule recurring tasks
  clean         Delete stale .claude/.mullet dirs (dry-run by default)

examples:
  mullet "list the files"                         # basic task
  mullet --pool backend "fix the auth bug"         # with project context
  mullet --speed fast "implement user auth"        # skip local, use Kimi
  mullet --provider kimi "refactor this module"   # explicit provider
  mullet --json "explain this" | jq .response     # JSON output
  cat file.py | mullet "review this code"         # pipe content
  mullet serve --port 8080                        # start API server
  mullet explain "complex task"                   # show routing analysis
  mullet "review @file:main.py and @diff"         # context references
  mullet watch "run tests and fix failures"       # re-run on file changes
  mullet doctor                                   # system health check
  mullet models                                   # list models + pricing""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument("--pool", "--client", "-c", dest="client", help="Knowledge pool name for project context")
    parser.add_argument("--provider", "-p", help="Force provider: local, kimi, haiku, sonnet, opus, opus-4-8, opus-4-7, gemini, bedrock, openrouter, or URL")
    parser.add_argument("--providers", help="Comma-separated provider allowlist — restricts routing to only these. e.g. 'kimi,haiku,sonnet' excludes local and gemini")
    parser.add_argument("--tier", choices=["LOW", "STANDARD", "ADVANCED", "CRITICAL", "PREMIUM"], help="Force routing tier")
    parser.add_argument("--local-only", action="store_true", help="Route ALL compute to local GPU")
    parser.add_argument("--speed", choices=["economy", "balanced", "fast", "max", "premium"],
                        default=None,
                        help="Speed mode: economy (local first, default), balanced (cloud for simple), "
                             "fast (Kimi always), max (Sonnet for all), premium (Opus-first, auto-downgrade)")
    parser.add_argument("--no-preflight", action="store_true",
                        help="Skip the pre-flight budget prompt for large tasks or loops")
    parser.add_argument("--max-iterations", type=int, default=90, help="Iteration budget")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--base-url", help="Override provider base URL")
    parser.add_argument("--verbose", action="store_true", help="Show routing decisions and token usage")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    parser.add_argument("--tools", action="store_true", help="List available tools and exit")
    parser.add_argument("--skills", action="store_true", dest="list_skills", help="List available skills and exit")
    parser.add_argument("--export", help="Export conversation to file (json or md format)")
    parser.add_argument("--no-claude", dest="no_claude", action="store_true", help="Skip .claude/ scaffolding during init")
    parser.add_argument("--no-memory", dest="no_memory", action="store_true", help="Skip project memory seeding during init")
    parser.add_argument("--no-pool", dest="no_pool", action="store_true", help="Skip pool creation during init")
    parser.add_argument("--force", dest="force", action="store_true", help="Overwrite existing files during init")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Print planned actions without writing files")
    parser.add_argument("--install-help", action="store_true", help="Show one-line install instructions and exit")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument("--banner", action="store_true", help="Show ASCII art banner with version")

    # Everything else is captured as remaining args
    parser.add_argument("remaining", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    return parser


def get_opencode_declared_context(model_id: str) -> int | None:
    """Return the context length ~/.config/opencode/opencode.json declares
    for a local (lmstudio provider) model, or None if not configured there.

    Used to catch a real, easy-to-miss failure mode: LM Studio's actually
    *loaded* context (queryable via get_local_loaded_context) can silently
    be far smaller than what opencode.json's own "limit.context" declares
    for the same model — e.g. a model reloaded with LM Studio's default
    Context Length setting rather than the value opencode.json assumes.
    Every real coding task then fails with an opencode-side context-overflow
    error (a "n_keep >= n_ctx" style message), even for short tasks, since
    opencode's own system prompt/tool definitions alone can exceed a small
    loaded context.
    """
    import json as _json
    try:
        cfg_path = os.path.expanduser("~/.config/opencode/opencode.json")
        with open(cfg_path) as fh:
            cfg = _json.load(fh)
        models = ((cfg.get("provider") or {}).get("lmstudio") or {}).get("models", {})
        entry = models.get(model_id)
        if entry:
            return entry.get("limit", {}).get("context")
    except Exception:
        pass
    return None


def get_local_loaded_context(base_url: str | None = None) -> int:
    """Query the actual loaded context length from a local LM Studio server.

    Uses /api/v0/models which returns `loaded_context_length` for each loaded model.
    Falls back to 0 if the endpoint is unavailable or no model is loaded.
    ``base_url`` is the server root and defaults to ``LOCAL_LLM_URL`` without ``/v1``.
    """
    import urllib.request as _req
    import json as _json
    if base_url is None:
        from mullet.core.env import local_llm_host
        base_url = local_llm_host()
    try:
        with _req.urlopen(f"{base_url}/api/v0/models", timeout=3) as resp:
            data = _json.loads(resp.read())
        loaded = [
            m.get("loaded_context_length", 0)
            for m in data.get("data", [])
            if m.get("state") == "loaded" and m.get("type") == "llm"
        ]
        return max(loaded) if loaded else 0
    except Exception:
        return 0


_PROVIDER_NORMALISE = {
    "anthropic": "sonnet",
    "moonshot": "kimi",
    "google": "gemini",
    "lmstudio": "local",
    "ollama": "local",
}


def _parse_providers_allowlist(args) -> frozenset[str] | None:
    """Parse --providers CSV into a frozenset of normalised names, or None."""
    raw = getattr(args, "providers", None)
    if not raw:
        return None
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    normalised = {_PROVIDER_NORMALISE.get(p, p) for p in parts}
    return frozenset(normalised)


_SPEED_PRIORITY = {
    # Balanced: OpenRouter is the default cloud workhorse (cheap, reliable).
    # Anthropic (Haiku/Sonnet) is substituted when budget pressure <60% — it's prepaid
    # so using it saves spend on OpenRouter/Kimi. Max/premium are Anthropic-first.
    "fast":    ["openrouter", "haiku", "kimi"],
    "max":     ["sonnet", "opus", "haiku", "openrouter", "kimi"],
    "economy": ["local", "openrouter", "kimi", "haiku"],
    "balanced": ["openrouter", "haiku", "kimi", "sonnet"],
    "premium": ["opus", "sonnet", "haiku", "openrouter", "kimi"],
}


def _pick_speed_provider(speed_mode: str, allowed: frozenset[str] | None) -> str | None:
    """Return the best provider for a speed mode, gated by the allowlist."""
    options = _SPEED_PRIORITY.get(speed_mode)
    if not options:
        return None
    if allowed is None:
        return options[0]
    for candidate in options:
        if candidate in allowed:
            return candidate
    raise ValueError(
        f"Speed mode '{speed_mode}' requires one of {options}, "
        f"but --providers is restricted to {sorted(allowed)}"
    )


def resolve_provider(args, allowed_providers: frozenset[str] | None = None) -> Provider | None:
    """Resolve LLM provider. Priority: session override > --provider > --speed > auto-detect. All gated by --providers allowlist."""
    if allowed_providers is None:
        allowed_providers = _parse_providers_allowlist(args)

    # Session override injection — only when caller hasn't already pinned one
    if not getattr(args, "provider", None) and not getattr(args, "speed", None):
        try:
            from mullet.core.session_state import get_override
            override = get_override()
            if override and override.get("provider"):
                args.provider = override["provider"]
                if override.get("model") and not getattr(args, "model", None):
                    args.model = override["model"]
                print(f"[Opus override] active — routing {args.provider}"
                      f"{' (' + override['model'] + ')' if override.get('model') else ''}",
                      file=sys.stderr)
        except Exception as _exc:
            logger.debug("session override lookup failed: %s", _exc)

    # Budget-aware routing
    try:
        budget = check_budget()
        if budget.status == BudgetStatus.PAUSE:
            print(f"[Token Guard] {budget.message}", file=sys.stderr)
            return None
        provider_name = getattr(args, "provider", None)
        # If override is pinning opus but pressure is climbing, auto-downgrade.
        if provider_name in ("opus", "opus-4-8", "opus-4-7") and budget.pressure > 0.55:
            downgrade = "sonnet" if budget.pressure < 0.80 else "haiku"
            print(f"[Opus override] pressure {budget.pressure:.2f} — downgrading to {downgrade} "
                  f"(override remains active)", file=sys.stderr)
            args.provider = downgrade
        elif budget.status == BudgetStatus.THROTTLE and provider_name in ("anthropic", "sonnet", "opus", "opus-4-8", "opus-4-7"):
            print("[Token Guard] Budget warning — switching to gemini", file=sys.stderr)
            args.provider = "gemini"
    except Exception:
        pass  # Guard unavailable, proceed normally

    # Explicit base URL
    if args.base_url:
        return OpenAICompatibleProvider(
            base_url=args.base_url,
            model=args.model or "default",
        )

    # Named provider
    provider_name = getattr(args, "provider", None)
    if provider_name:
        # Conflict check: --provider X must be within --providers allowlist if set
        if allowed_providers is not None:
            normalised_single = _PROVIDER_NORMALISE.get(provider_name, provider_name)
            if normalised_single not in allowed_providers:
                raise ValueError(
                    f"Conflicting provider intents: --provider={provider_name!r} "
                    f"is not in --providers allowlist {sorted(allowed_providers)}"
                )
        if provider_name in ("anthropic", "sonnet", "opus", "opus-4-7", "opus-4-8", "haiku"):
            model_map = {
                "sonnet": "claude-sonnet-4-6",
                "opus": "claude-opus-4-8",
                "opus-4-8": "claude-opus-4-8",
                "opus-4-7": "claude-opus-4-7",
                "haiku": "claude-haiku-4-5",
                "anthropic": "claude-sonnet-4-6",
            }
            cli_model_map = {"sonnet": "sonnet", "opus": "opus", "opus-4-8": "opus", "opus-4-7": "opus", "haiku": "haiku", "anthropic": "sonnet"}
            default_model = model_map.get(provider_name, "claude-sonnet-4-6")
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                from mullet.core.providers.anthropic import AnthropicProvider
                return AnthropicProvider(api_key=api_key, model=args.model or default_model)
            # Fall back to the user's own Claude CLI (subprocess)
            from mullet.core.providers.claude_cli import ClaudeCLIProvider, is_claude_cli_available
            if is_claude_cli_available():
                cli_model = cli_model_map.get(provider_name, "sonnet")
                return ClaudeCLIProvider(model=args.model or cli_model)
            print("Error: ANTHROPIC_API_KEY not set and Claude CLI not available.", file=sys.stderr)
            return None
        elif provider_name in ("gemini", "google"):
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("Error: GOOGLE_API_KEY not set.", file=sys.stderr)
                return None
            try:
                from mullet.core.providers.gemini import GeminiProvider
                return GeminiProvider(api_key=api_key, model=args.model or "gemini-2.5-flash")
            except ImportError:
                print("Error: google-genai not installed. Run: pip install google-genai", file=sys.stderr)
                return None
        elif provider_name == "bedrock":
            try:
                from mullet.core.providers.bedrock import BedrockProvider
                return BedrockProvider(model_id=args.model or "anthropic.claude-sonnet-4-6-20250514-v1:0")
            except ImportError:
                print("Error: boto3 not installed. Run: pip install boto3", file=sys.stderr)
                return None
        elif provider_name in ("local", "lmstudio", "ollama"):
            return detect_local_provider()
        elif provider_name in ("openrouter",):
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                print("Error: OPENROUTER_API_KEY not set.", file=sys.stderr)
                return None
            from mullet.core.providers.openrouter import OpenRouterProvider
            return OpenRouterProvider(
                api_key=api_key,
                model=args.model or "google/gemini-2.5-flash",
            )
        elif provider_name in ("kimi", "moonshot"):
            api_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
            if not api_key:
                print("Error: KIMI_API_KEY or MOONSHOT_API_KEY not set.", file=sys.stderr)
                return None
            from mullet.core.providers.kimi import KimiProvider
            kw = {"model": args.model} if getattr(args, "model", None) else {}
            return KimiProvider(api_key=api_key, **kw)
        elif provider_name.startswith("http"):
            return OpenAICompatibleProvider(base_url=provider_name, model=args.model or "default")
        else:
            print(f"Error: Unknown provider '{provider_name}'. "
                  f"Available: local, anthropic, gemini, bedrock, openrouter, kimi, or a URL.",
                  file=sys.stderr)
            return None

    # Speed mode overrides auto-detect routing — gated by allowlist
    speed_mode = getattr(args, "speed", None) or "economy"
    picked = _pick_speed_provider(speed_mode, allowed_providers)
    if picked:
        from mullet.core.router.complexity import speed_mode_badge
        if picked == "kimi":
            kimi_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
            if kimi_key:
                from mullet.core.providers.kimi import KimiProvider
                print(f"  speed={speed_mode_badge(speed_mode)} → Kimi K2", file=sys.stderr)
                return KimiProvider(api_key=kimi_key)
        elif picked == "sonnet":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                from mullet.core.providers.anthropic import AnthropicProvider
                print(f"  speed={speed_mode_badge(speed_mode)} → Sonnet", file=sys.stderr)
                return AnthropicProvider(api_key=api_key, model=args.model or "claude-sonnet-4-6")
        elif picked == "opus":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                from mullet.core.providers.anthropic import AnthropicProvider
                opus_model = args.model or "claude-opus-4-8"
                print(f"  speed={speed_mode_badge(speed_mode)} → Opus ({opus_model})", file=sys.stderr)
                return AnthropicProvider(api_key=api_key, model=opus_model)
        elif picked == "haiku":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                from mullet.core.providers.anthropic import AnthropicProvider
                print(f"  speed={speed_mode_badge(speed_mode)} → Haiku", file=sys.stderr)
                return AnthropicProvider(api_key=api_key, model=args.model or "claude-haiku-4-5")
        elif picked == "gemini":
            gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if gemini_key:
                try:
                    from mullet.core.providers.gemini import GeminiProvider
                    print(f"  speed={speed_mode_badge(speed_mode)} → Gemini", file=sys.stderr)
                    return GeminiProvider(api_key=gemini_key, model=args.model or "gemini-2.5-flash")
                except ImportError:
                    pass
        elif picked == "local":
            local = detect_local_provider()
            if local:
                return local

    # Auto-detect: try Anthropic key first, then Gemini, then local — gated by allowlist
    allow_anthropic = allowed_providers is None or "sonnet" in allowed_providers
    allow_gemini = allowed_providers is None or "gemini" in allowed_providers
    allow_local = allowed_providers is None or "local" in allowed_providers

    if not getattr(args, "local_only", False):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key and allow_anthropic:
            from mullet.core.providers.anthropic import AnthropicProvider
            return AnthropicProvider(api_key=api_key, model=args.model or "claude-sonnet-4-6")

        gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if gemini_key and allow_gemini:
            try:
                from mullet.core.providers.gemini import GeminiProvider
                return GeminiProvider(api_key=gemini_key, model=args.model or "gemini-2.5-flash")
            except ImportError as _exc:
                logger.warning("cli/main: GeminiProvider import failed: %s", _exc)

    if allow_local:
        return detect_local_provider()
    return None


def build_tool_registry() -> ToolRegistry:
    """Build the default tool registry with built-in tools."""
    registry = ToolRegistry()
    terminal.register(registry)
    file_ops.register(registry)
    web.register(registry)
    delegate.register(registry)
    code_exec.register(registry)
    browser.register(registry)
    session_tool.register(registry)
    git_tool.register(registry)
    test_runner.register(registry)
    vision.register(registry)
    docker.register(registry)
    lsp_tool.register(registry)
    todo_tool.register(registry)
    clarify_tool.register(registry)
    memory_tool.register(registry)
    patch_tool.register(registry)
    image_gen_tool.register(registry)
    skill_mgr_tool.register(registry)
    notepad_tool.register(registry)
    python_repl_tool.register(registry)
    shared_mem_tool.register(registry)
    skill_invoke_tool.register(registry)

    # Load plugins
    try:
        from mullet.core.plugins import load_plugins
        loaded = load_plugins(tool_registry=registry)
        if loaded > 0:
            logger.info("Loaded %d plugins", loaded)
    except Exception as _exc:
        logger.warning("cli/main: Failed to load plugins: %s", _exc)

    return registry


def build_agent(args, task: str = "") -> tuple[AgentRunner, str, list[str]]:
    """Build a fully wired agent with provider, tools, knowledge, and skills.

    Uses the Router for intelligent provider selection when no --provider
    or --base-url is explicitly set.

    Returns (agent, system_prompt, toolsets).
    """
    # Parse --providers allowlist once, pass it through every routing path
    allowed_providers = _parse_providers_allowlist(args)

    # Use config-driven stack builder for intelligent provider selection
    if not args.base_url and not getattr(args, "provider", None):
        try:
            from mullet.config.loader import build_stack
            stack = build_stack()
            router = stack["router"]

            from mullet.core.router.resolver import ProviderResolver
            resolver = ProviderResolver(router)
            provider, decision = resolver.route_and_resolve(
                task=task or "general task",
                force_tier=getattr(args, "tier", None),
                allowed_providers=allowed_providers,
            )
            if provider and args.verbose:
                print(f"Router: {decision.tier.value} → {decision.provider}/{decision.model} "
                      f"({', '.join(decision.reasons)})", file=sys.stderr)
        except Exception as e:
            logger.debug("Router/stack failed, falling back: %s", e)
            provider = None

        if not provider:
            provider = resolve_provider(args, allowed_providers=allowed_providers)
    else:
        provider = resolve_provider(args, allowed_providers=allowed_providers)

    if not provider:
        print("Error: No LLM provider found.\n", file=sys.stderr)
        print("Options:", file=sys.stderr)
        print("  1. Start LM Studio (port 1234) or Ollama (port 11434), or set LOCAL_LLM_URL / OLLAMA_URL", file=sys.stderr)
        print("  2. Set ANTHROPIC_API_KEY for Claude", file=sys.stderr)
        print("  3. Set OPENROUTER_API_KEY for OpenRouter", file=sys.stderr)
        print("  4. Use --base-url http://your-endpoint/v1", file=sys.stderr)
        print("  5. Use --provider anthropic (if API key is set)", file=sys.stderr)
        print("\nSee mullet.yaml.example for full configuration.", file=sys.stderr)
        sys.exit(2)

    if args.verbose:
        print(f"Provider: {provider.name} ({getattr(provider, 'model', '?')})", file=sys.stderr)

    # Strip escape hatch prefix (quick:, simple:, just:)
    if task:
        from mullet.core.router.complexity import strip_escape_prefix
        task, _ = strip_escape_prefix(task)

    registry = build_tool_registry()
    from mullet.core.toolset_policy import apply_policy
    toolsets = apply_policy(
        ["terminal", "file", "web", "code", "browser", "delegation", "session", "git", "testing", "vision", "docker", "lsp"],
        getattr(provider, "name", None),
    )
    from mullet.core.agent.prompts import build_base_prompt
    model_name = getattr(provider, "model", "") if provider else ""
    system_prompt = build_base_prompt(project_name=args.client, model=model_name)

    # Knowledge layer
    if args.client:
        try:
            from mullet.core.knowledge.manager import (
                KnowledgeManager,
                build_knowledge_environment,
            )
            from mullet.core.knowledge import tools as knowledge_tools

            if task:
                _km, context, system_prompt = build_knowledge_environment(
                    args.client, task, registry, system_prompt,
                )
                toolsets.append("trove")
                if args.verbose:
                    print(f"Knowledge: {context.chunks_used} chunks, "
                          f"{context.memories_used} memories", file=sys.stderr)
            else:
                km = KnowledgeManager(args.client)
                knowledge_tools.register(registry, km)
                toolsets.append("trove")
        except Exception as e:
            print(f"Warning: Could not load client '{args.client}': {e}", file=sys.stderr)

    # Context references (@file:path, @diff, @url:url, etc.)
    if task and "@" in task:
        from mullet.core.agent.context_refs import resolve_context_references
        from mullet.core.agent.compression import get_context_length
        ctx_result = resolve_context_references(
            task, context_length=get_context_length(model_name),
        )
        if ctx_result.expanded:
            task = ctx_result.message
            if args.verbose:
                print(f"Context: {len(ctx_result.references)} refs, "
                      f"~{ctx_result.injected_tokens} tokens", file=sys.stderr)
        if ctx_result.warnings:
            for w in ctx_result.warnings:
                print(f"Warning: {w}", file=sys.stderr)
        if ctx_result.blocked:
            print("Error: Context too large for model. Use fewer @references.", file=sys.stderr)
            sys.exit(1)

    # Skills — load registry and check for skill match
    try:
        from mullet.core.skills.loader import SkillLoader
        from mullet.core.skills.registry import SkillRegistry

        loader = SkillLoader()
        skill_registry = SkillRegistry(loader=loader)
        skill_registry.load_all()

        # Check if task matches a skill
        if task:
            match = skill_registry.match(task)
            if match:
                invocation = skill_registry.invoke(match.skill.metadata.name)
                system_prompt += f"\n\n{invocation.content}"
                # Store matched skill for pipeline chaining
                args._matched_skill = match.skill
                if args.verbose:
                    print(f"Skill: {match.skill.metadata.name} ({match.match_type})", file=sys.stderr)

        # Add skill index to system prompt
        index = skill_registry.build_index_prompt()
        if index:
            system_prompt += f"\n\n{index}"
    except Exception as e:
        logger.debug("Skills not loaded: %s", e)

    # Inject persistent memory into system prompt
    try:
        from mullet.core.agent.memory import AgentMemory
        km = _get_knowledge(args)
        memory = AgentMemory(knowledge=km)
        memory_prompt = memory.build_memory_prompt()
        if memory_prompt:
            system_prompt += f"\n\n{memory_prompt}"
            if args.verbose:
                print(f"Memory: {memory.count} entries loaded", file=sys.stderr)
    except Exception as _exc:
        logger.warning("cli/main: Failed to inject persistent memory: %s", _exc)

    agent = AgentRunner(
        provider=provider,
        tool_registry=registry,
        toolsets=toolsets,
        system_prompt=system_prompt,
        max_iterations=args.max_iterations,
        allowed_providers=allowed_providers,
    )

    # Infer tier from provider name
    pname = getattr(provider, "name", "")
    if pname in ("local", "lmstudio", "ollama"):
        agent.tier_used = "LOW"
    elif pname in ("openrouter",):
        agent.tier_used = "STANDARD"
    elif pname in ("anthropic", "bedrock"):
        agent.tier_used = "ADVANCED"

    # Wire delegation so agent can spawn child agents
    delegate.set_parent_agent(agent)

    return agent, system_prompt, toolsets


def _run_init(args=None):
    """Initialize a project with mullet.yaml config."""
    from mullet.cli.init_project import run_init
    if args is None:
        import argparse
        args = argparse.Namespace(
            remaining=[],
            target_dir=None,
            pool=None,
            client=None,
            providers=None,
            model=None,
            speed=None,
            force=False,
            dry_run=False,
            no_claude=False,
            no_memory=False,
            no_pool=True,
        )
    run_init(args)


def _run_doctor(full: bool = False, fix: bool = False) -> int:
    """Check system health and report status of all components.

    Read-only. Without --full it makes no network calls, so it stays quick.

    Args:
        full: If True, also check GitHub for a newer Mullet and probe the
              provider APIs. Slower, and needs the network.
        fix: If True, attempt auto-remediation for non-passing checks.

    Returns:
        Exit code: 0 if all checks pass (or all fixed when fix=True), 1 otherwise.
    """
    import mullet
    from mullet.cli.doctor_checks import claude_setup_checks, gpu_checks, spending_checks, toolchain_checks

    print(f"Mullet v{mullet.__version__} — System Health Check\n")

    if full:
        try:
            from mullet.cli.update_checker import check_for_update
            _upd = check_for_update(timeout=3.0)
            if _upd["update_available"]:
                print(f"  ⚠  {_upd['message']}\n")
        except Exception:
            pass

    # Python, Claude Code CLI, git, the job sandbox (bubblewrap and socat on
    # Linux) and a warning when ANTHROPIC_API_KEY would override subscription login.
    checks = toolchain_checks()

    # Local LLM
    local = detect_local_provider()
    if local:
        if local.health_check():
            ctx_note = ""
            ctx_status = "ok"
            try:
                loaded_ctx = get_local_loaded_context()
                if loaded_ctx > 0:
                    ctx_note = f", {loaded_ctx // 1000}K ctx"
                    declared_ctx = get_opencode_declared_context(local.model)
                    if declared_ctx and loaded_ctx < declared_ctx:
                        ctx_note += (
                            f" ⚠ opencode.json declares {declared_ctx // 1000}K for "
                            f"this model — increase Context Length in LM Studio and "
                            f"reload, or real tasks will fail with a context-overflow error"
                        )
                        ctx_status = "warn"
                    elif loaded_ctx < 4096:
                        ctx_note += " ⚠ very small"
                        ctx_status = "warn"
            except Exception:
                pass
            checks.append(("Local LLM", f"{local.name} ({local.model}{ctx_note})", ctx_status))
        else:
            checks.append(("Local LLM", "Running but no model loaded — load a model in LM Studio", "warn"))
    else:
        checks.append(("Local LLM", "Not found (start LM Studio/Ollama)", "warn"))

    # The GPU: free VRAM, temperature, who holds Mullet's GPU lease, LM Studio
    # on the GPU, and the power limit.
    checks.extend(gpu_checks())

    # No spending limits set; with --full, paid providers running low on credit.
    checks.extend(spending_checks(full))

    # Providers allowed to run commands or touch files on this machine.
    from mullet.cli.doctor_checks import toolset_checks
    checks.extend(toolset_checks())

    # Google Gemini
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        checks.append(("Google Gemini", "API key set", "ok"))
    else:
        checks.append(("Google Gemini", "GOOGLE_API_KEY not set", "info"))

    # OpenRouter
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openrouter_absent_note = None
    if openrouter_key:
        checks.append(("OpenRouter API", "Key set", "ok"))
        if full:
            try:
                import json as _json
                import time as _time
                import urllib.request
                openrouter_url = "https://openrouter.ai/api/v1/models"
                req = urllib.request.Request(
                    openrouter_url,
                    headers={"Authorization": f"Bearer {openrouter_key}"},
                    method="GET",
                )
                start = _time.monotonic()
                with urllib.request.urlopen(req, timeout=5) as resp:
                    latency_ms = int((_time.monotonic() - start) * 1000)
                    if resp.status == 200:
                        data = _json.loads(resp.read().decode("utf-8"))
                        model_count = len(data.get("data", []))
                        checks.append(("OpenRouter API probe", f"{model_count} models available ({latency_ms}ms)", "ok"))
                    else:
                        checks.append(("OpenRouter API probe", f"HTTP {resp.status}", "warn"))
            except Exception as _exc:
                checks.append(("OpenRouter API probe", f"reachable check failed: {_exc}", "warn"))
    else:
        openrouter_absent_note = ("OpenRouter API", "OPENROUTER_API_KEY not set (optional)", "info")

    # Kimi K2 (Moonshot) — primary delegation target.
    # Note: when key is absent we still print an info line so the user sees the gap,
    # but it is intentionally NOT added to `checks` so it doesn't count against the
    # X/Y passed denominator (users without a Kimi key aren't penalised).
    kimi_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
    kimi_absent_note = None
    if kimi_key:
        which = "KIMI_API_KEY" if os.environ.get("KIMI_API_KEY") else "MOONSHOT_API_KEY"
        checks.append(("Kimi API", f"{which} set", "ok"))
        if full:
            # Optional reachability probe — only when --full is passed.
            try:
                import urllib.request
                kimi_url = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
                req = urllib.request.Request(
                    f"{kimi_url.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {kimi_key}"},
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        checks.append(("Kimi API probe", "reachable (models endpoint 200 OK)", "ok"))
                    else:
                        checks.append(("Kimi API probe", f"HTTP {resp.status}", "warn"))
            except Exception as _exc:
                checks.append(("Kimi API probe", f"reachable check failed: {_exc}", "warn"))
    else:
        kimi_absent_note = ("Kimi API", "KIMI_API_KEY not set", "info")

    # Knowledge engine
    try:
        from mullet.core.knowledge.manager import _get_engine_module
        client_mod = _get_engine_module("client")
        clients = client_mod.discover_clients()
        checks.append(("Knowledge Engine", f"Built-in, {len(clients)} clients", "ok"))
    except ImportError:
        checks.append(("Knowledge Engine", "Not available (missing dependencies)", "warn"))
    except Exception as e:
        checks.append(("Knowledge Engine", f"Error: {e}", "warn"))

    # Skills
    try:
        from mullet.core.skills.loader import SkillLoader
        from mullet.core.skills.registry import SkillRegistry
        loader = SkillLoader()
        reg = SkillRegistry(loader=loader)
        reg.load_all()
        skills = reg.list_skills()
        checks.append(("Skills", f"{len(skills)} loaded", "ok"))
    except Exception:
        checks.append(("Skills", "Error loading", "warn"))

    # Tools
    registry = build_tool_registry()
    checks.append(("Tools", f"{len(registry.list_tools())} available", "ok"))

    # Config
    from pathlib import Path
    from mullet.cli.doctor_checks import config_file, database_path
    config_paths = [Path("mullet.yaml"), config_file("config.yaml")]
    config_found = any(p.exists() for p in config_paths)
    if config_found:
        checks.append(("Config", "mullet.yaml found", "ok"))
    else:
        checks.append(("Config", "Using defaults (no mullet.yaml)", "info"))

    # Config validation
    if config_found:
        try:
            from mullet.config.loader import load_from_yaml
            from mullet.config.validate import validate_config
            yaml_data = load_from_yaml()
            validation = validate_config(yaml_data)
            if validation.valid and not validation.issues:
                checks.append(("Config Valid", "No issues", "ok"))
            elif validation.valid:
                checks.append(("Config Valid", f"{len(validation.issues)} warnings", "info"))
            else:
                checks.append(("Config Valid", f"{len([i for i in validation.issues if i.severity == 'error'])} errors", "warn"))
        except Exception as _exc:
            checks.append(("Config Valid", f"Error validating config: {_exc}", "warn"))

    # Database
    db_path = database_path()
    if db_path.exists():
        checks.append(("Database", str(db_path), "ok"))
    else:
        checks.append(("Database", "Will be created on first use", "info"))

    # Project detection
    try:
        from mullet.core.agent.project_detect import detect_project
        project = detect_project()
        if project.language != "unknown":
            detail = project.language
            if project.framework:
                detail += f" ({project.framework})"
            checks.append(("Project", detail, "ok"))
    except Exception as _exc:
        logger.warning("cli/main: Failed to detect project: %s", _exc)

    # Claude Code: what `mullet setup-claude` recorded, and hooks an older
    # setup-claude left pointing into a checkout. Never runs `claude mcp`.
    checks.extend(claude_setup_checks())

    # Security posture
    try:
        from mullet.server.auth import _record_failure, _rate_limit_check  # noqa: F401  (importability probe)
        from mullet.server.mcp_handlers import _sanitize_paths  # noqa: F401  (importability probe)
        checks.append(("Auth rate limiting", "Active (5/60s → 5s delay, 10/60s → 30s, 20/60s → 60s)", "ok"))
        checks.append(("Path traversal guard", "Active (files + target_dir sanitized)", "ok"))
    except ImportError as e:
        checks.append(("Security guards", f"Import error: {e}", "warn"))

    try:
        from mullet.core.providers.openai_compat import _consecutive_failures_lock  # noqa: F401  (importability probe)
        checks.append(("Provider failure lock", "Thread-safe (_consecutive_failures_lock present)", "ok"))
    except ImportError:
        checks.append(("Provider failure lock", "Not found", "warn"))

    # pricing.yaml — single source of truth for model pricing
    try:
        from mullet.core.providers.pricing import load_pricing, _PRICING_YAML
        _pricing = load_pricing()
        _n_models = len(_pricing)
        if _n_models >= 30:
            checks.append(("pricing.yaml", f"{_n_models} models, SSOT active ({_PRICING_YAML.name})", "ok"))
        else:
            checks.append(("pricing.yaml", f"Only {_n_models} models — may be incomplete", "warn"))
    except FileNotFoundError:
        checks.append(("pricing.yaml", "Missing — run: pip install -e .", "warn"))
    except Exception as _e:
        checks.append(("pricing.yaml", f"Load error: {_e}", "warn"))

    # Pool coverage — count indexed pools + flag config/exclude issues
    try:
        from mullet.core.env import get_pools_dir
        from mullet.cli.pool_audit import _check_config_validity, _check_exclude_patterns, _EMPTY_CHROMA_SIZE_MAX
        pools_dir = get_pools_dir()
        if pools_dir.exists():
            pool_dirs = sorted(d for d in pools_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
            indexed_count = 0
            config_issues = 0
            for pool_dir in pool_dirs:
                try:
                    # "Indexed" = chroma.sqlite3 exists and is larger than the empty-init sentinel
                    chroma = pool_dir / "index" / "chroma.sqlite3"
                    if chroma.exists() and chroma.stat().st_size > _EMPTY_CHROMA_SIZE_MAX:
                        indexed_count += 1
                    # Config validity check (fast — just reads YAML)
                    import yaml as _yaml
                    for cfg_name in ("config.yaml", "client.yaml"):
                        cfg_path = pool_dir / cfg_name
                        if cfg_path.exists():
                            try:
                                cfg = _yaml.safe_load(cfg_path.read_text()) or {}
                                issues = _check_config_validity(cfg) + _check_exclude_patterns(pool_dir, cfg)
                                if issues:
                                    config_issues += 1
                            except Exception:
                                config_issues += 1
                            break
                except Exception:
                    pass
            total = len(pool_dirs)
            if total == 0:
                checks.append(("Pool coverage", "No pools configured", "info"))
            else:
                detail = f"{indexed_count}/{total} indexed"
                if config_issues:
                    detail += f", {config_issues} config issue{'s' if config_issues != 1 else ''} — run: mullet pool health"
                status = "ok" if indexed_count == total and config_issues == 0 else "warn" if config_issues else "info"
                checks.append(("Pool coverage", detail, status))
        else:
            checks.append(("Pool coverage", "No pools directory", "info"))
    except Exception as _exc:
        checks.append(("Pool coverage", f"Error checking pools: {_exc}", "warn"))

    # Print results
    icons = {"ok": "\033[32m✓\033[0m", "warn": "\033[33m!\033[0m", "info": "\033[36m·\033[0m"}
    for name, detail, status in checks:
        icon = icons.get(status, "·")
        print(f"  {icon} {name:20s} {detail}")

    # Surface optional notes that should be visible but not counted in the denominator
    # (e.g. Kimi key absence — informative, not a failure).
    if kimi_absent_note is not None:
        name, detail, status = kimi_absent_note
        icon = icons.get(status, "·")
        print(f"  {icon} {name:20s} {detail}")

    if openrouter_absent_note is not None:
        name, detail, status = openrouter_absent_note
        icon = icons.get(status, "·")
        print(f"  {icon} {name:20s} {detail}")

    print()
    ok_count = sum(1 for _, _, s in checks if s == "ok")
    print(f"{ok_count}/{len(checks)} checks passed")

    # Routing quality summary (shown only when data is available — not a hard check)
    try:
        import json as _jq
        from mullet.core.routing_history import history_path
        from mullet.core.router.decision import QUALITY_THRESHOLD, QUALITY_MIN_SAMPLES
        _hist_path = history_path()
        if _hist_path.exists():
            _raw: dict[str, list[bool]] = {}
            for _line in _hist_path.read_text(encoding="utf-8").splitlines():
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _d = _jq.loads(_line)
                except _jq.JSONDecodeError:
                    continue
                _p = _d.get("provider", "")
                if _p:
                    _raw.setdefault(_p, []).append(bool(_d.get("success", False)))
            if _raw:
                print("\nRouting quality (recent history):")
                for _pname, _outcomes in sorted(_raw.items()):
                    _n = len(_outcomes)
                    _rate = sum(_outcomes) / _n
                    if _n < QUALITY_MIN_SAMPLES:
                        _icon = "\033[36m·\033[0m"
                        _suffix = f"  ({_n} sample{'s' if _n != 1 else ''}, need {QUALITY_MIN_SAMPLES} to filter)"
                    elif _rate < QUALITY_THRESHOLD:
                        _icon = "\033[33m!\033[0m"
                        _suffix = f"  ← below {QUALITY_THRESHOLD:.0%} threshold (router skips)"
                    else:
                        _icon = "\033[32m✓\033[0m"
                        _suffix = ""
                    print(f"  {_icon} {_pname:<20} {_rate:.0%} ({_n} tasks){_suffix}")
    except Exception:
        pass  # routing quality is informational — never fail doctor over it
    else:
        # Suggest purge if --fix requested and providers are below threshold
        if fix and _raw:
            _below = [
                _pname for _pname, _outcomes in _raw.items()
                if len(_outcomes) >= QUALITY_MIN_SAMPLES
                and sum(_outcomes) / len(_outcomes) < QUALITY_THRESHOLD
            ]
            if _below:
                print()
                print("  Routing quality hints (use after fixing the underlying issue):")
                for _pname in _below:
                    print(f"    mullet routing purge-provider {_pname} --yes")

    if not local and not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠ No local LLM found: delegation needs one, or a provider key such as "
              "OPENROUTER_API_KEY. Mullet jobs need only the claude CLI.")

    return 0 if ok_count == len(checks) else 1


def _get_knowledge(args):
    """Load KnowledgeManager if --client is specified."""
    if not args.client:
        return None
    try:
        from mullet.core.knowledge.manager import KnowledgeManager
        return KnowledgeManager(args.client)
    except Exception:
        return None


def _maybe_preflight(task: str, args) -> None:
    """Show a one-shot budget snapshot before large tasks. Skips silently if disabled."""
    if getattr(args, "no_preflight", False) or os.environ.get("MULLET_NO_PREFLIGHT"):
        return
    if getattr(args, "json_output", False):
        return
    if not sys.stderr.isatty():
        return
    try:
        from mullet.core.budget_advisor import (
            should_preflight,
            recommend_lane,
            format_preflight_message,
        )
        iters = int(getattr(args, "max_iterations", 1) or 1)
        if not should_preflight(len(task or ""), iters):
            return
        user_pref = getattr(args, "speed", None)
        valid = {"premium", "balanced", "economy", "fast", "max"}
        pref = user_pref if user_pref in valid else None
        rec = recommend_lane(len(task or ""), iters, user_preference=pref)
        sys.stderr.write(format_preflight_message(rec, []) + "\n")
        sys.stderr.flush()
    except Exception as _exc:
        logger.debug("preflight skipped: %s", _exc)


def _delegation_refused(args=None) -> bool:
    """Say so and return True when delegation is off (MULLET_DELEGATION or the strategy).

    Every CLI path that sends work to a model provider (task runs, review,
    watch, the REPL, live-test) checks this before
    building a provider, so nothing is dispatched while delegation is off.
    """
    from mullet.core.env import delegation_enabled, delegation_off_message
    if delegation_enabled():
        return False
    if args is not None and getattr(args, "json_output", False):
        import json as _json
        print(_json.dumps({"success": False, "error": delegation_off_message(), "delegation": "off"}))
    else:
        print(delegation_off_message(), file=sys.stderr)
    return True


def run_task(task: str, args) -> int:
    """Run a single task and return exit code."""
    if _delegation_refused(args):
        return 2
    args._last_task = task  # Store for quality scoring

    _maybe_preflight(task, args)

    agent, _, _ = build_agent(args, task)

    # Track file changes during task execution
    from mullet.core.agent.diff import ChangeTracker
    _tracker = ChangeTracker(watch_dirs=[os.getcwd()])
    _tracker.snapshot_before()

    is_json = getattr(args, "json_output", False)
    import time as _time
    _start_time = _time.time()

    # Show brief provider info (not verbose — just a hint)
    if not is_json:
        pname = getattr(agent.provider, "name", "?")
        tier = getattr(agent, "tier_used", "")
        sys.stderr.write(f"\033[90m{pname}" + (f" ({tier})" if tier else "") + "\033[0m\n")
        sys.stderr.flush()

    # Streaming: show tokens and tool calls as they happen (disabled for JSON)
    streamed_content = []

    def _on_token(delta: str):
        if not is_json:
            streamed_content.append(delta)
            sys.stdout.write(delta)
            sys.stdout.flush()

    def _on_tool(name: str, preview: str):
        indicator = f"\n\033[36m⚡ {name}\033[0m"
        if preview:
            indicator += f" {preview[:60]}"
        indicator += "\n"
        sys.stderr.write(indicator)
        sys.stderr.flush()

    def _on_tool_result(name: str, preview: str):
        sys.stderr.write(f"\033[32m  ✓ {name}\033[0m {preview[:80]}\n")
        sys.stderr.flush()

    agent.on_token = _on_token
    agent.on_tool_call = _on_tool
    agent.on_tool_result = _on_tool_result

    try:
        result = agent.run(task)
    except Exception as e:
        from mullet.core.agent.errors import format_error
        print(format_error(e), file=sys.stderr)
        return 1

    duration = _time.time() - _start_time

    # If JSON mode, use structured output
    if is_json:
        _print_result(result, args, duration=duration)
    elif streamed_content:
        sys.stdout.write("\n")
    else:
        print(result.final_response)

    if getattr(args, "verbose", False):
        _print_verbose(result, args)

    # Show file changes (only if tools were used and files actually changed)
    if result.tool_calls_made > 0 and not is_json:
        _tracker.snapshot_after()
        changes_text = _tracker.format_changes()
        if changes_text.strip():
            print(changes_text, file=sys.stderr)

    # Record routing outcome for learning
    try:
        from mullet.core.router.learning import record_outcome
        # Use quality score to weight the learning signal
        quality_weighted_success = result.completed
        try:
            from mullet.core.agent.quality import score_result
            q = score_result(result, task=task)
            quality_weighted_success = q.overall >= 50  # Quality threshold
        except Exception:
            pass
        record_outcome(
            task=task,
            tier=getattr(result, "tier_used", ""),
            provider=result.provider_used,
            success=quality_weighted_success,
            tool_calls=result.tool_calls_made,
            tokens=result.tokens_used.total_tokens,
        )
    except Exception as _exc:
        logger.warning("cli/main: Failed to record routing outcome: %s", _exc)

    # Export conversation if requested
    export_path = getattr(args, "export", None)
    if export_path:
        from mullet.core.agent.export import export_json, export_markdown
        if export_path.endswith(".md"):
            export_markdown(result, task, export_path)
        else:
            export_json(result, task, export_path)
        print(f"\nConversation exported to: {export_path}", file=sys.stderr)

    # Skill pipeline auto-chain: if the invoked skill has next_skill, run it
    matched_skill = getattr(args, "_matched_skill", None)
    if matched_skill and result.completed:
        next_name = matched_skill.metadata.next_skill
        if next_name:
            # Cycle/depth guard for skill pipeline recursion
            visited = getattr(args, "_visited_skills", None)
            if visited is None:
                visited = set()
                args._visited_skills = visited
            if next_name in visited:
                logger.warning("Skill pipeline cycle detected: %s already visited — breaking chain", next_name)
            elif len(visited) >= 10:
                logger.warning("Skill pipeline depth limit (10) reached — breaking chain")
            else:
                visited.add(next_name)
                try:
                    from mullet.core.skills.loader import SkillLoader
                    from mullet.core.skills.registry import SkillRegistry
                    loader = SkillLoader()
                    registry = SkillRegistry(loader=loader)
                    registry.load_all()
                    next_invocation = registry.invoke(next_name)
                    if next_invocation:
                        handoff_context = ""
                        if matched_skill.metadata.handoff:
                            from mullet.core.skills.invoker import resolve_handoff_artifact
                            artifact = resolve_handoff_artifact(matched_skill.metadata.handoff)
                            if artifact:
                                handoff_context = f"\n\nHandoff artifact from previous skill:\n{Path(artifact).read_text()[:4000]}"
                        print(f"\n\033[34m▶ Pipeline → {next_name}\033[0m", file=sys.stderr)
                        next_task = f"{task}{handoff_context}"
                        return run_task(next_task, args)
                except Exception as e:
                    logger.debug("Skill pipeline chain failed: %s", e)

    return 0 if result.completed else 1


def _print_result(result: AgentResult, args, duration: float = 0):
    """Print the agent result (for non-streaming mode)."""
    if getattr(args, "json_output", False):
        import json
        from mullet.core.agent.cost import estimate_cost, estimate_savings
        cost = estimate_cost(result.provider_used, result.tier_used or "", result.tokens_used)
        savings = estimate_savings(result.provider_used, result.tier_used or "", result.tokens_used)
        print(json.dumps({
            "completed": result.completed,
            "response": result.final_response,
            "tool_calls": result.tool_calls_made,
            "iterations": result.iterations_used,
            "tokens": {
                "input": result.tokens_used.input_tokens,
                "output": result.tokens_used.output_tokens,
                "total": result.tokens_used.total_tokens,
            },
            "provider": result.provider_used,
            "tier": result.tier_used,
            "cost": round(cost, 6),
            "savings": round(savings, 6),
            "duration": round(duration, 2),
        }, indent=2))
    else:
        print(result.final_response)
        _print_verbose(result, args)


def _print_verbose(result: AgentResult, args):
    """Print verbose stats (always to stderr)."""
    if not getattr(args, "verbose", False):
        return
    from mullet.core.agent.cost import format_cost_summary
    cost_line = format_cost_summary(
        result.provider_used,
        getattr(result, "tier_used", ""),
        result.tokens_used,
    )
    print("\n---", file=sys.stderr)
    print(f"Tools called: {result.tool_calls_made}", file=sys.stderr)
    print(f"Iterations: {result.iterations_used}", file=sys.stderr)
    print(f"Tokens: {result.tokens_used.total_tokens:,}", file=sys.stderr)
    print(f"Provider: {result.provider_used} ({result.tier_used or '?'})", file=sys.stderr)
    print(f"{cost_line}", file=sys.stderr)

    # Quality score
    try:
        from mullet.core.agent.quality import score_result
        task = getattr(args, "_last_task", "")
        quality = score_result(result, task=task)
        print(f"Quality: {quality.overall:.0f}/100", file=sys.stderr)
    except Exception as _exc:
        logger.warning("cli/main: Failed to score result quality: %s", _exc)


def _run_swarm_command(goal: str, raw_args: list[str], parsed_args) -> None:
    """Handle `mullet swarm "goal" [--files 'glob'] [--concurrency N] [--dry-run]` subcommand."""
    import asyncio
    import glob as _glob
    from mullet.core.orchestration.swarm import run_swarm
    from mullet.core.orchestration.swarm_types import SwarmContext

    if not goal:
        print("Usage: mullet swarm \"<goal>\" [--files '<glob>'] [--items a b c] [--concurrency N] [--dry-run]", file=sys.stderr)
        sys.exit(1)

    # Parse flags from raw_args
    files: list[str] = []
    items: list[str] = []
    concurrency = 20
    dry_run = "--dry-run" in raw_args
    output_shape = "markdown report"

    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        if arg in ("--files", "-f") and i + 1 < len(raw_args):
            pattern = raw_args[i + 1]
            files = sorted(_glob.glob(pattern, recursive=True))
            i += 2
        elif arg == "--items" and i + 1 < len(raw_args):
            i += 1
            while i < len(raw_args) and not raw_args[i].startswith("--"):
                items.append(raw_args[i])
                i += 1
        elif arg in ("--concurrency", "-c") and i + 1 < len(raw_args):
            concurrency = int(raw_args[i + 1])
            i += 2
        elif arg == "--output-shape" and i + 1 < len(raw_args):
            output_shape = raw_args[i + 1]
            i += 2
        else:
            i += 1

    ctx = SwarmContext(files=files, items=items, concurrency=concurrency, output_shape=output_shape)

    print(f"Swarm: {goal!r}")
    if files:
        print(f"  Files: {len(files)} matched")
    if items:
        print(f"  Items: {len(items)}")
    print(f"  Concurrency: {concurrency}")
    if dry_run:
        print("  Mode: dry-run (decompose only)")

    def _progress(result, completed, total):
        status = "✓" if result.success else "✗"
        elapsed = f"{result.elapsed_seconds:.1f}s" if result.elapsed_seconds else ""
        err = f" ({result.error})" if not result.success and result.error else ""
        print(f"  [{completed}/{total}] {status} {result.task_id}{err} {elapsed}", file=sys.stderr, flush=True)

    result = asyncio.run(run_swarm(goal, ctx, dry_run=dry_run, on_progress=_progress))

    if dry_run:
        print(f"\n{len(result.tasks)} tasks planned:")
        for t in result.tasks:
            print(f"  [{t.id}] {t.instruction[:80]}")
        if result.estimated_input_tokens:
            print(f"\nEstimated: ~{result.estimated_input_tokens:,} input tokens"
                  f"  ≈ ${result.estimated_cost_usd:.4f}")
        return

    succeeded = sum(1 for r in result.results if r.success)
    print(f"\nCompleted: {succeeded}/{len(result.results)} tasks succeeded in {result.wall_clock_seconds:.1f}s")
    if result.partial_failure:
        print(f"  WARNING: {len(result.results) - succeeded} task(s) failed")
    print(f"  Tokens: {result.total_tokens.input_tokens} in / {result.total_tokens.output_tokens} out")
    print("\n--- SYNTHESIS ---\n")
    print(result.synthesis)


def _run_models_command(args: list[str]) -> None:
    """Handle `mullet models [list|refresh] [--provider PROVIDER]` subcommand."""
    from mullet.core.providers.model_registry import get_registry

    subaction = args[0] if args else "list"
    provider_filter: str | None = None
    for i, a in enumerate(args):
        if a in ("--provider", "-p") and i + 1 < len(args):
            provider_filter = args[i + 1]

    registry = get_registry()

    if subaction == "refresh":
        print("Refreshing model registry from provider APIs...")
        counts = registry.refresh_all()
        for provider, count in counts.items():
            print(f"  {provider}: {count} models")
        print("Done.")
        return

    # Default: list
    all_models = registry.list_all()
    for provider, models in all_models.items():
        if provider_filter and provider_filter.lower() != provider:
            continue
        print(f"\n{provider.upper()} ({len(models)} models)")
        print(f"  {'ID':<40} {'TIER':<12} {'CTX':<10} {'IN/1K':>8}  {'OUT/1K':>8}")
        print("  " + "-" * 82)
        for m in sorted(models, key=lambda x: x.output_cost_per_1k):
            ctx = f"{m.context_length // 1000}K" if m.context_length >= 1000 else str(m.context_length)
            print(
                f"  {m.id:<40} {m.tier:<12} {ctx:<10} ${m.input_cost_per_1k:>7.4f}  ${m.output_cost_per_1k:>7.4f}"
            )


def main():
    """Entry point for the ``mullet`` CLI — delegates to ``_main_inner``."""
    try:
        _main_inner()
    except KeyboardInterrupt:
        sys.exit(130)
    except SystemExit:
        raise  # Let sys.exit() through
    except Exception as e:
        from mullet.core.agent.errors import format_error
        print(format_error(e), file=sys.stderr)
        # Unattended/overnight runs need full diagnostics, not just a friendly summary.
        if os.environ.get("MULLET_TRACEBACK", "").lower() in ("1", "true", "yes"):
            traceback.print_exc()
        logger.debug("Fatal error", exc_info=True)
        sys.exit(1)


_INSTALL_HELP = """\
Install Mullet from GitHub on Linux or macOS (needs git, and uv or pipx):

  curl -fsSL https://raw.githubusercontent.com/tomgould/mullet-public/main/scripts/install.sh | bash

The installer runs one of these, which you can also run yourself:

  uv tool install git+https://github.com/tomgould/mullet-public.git
  pipx install git+https://github.com/tomgould/mullet-public.git

With an optional extra (gemini, bedrock, postgres or all):

  uv tool install "mullet[gemini] @ git+https://github.com/tomgould/mullet-public.git"

Then run `mullet doctor` and `mullet setup-claude`.

Mullet is not on PyPI: the "mullet" package there is an unrelated project."""


def _run_packaged(module: str, args: list[str]) -> int:
    """Run one of Mullet's own modules as a script with this interpreter.

    ``python -m`` would put the current directory first on the import path, so
    a stray ``argparse.py`` there would run in place of the real one. The child
    drops that entry before importing anything.
    """
    import subprocess as _sp
    code = (
        "import sys; sys.path.pop(0); import runpy; "
        f"runpy.run_module({module!r}, run_name='__main__', alter_sys=True)"
    )
    # -u: the reindexer's progress output should appear as it happens
    return _sp.run([sys.executable, "-u", "-c", code, *args]).returncode


def _cmd_paths(argv: list[str]) -> int:
    """Print Mullet's directories as ``name=path`` lines, or one bare path."""
    from mullet.core import paths as _paths
    dirs = {
        "config": _paths.config_dir,
        "state": _paths.state_dir,
        "data": _paths.data_dir,
        "cache": _paths.cache_dir,
        "pools": _paths.pools_dir,
    }
    if not argv:
        for name, fn in dirs.items():
            print(f"{name}={fn()}")
        return 0
    if len(argv) == 1 and argv[0] in dirs:
        print(dirs[argv[0]]())
        return 0
    print(f"usage: mullet paths [{'|'.join(dirs)}]", file=sys.stderr)
    return 2


def _main_inner():
    if len(sys.argv) >= 2 and sys.argv[1] == "hook":
        from mullet.cli.hook_cli import cmd_hook
        sys.exit(cmd_hook(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "setup-claude":
        from mullet.cli.setup_claude import cmd_setup_claude
        sys.exit(cmd_setup_claude(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "preset":
        from mullet.cli.preset_cli import cmd_preset
        return cmd_preset(sys.argv[2:])
    if len(sys.argv) >= 2 and sys.argv[1] == "job":
        from mullet.cli.job_cli import cmd_job
        sys.exit(cmd_job(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "local":
        from mullet.cli.local_cli import cmd_local
        sys.exit(cmd_local(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "night":
        from mullet.cli.night_cli import cmd_night
        sys.exit(cmd_night(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "ladder":
        from mullet.cli.ladder_cli import cmd_ladder
        sys.exit(cmd_ladder(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "migrate-state":
        from mullet.cli.migrate_state import cmd_migrate_state
        sys.exit(cmd_migrate_state(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "paths":
        sys.exit(_cmd_paths(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "reap":
        from mullet.cli.reap_cmd import cmd_reap
        sys.exit(cmd_reap(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "strategy":
        from mullet.cli.strategy_cli import cmd_strategy
        sys.exit(cmd_strategy(sys.argv[2:]))

    parser = build_parser()
    args = parser.parse_args()

    if args.install_help:
        print(_INSTALL_HELP)
        return

    if args.version:
        if args.banner:
            from mullet.cli.banner import render_banner
            print(render_banner(no_color=not sys.stdout.isatty()))
        else:
            import mullet as _mullet_pkg
            print(f"mullet {_mullet_pkg.__version__}")
        return

    from mullet.config.logging import setup_logging
    setup_logging(verbose=args.verbose)

    # Info commands
    if args.tools:
        registry = build_tool_registry()
        print("Available tools:")
        for name in sorted(registry.list_tools()):
            print(f"  - {name}")
        print(f"\nToolsets: {', '.join(registry.list_toolsets())}")
        print(f"Profiles: {', '.join(registry.list_profiles())}")
        return

    if args.list_skills:
        try:
            from mullet.core.skills.loader import SkillLoader
            from mullet.core.skills.registry import SkillRegistry
            loader = SkillLoader()
            reg = SkillRegistry(loader=loader)
            reg.load_all()
            print("Available skills:")
            for s in reg.list_skills():
                mode_tag = f" [{s['mode']}]" if s.get("mode") else ""
                print(f"  /{s['name']}{mode_tag} — {s.get('description', '')}")
        except Exception as e:
            print(f"Could not load skills: {e}", file=sys.stderr)
        return

    # Extract command and task from remaining args
    remaining = args.remaining or []
    command = remaining[0] if remaining and remaining[0] in SUBCOMMANDS else None
    if command:
        task_parts = remaining[1:]
    else:
        task_parts = remaining
    task = " ".join(task_parts) if task_parts else ""

    # Subcommand dispatch
    if command == "doctor":
        if task in ("--help", "-h", "help"):
            print("Usage: mullet doctor [--full] [--fix]")
            print()
            print("Run the Mullet system health check (provider availability,")
            print("pool coverage, Claude integration, etc.).")
            print()
            print("Options:")
            print("  --full   Include slow / network-dependent checks (provider probes).")
            print("  --fix    Apply automated fixes for issues with known remediation.")
            sys.exit(0)
        full = "--full" in task_parts or "--full" in sys.argv
        fix = "--fix" in task_parts or "--fix" in sys.argv
        return _run_doctor(full=full, fix=fix)

    if command == "init":
        _run_init(args)
        return

    if command == "update":
        from mullet.cli.init_project import run_update
        run_update(args)
        return 0

    if command == "setup-claude":
        # Reached when global options come first (`mullet --verbose setup-claude`).
        from mullet.cli.setup_claude import cmd_setup_claude
        sys.exit(cmd_setup_claude(task_parts))

    if command == "models":
        _run_models_command(task_parts)
        return

    if command == "live-test":
        if task not in ("--help", "-h", "help") and _delegation_refused(args):
            sys.exit(2)
        from mullet.cli.live_test import run_cli as _run_live_test_cli
        sys.exit(_run_live_test_cli(task_parts))

    if command == "clean":
        if any(p in task_parts for p in ("-h", "--help", "help")):
            print("Usage: mullet clean [--force] [--max-age=SECONDS]")
            print()
            print("Remove stale agent entries from the HUD state file.")
            print()
            print("Options:")
            print("  --force          Clear ALL agents, not just stale ones")
            print("  --max-age=N      Treat agents older than N seconds as stale (default: 1800)")
            print()
            print("Examples:")
            print("  mullet clean                  # remove agents idle > 30 min")
            print("  mullet clean --max-age=600    # stricter (10 min)")
            print("  mullet clean --force          # nuke everything")
            sys.exit(0)
        from mullet.hud.agent_state import cleanup_stale_agents, clear_agents
        force = "--force" in task_parts
        max_age = 1800
        for part in task_parts:
            if part.startswith("--max-age="):
                try:
                    max_age = int(part.split("=", 1)[1])
                except ValueError:
                    pass
        if force:
            clear_agents()
            print("Cleared all agents from HUD.")
        else:
            removed = cleanup_stale_agents(max_age_seconds=max_age)
            if removed:
                print(f"Cleaned {removed} stale agent(s) from HUD (older than {max_age}s).")
            else:
                print("No stale agents found.")
        sys.exit(0)

    if command == "history":
        from mullet.core.agent.session import list_recent_sessions, search_sessions
        import datetime
        if task:
            # Search mode
            results = search_sessions(task, max_results=10)
            if results:
                print(f"Search results for '{task}':\n")
                for r in results:
                    ts = datetime.datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M") if r["timestamp"] else "?"
                    print(f"  [{ts}] ({r['role']}): {r['content'][:100]}")
            else:
                print(f"No results for '{task}'")
        else:
            # List mode
            sessions = list_recent_sessions(limit=10)
            if sessions:
                print("Recent sessions:\n")
                for s in sessions:
                    ts = datetime.datetime.fromtimestamp(s["created_at"]).strftime("%Y-%m-%d %H:%M") if s.get("created_at") else "?"
                    title = s.get("title") or s["id"][:20]
                    print(f"  {ts}  {s['provider'] or '?':12s}  {s['messages']} msgs  {title}")
            else:
                print("No sessions found. Use the REPL to create sessions.")
        return

    if command == "config":
        from mullet.cli.config_cli import cmd_config
        sys.exit(cmd_config(task_parts))

    if command == "stats":
        if task in ("--help", "-h", "help"):
            print("Usage: mullet stats")
            print()
            print("Print aggregate usage stats from session history:")
            print("  Sessions, Messages, Period, per-provider breakdown.")
            print()
            print("See also: `mullet history` (per-session), `mullet logs` (LLM calls).")
            sys.exit(0)
        from mullet.cli.stats import format_stats
        print(format_stats())
        return

    if command == "override":
        from mullet.cli.override_cmd import run as run_override
        sys.exit(run_override(remaining[1:] if len(remaining) > 1 else []))

    if command == "observations":
        from mullet.cli.observations_cmd import run as run_observations
        sys.exit(run_observations(remaining[1:] if len(remaining) > 1 else []))

    if command == "decisions":
        from mullet.cli.decisions_cmd import run as run_decisions
        sys.exit(run_decisions(remaining[1:] if len(remaining) > 1 else []))

    if command == "routing":
        from mullet.cli.routing_cmd import run as run_routing
        sys.exit(run_routing(remaining[1:] if len(remaining) > 1 else []))

    if command == "review-session":
        from mullet.cli.review_session_cmd import run as run_review
        sys.exit(run_review(remaining[1:] if len(remaining) > 1 else []))

    if command == "memory-sync":
        from mullet.cli.memory_sync_cmd import run as run_memory_sync
        sys.exit(run_memory_sync(remaining[1:] if len(remaining) > 1 else []))

    if command == "logs":
        if task in ("--help", "-h", "help"):
            print("Usage: mullet logs [tail [-n N] | stats | suggest]")
            print()
            print("Inspect LLM interaction history written by the provider routing layer.")
            print()
            print("Subcommands:")
            print("  tail [-n N]   Show the last N entries (default 20). Default if omitted.")
            print("  stats         Aggregate per-provider/model usage + latency + cost.")
            print("  suggest       LLM-flagged suggestions surfaced from recent calls.")
            sys.exit(0)
        from mullet.cli.logs_cmd import run_logs_tail, run_logs_stats, run_logs_suggest
        sub = remaining[1] if len(remaining) > 1 else "tail"
        if sub == "stats":
            run_logs_stats()
        elif sub == "suggest":
            run_logs_suggest()
        else:
            n = 20
            if "-n" in remaining:
                try:
                    n = int(remaining[remaining.index("-n") + 1])
                except (ValueError, IndexError):
                    pass
            run_logs_tail(n=n)
        return

    if command == "plugins":
        from mullet.core.plugins import list_plugins
        from pathlib import Path
        registry = build_tool_registry()  # This triggers plugin loading
        plugins = list_plugins()
        print("\033[1mMullet Plugins\033[0m\n")
        if plugins:
            for p in plugins:
                source = p.get("source", "entry-point") if isinstance(p, dict) else "entry-point"
                print(f"  \033[32m✓\033[0m {p['name']:25s} ({source})")
        else:
            print("  No plugins loaded.")
        print()
        # Show plugin directories
        from mullet.core.paths import data_dir
        dirs = [data_dir() / "plugins", Path(".mullet") / "plugins"]
        print("  Plugin directories:")
        for d in dirs:
            exists = "exists" if d.exists() else "create to add plugins"
            print(f"    {d} ({exists})")
        # Skill stats summary
        try:
            from mullet.core.skills.stats import top_skills
            rows = top_skills(5)
            if rows:
                print("\n  Top skills (invocations):")
                for r in rows:
                    rate = f"{r['success_rate']:.0%}" if r["success_rate"] is not None else "n/a"
                    print(f"    {r['name']:30s} {r['invocations']:4d} invocations  success: {rate}")
        except Exception:
            pass
        print("\n  See: mullet docs or CONTRIBUTING.md for plugin authoring guide.")
        return

    if command == "budget":
        if task in ("--help", "-h", "help"):
            print("Usage: mullet budget [--refresh|-r] [--json] [--watch|-w]")
            print()
            print("Show per-provider balance + spend cap status.")
            print()
            print("Options:")
            print("  --refresh, -r    Force a fresh probe of every provider (slower).")
            print("  --json           Output machine-readable JSON instead of the table.")
            print("  --watch, -w      Refresh every 30 seconds; Ctrl-C to exit.")
            print()
            print("See also: `mullet config budget` (set caps), `mullet cost` (pricing).")
            sys.exit(0)
        import asyncio
        import time as _time
        from mullet.core.providers.budget_probe import probe_all

        refresh = "--refresh" in task_parts or "-r" in task_parts
        json_out = "--json" in task_parts
        watch = "--watch" in task_parts or "-w" in task_parts

        def _print_budget_table(balances: dict, console, *, first: bool = False) -> None:
            from rich.table import Table
            from datetime import datetime as _dt
            ts = _dt.now().strftime("%H:%M:%S")
            title = f"Provider Budget Status  [{ts}]"
            table = Table(title=title, show_header=True, header_style="bold cyan")
            table.add_column("Provider", style="bold", min_width=25)
            table.add_column("Balance / Status", min_width=35)
            table.add_column("Spend Cap", justify="right", min_width=10)
            table.add_column("Priority", justify="center", min_width=10)
            table.add_column("Available", justify="center")
            try:
                from mullet.core.providers.spend_caps import load_budget_config
                _caps = load_budget_config().caps
            except Exception:
                _caps = {}
            low_warnings = []
            for k, b in balances.items():
                avail_icon = "✓" if b.has_budget else "✗"
                avail_style = "green" if b.has_budget else "red"
                priority = f"{b.priority_score:.2f}"
                summary = b.summary_line()
                low_style = " [yellow]⚠ LOW[/yellow]" if b.is_low else ""
                cap = _caps.get(k)
                cap_str = f"${cap.cap_usd:.0f}" if cap else "[dim]-[/dim]"
                table.add_row(b.name, summary + low_style, cap_str, priority, f"[{avail_style}]{avail_icon}[/{avail_style}]")
                if b.is_low:
                    low_warnings.append(b.name)
            console.print(table)
            if low_warnings:
                console.print(f"[bold yellow]⚠  Low balance: {', '.join(low_warnings)}[/bold yellow]")
            hint = "--watch/-w for live refresh" if not watch else "Ctrl+C to exit watch mode"
            console.print(f"[dim]Cache TTL: 5 min. Use --refresh/-r to re-probe. {hint}[/dim]")
            try:
                from mullet.core.providers.cost_history import get_percentiles
                stats = get_percentiles(window_days=30)
                if stats["n"] >= 2:
                    print(f"  30-day cost distribution: p50=${stats['p50']:.3f} p95=${stats['p95']:.3f} mean=${stats['mean']:.3f} (n={stats['n']})")
            except Exception:
                pass

        balances = asyncio.run(probe_all(force_refresh=refresh))

        if json_out:
            import json
            out = {}
            for k, b in balances.items():
                out[k] = {
                    "name": b.name,
                    "remaining_usd": b.credits_remaining_usd,
                    "used_usd": b.credits_used_usd,
                    "total_usd": b.credits_total_usd,
                    "is_subscription": b.is_subscription,
                    "is_free": b.is_free,
                    "has_budget": b.has_budget,
                    "priority_score": round(b.priority_score, 3),
                    "window_label": b.window_label,
                    "window_resets_at": b.window_resets_at,
                    "error": b.error,
                    "summary": b.summary_line(),
                }
            print(json.dumps(out, indent=2))
            return

        from rich.console import Console
        console = Console()

        if not watch:
            _print_budget_table(balances, console, first=True)
            return

        # Watch mode: refresh every 60s until Ctrl+C
        try:
            import os as _os
            while True:
                _os.system("clear" if _os.name != "nt" else "cls")
                _print_budget_table(balances, console)
                _time.sleep(60)
                balances = asyncio.run(probe_all(force_refresh=True))
        except KeyboardInterrupt:
            console.print("\n[dim]Watch mode stopped.[/dim]")
        return

    if command == "cost":
        from mullet.core.providers.model_info import get_model_info
        print("\033[1mMullet Cost Guide\033[0m — estimated cost per 1K tokens\n")
        # Refresh 2026-05-19 — see docs/MODEL_ROUTING.md for the canonical
        # cost+routing table. Pricing per 1K tokens (input+output combined).
        tiers = [
            ("\033[32mLOCAL (free)\033[0m", [
                ("Local LLM", "local"),
            ]),
            ("\033[33mCHEAP (sub-$0.01/1K)\033[0m", [
                ("Gemini 2.5 Flash", "gemini-2.5-flash"),
                ("DeepSeek V4 Flash", "deepseek/deepseek-v4-flash"),
                ("DeepSeek V4 Pro", "deepseek/deepseek-v4-pro"),
                ("Qwen3 Coder", "qwen/qwen3-coder"),
                ("GLM 4.6", "z-ai/glm-4.6"),
                ("Kimi K2.7-Code", "kimi-k2.7-code"),
                ("Kimi K2.6", "kimi-k2.6"),
            ]),
            ("\033[31mPREMIUM\033[0m", [
                ("Claude Sonnet 4.5", "anthropic/claude-sonnet-4.5"),
                ("Claude Opus 4.1", "anthropic/claude-opus-4.1"),
                ("GPT-5", "openai/gpt-5"),
                ("Gemini 2.5 Pro", "google/gemini-2.5-pro"),
                ("Grok 4.20", "x-ai/grok-4.20"),
            ]),
        ]
        for tier_name, models in tiers:
            print(f"  {tier_name}")
            for display, model_id in models:
                info = get_model_info(model_id)
                total = info.input_cost_per_1k + info.output_cost_per_1k
                ctx = f"{info.context_length//1000}K"
                print(f"    {display:20s} ${total:.4f}/1K  ({ctx} context)")
            print()
        print("  Mullet routes to the cheapest tier that can handle each task.")
        print("  Use 'quick:' prefix to force LOCAL: mullet \"quick: do something\"")
        print("  Full routing table: docs/MODEL_ROUTING.md")
        return

    if command == "completion":
        from mullet.cli.completion import get_completion
        shell = task_parts[0] if task_parts else "bash"
        print(get_completion(shell))
        return

    if command == "explain":
        if not task:
            print("Usage: mullet explain \"your task here\"", file=sys.stderr)
            sys.exit(1)
        from mullet.core.router.complexity import extract_signals, score_complexity, score_to_tier, is_simple_message, confidence_breakdown
        from mullet.core.agent.cost import estimate_cost, estimate_savings
        from mullet.core.agent.types import TokenUsage

        is_simple = is_simple_message(task)
        signals = extract_signals(task)
        score = score_complexity(signals)
        tier = score_to_tier(score)

        print(f"\033[1mRouting Analysis\033[0m: \"{task[:60]}{'...' if len(task) > 60 else ''}\"\n")
        print(f"  Simple message: {'Yes' if is_simple else 'No'}")
        print(f"  Complexity score: {score:.1f}/100")
        print(f"  Routing tier: \033[{'32' if tier.value == 'LOW' else '33' if tier.value == 'STANDARD' else '31'}m{tier.value}\033[0m")
        print()
        print("  Signals detected:")
        
        breakdown = confidence_breakdown(signals)
        
        def print_signal_group(title, signals_dict):
            """Print a labeled group of routing signals with their contribution score."""
            if signals_dict:
                print(f"    {title}:")
                for signal, score in signals_dict.items():
                    print(f"      - {signal}: {score:+.1f}")

        print_signal_group("Lexical Signals", breakdown["lexical_signals"])
        print_signal_group("Structural Signals", breakdown["structural_signals"])
        print_signal_group("Context Signals", breakdown["context_signals"])
        print(f"    Total score: {breakdown['total_score']:.1f}")

        # Show routing learning suggestion if available
        try:
            from mullet.core.router.learning import suggest_tier
            learned = suggest_tier(task)
            if learned:
                print(f"    ~ Learning suggests: {learned} (from past outcomes)")
        except Exception as _exc:
            logger.warning("cli/main: Failed to retrieve learning suggestion: %s", _exc)
        print()

        est = TokenUsage(input_tokens=2000, output_tokens=1000, total_tokens=3000)
        if tier.value == "LOW":
            print("  Provider: \033[32mLocal GPU (free)\033[0m")
            print("  Estimated cost: $0.00")
            savings = estimate_savings("local", "", est)
            print(f"  Savings vs premium: ${savings:.4f}")
        elif tier.value == "STANDARD":
            print("  Provider: Cheap cloud (Haiku/Flash)")
            cost = estimate_cost("openrouter", "haiku", est)
            print(f"  Estimated cost: ${cost:.4f}")
        else:
            print("  Provider: \033[31mPremium cloud (Sonnet/Opus)\033[0m")
            cost = estimate_cost("anthropic", "sonnet", est)
            print(f"  Estimated cost: ${cost:.4f}")

        # Cost comparison across tiers
        print()
        print("  Cost comparison (est. 3K tokens):")
        try:
            from mullet.core.providers.model_info import get_pricing
            tiers = [
                ("LOCAL", "local", "\033[32m"),
                ("STANDARD", "gemini-2.5-flash", "\033[33m"),
                ("ADVANCED", "claude-sonnet-4-6", "\033[31m"),
                ("CRITICAL", "claude-opus-4-8", "\033[35m"),
            ]
            for t_name, model, color in tiers:
                inp, out = get_pricing(model)
                cost = (2000/1000)*inp + (1000/1000)*out
                marker = " ← selected" if t_name == tier.value else ""
                print(f"    {color}{t_name:8s}\033[0m ${cost:.4f}{marker}")
        except Exception as _exc:
            logger.warning("cli/main: Failed to retrieve model pricing: %s", _exc)

        # Show project context if detected
        try:
            from mullet.core.agent.project_detect import detect_project
            project = detect_project()
            if project.language != "unknown":
                print()
                parts = [f"  Project: {project.language}"]
                if project.framework:
                    parts[0] += f" ({project.framework})"
                if project.test_command:
                    parts.append(f"  Test: {project.test_command}")
                print("\n".join(parts))
        except Exception as _exc:
            logger.warning("cli/main: Failed to detect project in explain: %s", _exc)
        return

    if command == "watch":
        if task in ("--help", "-h", "help"):
            print("Usage: mullet watch \"<task>\" [--glob '<pattern>'] [--interval N]")
            print("Watch files and re-run task on changes.")
            sys.exit(0)
        if not task:
            print("Usage: mullet watch \"your task here\"", file=sys.stderr)
            sys.exit(1)
        if _delegation_refused(args):
            sys.exit(2)
        from mullet.cli.watch import run_watch
        run_watch(task, args)
        return

    if command == "review":
        if task in ("--help", "-h", "help"):
            print("Usage: mullet review [extra-context]")
            print()
            print("Review the current git changes using the agent. Picks the first")
            print("non-empty source of changes in this order:")
            print("  1. staged changes (`git diff --staged`)")
            print("  2. unstaged changes (`git diff`)")
            print("  3. the most recent commit (`git log -1 -p`)")
            print()
            print("Any task text passed as args is prefixed to the review prompt")
            print("so you can guide the focus, e.g.:")
            print("  mullet review 'focus on security'")
            sys.exit(0)
        # Review recent git changes using the agent
        import subprocess as _sp
        diff_result = _sp.run(["git", "diff", "--staged"], capture_output=True, text=True, timeout=10)
        diff = diff_result.stdout.strip()
        if not diff:
            diff_result = _sp.run(["git", "diff"], capture_output=True, text=True, timeout=10)
            diff = diff_result.stdout.strip()
        if not diff:
            diff_result = _sp.run(["git", "log", "-1", "-p"], capture_output=True, text=True, timeout=10)
            diff = diff_result.stdout.strip()
        if not diff:
            print("No changes to review.", file=sys.stderr)
            sys.exit(0)
        # Truncate large diffs
        if len(diff) > 15000:
            diff = diff[:15000] + "\n\n[... truncated]"
        review_task = (
            f"Review this code change. Check for bugs, security issues, and improvements.\n\n"
            f"```diff\n{diff}\n```"
        )
        if task:
            review_task = f"{task}\n\n{review_task}"
        sys.exit(run_task(review_task, args))

    if command == "mcp":
        if task in ("--help", "-h", "help"):
            print("Usage: mullet mcp")
            print("Start MCP server for Claude Code integration (stdio transport).")
            sys.exit(0)
        from mullet.server.mcp import run_mcp_server
        run_mcp_server()
        return

    if command == "serve":
        if task in ("--help", "-h", "help"):
            print("Usage: mullet serve [--host HOST] [--port PORT] [--show-token]")
            print("Start the Mullet API server (REST + WebSocket). Default: 127.0.0.1:3000")
            print("Requests need this install's token (Authorization: Bearer <token>); --show-token prints it.")
            print("Settings: server.require_token, server.allowed_origins, server.allowed_hosts in the config file.")
            sys.exit(0)
        if task == "--show-token" or "--show-token" in task_parts:
            from mullet.server.guard import get_or_create_token, token_path
            print(f"Token file: {token_path()}")
            print(get_or_create_token())
            return
        host = "127.0.0.1"
        port = 3000
        # Parse serve-specific args from task_parts
        for i, arg in enumerate(task_parts):
            if arg == "--host" and i + 1 < len(task_parts):
                host = task_parts[i + 1]
            if arg == "--port" and i + 1 < len(task_parts):
                try:
                    port = int(task_parts[i + 1])
                except ValueError:
                    print("Error: --port requires an integer value", file=sys.stderr)
                    return
        from mullet.server.app import run_server
        print(f"Starting Mullet server on {host}:{port}")
        run_server(host=host, port=port)
        return

    if command == "login":
        if task in ("--help", "-h", "help"):
            print("Usage: mullet login")
            from mullet.core.paths import config_dir
            print(f"Authenticate with a Personal Access Token (stored at {config_dir() / 'credentials.json'}).")
            sys.exit(0)
        print("mullet login: Paste your Personal Access Token:")
        token = input("> ").strip()
        if not token.startswith("mllt_"):
            print("Warning: Token should start with 'mllt_'", file=sys.stderr)

        from mullet.core.paths import config_dir
        cred_dir = config_dir()
        os.makedirs(cred_dir, mode=0o700, exist_ok=True)
        import json
        cred_path = os.path.join(cred_dir, "credentials.json")
        fd = os.open(cred_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"token": token}, f)
        print(f"Token saved to {cred_path}")

        # Verify the token if a server is running
        try:
            from mullet.server.db import init_db
            from mullet.server.auth import verify_pat
            db = init_db()
            user = verify_pat(token, db)
            if user:
                print(f"Authenticated as: {user['email']} (role: {user['role']})")
            else:
                print("Warning: Token could not be verified against local database")
        except Exception:
            print("Token saved (verification requires a running server)")
        return

    if command == "ingest":
        if task in ("--help", "-h", "help"):
            print("Usage: mullet ingest [<directory>]")
            print("Ingest code files from a directory into the active pool. Defaults to cwd.")
            sys.exit(0)
        from pathlib import Path as _Path
        from mullet.knowledge.engine.client import Client
        target = _Path(task or ".").resolve()
        if not target.is_dir():
            print(f"Error: {target} is not a directory.", file=sys.stderr)
            sys.exit(1)
        print(f"Ingesting code from: {target}")
        try:
            # Use a minimal client config for code chunking
            client = Client(_Path("."), {"name": "cli-ingest", "machine_name": "cli"})
            chunks = client.get_code_chunks([target])
            print(f"Extracted {len(chunks)} code chunks from {target}")
            for chunk in chunks[:5]:
                meta = chunk.get("metadata", {})
                print(f"  {meta.get('symbol_type', '?'):10} {meta.get('symbol_name', '?'):30} {meta.get('source_file', '?')}")
            if len(chunks) > 5:
                print(f"  ... and {len(chunks) - 5} more")
        except Exception as e:
            print(f"Ingestion error: {e}", file=sys.stderr)
        return

    if command == "reindex":
        # Everything after "reindex" goes to the reindexer as given; a leading
        # bare word is the pool name (`mullet reindex mypool --graph`).
        raw_args = sys.argv[sys.argv.index("reindex") + 1:] if "reindex" in sys.argv else []
        if raw_args and not raw_args[0].startswith("-"):
            raw_args = ["--pool", *raw_args]
        sys.exit(_run_packaged("mullet.knowledge.reindex", raw_args))

    if command == "pool":
        sub = task_parts[0] if task_parts else "list"
        if sub in ("--help", "-h", "help") or (len(task_parts) > 1 and task_parts[1] in ("--help", "-h")):
            print("Usage: mullet pool [create|list|delete <pool>|docs <pool>|audit|health [pool]|prune [--yes]|tune [--pool NAME] [--dry-run]]")
            print("  create           Interactive wizard to create a new pool")
            print("  list             List all pools")
            print("  delete <pool>    Delete a pool and its index (prompts unless --yes)")
            print("  docs <pool>      Show indexed documents for a pool")
            print("  audit            Audit all pools for health and metadata")
            print("  health [pool|--all]  Quick health check (--all validates configs + exclude patterns)")
            print("  validate [pool|--all] Deep validation + actionable fix commands for every issue")
            print("  prune            Remove pools with no config.yaml")
            print("  tune             Tune pool retrieval settings")
            sys.exit(0)
        if sub == "create":
            from scripts.create_pool import create_pool
            create_pool()
        elif sub == "list":
            from mullet.server.mcp_knowledge import handle_list_pools
            print(handle_list_pools({}))
        elif sub == "docs":
            pool = task_parts[1] if len(task_parts) > 1 else None
            if not pool:
                print("Usage: mullet pool docs <pool_name>", file=sys.stderr)
                sys.exit(1)
            from mullet.knowledge.engine.client import load_client
            from mullet.server.mcp_knowledge import _get_pools_dir
            from mullet.knowledge.engine.ingest import build_file_manifest
            try:
                client = load_client(pool, clients_dir=_get_pools_dir())
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(2)
            extensions = client.get_document_extensions()
            exclude = client.documents_config.get("exclude_patterns", [])
            manifest = build_file_manifest(client.documents_path, extensions, exclude_patterns=exclude)
            print(f"**{pool}** — {len(manifest)} indexed documents:")
            for path in sorted(manifest.keys()):
                size = manifest[path]["size"]
                print(f"  {path} ({size:,} bytes)")
        elif sub == "audit":
            from mullet.cli.pool_audit import run_pool_audit
            from mullet.server.mcp_knowledge import _get_pools_dir
            run_pool_audit(_get_pools_dir())
        elif sub == "health":
            from mullet.cli.pool_audit import run_pool_health
            from mullet.server.mcp_knowledge import _get_pools_dir
            # --all: explicit all-pools flag (same as no filter)
            if "--all" in sys.argv:
                pool_filter = None
            else:
                pool_filter = task_parts[1] if len(task_parts) > 1 else None
            workers = 1
            if "--workers" in sys.argv:
                try:
                    idx = sys.argv.index("--workers")
                    workers = int(sys.argv[idx + 1])
                except (ValueError, IndexError):
                    pass
            rc = run_pool_health(_get_pools_dir(), pool_name=pool_filter, workers=workers)
            sys.exit(rc)
        elif sub == "validate":
            from mullet.server.mcp_knowledge import _get_pools_dir
            from mullet.cli.pool_audit import run_pool_validate
            args_rest = [a for a in sys.argv[sys.argv.index("validate") + 1:] if not a.startswith("-")]
            pool_filter = args_rest[0] if args_rest else None
            rc = run_pool_validate(_get_pools_dir(), pool_name=pool_filter)
            sys.exit(rc)
        elif sub == "delete":
            pool_name = task_parts[1] if len(task_parts) > 1 else None
            if not pool_name:
                print("Usage: mullet pool delete <pool_name> [--yes]", file=sys.stderr)
                sys.exit(1)
            from mullet.server.mcp_knowledge import handle_delete_pool, _get_pools_dir
            yes = "--yes" in sys.argv
            if not yes:
                confirm = input(f"Delete pool '{pool_name}' and all its data? [y/N] ").strip().lower()
                yes = confirm in ("y", "yes")
            if not yes:
                print("Aborted.")
                sys.exit(0)
            result = handle_delete_pool({"pool_name": pool_name, "confirm": True})
            print(result)
            if result.startswith("Error"):
                sys.exit(1)
        elif sub == "prune":
            from mullet.cli.pool_audit import run_pool_prune
            from mullet.server.mcp_knowledge import _get_pools_dir
            run_pool_prune(_get_pools_dir(), yes="--yes" in sys.argv)
        elif sub == "tune":
            from mullet.cli.pool_tune import run_pool_tune
            from mullet.server.mcp_knowledge import _get_pools_dir
            pool_filter = None
            if "--pool" in sys.argv:
                idx = sys.argv.index("--pool")
                pool_filter = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
            run_pool_tune(_get_pools_dir(), pool_filter=pool_filter, dry_run="--dry-run" in sys.argv)
        else:
            print("Usage: mullet pool [create|list|docs <pool>|audit|health [pool]|prune [--yes]|tune [--pool NAME] [--dry-run]]")
        return

    if command == "docs":
        from mullet.cli import docs_cmd
        sys.exit(docs_cmd.docs.main(args=task_parts, prog_name="mullet docs"))

    if command == "evaluate":
        pool = task_parts[0] if task_parts else None
        if not pool or pool in ("--help", "-h", "help"):
            print("Usage: mullet evaluate <pool_name> [--tags T] [--rag] [--compare A,B] [--json] [--verbose]")
            print("       mullet evaluate --list")
            print("Run quality evaluation against a pool using its eval dataset.")
            sys.exit(0 if pool in ("--help", "-h", "help") else 1)
        raw_args = sys.argv[sys.argv.index("evaluate") + 1:]
        if not raw_args[0].startswith("-"):
            raw_args = ["--client", *raw_args]
        sys.exit(_run_packaged("mullet.knowledge.evaluate", raw_args))

    if command == "search":
        if not task:
            print("Usage: mullet search <query>", file=sys.stderr)
            sys.exit(1)
        pool = None
        if "--pool" in sys.argv:
            idx = sys.argv.index("--pool")
            pool = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        # Strip --pool <value> from task_parts so the query isn't contaminated
        clean_parts = []
        skip_next = False
        for part in task_parts:
            if skip_next:
                skip_next = False
                continue
            if part == "--pool":
                skip_next = True
                continue
            clean_parts.append(part)
        query = " ".join(clean_parts)
        if not query:
            print("Usage: mullet search <query>", file=sys.stderr)
            sys.exit(1)
        from mullet.server.mcp_knowledge import handle_search
        print(handle_search({"query": query, "pool_name": pool, "top_k": 10}))
        return

    if command == "graph":
        sub = task.split()[0] if task else "stats"
        rest = task[len(sub):].strip() if task else ""
        if sub == "build":
            import time

            pool = None
            if "--pool" in sys.argv:
                idx = sys.argv.index("--pool")
                pool = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
            elif "-c" in sys.argv:
                idx = sys.argv.index("-c")
                pool = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
            if not pool:
                print("Usage: mullet graph build --pool <name>", file=sys.stderr)
                sys.exit(1)

            from mullet.knowledge.engine.client import load_client
            from mullet.knowledge.graph.store import KnowledgeGraph
            from mullet.server.mcp_knowledge import handle_build_graph, _jobs

            client = load_client(pool)
            result = handle_build_graph({"pool_name": pool, "rebuild": False})
            # Parse Job ID from result string
            job_id = None
            if "Job ID: `" in result:
                job_id = result.split("Job ID: `")[1].split("`")[0]
            if not job_id:
                print(result)
                sys.exit(1)

            start_time = time.time()
            last_entities = -1
            last_rels = -1
            while True:
                time.sleep(5)
                job = _jobs.get(job_id)
                if not job:
                    print("\nJob lost.")
                    sys.exit(1)
                status = job.get("status", "unknown")
                if status == "failed":
                    print(f"\nGraph build failed: {job.get('error', 'unknown error')}")
                    sys.exit(1)
                if status == "completed":
                    elapsed = int(time.time() - start_time)
                    kg = KnowledgeGraph(client.index_path)
                    kg.load()
                    s = kg.stats
                    entities = s.get("entity_count", 0)
                    rels = s.get("relationship_count", 0)
                    kg.close()
                    print(f"\r  {pool:20s} building graph  entities={entities} rel={rels}  [{elapsed}s]\n")
                    sys.exit(0)

                kg = KnowledgeGraph(client.index_path)
                kg.load()
                s = kg.stats
                entities = s.get("entity_count", 0)
                rels = s.get("relationship_count", 0)
                kg.close()
                if entities != last_entities or rels != last_rels:
                    last_entities = entities
                    last_rels = rels
                    elapsed = int(time.time() - start_time)
                    print(f"\r  {pool:20s} building graph  entities={entities} rel={rels}  [{elapsed}s]", end="", flush=True)
        elif sub == "search" and rest:
            print(f"Graph search for: {rest}")
            print("Note: Requires a built knowledge graph. Run 'mullet graph build' first.")
        elif sub == "stats":
            try:
                from mullet.knowledge.graph.store import KnowledgeGraph
                kg = KnowledgeGraph(".")
                kg.load()
                s = kg.stats
                print(f"Entities: {s.get('entity_count', 0)}, Relationships: {s.get('relationship_count', 0)}")
                kg.close()
            except Exception:
                print("No knowledge graph found. Run 'mullet graph build' to create one.")
        else:
            print("Usage: mullet graph [build|search <query>|stats]")
        return

    if command == "source":
        sub = task.split()[0] if task else "list"
        rest = task[len(sub):].strip() if task else ""
        if sub == "add" and rest.startswith("github"):
            repo = rest.replace("github", "").strip()
            print("Add this to your config.yaml sources section:")
            print("  sources:")
            print("    github:")
            print(f"      - owner: {repo.split('/')[0] if '/' in repo else repo}")
            if "/" in repo:
                print(f"        repo: {repo.split('/')[1]}")
        elif sub == "add" and rest.startswith("web"):
            url = rest.replace("web", "").strip()
            print("Add this to your config.yaml sources section:")
            print("  sources:")
            print("    web:")
            print(f"      - url: {url}")
            print("        max_pages: 500")
            print("        sync_on_ingest: true      # re-crawl on every reindex")
            print("        # use_sitemap: true       # seed from /sitemap.xml")
            print("        # download_types: [pdf, docx]  # index linked documents")
        elif sub == "list":
            print("Sources are configured in config.yaml. Example:")
            print("  sources:")
            print("    github:")
            print("      - owner: anthropics")
            print("        repo: claude-code")
            print("    web:")
            print("      - url: https://docs.example.com")
        else:
            print("Usage: mullet source [add github <owner/repo>|add web <url>|list]")
        return

    if command == "schedule":
        if task in ("--help", "-h", "help"):
            print("Usage: mullet schedule {list|add <name> <interval> <command>|remove <name>}")
            print("  list                                   List scheduled tasks (default)")
            print("  add <name> <interval> <command>        Add a recurring task")
            print("  remove <name>                          Remove a scheduled task by name")
            print()
            print("Examples:")
            print("  mullet schedule add nightly-reindex '0 3 * * *' 'mullet reindex --pool dev'")
            print("  mullet schedule list")
            print("  mullet schedule remove nightly-reindex")
            return
        from mullet.core.orchestration.scheduler import TaskScheduler
        sched = TaskScheduler()
        sub = task.split()[0] if task else "list"
        rest = task[len(sub):].strip() if task else ""
        if sub == "add" and rest:
            parts = rest.split(None, 2)
            if len(parts) >= 3:
                sched.add(parts[0], parts[1], parts[2])
                print(f"Scheduled: {parts[0]} every {parts[1]}")
            else:
                print("Usage: mullet schedule add <name> <interval> <command>")
        elif sub == "remove" and rest:
            sched.remove(rest.strip())
            print(f"Removed: {rest.strip()}")
        else:
            for t in sched.list_tasks():
                status = "paused" if not t.enabled else f"next: {t.next_run or 'now'}"
                print(f"  {t.name}  [{t.schedule}]  runs: {t.run_count}  {status}")
            if not sched.list_tasks():
                print("No scheduled tasks. Use: mullet schedule add <name> <interval> <command>")
        return

    if command == "research":
        from mullet.core.orchestration.autoresearch import ResearchPipeline
        if not task or task in ("--help", "-h", "help"):
            print("Usage: mullet research <question>")
            print("Deep research mode: iterates with web + knowledge sources until confident.")
            sys.exit(0 if task in ("--help", "-h", "help") else 1)
        pipeline = ResearchPipeline(task)
        print(f"Research pipeline created for: {task}")
        print(f"Max iterations: {pipeline.max_iterations}, min score: {pipeline.min_score}")
        print("Use the agent to add research iterations.")
        return

    if command == "hud":
        from mullet.cli.hud import cmd_hud
        sys.exit(cmd_hud(task_parts))

    if command == "hud-claude":
        from mullet.cli.hud import cmd_hud_claude
        sys.exit(cmd_hud_claude(task_parts))

    # Handle piped stdin
    if not sys.stdin.isatty():
        stdin_content = sys.stdin.read().strip()
        if stdin_content:
            if task:
                # Task from args + content from stdin = combine
                task = f"{task}\n\n--- Piped content ---\n{stdin_content}"
            else:
                # No task arg — use stdin as the task
                task = stdin_content

    # Default: direct task or REPL
    if task:
        sys.exit(run_task(task, args))

    # No task — REPL mode
    if _delegation_refused(args):
        sys.exit(2)
    provider = resolve_provider(args, allowed_providers=_parse_providers_allowlist(args))
    if not provider:
        print("Mullet — Cloud in the front, Local GPU in the back.\n")
        print("No LLM provider found. To get started:\n")
        print("  1. Start LM Studio or Ollama for local inference")
        print("  2. Or set ANTHROPIC_API_KEY for Claude")
        print("  3. Or set GOOGLE_API_KEY for Gemini")
        print("  4. Or use --base-url http://your-endpoint/v1\n")
        print("Run 'mullet doctor' to check system status.")
        print("Run 'mullet --help' for all options.")
        sys.exit(2)

    from mullet.cli.repl import run_repl
    registry = build_tool_registry()

    from mullet.core.agent.prompts import build_base_prompt
    model_name = getattr(provider, "model", "")
    system_prompt = build_base_prompt(model=model_name)
    if args.client:
        try:
            from mullet.core.knowledge.manager import KnowledgeManager
            from mullet.core.knowledge import tools as knowledge_tools
            # REPL keeps the KM alive for the session — knowledge_session would
            # close it as soon as we exit this block.
            km = KnowledgeManager(args.client)
            knowledge_tools.register(registry, km)
            system_prompt += f"\n\nYou are working on project: {args.client}"
        except Exception as e:
            print(f"Warning: Could not load client '{args.client}': {e}", file=sys.stderr)

    run_repl(
        provider=provider,
        tool_registry=registry,
        toolsets=["terminal", "file", "web"] + (["trove"] if args.client else []),
        system_prompt=system_prompt,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        allowed_providers=_parse_providers_allowlist(args),
    )


if __name__ == "__main__":
    main()
