"""Model metadata — context windows, pricing, and capabilities.

Provides a lookup table for common models across providers.
Used by the router for token budget estimation and the cost tracker
for accurate cost calculations.

Pricing is per 1K tokens, updated to reflect current rates.

Capability flags (supports_tools / supports_vision / supports_streaming)
are sourced from ModelConstraints — see ``mullet.core.providers.constraints``.
ModelInfo exposes back-compat properties that route to the constraint
registry so callers keep working, but the single source of truth lives in
ModelConstraints. If a flag is missing here, add it to STATIC_CONSTRAINTS.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Metadata for a single model.

    Capability flags delegate to ``ConstraintRegistry`` — never store a local
    override here (two sources of truth → silent drift).
    """

    context_length: int
    input_cost_per_1k: float  # $ per 1K input tokens
    output_cost_per_1k: float  # $ per 1K output tokens
    # model_id is filled in by get_model_info after lookup so property
    # accessors below can consult ConstraintRegistry. When set to "" we
    # fall back to the permissive default.
    model_id: str = ""

    @property
    def supports_tools(self) -> bool:
        """Back-compat: route to ConstraintRegistry.get(model_id).supports_tools."""
        from mullet.core.providers.constraints import get_constraint_registry
        return get_constraint_registry().get(self.model_id).supports_tools

    @property
    def supports_vision(self) -> bool:
        """Back-compat: route to ConstraintRegistry.get(model_id).supports_vision."""
        from mullet.core.providers.constraints import get_constraint_registry
        return get_constraint_registry().get(self.model_id).supports_vision

    @property
    def supports_streaming(self) -> bool:
        """Back-compat: route to ConstraintRegistry.get(model_id).supports_streaming."""
        from mullet.core.providers.constraints import get_constraint_registry
        return get_constraint_registry().get(self.model_id).supports_streaming


# Comprehensive model database — keyed by common model name/ID
# Pricing as of mid-2025. Update periodically.
#
# Capability flags live in ``mullet.core.providers.constraints``. Do NOT pass
# supports_* kwargs here — use STATIC_CONSTRAINTS in constraints.py.
MODEL_DATABASE: dict[str, ModelInfo] = {
    # Anthropic (2026 pricing per 1K tokens)
    "claude-opus-4-7": ModelInfo(1_000_000, 0.005, 0.025),
    "claude-opus-4-8": ModelInfo(1_000_000, 0.005, 0.025),   # released 2026-05-28; Fable 5 safety fallback
    "claude-fable-5": ModelInfo(1_000_000, 0.010, 0.050),    # Mythos-class tier, 128K max output, released 2026-06-09
    "claude-opus-4-6": ModelInfo(1_000_000, 0.005, 0.025),
    "claude-sonnet-4-6": ModelInfo(1_000_000, 0.003, 0.015),
    "claude-sonnet-5": ModelInfo(1_000_000, 0.002, 0.01),
    "claude-haiku-4-5": ModelInfo(200_000, 0.001, 0.005),
    "claude-haiku-4-5-20251001": ModelInfo(200_000, 0.001, 0.005),
    "claude-sonnet-4-5": ModelInfo(1_000_000, 0.003, 0.015),

    # OpenAI
    "gpt-5": ModelInfo(400_000, 0.00125, 0.01),    # $1.25/$10 per M — reasoning-by-default
    "gpt-5-mini": ModelInfo(400_000, 0.00025, 0.002), # $0.25/$2 per M (estimate)
    "gpt-4o": ModelInfo(128_000, 0.0025, 0.010),
    "gpt-4o-mini": ModelInfo(128_000, 0.00015, 0.0006),
    "gpt-4-turbo": ModelInfo(128_000, 0.01, 0.03),
    "o1": ModelInfo(200_000, 0.015, 0.060),
    "o1-mini": ModelInfo(128_000, 0.003, 0.012),
    "o3-mini": ModelInfo(200_000, 0.0011, 0.0044),

    # Google (2026 pricing per 1K tokens)
    "gemini-2.5-pro": ModelInfo(1_000_000, 0.00125, 0.01),
    "gemini-2.5-flash": ModelInfo(1_000_000, 0.00015, 0.00035),
    "gemini-2.5-flash-lite": ModelInfo(1_000_000, 0.0001, 0.0004),
    "gemini-3.5-flash": ModelInfo(1_048_576, 0.0015, 0.009),
    "gemini-3.1-pro": ModelInfo(2_000_000, 0.002, 0.012),
    "gemini-2.0-flash": ModelInfo(1_000_000, 0.0001, 0.0004),

    # Meta (via OpenRouter or local)
    "llama-3.3-70b": ModelInfo(128_000, 0.0003, 0.0003),
    "llama-3.1-8b": ModelInfo(128_000, 0.00005, 0.00005),
    "llama-4-scout": ModelInfo(512_000, 0.00015, 0.0006),
    "llama-4-maverick": ModelInfo(1_000_000, 0.0003, 0.0012),

    # Mistral
    "mistral-large": ModelInfo(128_000, 0.002, 0.006),
    "mistral-small": ModelInfo(128_000, 0.0002, 0.0006),
    "mistral-nemo": ModelInfo(128_000, 0.0, 0.0),  # Local model
    "codestral": ModelInfo(256_000, 0.0003, 0.0009),

    # DeepSeek
    "deepseek-v3": ModelInfo(128_000, 0.00027, 0.0011),

    # Kimi (Moonshot AI) — pricing per 1K tokens.
    # Ref: https://platform.kimi.ai/docs/models
    # K2.6 / K2.5 are the current generation (native multimodal, 256K).
    # NOTE: 0711-preview is 128K context (NOT 256K like the rest of the K2 family).
    "kimi-k2.6": ModelInfo(262_144, 0.0006, 0.0025),
    "kimi-k2.5": ModelInfo(262_144, 0.0006, 0.003),
    "kimi-k2.7-code": ModelInfo(262_144, 0.00095, 0.004),    # released 2026-06-12; code-first, ~30% fewer reasoning tokens; thinking always on
    "kimi-k2.7-code-highspeed": ModelInfo(262_144, 0.0019, 0.008),  # 5-6x faster output, ~2x price; thinking always on
    "kimi-k2-0711-preview": ModelInfo(131_072, 0.0006, 0.0025),  # deprecated 2026-05-25
    "kimi-k2-0905-preview": ModelInfo(262_144, 0.0006, 0.0025),  # deprecated 2026-05-25
    "kimi-k2-turbo-preview": ModelInfo(262_144, 0.00115, 0.008),  # deprecated 2026-05-25
    "kimi-k2-thinking": ModelInfo(262_144, 0.0006, 0.0025),       # deprecated 2026-05-25
    "kimi-k2-thinking-turbo": ModelInfo(262_144, 0.00115, 0.008),  # deprecated 2026-05-25
    # Deprecated Moonshot models — kept for alias resolution only.
    # kimi-latest discontinued 2026-01-28; kimi-thinking-preview discontinued 2025-11-11.
    "kimi-latest": ModelInfo(256_000, 0.0006, 0.0025),

    # ── Alibaba Cloud / DashScope / Qwen ──────────────────────────────────────
    # Ref: https://www.alibabacloud.com/en/product/tongyi-qianwen (DashScope pricing)
    # All models are OpenAI-compatible via dashscope.aliyuncs.com/compatible-mode/v1
    "qwen-turbo":            ModelInfo(1_000_000, 0.0002,  0.0006),   # cheapest, fastest
    "qwen-plus":             ModelInfo(  131_072, 0.0004,  0.0012),   # balanced general use
    "qwen-max":              ModelInfo(   32_768, 0.0016,  0.0064),   # highest quality
    "qwen-long":             ModelInfo(10_000_000, 0.00005, 0.0002),  # ultra-cheap, huge ctx
    "qwen2.5-72b-instruct":  ModelInfo(  131_072, 0.0005,  0.002),    # open-weight large
    "qwen2.5-32b-instruct":  ModelInfo(  131_072, 0.0003,  0.0012),   # open-weight mid
    "qwen2.5-coder-32b-instruct": ModelInfo(131_072, 0.0003, 0.0012), # code-specialised
    "qwq-32b":               ModelInfo(  131_072, 0.0006,  0.0024),   # reasoning/thinking

    # ── OpenRouter model IDs (provider/model format) ─────────────────────────
    # Pricing per 1K tokens (USD/1K) — multiply by 1000 to compare to $/M.
    # Verified against OpenRouter live API 2026-05-18. Sorted by tier.

    # Frontier (max quality, max cost)
    "anthropic/claude-opus-4.1":             ModelInfo(  200_000, 0.015, 0.075),   # $15/$75 per M — premier synthesis
    "openai/gpt-5":                          ModelInfo(  400_000, 0.00125, 0.01),  # $1.25/$10 — reasoning-by-default
    "anthropic/claude-sonnet-4.5":           ModelInfo(1_000_000, 0.003, 0.015),   # $3/$15 — 1M ctx, balanced
    "google/gemini-2.5-pro":                 ModelInfo(1_048_576, 0.00125, 0.01),  # $1.25/$10 — 1M ctx, reasoning
    "x-ai/grok-4.20":                        ModelInfo(2_000_000, 0.00125, 0.0025),# $1.25/$2.50 — 2M ctx
    "google/gemini-2.5-flash":               ModelInfo(1_000_000, 0.0003, 0.0025), # $0.30/$2.50 — fast, no reasoning by default

    # Strong cheap coders — Chinese open-weights
    "deepseek/deepseek-v3.2-exp":            ModelInfo(  163_840, 0.00027, 0.00041),  # $0.27/$0.41 — was best $/quality (superseded by V4 Flash)
    "deepseek/deepseek-v4-flash":            ModelInfo(1_000_000, 0.000098, 0.000196), # $0.098/$0.196 — 1M ctx, new cheap workhorse
    "deepseek/deepseek-v4-pro":              ModelInfo(1_000_000, 0.000435, 0.00087),  # $0.435/$0.87 — 1M ctx, advanced MoE
    "z-ai/glm-5.1":                          ModelInfo(  202_752, 0.00098, 0.00308),   # $0.98/$3.08 — strong coder
    "minimax/minimax-m3":                    ModelInfo(1_048_576, 0.00030, 0.00120),   # $0.30/$1.20 — 1M ctx multimodal (launch promo)
    "qwen/qwen3-coder-next":                 ModelInfo(  203_000, 0.00100, 0.00320),   # $1.00/$3.20 — 80B/3B MoE coder
    "qwen/qwen3-coder":                      ModelInfo(1_048_576, 0.00022, 0.0018),   # $0.22/$1.80 — 1M ctx, code-tuned
    "z-ai/glm-4.6":                          ModelInfo(  202_752, 0.00043, 0.00174),  # $0.43/$1.74 — strong general

    # Deprecated but kept for callers — superseded by deepseek-v3.2-exp.
    # Pricing refreshed to current OpenRouter live values 2026-05-19 so the
    # sync_openrouter_pricing.py check reports clean.
    "deepseek/deepseek-chat":                ModelInfo(163_840, 0.00032, 0.00089),  # DeepSeek V3 (legacy, deprecated by v3.2-exp)
    "meta-llama/llama-4-scout":              ModelInfo(512_000, 0.00008, 0.00030),  # 512K ctx
    "meta-llama/llama-4-maverick":           ModelInfo(1_000_000, 0.00015, 0.00060),
    "qwen/qwen-2.5-coder-32b-instruct":      ModelInfo(131_072, 0.00066, 0.00100),
    "qwen/qwen3-30b-a3b":                    ModelInfo(131_072, 0.00009, 0.00045),
    "mistralai/devstral-small":              ModelInfo(131_072, 0.00010, 0.00030),
    "x-ai/grok-3-mini-beta":                 ModelInfo(131_072, 0.00030, 0.00050),  # (not on OR — direct only)

    # Google Gemma (local/API)
    "gemma-4-27b": ModelInfo(128_000, 0.0, 0.0),
    "gemma-4-31b": ModelInfo(128_000, 0.0, 0.0),
    "gemma-3-27b": ModelInfo(128_000, 0.0, 0.0),
    "gemma-2-27b": ModelInfo(8_192, 0.0, 0.0),

    # Qwen local models
    "qwen2.5-coder-3b": ModelInfo(128_000, 0.0, 0.0),
    "qwen2.5-coder-1.5b": ModelInfo(128_000, 0.0, 0.0),
    "qwen3.5-9b": ModelInfo(128_000, 0.0, 0.0),
    "qwen3.5": ModelInfo(128_000, 0.0, 0.0),
    "qwen3-coder-30b-a3b": ModelInfo(262_144, 0.0, 0.0),       # MoE coder, downloaded 2026-06-13
    "qwen3-4b-thinking-2507": ModelInfo(262_144, 0.0, 0.0),    # tiny reasoning, VRAM-resident
    "qwen3-vl-8b-instruct": ModelInfo(262_144, 0.0, 0.0),      # vision, VRAM-resident

    # Current LM Studio set. All three support 262K, but Mullet loads them at
    # 32K (lmstudio_manager.LOAD_CONTEXT_LENGTH), so 32K is the usable window.
    "ornith-1.5-35b-a3b": ModelInfo(32_768, 0.0, 0.0),         # MoE reasoning model, first local preference
    "qwen3.6-35b-a3b": ModelInfo(32_768, 0.0, 0.0),            # MoE general
    "ornith-1.5-9b": ModelInfo(32_768, 0.0, 0.0),              # dense reasoning model

    # Microsoft Phi local models
    "phi-4-mini": ModelInfo(128_000, 0.0, 0.0),
    "phi-4": ModelInfo(128_000, 0.0, 0.0),
    "phi-3": ModelInfo(128_000, 0.0, 0.0),



    # Local defaults (free, various context sizes)
    "default": ModelInfo(32_000, 0.0, 0.0),
    "local": ModelInfo(32_000, 0.0, 0.0),
}

# Aliases for common model name patterns
_ALIASES = {
    "sonnet": "claude-sonnet-5",
    "sonnet-5": "claude-sonnet-5",
    "opus": "claude-opus-4-8",   # plain "opus" → current best
    "opus-4-8": "claude-opus-4-8",
    "opus-4-7": "claude-opus-4-7",
    "opus-4-6": "claude-opus-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "claude-3-5-sonnet": "claude-sonnet-4-5",
    "claude-3-opus": "claude-opus-4-6",
    "claude-opus-4-7": "claude-opus-4-7",
    "fable": "claude-fable-5",
    "fable-5": "claude-fable-5",
    "gpt4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "flash": "gemini-2.5-flash",
    "gemini-flash": "gemini-2.5-flash",
    "gemini-pro": "gemini-3.1-pro",
    "llama-70b": "llama-3.3-70b",
    "llama-8b": "llama-3.1-8b",
    "deepseek-chat": "deepseek-v3",

    "kimi": "kimi-k2.7-code",   # plain "kimi" → current best
    "kimi-k2": "kimi-k2.7-code",  # generic K2 family → current best
    "kimi-2.5": "kimi-k2.5",
    "kimi-2.6": "kimi-k2.6",
    "kimi-k2.5": "kimi-k2.5",
    "kimi-k2.6": "kimi-k2.6",
    "k2.7": "kimi-k2.7-code",
    "kimi-k2.7-speed": "kimi-k2.7-code-highspeed",
    "k2.7-speed": "kimi-k2.7-code-highspeed",
    "kimi-k2.7-code-highspeed": "kimi-k2.7-code-highspeed",
    "kimi-k2-0905": "kimi-k2-0905-preview",
    "kimi-k2-0711": "kimi-k2-0711-preview",
    "kimi-k2-turbo": "kimi-k2-turbo-preview",  # TODO: verify kimi-k2-turbo (non-preview) exists in Moonshot API
    "kimi-k2-thinking": "kimi-k2-thinking",
    "kimi-thinking": "kimi-k2-thinking",
    "kimi-thinking-turbo": "kimi-k2-thinking-turbo",
    "qwen":          "qwen-plus",
    "qwen-coder":    "qwen2.5-coder-32b-instruct",
    "qwen-think":    "qwq-32b",
    "qwq":           "qwq-32b",
}


def _with_id(info: ModelInfo, model_id: str) -> ModelInfo:
    """Return a copy of ``info`` stamped with ``model_id`` so back-compat
    properties can consult the constraint registry.

    Shallow-copy via dataclasses.replace keeps MODEL_DATABASE entries pristine
    (the DB is shared module-level state; mutating in place would leak across
    lookups).
    """
    from dataclasses import replace
    if info.model_id == model_id:
        return info
    return replace(info, model_id=model_id)


def get_model_info(model: str) -> ModelInfo:
    """Look up model metadata by name.

    Handles exact matches, aliases, and fuzzy substring matching.
    Returns a default for unknown models. The returned ModelInfo is stamped
    with the resolved model_id so capability-flag back-compat properties
    can defer to ConstraintRegistry.
    """
    if not model:
        return _with_id(MODEL_DATABASE["default"], "default")

    model_lower = model.lower().strip()

    # Exact match
    if model_lower in MODEL_DATABASE:
        return _with_id(MODEL_DATABASE[model_lower], model_lower)

    # Alias match
    alias = _ALIASES.get(model_lower)
    if isinstance(alias, ModelInfo):
        return alias
    if isinstance(alias, str) and alias in MODEL_DATABASE:
        return _with_id(MODEL_DATABASE[alias], alias)

    # Fuzzy: check if model name contains a known key
    for key, info in MODEL_DATABASE.items():
        if key in model_lower or model_lower in key:
            return _with_id(info, key)

    # Strip provider prefix (e.g., "anthropic/claude-sonnet-4-6", "openai/gpt-4o")
    # Common patterns: "provider/model", "provider.model", "model:tag"
    stripped = _strip_provider_prefix(model_lower)
    if stripped != model_lower:
        return get_model_info(stripped)

    return _with_id(MODEL_DATABASE["default"], model_lower)


def _strip_provider_prefix(model: str) -> str:
    """Strip common provider prefixes from model names.

    Handles: anthropic/claude-..., google/gemini-..., openai/gpt-...,
    meta-llama/..., mistralai/..., deepseek/..., etc.
    Also strips version suffixes like -20250514 and Ollama tags like :latest.
    """
    import re

    # Provider prefix (before /)
    if "/" in model:
        parts = model.split("/", 1)
        model = parts[-1]  # Take everything after provider

    # Ollama tag (after :)
    if ":" in model:
        model = model.split(":")[0]

    # Version date suffix (e.g., -20250514, -v1, -v2.1)
    model = re.sub(r'-\d{8}$', '', model)
    model = re.sub(r'-v\d+(\.\d+)?$', '', model)

    return model


def get_context_length(model: str) -> int:
    """Get context window size for a model."""
    return get_model_info(model).context_length


def get_pricing(model: str) -> tuple[float, float]:
    """Get (input_cost_per_1k, output_cost_per_1k) for a model."""
    info = get_model_info(model)
    return info.input_cost_per_1k, info.output_cost_per_1k


def _apply_pricing_from_yaml() -> None:
    """Overwrite MODEL_DATABASE cost fields from pricing.yaml (the SSOT).

    Runs once at module load. Free/local models (cost == 0.0) are left untouched.
    Models present in MODEL_DATABASE but absent from pricing.yaml keep their
    MODEL_DATABASE values until pricing.yaml is updated.
    """
    try:
        from mullet.core.providers.pricing import load_pricing
        yaml_pricing = load_pricing()
    except Exception as exc:
        logger.warning("pricing.yaml override failed — MODEL_DATABASE costs unchanged: %s", exc)
        return
    for model_id, info in MODEL_DATABASE.items():
        if model_id in yaml_pricing:
            entry = yaml_pricing[model_id]
            info.input_cost_per_1k = entry["input_per_1k"]
            info.output_cost_per_1k = entry["output_per_1k"]

_apply_pricing_from_yaml()
