"""Unified model constraint layer.

Replaces scattered ad-hoc per-model API parameter patches with a central
registry. shape_request() applies constraints before every wire call;
parse_constraint_error() enables self-healing on novel rejections.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LEARNED_CONSTRAINTS_PATH = Path.home() / ".mullet" / "learned_constraints.json"
LEARNED_TTL_DAYS = 30


@dataclass
class ModelConstraints:
    """Per-model API parameter constraints."""

    omit_params: frozenset[str] = field(default_factory=frozenset)
    rename_params: dict[str, str] = field(default_factory=dict)
    max_context: int = 1_000_000
    max_output_tokens: int | None = None
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_json_mode: bool = True
    supports_system_prompt: bool = True
    supports_vision: bool = False
    temperature_range: tuple[float, float] | None = (0.0, 2.0)
    rate_limit_rpm: int | None = None
    thinking_required: bool = False
    notes: str = ""

    def can_serve(self, needs: dict) -> bool:
        """Return True if this model can handle the described request.

        needs keys:
          context_tokens (int): estimated input token count
          requires_tools (bool): task needs tool calling
          requires_streaming (bool): task needs streaming
        """
        if needs.get("context_tokens", 0) > self.max_context:
            return False
        if needs.get("requires_tools") and not self.supports_tools:
            return False
        if needs.get("requires_streaming") and not self.supports_streaming:
            return False
        return True


# ---------------------------------------------------------------------------
# Static constraint table
# ---------------------------------------------------------------------------

STATIC_CONSTRAINTS: dict[str, ModelConstraints] = {
    # Kimi K2.6 / K2.5 — latest multimodal (text+image+video), 256k context.
    # Ref: https://platform.kimi.ai/docs/pricing/chat-k26
    "kimi-k2.6": ModelConstraints(
        omit_params=frozenset(),
        max_context=262_144,
        max_output_tokens=262_144,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=True,
        temperature_range=(0.0, 2.0),
        notes="Kimi K2.6: native multimodal, thinking+non-thinking, default max_tokens=32768, temperature=1.0 recommended",
    ),
    "kimi-k2.5": ModelConstraints(
        omit_params=frozenset(),
        max_context=262_144,
        max_output_tokens=262_144,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=True,
        temperature_range=(0.0, 2.0),
        notes="Kimi K2.5: native multimodal, thinking+non-thinking, temperature=1.0 recommended",
    ),
    "kimi-k2.7-code": ModelConstraints(
        omit_params=frozenset(),
        max_context=262_144,
        max_output_tokens=262_144,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=True,
        temperature_range=(0.0, 2.0),
        thinking_required=True,
        notes="Kimi K2.7 Code (released 2026-06-12): code-first, native multimodal, ~30% fewer reasoning tokens; thinking ALWAYS ON — disabling it via API returns an error",
    ),
    "kimi-k2.7-code-highspeed": ModelConstraints(
        omit_params=frozenset(),
        max_context=262_144,
        max_output_tokens=262_144,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=True,
        temperature_range=(0.0, 2.0),
        thinking_required=True,
        notes="Kimi K2.7 Code HighSpeed: 5-6x faster output, ~2x price; thinking ALWAYS ON",
    ),
    # Kimi K2 preview series — scheduled for discontinuation 2026-05-25. Use
    # kimi-k2.6 for new work. Ref: https://platform.kimi.ai/docs/models
    # "Does not support vision functionality" — explicit in pricing/chat-k2.
    "kimi-k2-0711-preview": ModelConstraints(
        omit_params=frozenset(),
        max_context=131_072,
        max_output_tokens=131_072,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=False,
        temperature_range=(0.0, 2.0),
        notes="Kimi K2 0711 preview: 128K context (NOT 256K); discontinuing 2026-05-25",
    ),
    "kimi-k2-0905-preview": ModelConstraints(
        omit_params=frozenset(),
        max_context=262_144,
        max_output_tokens=262_144,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=False,
        temperature_range=(0.0, 2.0),
        notes="Kimi K2 0905 preview: enhanced agentic coding; discontinuing 2026-05-25",
    ),
    "kimi-k2-turbo-preview": ModelConstraints(
        omit_params=frozenset(),
        max_context=262_144,
        max_output_tokens=262_144,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=False,
        temperature_range=(0.0, 2.0),
        notes="Kimi K2 turbo preview: 60-100 tok/s, aligned with K2-0905; discontinuing 2026-05-25",
    ),
    "kimi-k2-thinking": ModelConstraints(
        omit_params=frozenset(),
        max_context=262_144,
        max_output_tokens=262_144,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=False,
        temperature_range=(0.0, 2.0),
        notes="Kimi K2 thinking: reasoning model (thinking always on); discontinuing 2026-05-25",
    ),
    "kimi-k2-thinking-turbo": ModelConstraints(
        omit_params=frozenset(),
        max_context=262_144,
        max_output_tokens=262_144,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=False,
        temperature_range=(0.0, 2.0),
        notes="Kimi K2 thinking turbo: reasoning+high-speed; discontinuing 2026-05-25",
    ),
    # Moonshot v1 legacy models — deprecated; kept so constraint checks reject
    # them for large prompts rather than silently accepting via permissive default.
    # temperature must be omitted (API rejects it); tools not supported.
    "moonshot-v1-8k": ModelConstraints(
        omit_params=frozenset({"temperature"}),
        max_context=8_192,
        supports_tools=False,
        supports_streaming=True,
        notes="Moonshot v1 8k (deprecated) — 8192 token context, no tools",
    ),
    "moonshot-v1-32k": ModelConstraints(
        omit_params=frozenset({"temperature"}),
        max_context=32_768,
        supports_tools=False,
        supports_streaming=True,
        notes="Moonshot v1 32k (deprecated) — 32768 token context, no tools",
    ),
    "moonshot-v1-128k": ModelConstraints(
        omit_params=frozenset({"temperature"}),
        max_context=131_072,
        supports_tools=False,
        supports_streaming=True,
        notes="Moonshot v1 128k (deprecated) — 131072 token context, no tools",
    ),
    # Anthropic models
    "claude-sonnet-4-6": ModelConstraints(
        omit_params=frozenset(),
        max_context=1_000_000,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=True,
        temperature_range=(0.0, 1.0),
        notes="Claude Sonnet 4.6",
    ),
    "claude-sonnet-5": ModelConstraints(
        omit_params=frozenset(),
        max_context=1_000_000,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=True,
        temperature_range=(0.0, 1.0),
        notes="Claude Sonnet 5",
    ),
    "claude-haiku-4-5-20251001": ModelConstraints(
        omit_params=frozenset(),
        max_context=200_000,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=True,
        temperature_range=(0.0, 1.0),
        notes="Claude Haiku 4.5",
    ),
    "claude-opus-4-7": ModelConstraints(
        omit_params=frozenset({"temperature"}),
        max_context=1_000_000,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=True,
        temperature_range=None,
        notes="Claude Opus 4.7: rejects the 'temperature' param (400 invalid_request_error) — must be omitted",
    ),
    "claude-opus-4-8": ModelConstraints(
        omit_params=frozenset({"temperature"}),
        max_context=1_000_000,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=True,
        temperature_range=None,
        notes="Claude Opus 4.8 (released 2026-05-28): omits 'temperature' like Opus 4.7; serves as Fable 5 safety fallback",
    ),
    "claude-fable-5": ModelConstraints(
        omit_params=frozenset({"temperature"}),
        max_context=1_000_000,
        max_output_tokens=128_000,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=True,
        temperature_range=None,
        notes="Claude Fable 5 (Mythos-class, released 2026-06-09): 1M ctx, 128K max output; omits 'temperature'",
    ),
    # Google Gemini
    "gemini-2.5-flash": ModelConstraints(
        omit_params=frozenset(),
        max_context=1_000_000,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=True,
        temperature_range=(0.0, 2.0),
        notes="Gemini 2.5 Flash",
    ),
    "gemini-2.5-pro": ModelConstraints(
        omit_params=frozenset(),
        max_context=1_000_000,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=True,
        temperature_range=(0.0, 2.0),
        notes="Gemini 2.5 Pro",
    ),
    "gemini-3.5-flash": ModelConstraints(
        omit_params=frozenset(),
        max_context=1_048_576,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=True,
        temperature_range=(0.0, 2.0),
        notes="Gemini 3.5 Flash",
    ),
    "gemini-3.1-pro": ModelConstraints(
        omit_params=frozenset(),
        max_context=2_000_000,
        supports_tools=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_vision=True,
        temperature_range=(0.0, 2.0),
        notes="Gemini 3.1 Pro",
    ),
    # OpenAI o1-series — no temperature, no streaming, no system prompt
    "o1-mini": ModelConstraints(
        omit_params=frozenset({"temperature", "top_p"}),
        max_context=128_000,
        supports_system_prompt=False,
        supports_streaming=False,
        temperature_range=None,
        notes="OpenAI o1-mini: no temperature/top_p, no streaming, no system",
    ),
    "o1-preview": ModelConstraints(
        omit_params=frozenset({"temperature", "top_p"}),
        max_context=128_000,
        supports_system_prompt=False,
        supports_streaming=False,
        temperature_range=None,
        notes="OpenAI o1-preview: no temperature/top_p, no streaming, no system",
    ),
    # OpenAI GPT-4o family
    "gpt-4o": ModelConstraints(
        omit_params=frozenset(),
        max_context=128_000,
        supports_tools=True,
        supports_vision=True,
        temperature_range=(0.0, 2.0),
        notes="GPT-4o",
    ),
    "gpt-4o-mini": ModelConstraints(
        omit_params=frozenset(),
        max_context=128_000,
        supports_tools=True,
        temperature_range=(0.0, 2.0),
        notes="GPT-4o mini",
    ),
}


class ConstraintRegistry:
    """Thread-safe registry of ModelConstraints with static entries and runtime learning."""

    def __init__(self) -> None:
        self._static: dict[str, ModelConstraints] = {}
        self._learned: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load_static()
        self._load_learned()

    def get(self, model_id: str) -> ModelConstraints:
        """Return constraints for model_id.

        Lookup order:
        1. Exact match in static table
        2. Apply any learned omit_params on top
        3. Permissive default for unknown models
        """
        # 1. Exact static match
        base: ModelConstraints | None = self._static.get(model_id)

        # 2. Merge learned omit_params
        with self._lock:
            learned_entry = self._learned.get(model_id)

        if learned_entry:
            extra_omit = frozenset(learned_entry.get("omit_params", []))
            if extra_omit:
                src = base if base is not None else self._permissive_default()
                return ModelConstraints(
                    omit_params=src.omit_params | extra_omit,
                    rename_params=src.rename_params,
                    max_context=src.max_context,
                    max_output_tokens=src.max_output_tokens,
                    supports_tools=src.supports_tools,
                    supports_streaming=src.supports_streaming,
                    supports_json_mode=src.supports_json_mode,
                    supports_system_prompt=src.supports_system_prompt,
                    supports_vision=src.supports_vision,
                    temperature_range=src.temperature_range,
                    rate_limit_rpm=src.rate_limit_rpm,
                    thinking_required=src.thinking_required,
                    notes=src.notes + " [+learned]",
                )

        # 3. Prefix match — e.g. "moonshot-v1-8k-custom" → "moonshot-v1-8k"
        if base is None:
            for key, constraints in self._static.items():
                if model_id.startswith(key + "-") or model_id.startswith(key + "."):
                    base = constraints
                    break

        # 4. Permissive default for unknown models
        return base if base is not None else self._permissive_default()

    def learn(self, model_id: str, omit_param: str, reason: str) -> None:
        """Add omit_param to this model's learned constraints and persist to disk.

        Called when a provider API rejects a parameter at runtime.
        Mutations are visible immediately via get().
        """
        with self._lock:
            if model_id not in self._learned:
                self._learned[model_id] = {"omit_params": [], "reason": reason, "ts": time.time()}
            entry = self._learned[model_id]
            if omit_param not in entry["omit_params"]:
                entry["omit_params"].append(omit_param)
            entry["ts"] = time.time()
            entry["reason"] = reason
        self._save_learned()
        logger.info("constraint-learned model=%s omit=%s reason=%r", model_id, omit_param, reason)

    def _load_static(self) -> None:
        self._static = dict(STATIC_CONSTRAINTS)

    def _load_learned(self) -> None:
        """Load learned constraints from disk, silently skipping entries older than TTL."""
        if not LEARNED_CONSTRAINTS_PATH.exists():
            return
        try:
            raw = json.loads(LEARNED_CONSTRAINTS_PATH.read_text())
            cutoff = time.time() - (LEARNED_TTL_DAYS * 24 * 3600)
            with self._lock:
                for model_id, entry in raw.items():
                    if entry.get("ts", 0) >= cutoff:
                        self._learned[model_id] = entry
        except Exception as exc:
            logger.debug("Failed to load learned constraints: %s", exc)

    def _save_learned(self) -> None:
        """Atomically write learned constraints to disk."""
        LEARNED_CONSTRAINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = dict(self._learned)
        fd, tmp = tempfile.mkstemp(dir=LEARNED_CONSTRAINTS_PATH.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, LEARNED_CONSTRAINTS_PATH)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    @staticmethod
    def _permissive_default() -> ModelConstraints:
        """Default for unknown models: all capabilities enabled, large context."""
        return ModelConstraints()


def shape_request(model_id: str, raw: dict) -> dict:
    """Apply model constraints to a raw request kwargs dict.

    Applies in order:
    1. Strip keys in omit_params
    2. Rename keys in rename_params
    3. Clamp max_tokens to max_output_tokens ceiling
    4. Clamp or remove temperature per temperature_range
    5. Log one INFO line describing what changed (only if something changed)

    Returns a new dict — never mutates raw.
    """
    constraints = get_constraint_registry().get(model_id)
    shaped = copy(raw)
    stripped: list[str] = []
    renamed: list[str] = []
    clamped: list[str] = []

    # Step 1: strip forbidden params
    for p in list(constraints.omit_params):
        if p in shaped:
            del shaped[p]
            stripped.append(p)

    # Step 2: rename params
    for src, dst in constraints.rename_params.items():
        if src in shaped:
            shaped[dst] = shaped.pop(src)
            renamed.append(f"{src}->{dst}")

    # Step 3: clamp max_tokens
    if constraints.max_output_tokens is not None:
        mt = shaped.get("max_tokens")
        if mt is not None and mt > constraints.max_output_tokens:
            shaped["max_tokens"] = constraints.max_output_tokens
            clamped.append(f"max_tokens:{mt}->{constraints.max_output_tokens}")

    # Step 4: handle temperature (only acts if not already stripped in step 1)
    if "temperature" in shaped:
        if constraints.temperature_range is None:
            del shaped["temperature"]
            stripped.append("temperature")
        else:
            lo, hi = constraints.temperature_range
            t = shaped["temperature"]
            clamped_t = max(lo, min(hi, t))
            if clamped_t != t:
                shaped["temperature"] = clamped_t
                clamped.append(f"temperature:{t}->{clamped_t}")

    # Step 5: enforce thinking_required — strip enable_thinking=False
    if constraints.thinking_required and shaped.get("enable_thinking") is False:
        del shaped["enable_thinking"]
        stripped.append("enable_thinking")
        logger.warning(
            "constraint-shaping model=%s: removed enable_thinking=False because "
            "thinking is always required for this model",
            model_id,
        )

    if stripped or renamed or clamped:
        logger.info(
            "constraint-shaping model=%s stripped=%s renamed=%s clamped=%s",
            model_id, stripped, renamed, clamped,
        )

    return shaped


def parse_constraint_error(provider_name: str, exception: Exception) -> tuple[str, str] | None:
    """Map a provider API error to (param_to_omit, reason) if it is a known constraint violation.

    Returns None for unrecognized errors — caller must not retry on constraint grounds.
    """
    msg = str(exception).lower()
    name = provider_name.lower()

    # Moonshot/Kimi: temperature rejection on v1 chat models
    if name in ("kimi", "moonshot", "custom"):
        if "invalid temperature: only 1 is allowed" in msg:
            return ("temperature", "moonshot v1 rejects non-default temperature")

    # OpenAI o1-series: unsupported parameter errors
    if name == "openai":
        if "unsupported parameter" in msg:
            import re
            match = re.search(r"unsupported parameter[:\s]+['\"]?(\w+)['\"]?", msg)
            if match:
                return (match.group(1), f"openai model rejected parameter: {match.group(1)}")
            return ("temperature", "openai model rejected parameter")
        if "does not support" in msg and "temperature" in msg:
            return ("temperature", "model does not support temperature")

    # Anthropic error shapes (via anthropic SDK):
    #   - overloaded_error (529) → transient, retryable; NOT a constraint
    #   - invalid_request_error with "unsupported_value" / "Invalid" and a field
    #     name → learnable constraint
    if name == "anthropic":
        import re
        # Overloaded / rate-limit — not a constraint; let the caller's retry handle it
        if "overloaded_error" in msg or "overloaded" in msg:
            return None

        # Non-constraint errors
        if "could not parse json body" in msg:
            return None
        if re.search(r"messages\.\d+\.role must be", msg):
            return None

        # Specific constraint errors
        if "top_p must be between 0 and 1" in msg:
            return ("top_p", f"{name}: top_p out of range")
        if "max_tokens is too large" in msg:
            return ("max_tokens", f"{name}: max_tokens exceeds limit")

        # Parameter deprecation: e.g. opus-4-7 rejects temperature with
        #   "`temperature` is deprecated for this model."
        match = re.search(r"[`'\"]?(\w+)[`'\"]?\s+is\s+deprecated", msg)
        if match:
            return (match.group(1), f"{name}: parameter deprecated for this model")

        # Generic constraint errors
        if "unsupported_value" in msg or "invalid_request_error" in msg:
            # Look for a field reference like "temperature", "top_p", "frequency_penalty"
            # in messages like: "temperature: Input must be..."
            # or: "Invalid value for 'temperature'"
            match = re.search(r"[`'\"]?(\w+)[`'\"]?\s*:?\s*(?:input must|is not allowed|is not supported)", msg)
            if match:
                return (match.group(1), f"{name} rejected parameter: {match.group(1)}")
            match = re.search(r"invalid value for [`'\"]?(\w+)[`'\"]?", msg)
            if match:
                return (match.group(1), f"{name} invalid value for: {match.group(1)}")

    # Gemini error shapes (via google-genai SDK):
    #   - "400 INVALID_ARGUMENT" with field info → learnable
    #   - "RESOURCE_EXHAUSTED" / "429" / "quota" → transient; NOT a constraint
    #   - Safety-filter blocks (finish_reason=SAFETY) → not a param issue
    if name == "gemini":
        import re
        # Non-constraint errors (transient or auth)
        if "resource_exhausted" in msg or "quota" in msg:
            return None
        # Enhanced safety check to include finish_reason=SAFETY
        if "safety" in msg and ("block" in msg or "finish_reason=safety" in msg):
            return None
        if "api key not valid" in msg:
            return None

        # Constraint errors
        if "invalid_argument" in msg or "invalid argument" in msg or "400" in msg:
            # New pattern for response generation config
            match = re.search(r"invalid response generation configuration for field '(\w+)'", msg)
            if match:
                return (match.group(1), f"gemini rejected generation configuration field: {match.group(1)}")

            # Existing patterns
            # Match patterns like: "invalid field 'foo'" or "parameter 'foo'" or "'foo' is not"
            match = re.search(r"(?:invalid field|parameter|unknown field)\s*[`'\"]?(\w+)[`'\"]?", msg)
            if match:
                return (match.group(1), f"gemini rejected parameter: {match.group(1)}")
            match = re.search(r"[`'\"](\w+)[`'\"]\s+is not\s+(?:a valid|supported|recognized)", msg)
            if match:
                return (match.group(1), f"gemini rejected parameter: {match.group(1)}")

    return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: ConstraintRegistry | None = None
_registry_lock = threading.Lock()


def get_constraint_registry() -> ConstraintRegistry:
    """Return the module-level ConstraintRegistry singleton."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = ConstraintRegistry()
    return _registry
