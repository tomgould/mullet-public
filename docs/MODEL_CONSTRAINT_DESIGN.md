# Model Constraint Layer — Design

_Status: shipped and wired across OpenAI-compatible, Anthropic, and Gemini providers. See commits 2026-04-20/21 (aafc633 → 6696fde)._

## 1. Problem statement

Third-party LLM APIs impose per-model constraints that differ wildly:
temperature must be pinned to 1.0 on Moonshot `moonshot-v1-*`, omitted
entirely on OpenAI `o1-*`, accepted 0–2 on most others; param names
differ (`max_tokens` vs `max_output_tokens`); capabilities (tools,
streaming, system prompts, vision, context window) vary per model.
Today's fragmented approach — a `_FIXED_TEMPERATURE_MODELS` set, a
literal-error-string retry, hardcoded per-provider quirks — required
multi-file edits for every new model variant and produced silent 400s
when we missed one. This document describes the unified layer that
replaces it.

## 2. Constraint data model

A `ModelConstraints` dataclass in `mullet/core/providers/constraints.py`:

```python
@dataclass
class ModelConstraints:
    omit_params: frozenset[str]   # params to strip entirely before wire
    rename_params: dict[str, str] # source → target name map
    max_context: int              # input token ceiling
    max_output_tokens: int | None # output ceiling (None = provider default)
    supports_tools: bool
    supports_streaming: bool
    supports_json_mode: bool
    supports_system_prompt: bool
    supports_vision: bool
    temperature_range: tuple[float, float] | None  # None = omit entirely
    rate_limit_rpm: int | None
    notes: str                    # operator-facing reason
```

### Example entries (seeded in `STATIC_CONSTRAINTS`)

| Model | omit_params | temperature_range | tools | max_context |
|---|---|---|---|---|
| `moonshot-v1-8k` | `{temperature}` | `None` (omit) | ✗ | 8192 |
| `moonshot-v1-128k` | `{temperature}` | `None` | ✗ | 131072 |
| `kimi-k2.5` | `{}` | `(0.0, 2.0)` | ✓ | 262144 |
| `kimi-k2.6` | `{}` | `(0.0, 2.0)` | ✓ | 262144 |
| `kimi-k2.7-code` | `{}` | `(0.0, 2.0)` | ✓ | 262144 |
| `o1-mini` | `{temperature, top_p}` | `None` | ✗ | 128000 |
| `claude-sonnet-4-6` | `{}` | `(0.0, 1.0)` | ✓ | 1000000 |
| `gemini-2.5-flash` | `{}` | `(0.0, 2.0)` | ✓ | 1000000 |

## 3. Registry structure & seeding

`ConstraintRegistry` is a thread-safe module-level singleton with three
data sources merged in priority order:

1. **Static** (`STATIC_CONSTRAINTS` dict, ~14 entries). Loaded at
   import time.
2. **Prefix match**. Unknown `moonshot-v1-*` variants fall back to
   `moonshot-v1-128k` constraints so future Moonshot SKUs inherit the
   temperature-omit rule automatically.
3. **Learned**. Entries persisted to
   `~/.mullet/learned_constraints.json` with a 30-day TTL. Loaded on
   singleton init; stale entries are silently dropped.

Unknown models fall back to `_permissive_default()` — all capabilities
enabled, `max_context=1_000_000`, `temperature_range=(0.0, 2.0)` — so
Mullet never blocks a novel model outright.

## 4. Request-shaping pipeline

`shape_request(model_id: str, raw: dict) -> dict` is the single
chokepoint called before every wire call. Pipeline:

1. **Strip** — remove any key in `omit_params`.
2. **Rename** — map keys via `rename_params` (e.g. `max_tokens` →
   `max_output_tokens` for Gemini, though today Gemini is handled via
   its native config shape).
3. **Clamp output tokens** — if `max_output_tokens` is set and the
   request exceeds it, clamp.
4. **Temperature** — if `temperature_range is None`, strip the key; if
   a range is set and the value is outside, clamp to nearest bound.
5. **Log once** — one INFO line like
   `constraint-shaping model=X stripped=['temperature'] renamed=[] clamped=[]`
   whenever any transform was applied.

Integration sites:

| File | Function | Location |
|---|---|---|
| `openai_compat.py` | `create`, `create_stream` | Right after kwargs are built, before `_call_with_retry` |
| `anthropic.py` | `create`, `create_stream` | After kwargs assembly, before dispatch |
| `gemini.py` | `create` | Pseudo-kwargs → `GenerateContentConfig` (Gemini uses a config object, not flat kwargs) |

## 5. Novel-constraint self-healing

On an API 400 we don't know about, a per-provider `parse_constraint_error`
maps the exception to `(param, reason)` — then the provider layer:

```
try:
    response = client.x.create(**kwargs)
except Exception as e:
    fix = parse_constraint_error(self.name, e)
    if fix is None:
        raise
    param, reason = fix
    registry.learn(self.model, param, reason)   # persists to disk
    kwargs = shape_request(self.model, kwargs)  # re-shape with new constraint
    response = client.x.create(**kwargs)        # retry ONCE
```

### Seeded parsers

| Provider | Known patterns |
|---|---|
| Kimi/Moonshot | `"invalid temperature: only 1 is allowed"` → omit `temperature` |
| OpenAI | `"Unsupported parameter: <name>"` (o1 series) |
| Anthropic | `invalid_request_error` with `Invalid value for '<param>'` |
| Gemini | `INVALID_ARGUMENT` with `unknown field '<param>'` |

Transient errors (Anthropic `overloaded_error`, Gemini `RESOURCE_EXHAUSTED`,
Gemini safety-filter blocks) intentionally return `None` — the provider's
normal retry/backoff logic handles them.

## 6. Fallback chain pre-filter

`mcp_handlers._build_fallback_chain` now consults
`ConstraintRegistry.get(model).can_serve(needs)` before adding any model
to the chain:

```python
needs = {"context_tokens": prompt_tokens, "requires_tools": True, "requires_streaming": False}
kimi_model = registry.get_best_model("kimi", tier)
if constraint_registry.get(kimi_model).can_serve(needs):
    fallbacks.append(KimiProvider(api_key=kimi_key, model=kimi_model))
```

This replaces the previous tier-list filter that only excluded
`moonshot-v1-*` from CRITICAL/ADVANCED — now the check is
uniform across tiers and based on actual capability instead of tier-name
heuristics.

## 7. Test strategy

- **Unit (fast)**: shape (strip/rename/clamp/no-mutate), registry (static, prefix, learned, TTL, accumulation), error parsers (each provider + unknown), can_serve matrix. All mocked — no real API.
- **Integration (wiring)**: per-provider tests assert `shape_request` is called, INFO wire log emits, error-learning retry fires exactly once. Mocks the SDK client.
- **E2E (opt-in)**: `-m live` tests spawn real CLI + real LLM. Default-excluded from the suite via `addopts = ['-m', 'not live']`. Opt in with `pytest -m live`.
- **Learned-state isolation**: session-scope fixture monkeypatches
  `LEARNED_CONSTRAINTS_PATH` to a tmp path and resets the singleton, so
  no test pollutes `~/.mullet/learned_constraints.json` or leaks into
  siblings.

## 9. Migration path

This rolled in **without breaking the 8 prior commits** by:

1. Shipping `constraints.py` + static seed tables first (no callers).
2. Wiring `openai_compat` to call `shape_request`, keeping the old
   `_FIXED_TEMPERATURE_MODELS` set as a belt-and-braces safety net.
3. Switching the openai_compat retry path from literal-string match to
   `parse_constraint_error`. Deleting the old constant only after the
   new parser had a test demonstrating coverage.
4. Adding the `_build_fallback_chain` `can_serve` check as an
   additional filter (not a replacement), then tightening it over the
   next commit.
5. Wiring anthropic.py and gemini.py once the pattern was validated in
   openai_compat.
6. Finally, extending the per-provider parsers beyond stubs.

At every step the test suite remained green (`-m 'not live'`).

## 10. Open questions / risks

- **Regex fragility**: the parsers use regex on error strings — if
  OpenAI/Anthropic/Gemini change the phrasing of a 400 message, the
  parser returns `None` and the caller's generic retry takes over
  (safe fallback). Monitor `interaction_log` for repeated unparseable
  400s and extend the parsers.
- **Learned entry poisoning**: if `self.model` says kimi-k2.5 but the
  wire actually sends something else (e.g. silent fallback), the
  learned constraint ends up keyed to the wrong model. The HUD/wire
  consistency fix (commit 4a52504) + the wire-level INFO log mitigate
  this, but a defense-in-depth option would be to key learning off the
  model id extracted from the error response rather than `self.model`.
- **ModelInfo vs ModelConstraints duplication**: ✅ RESOLVED.
  `ModelInfo.supports_tools` and `ModelInfo.supports_streaming` now
  delegate to `ConstraintRegistry.get(model_id).supports_*`
  (v1.8.1, commit `f5b7a7f`). The dataclass fields remain for back-compat
  but are no longer duplicated in `MODEL_DATABASE` entries.
- **Streaming vs non-streaming asymmetry**: constraint self-healing
  currently only fires on the non-streaming `_call_with_retry` path.
  Adding it to streams requires the SDK to expose the raw error before
  the stream is consumed — investigate per provider.
- **Provider detection via `self.name`**: `parse_constraint_error`
  dispatches by provider name string. If a new provider is added with
  a custom name, its errors go unparsed until a parser is added. A
  more robust approach would be an explicit `error_parser` attribute
  on each Provider subclass.
