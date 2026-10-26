# Model Routing — Cost, Competency, Recommendations

Canonical reference for which model to use for which task. Verified against
live OpenRouter pricing + ping probes on **2026-06-16**.

This page covers routing for the legacy single-task path: `mullet "<task>"`, the API and the delegation tools (`mullet_run`, `mullet_delegate`, `mullet_swarm`). These run only when `MULLET_DELEGATION=on` is set (the default is off). `mullet_opencode` runs as a single-step job on the ladder, so this routing does not choose its model. `mullet job` does not use it either: job steps go up the delegation ladder (free, local, Kimi, OpenRouter, Claude), described under "Delegation ladder" in the README.

Authoritative source code:
- Cost table: `mullet/core/providers/model_info.py`
- Routing logic: `mullet/core/router/routing_matrix.py::select_openrouter_model`

---

## Available models

| Provider     | Model ID                              | In $/M | Out $/M | Context | Probe latency | Notes |
|--------------|---------------------------------------|-------:|--------:|--------:|--------------:|---|
| lmstudio     | ornith-1.5-35b-a3b                    |   0.00 |    0.00 |   32K¹  |    not probed | Free, local MoE (experts in RAM), ~20 tok/s, reasoning model |
| moonshot     | kimi-k2.7-code (direct API)           |   0.95 |    4.00 |   262K  |        982 ms | Default for opencode; thinking always ON |
| openrouter   | deepseek/deepseek-v4-flash            |   0.05 |    0.25 | **1M**  |         ~1 s | **Best $/quality coder** |
| openrouter   | deepseek/deepseek-v4-pro              |   0.20 |    1.00 | **1M**  |       1200 ms | Premium V4, advanced tasks |
| openrouter   | deepseek/deepseek-v3.2-exp            |   0.27 |    0.41 |   164K  |       1091 ms | Previous default (still usable) |
| openrouter   | qwen/qwen3-coder                      |   0.22 |    1.80 | **1M**  |       1265 ms | Code-tuned, huge ctx |
| openrouter   | z-ai/glm-4.6                          |   0.43 |    1.74 |   203K  |       4016 ms | Strong general, cheap |
| openrouter   | google/gemini-2.5-flash               |   0.30 |    2.50 | **1M**  |          ~1 s | Fast, no reasoning by default |
| openrouter   | x-ai/grok-4.20                        |   1.25 |    2.50 | **2M**  |        768 ms | Largest context available |
| openrouter   | anthropic/claude-sonnet-4.5           |   3.00 |   15.00 | **1M**  |       1894 ms | Premium synthesis, 1M ctx |
| openrouter   | openai/gpt-5                          |   1.25 |   10.00 |   400K  |       3120 ms | Reasoning-by-default (see ⚠ below) |
| openrouter   | google/gemini-2.5-pro                 |   1.25 |   10.00 | **1M**  |          ~3 s | Reasoning-by-default |
| openrouter   | anthropic/claude-opus-4.1             |  15.00 |   75.00 |   200K  |       3120 ms | Frontier max-quality (expensive) |

¹ The context Mullet loads it with (`lms load --context-length 32768`); the model supports 262K.

### ⚠ Reasoning-by-default models

`openai/gpt-5`, `google/gemini-2.5-pro`, `deepseek/deepseek-v4-pro` consume the
`max_tokens` budget on internal reasoning before producing visible output. A
probe with `max_tokens: 8` returned **empty content** for GPT-5 / Gemini-2.5-Pro.

Mitigations:
- Set `max_tokens >= 200` so reasoning has headroom.
- For GPT-5: pass `reasoning: {"effort": "low"}` to cap reasoning spend.
- For trivial tasks prefer `gemini-2.5-flash` or `claude-sonnet-4.5` (no
  reasoning by default).

---

## Routing matrix — which model for which task

Implemented in `select_openrouter_model` (`mullet/core/router/routing_matrix.py`).

| Task signal                                  | Default (balanced)         | Premium (`max` / `premium`)   |
|---------------------------------------------|----------------------------|-------------------------------|
| `needs_reasoning=True` (math, debugging, planning) | `deepseek/deepseek-v4-pro`   | `openai/gpt-5`                |
| `input_tokens > 500K` (huge codebase context) | `qwen/qwen3-coder` (1M ctx)  | `anthropic/claude-sonnet-4.5` |
| `complexity_tier == CRITICAL` (production-shape decisions, multi-file refactors) | `anthropic/claude-sonnet-4.5` | `anthropic/claude-opus-4.1`   |
| `complexity_tier == ADVANCED`               | `deepseek/deepseek-v4-pro`   | `anthropic/claude-sonnet-4.5` |
| `input_tokens > 100K`                       | `deepseek/deepseek-v4-flash` | `deepseek/deepseek-v4-flash`  |
| `STANDARD` / `LOW`                          | `deepseek/deepseek-v4-flash` | `deepseek/deepseek-v4-flash`  |

The local LLM path (`select_local_model`) is unchanged — it picks among
loaded LM Studio models classified by parameter count via `_classify_local_model`.

---

## Competency rankings

Rankings are informed by:
- Public benchmarks (SWE-bench Verified, LiveCodeBench, MMLU-Pro, AIME, GPQA)
- Practical use over many Mullet sessions
- Tool-calling reliability (opencode delegate work)
- Cost-effectiveness vs nearest alternative

### Code generation / editing

| Rank | Model                            | Why                                         |
|-----:|----------------------------------|---------------------------------------------|
| 1    | `anthropic/claude-opus-4.1`      | Highest absolute quality, expensive         |
| 2    | `anthropic/claude-sonnet-4.5`    | Near-Opus quality at 1/5 the cost, 1M ctx   |
| 3    | `deepseek/deepseek-v3.2-exp`     | Best $/quality — competitive on most tasks |
| 4    | `qwen/qwen3-coder`               | Code-tuned, 1M ctx for big repos           |
| 5    | `kimi-k2.7-code` (moonshot direct) | Strong coder, thinking ON, cheaper than openrouter Anthropic |
| 6    | `z-ai/glm-4.6`                   | Competent fallback                          |
| 7    | `lmstudio/ornith-1.5-35b-a3b`    | Free, local MoE, ~20 tok/s                  |

### Reasoning / debugging / planning

| Rank | Model                            | Why                                         |
|-----:|----------------------------------|---------------------------------------------|
| 1    | `openai/gpt-5`                   | Best-in-class reasoning, premium            |
| 2    | `anthropic/claude-opus-4.1`      | Strong on multi-step synthesis              |
| 3    | `deepseek/deepseek-v4-pro`       | Cheap reasoning, transparent CoT            |
| 4    | `google/gemini-2.5-pro`          | Built-in reasoning, 1M ctx                  |

### Long-context (codebase summarisation, multi-file review)

| Rank | Model                            | Context | $/M in–out |
|-----:|----------------------------------|--------:|------------|
| 1    | `x-ai/grok-4.20`                 |     2M  | $1.25/$2.50 |
| 2    | `anthropic/claude-sonnet-4.5`    |     1M  | $3/$15 |
| 3    | `qwen/qwen3-coder`               |     1M  | $0.22/$1.80 |
| 4    | `google/gemini-2.5-pro`          |     1M  | $1.25/$10 |

### Fastest (low-latency)

| Rank | Model                            | Probe latency |
|-----:|----------------------------------|--------------:|
| 1    | `x-ai/grok-4.20`                 |        768 ms |
| 2    | `kimi-k2.7-code` (direct)        |        982 ms |
| 3    | `deepseek/deepseek-v3.2-exp`     |       1091 ms |
| 4    | `qwen/qwen3-coder`               |       1265 ms |

---

## Speed-mode → model mapping (summary)

The `--speed` flag on `mullet` CLI biases the routing. Without it the CLI uses `economy`; the default `balanced-mix` preset sets `balanced` for the delegation tools.

| Speed mode    | Local LLM | Cheap cloud (DeepSeek/Qwen) | Premium (Sonnet/Opus/GPT-5) |
|---------------|-----------|-----------------------------|------------------------------|
| `economy`     | First     | Fallback                    | Never                        |
| `balanced` | Simple tasks | Most work          | Only for CRITICAL            |
| `fast`        | Never     | Always                      | Only for CRITICAL            |
| `max`         | Never     | Skipped                     | Always                       |
| `premium`     | Never     | Skipped                     | Opus-first; auto-downgrade   |

### Interaction with presets

A preset's `allowed_providers` is a filter applied before the routing matrix (e.g. `claude-only` restricts to sonnet/haiku/opus-4-8; `local-only` to local). The preset's `speed_mode` biases tier selection. See README.md § "Provider Presets" for details.

---

## Verification commands

Re-run these whenever the model line-up changes:

```bash
# 1. Live pricing pull (requires OPENROUTER_API_KEY in env or .env)
python3 scripts/refresh_pricing.py --dry-run   # uses OPENROUTER_API_KEY from env/.env

# 2. Ping each model via opencode (verifies auth + tool-calling)
opencode run -m openrouter/deepseek/deepseek-v4-flash "say ok"
opencode run -m moonshot/kimi-k2.7-code "say ok"
opencode run -m lmstudio/ornith-1.5-35b-a3b "say ok"

# 3. Inspect mullet's chosen route for a task (no actual call)
mullet explain "your task description here"
```

---

## Cost worked example

For a code review of a mid-sized API codebase (~110 source files, expected
~3-4K lines of MD output ≈ 50K-80K output tokens, input via tool-fetched
files ≈ 200K-500K input tokens):

| Model                            | Est. cost |
|----------------------------------|----------:|
| `lmstudio/ornith-1.5-35b-a3b`    | **$0.00** |
| `deepseek/deepseek-v3.2-exp`     | $0.10 – $0.20 |
| `qwen/qwen3-coder`               | $0.15 – $0.25 |
| `kimi-k2.7-code` (moonshot direct) | $0.45 – $1.00 |
| `anthropic/claude-sonnet-4.5`    | $1.50 – $4.00 |
| `anthropic/claude-opus-4.1`      | $7.50 – $20.00 |
| `openai/gpt-5`                   | $0.50 – $2.00 |

**Default pick for a thorough review of this size:**
`deepseek/deepseek-v3.2-exp` (cost-effective, strong on TS/Node).

**Quality-first pick:** `anthropic/claude-sonnet-4.5` (1M ctx fits whole repo).

---

## Maintenance

When a new model appears on OpenRouter that displaces an entry above, edit:
1. `mullet/core/providers/model_info.py` — add `ModelInfo(ctx, in_per_token, out_per_token)`.
2. `mullet/core/router/routing_matrix.py::select_openrouter_model` — adjust the branches.
3. `~/.config/opencode/opencode.json` openrouter `models` block — register the slug.
4. This file — update the matrix + competency tables.

Run the verification commands above and re-time the probe latencies. Commit
the changes as one "model line-up refresh YYYY-MM-DD" commit so it's easy to
roll back if a new model misbehaves in practice.

## Quality-score maintenance

No API exposes model quality, so `mullet/core/providers/quality.py` must be
maintained by hand — unlike pricing, which `scripts/refresh_pricing.py`
auto-refreshes from OpenRouter.

When to trigger a refresh pass:

1. **After `refresh_pricing.py` auto-adds a new model** — the model gets pricing
   but no `MODEL_QUALITY` entry, so it silently fails open (treated as passing
   every quality floor) until scored.
2. **Quarterly** — frontier benchmarks shift infrequently; a quarterly cadence
   catches stale scores as newer runs supersede old ones.

Adding or updating one entry:

- Look up the model on public benchmarks (SWE-Bench Verified/Pro, Terminal-Bench
  2.x, Artificial Analysis Intelligence Index) or vendor head-to-head comparisons.
- Anchor `coding`/`reasoning`/`general` scores (0–100) relative to `claude-sonnet-5`.
- Set `confidence` honestly:
  - `direct` — real benchmark exists for this exact model.
  - `inherited` — variant with a documented penalty (see design doc).
  - `estimated` — everything else.
- Always fill `source` with a citation (URL or benchmark name); never bare numbers.
- Set `is_reasoning_tuned=True` only for models architecturally built for extended
  reasoning. Today that is only `deepseek/deepseek-v4-pro` and `kimi-k2.7-code`. No
  Gemini model qualifies; no Claude/Anthropic model qualifies.

> **Known gap (2026-07-10):** no Gemini model has `is_reasoning_tuned=True`, so
> `meets_reasoning_gate` excludes every Gemini candidate for reasoning-tagged
> requests — the Gemini value-picker never fires, always falling back to the
> static ladder. Revisit in a future refresh: does `gemini-3.1-pro` deserve the
> flag, or does Gemini genuinely have no reasoning-tuned entry in the curated set?

After any `MODEL_QUALITY` edit, run:

```bash
python3 -m pytest tests/test_quality_scores.py tests/test_quality_floors.py tests/test_value_picker.py -q
```

Commit as one `quality-score refresh YYYY-MM-DD` commit.

## Routing history — observability and management

Mullet records every routed call to `.mullet/routing_history.jsonl` in the project directory you run it from (append-only).
The `mullet routing` CLI reads from this file:

```
mullet routing show                       # per-provider success rate, cost, latency table
mullet routing insights [--limit N]       # 5-section report: signatures, reliability, tiers, anti-recs, tails
mullet routing compare kimi openrouter    # side-by-side: success rate, cost, p50/p95 latency
mullet routing recent [--provider P] [--failed]  # filterable recent outcomes
mullet routing export [--format json|csv] [-o FILE] [--limit N]  # export history for analysis
mullet routing reset [--yes]              # clear all history (with confirmation)
mullet routing purge-provider kimi [--yes]  # remove one provider's entries
```

The quality routing filter (in `decision.py`) reads the same file: providers with
≥3 samples and <60% success rate are skipped for new tasks and a fallback is chosen.
Use `mullet routing show` to check which providers are below threshold.

## Cross-references

In-repo sources of truth (read these when modifying routing logic):

- `mullet/core/providers/model_info.py` — canonical price table (`MODEL_DATABASE`).
- `mullet/core/router/routing_matrix.py` — `select_openrouter_model` + `select_local_model` + `select_gemini_model`.
- `mullet/core/router/decision.py` — quality filter (`QUALITY_THRESHOLD`, `QUALITY_MIN_SAMPLES`).
- `mullet/core/providers/budget_probe.py` — provider tier + balance detection.
- `scripts/sync_openrouter_pricing.py` — drift checker (CI-friendly).
- `docs/LOOP_AGREEMENT.md` §2: the delegation ladder for interactive Claude Code sessions.
