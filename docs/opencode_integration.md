# opencode Integration Guide

This document covers the full opencode integration in Mullet: what it is, how to set it up, how it works under the hood, and how to use it from Claude Code.

---

## What is opencode and why does Mullet use it?

[opencode](https://opencode.ai) is an open-source terminal coding agent. Unlike `mullet_run` delegates (which are remote API calls with no filesystem access), opencode runs **locally** with full native access to your project files — it can read, write, move files, run tests, and execute shell commands directly.

Mullet uses opencode as a zero-Anthropic-token delegation target:

| Property | `mullet_run` / Kimi | `mullet_opencode` / opencode |
|---|---|---|
| Filesystem access | None (paste code into prompt) | Full native read/write/exec |
| Model | Kimi K2 via API | Cloud (Kimi/DeepSeek via OpenRouter or Moonshot) via `opencode run`, or a local LM Studio model via Mullet's own lightweight agent |
| Anthropic tokens | 0 | 0 |
| Latency | ~5–30 s | ~30–300 s for cloud dispatch (`opencode run` agent loop); local dispatch doesn't go through opencode at all and varies with GPU load — see "How it works" |
| Best for | Short edits, analysis | Multi-file refactors, doc sweeps, test runs |

When opencode is not installed, `mullet_opencode` automatically falls back to `mullet_run/Kimi` with a warning that filesystem access is unavailable.

---

## Prerequisites

### 1. Install opencode

```bash
npm i -g opencode-ai@latest
```

Verify: `opencode --version`

### 2. Moonshot API key (for Kimi K2)

The free `opencode/big-pickle` model works without a key. For Kimi K2 quality, you need a Moonshot key:

1. Sign up at [platform.moonshot.cn](https://platform.moonshot.cn) or [platform.kimi.ai](https://platform.kimi.ai)
2. Create an API key
3. Export it:
   ```bash
   export KIMI_API_KEY=sk-...
   # or equivalently:
   export MOONSHOT_API_KEY=sk-...
   ```

Mullet reads `KIMI_API_KEY` first, then `MOONSHOT_API_KEY`, then falls back to checking `.mcp.json` / `~/.mcp.json` under `env.KIMI_API_KEY`.

---

## Setup via `mullet setup-claude`

`mullet setup-claude` handles both steps automatically:

```bash
mullet setup-claude
```

At the opencode prompt, answer `y`. The installer will:

1. Run `npm i -g opencode-ai@latest` if opencode is not found on `PATH`.
2. Write `~/.config/opencode/opencode.json` with the Moonshot/Kimi provider config (see below).
3. Print a reminder to set `KIMI_API_KEY`.

`setup-claude` does not write a `provider.lmstudio` block. Add one yourself (see the config reference below) if you want local models declared for direct/manual `opencode run` invocations. Mullet's own automatic local-first routing (the lightweight agent path described under "How it works") doesn't need it — it talks to LM Studio directly.

For a non-interactive install (CI, dotfiles bootstrap):

```bash
mullet setup-claude --yes        # auto-accept all prompts
mullet setup-claude --check      # preview without writing anything
```

---

## `~/.config/opencode/opencode.json` reference

After setup, your config should look like this:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "moonshot/kimi-k2.7-code",
  "provider": {
    "moonshot": {
      "name": "Moonshot",
      "api": "https://api.moonshot.ai/v1",
      "env": ["KIMI_API_KEY"],
      "models": {
        "kimi-k2.7-code": {
          "name": "Kimi K2.7 Code",
          "tool_call": true,
          "limit": { "context": 262144, "output": 32768 },
          "cost": { "input": 0.95, "output": 4.0 }
        },
        "kimi-k2.6": {
          "name": "Kimi K2.6",
          "tool_call": true,
          "limit": { "context": 262144, "output": 32768 },
          "cost": { "input": 0.95, "output": 4.0 }
        },
        "kimi-k2.5": {
          "name": "Kimi K2.5",
          "tool_call": true,
          "limit": { "context": 262144, "output": 32768 },
          "cost": { "input": 0.6, "output": 3.0 }
        }
      }
    }
  }
}
```

Key fields:
- `"model"` — active default model, must use `"<provider-id>/<model-id>"` format.
- `"provider.moonshot.env"` — list of env var names opencode will read for the API key.
- `"models.<id>.limit.context"` — context window in tokens; opencode uses this to prevent overflows.

The `models` whitelist under `provider.moonshot` is **required** — without it opencode raises `ProviderModelNotFoundError` even for valid Moonshot models (see Troubleshooting).

### Local models: the `provider.lmstudio` block

To let opencode's own CLI talk to a local LM Studio server — relevant if you invoke `opencode run` directly against an `lmstudio/...` model, e.g. for manual testing — add a `provider.lmstudio` block alongside `provider.moonshot`:

```json
{
  "provider": {
    "lmstudio": {
      "api": "http://127.0.0.1:1234/v1",
      "options": { "timeout": 300000 },
      "models": {
        "qwen2.5-coder-14b-instruct": {
          "name": "Qwen2.5-Coder-14B-Instruct",
          "tool_call": true,
          "limit": { "context": 8192, "output": 4096 },
          "cost": { "input": 0, "output": 0 }
        }
      }
    }
  }
}
```

`options.timeout` is in milliseconds and matters: local generation is much slower per-token than cloud, and opencode's default client timeout is well under a minute — without raising it, a local model that's still mid-generation can have its connection cut and produce a truncated, unparseable response.

This block is **not** required for Mullet's own tier-based local-first routing to work. `detect_local_provider()` (`mullet/core/providers/local_detection.py`) probes LM Studio's HTTP API on `localhost:1234` directly via `LMStudioManager` and never reads `opencode.json` — it only matters when opencode's own CLI is the one making the call. Mullet's automatic local dispatch (the lightweight agent path, below) has its own equivalent timeout fix in a different place: `OpenAICompatibleProvider` in `mullet/core/providers/openai_compat.py` uses a 300s client timeout for local providers versus 60s for cloud, for the same reason.

---

## How it works

### CLI dispatch (`opencode run`)

Mullet used to drive opencode over its **ACP** (Agent Communication Protocol) JSON-RPC session interface. That's gone. ACP's `session/new` `model` parameter turned out to be silently ignored by opencode — every session dispatched to whatever `opencode.json`'s static top-level `model` field named, regardless of what was requested over the protocol (confirmed empirically: flipping the config default while requesting a different model over ACP still produced the config-default model's behaviour). That meant every bit of tier-based routing, escalation, and known-bad-model logic built on top of it had zero real effect on which model actually ran a task — only the config file mattered.

`opencode run` — a separate, one-shot CLI invocation with a real `--model` flag — was confirmed (same empirical method) to respect per-call model selection, so cloud dispatch now spawns it directly instead of opening an ACP session:

```
Mullet (Python)                         opencode (subprocess, one-shot)
────────────────────────────────────────────────────────────────────────
spawn: opencode run <task> --model X --format json --dir <cwd> --auto
                                        starts up, works the task
                                        ← {"type":"step_start", ...}
                                        ← {"type":"text", "part":{"text":...}}
                                        ← {"type":"tool_use", "part":{"tool":..., "state":...}}
                                        ← {"type":"step_finish", "part":{"tokens":..., "cost":...}}
                                        ...
                                        process exits (stdout EOF)
Mullet checks the real exit code: 0 = success, non-zero = failure
git status before/after → "files_changed"
```

`--format json` emits line-delimited JSON events on stdout. Mullet accumulates `text` chunks into the final response and sums `tokens`/`cost` across every `step_finish` event, since a task can involve several tool-call/response cycles before the final answer. An `error` event is opencode's own immediate, correctly-diagnosed failure and is raised right away rather than waiting out the silence timeout below. `--auto` runs non-interactively — there's no TTY to approve permission prompts.

Completion is the *process exiting*, not any particular event: stdout EOF, then the real exit code decides success/failure. If stdout goes silent for longer than `OPENCODE_SILENT_TIMEOUT_S` (240s by default) while the process is still alive, Mullet kills it and treats that as a transient failure eligible for escalation/retry.

Note: `opencode_run()`'s own docstring in `mullet/tools/opencode_tool.py` still says it runs "via opencode ACP" — that's a stale leftover comment from the previous implementation. The dispatch code itself no longer uses ACP anywhere.

### Local dispatch: the lightweight agent path

If the resolved model string starts with `lmstudio/`, `handle_mullet_opencode` doesn't call opencode at all. It runs the task through Mullet's own `AgentRunner` (`mullet/core/agent/runner.py`) instead, using a small toolset (`terminal`, `file`, `code`, `git`) and `build_base_prompt(model=..., cwd=cwd)` as the system prompt.

This exists because opencode's own CLI agent loop carries roughly 11.4K tokens of fixed system-prompt-plus-tool-schema overhead — too much for a local model's realistic context budget on top of an actual task. The lightweight `AgentRunner` path costs roughly 3.5K tokens for the same kind of overhead and reliably completes real agentic tasks locally, where the full opencode loop was failing outright.

If the lightweight run doesn't produce a usable result (local unreachable, the run raises, or `result.completed` is falsy), `handle_mullet_opencode` falls through to the normal `opencode_run()` cloud dispatch above — but with the model reassigned to `openrouter/deepseek/deepseek-v4-flash`, not the original local model, since local has already been tried and failed.

The local path's success shape is narrower than `opencode_run()`'s: no `session_id`, and `usage` is just `{"total_tokens": N}` — no input/output split, no `cost_usd` (local is free, and `AgentRunner` doesn't track the split the same way opencode does). See "Return value" below.

### Model resolution

`opencode_run()` resolves the active model in this order:

1. `model` argument if passed explicitly.
2. `~/.config/opencode/opencode.json` → `model` field.
3. Fallback constant `"opencode/big-pickle"` (free, no key required).

If `opencode` isn't on `PATH` at all, this resolution never runs — `opencode_run()` returns a `mullet_run`/Kimi fallback immediately (see Troubleshooting).

### Tier-based routing and presets

`handle_mullet_opencode` only resolves a model itself when the caller doesn't pass one explicitly. Order of precedence:

1. **Explicit `model` argument** — used as-is, including the `"auto"` sentinel, which is routed via `mullet.core.router.preflight.analyze_prompt` + `routing_matrix.select_kimi_model` instead of the tier ladder below.
2. **Active preset** (`PresetManager().active()`), for any preset other than the default `balanced-mix`:
   - `native_only`, or no `opencode_model` configured → refuse and return an error telling the caller to make the edit natively in Claude Code instead.
   - `opencode_model` configured → use that model outright, skipping the tier ladder.
3. **Tier ladder** (default `balanced-mix` preset, or no active preset) — resolves a tier (explicit `tier` argument if given, else `is_simple_message`/`recommend_tier` heuristics) and current budget pressure (`check_budget().pressure`), then:
   1. Tier `LOW` → prefer a reachable local LM Studio model (`_prefer_local_model`), else the configured default.
   2. Else, pressure ≥ 80% → `moonshot/kimi-k2.7-code` (thinking always on), **regardless of tier**.
   3. Else, tier `CRITICAL`, or tier `ADVANCED` on a task classified as reasoning-heavy → `openrouter/deepseek/deepseek-v4-pro`.
   4. Else, tier `ADVANCED` → `openrouter/deepseek/deepseek-v4-pro` (same model, non-reasoning path).
   5. Else (`STANDARD`, pressure < 80%) → prefer a reachable local model, else `openrouter/deepseek/deepseek-v4-flash`.

Whichever model comes out of this, if it starts with `lmstudio/` it's routed through the lightweight agent path above, not opencode.

### Risk flags

Delegate self-reports of `"completed": true` are not trustworthy on their own — a delegate can report success while silently corrupting a literal, flattening curly quotes to ASCII, changing a test's expected value, or fabricating a call argument. Whichever dispatch path was used, if the result reports `completed: true` with a non-empty `files_changed` list, `_attach_risk_flags()` (`mullet/server/mcp_handlers.py`) runs each changed file through `mullet/core/verification/diff_risk.py`'s deterministic, zero-LLM-cost classifier (`git show HEAD:<file>` vs the working tree) and attaches a `risk_flags` list to the result.

Flagged categories: `literal_change`, `quote_flattening`, `new_control_flow`, `assertion_on_literal`, `call_args_changed`, `comment_changed`, `test_assertion_changed`. An empty list means the classifier found nothing suspicious in a mechanical, regex/difflib sense — not a correctness guarantee. `risk_flags` is only present when there were files to check; a run with no `files_changed` won't have the key at all.

---

## AGENTS.md

`AGENTS.md` in the project root is an **identity file** for opencode agents. When opencode starts a session in the project directory, it reads `AGENTS.md` to understand:

- Its role: opencode agent (not Claude, not Opus).
- What tools it has: bash, file read/write, web fetch, code search.
- What it cannot do: call Mullet MCP tools, access Anthropic API.
- Project layout: key directories, test commands, code standards.
- Review and change protocols: read before edit, syntax check, run tests, report.

This prevents the agent from hallucinating Claude Code tool calls (`mcp__mullet__*`) and grounds it in the actual project structure.

If you fork Mullet or apply it to another project, update `AGENTS.md` to reflect that project's layout and conventions.

---

## Using `mullet_opencode` from Claude Code

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | string | required | Natural language coding task for the agent |
| `cwd` | string | `os.getcwd()` | Project directory for the run |
| `model` | string | resolved automatically — see "Tier-based routing and presets" | Explicit override, e.g. `"moonshot/kimi-k2.7-code"` or `"lmstudio/qwen2.5-coder-14b-instruct"`. The sentinel `"auto"` routes through the preflight/routing-matrix path instead of the tier ladder. |

That's the complete public schema. There is no `apply`, `commit_message`, `diff`, `has_changes`, or `timeout` parameter — an earlier iteration of this tool had them; they've since been removed. `mullet_opencode` never stages or commits on your behalf (see "Workflow example" below), and there's no caller-supplied timeout — the run is bounded internally by `OPENCODE_SILENT_TIMEOUT_S` (240s default) plus the model-escalation/retry logic in `opencode_run()`.

The handler also reads an undocumented `tier` key from the raw arguments dict, used internally to force a tier for testing/preset overrides. It is not part of the public MCP schema above and callers should not rely on it.

### Return value

On success (cloud dispatch via `opencode run`):

```json
{
  "completed": true,
  "response": "I've updated all type hints in ...",
  "duration": 47.2,
  "model": "moonshot/kimi-k2.7-code",
  "session_id": "ses_abc123",
  "usage": {
    "total_tokens": 18420,
    "input_tokens": 15200,
    "output_tokens": 3220,
    "cached_tokens": 0,
    "cost_usd": 0.019,
    "currency": "USD"
  },
  "files_changed": ["mullet/tools/opencode_tool.py"],
  "risk_flags": []
}
```

On success via the local lightweight-agent path (model resolved to `lmstudio/...`), the shape is narrower — no `session_id`, no cost/token split:

```json
{
  "completed": true,
  "response": "I've updated all type hints in ...",
  "duration": 12.4,
  "model": "lmstudio/qwen2.5-coder-14b-instruct",
  "files_changed": ["mullet/tools/opencode_tool.py"],
  "usage": { "total_tokens": 4820 },
  "risk_flags": []
}
```

`risk_flags` (see "Risk flags" above) is only present when `files_changed` is non-empty; an empty list means the deterministic diff classifier found nothing suspicious, not that the change is guaranteed correct.

On error (opencode not found, dispatch failure, context overflow, etc.):

```json
{
  "error": "opencode failed after 2 attempts: opencode: no output for 240s — API may be overloaded; ...",
  "error_reason": "transient",
  "files_changed": null
}
```

`error_reason` is one of `"context_overflow"`, `"auth"`, `"transient"`, `"unknown"` (from `_classify_error()` in `opencode_tool.py`) — useful for deciding whether a retry is worth it at all; an `"auth"` failure won't fix itself.

When the active preset blocks delegation (`native_only`, or a non-default preset without an `opencode_model`):

```json
{
  "completed": false,
  "error": "Preset 'claude-only' handles edits natively in Claude Code — opencode delegation is disabled for this preset. Make the edit directly in Claude Code."
}
```

### Workflow example

```python
# Delegate the task — opencode (or the local lightweight agent) edits files directly.
result = mcp__mullet__mullet_opencode(
    task="Add Google-style docstrings to all public functions in mullet/tools/opencode_tool.py",
)

# Inspect what actually changed and whether the diff-risk classifier flagged anything.
if result.get("completed"):
    print(result.get("files_changed"))
    if result.get("risk_flags"):
        print("review these hunks before trusting the change:", result["risk_flags"])

# There is no built-in apply/commit step — mullet_opencode never stages or
# commits for you. Review with `git diff`, then commit yourself:
#   git add <files> && git commit -m "..."
# or discard with:
#   git checkout -- .
```

### HUD tracking

While `mullet_opencode` is running (either dispatch path), the terminal HUD shows a live row:

```
 opencode-a3f7b1c2  moonshot/kimi-k2.7-code  Add Google-style docstrings...
```

For local dispatch the model column shows the `lmstudio/...` name instead. If a cloud attempt escalates to a different model mid-run (e.g. after a `context_overflow`), the row's model updates in place rather than showing a stale attempt-1 model. The row is removed when the run completes or errors.

---

## Troubleshooting

### `ProviderModelNotFoundError`

opencode requires models to be declared in the `provider.<id>.models` dict in `opencode.json`. If you specify e.g. `"moonshot/kimi-k2.7-code"` or `"lmstudio/qwen2.5-coder-14b-instruct"` but the config only has a top-level `"model": "..."` with no matching `models` block, opencode will reject it. This only bites when opencode's own CLI actually receives the model — Mullet's automatic local dispatch (the lightweight agent path) bypasses opencode entirely and never hits it.

Fix: ensure your `opencode.json` has the full `provider.<id>.models` block, as shown in the config reference above, for every model you dispatch to directly.

### `opencode run` never completes / times out

Check that opencode itself is reachable and correctly configured:

```bash
opencode --version
opencode run "reply with the word pong and nothing else" --model <provider>/<model-id> --format json --dir . --auto
```

The second command should stream a `text` event containing "pong" and then exit 0 within a few seconds. If it hangs, errors, or never exits, opencode or the target provider is in a bad state — check the provider's API key and, for local models, that LM Studio is actually loaded and serving on the expected port. Mullet's own side gives up after `OPENCODE_SILENT_TIMEOUT_S` (240s by default) of total stdout *silence*, not a fixed overall deadline — a task that's still actively streaming tool calls and text won't time out just for running long.

### `bufsize` — events arrive late or not at all

`opencode run` is a one-shot, non-interactive CLI call now, not a session — Mullet spawns it with `stdin=subprocess.DEVNULL` (nothing to keep open) and reads `stdout`/`stderr` via `select()` + `readline()` with `bufsize=1` (line-buffered). If you're touching `opencode_tool.py`'s subprocess spawn, don't change `bufsize` to `0` (unbuffered) or `-1` (block-buffered) — block buffering causes `readline()` to block until the OS pipe buffer fills, which may never happen for a short response.

### KIMI_API_KEY not found

The key must be set before the Mullet MCP server starts. Reload the server after exporting:

```bash
export KIMI_API_KEY=sk-...
# restart Claude Code or the MCP server
```

Or add it to `.mcp.json` under `env.KIMI_API_KEY` in the mullet server entry — Mullet reads it from there as a fallback.

### Fallback: mullet_run used instead of opencode

If the result contains `"fallback": "mullet_run/kimi"`, opencode is not on `PATH`. Install it:

```bash
npm i -g opencode-ai@latest
which opencode   # should now return a path
```

Then retry the `mullet_opencode` call.
