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

This guide covers the `mullet_opencode` MCP tool. It is separate from `mullet job`, whose only worker today is your own Claude Code. An opencode worker for jobs, inside the job sandbox and checked by the same gates, is coming. `MULLET_DELEGATION=off` switches `mullet_opencode` off with the other delegation tools.

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

Mullet reads `KIMI_API_KEY` first, then `MOONSHOT_API_KEY`. They come from the process environment or `~/.config/mullet/.env`, never from a project's `.env` or `.mcp.json`.

---

## Setup

`mullet setup-claude` doesn't install opencode. Install it and write its config yourself:

1. `npm i -g opencode-ai@latest` (needs Node.js 18 or later).
2. Write `~/.config/opencode/opencode.json` with the Moonshot/Kimi provider config (see below).
3. Set `KIMI_API_KEY` in `~/.config/mullet/.env` or your environment.

Add a `provider.lmstudio` block too (see the config reference below) if you want local models declared for direct/manual `opencode run` invocations. Mullet's own automatic local-first routing (the lightweight agent path described under "How it works") doesn't need it: it talks to LM Studio directly.

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
        "ornith-1.5-35b-a3b": {
          "name": "Ornith-1.5-35B-A3B",
          "tool_call": true,
          "limit": { "context": 32768, "output": 4096 },
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

### Single-step ladder job

`mullet_opencode` is implemented in `mullet/server/mcp_handlers.py` and two job modules:

- `mullet/jobs/delegate.py` creates a single-step Mullet job, approves it, and runs it in the foreground.
- `mullet/jobs/handback.py` brings the finished node's diff back into the caller's working tree.

When Claude Code calls `mullet_opencode`, Mullet works out a repository root from `cwd`, the paths in scope, or the server's current working directory. It builds a job with a single mechanical node whose goal is the task text and whose scope is the given paths (or the whole repo if none are given).

The tool call itself counts as approval, so the job is approved immediately and run in the foreground through the job runner (`mullet/jobs/runner.py`). The `timeout` argument bounds the run in minutes (default 20, maximum 60). The node is kind `mechanical`, so the ladder starts it on a cheap rung and only enables review when `ladder.is_risky()` says the step is risky. The user's stored profile supplies the gates, so the usual build and test checks still run.

If and only if the node status is `PASSED`, `mullet/jobs/handback.py` computes the diff between the job branch and its base and applies it to the caller's working tree with `git apply --3way`. Before applying, it checks that every target file is clean in the caller's tree and that the base commit is still the caller's HEAD's ancestor-equal content for those files. If either check fails, nothing is applied and the result status is `passed_not_applied`. After a successful apply, the hand-back unstages the paths so the changes are left in the working tree for the caller to review. Mullet never commits or pushes in the caller's repo.

If the runner cannot start (for example, no sandbox, no `socat`, no `claude`, no `opencode`), the tool returns `blocked` with the reason instead of falling back to a direct call.

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
| `files` | array | `[]` | File paths or globs the task may change (the job's scope). Defaults to the whole repo. |
| `cwd` | string | cwd | Project directory or repository root. Defaults to the git repo containing the files, else the server's working directory. |
| `timeout` | integer | 20 | Maximum run time in minutes (max 60). |
| `providers` | array | auto | Rungs to allow on the ladder: `kimi`, `openrouter`, `free`, `local`, `claude`. Omit to use the whole ladder. |
| `model` | string | ignored | Deprecated: accepted and ignored, with a note in the result. |
| `apply` | boolean | ignored | Deprecated: accepted and ignored, with a note in the result. |

The public schema is `task`, `files`, `cwd`, `timeout` and `providers`. Old keys such as `model`, `tier`, `speed_mode`, `provider`, `apply` and `commit_message` are still accepted for backward compatibility but have no effect.

### Return value

```json
{
  "status": "applied",
  "reason": null,
  "job_id": "job_abc123",
  "report_path": "/path/to/job/report.md",
  "rung": "kimi",
  "model": "moonshot/kimi-k2.7-code",
  "cost": 0.012,
  "gates": [
    "G0 scope: passed",
    "G1 secrets: passed",
    "G2 test_weakening: passed",
    "G3 risky edits: passed",
    "G4 build: passed",
    "G5 tests: passed",
    "G6 diff: passed",
    "G7 review: passed"
  ],
  "files_changed": ["mullet/tools/opencode_tool.py"],
  "notes": []
}
```

Possible `status` values:

- `applied` - the node passed and its diff was applied to the caller's working tree.
- `passed_not_applied` - the node passed but the diff could not be applied; `reason` explains why.
- `failed` - the node failed its gates.
- `blocked` - the job could not start or was blocked before running.
- `unverified` - the node finished but some gates could not verify the change.
- `timeout` - the run exceeded the timeout.

When the status is anything other than `applied` or `passed_not_applied`, no changes are written to the caller's tree.

### Workflow example

```python
# Delegate the task - mullet_opencode runs it as a gated job.
result = mcp__mullet__mullet_opencode(
    task="Add Google-style docstrings to all public functions in mullet/tools/opencode_tool.py",
)

# Inspect the result.
print(result["status"])
print(result.get("files_changed"))
if result.get("reason"):
    print(result["reason"])

# The changes are already in the working tree if status is 'applied'.
# Review with git diff, then commit yourself:
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

opencode requires models to be declared in the `provider.<id>.models` dict in `opencode.json`. If you specify e.g. `"moonshot/kimi-k2.7-code"` or `"lmstudio/ornith-1.5-35b-a3b"` but the config only has a top-level `"model": "..."` with no matching `models` block, opencode will reject it. This only bites when opencode's own CLI actually receives the model — Mullet's automatic local dispatch (the lightweight agent path) bypasses opencode entirely and never hits it.

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

The key must be in the environment, or in `~/.config/mullet/.env` (`$MULLET_CONFIG_DIR/.env` if that is set), before the Mullet MCP server starts. Restart Claude Code or the MCP server after adding it:

```bash
echo 'KIMI_API_KEY=sk-...' >> ~/.config/mullet/.env
# restart Claude Code or the MCP server
```

A key in a project's `.env` or `.mcp.json` is not read.

### Fallback: mullet_run used instead of opencode

If the result contains `"fallback": "mullet_run/kimi"`, opencode is not on `PATH`. Install it:

```bash
npm i -g opencode-ai@latest
which opencode   # should now return a path
```

Then retry the `mullet_opencode` call.
