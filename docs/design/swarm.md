# Kimi Agent Swarm — Design Reference

> Status: **IMPLEMENTED** — v1.6.0+  
> Source: `mullet/core/orchestration/`  
> Interface: the `mullet_swarm` MCP tool (`goal`, `files`, `items`, `concurrency`, `output_shape`, `dry_run`, `apply`, `target_dir`)

---

## 1. Overview

Swarm is **horizontal scaling on rung 2 of the delegation ladder**. It decomposes a high-level goal into independent sub-tasks, dispatches them concurrently to Kimi (and optionally other providers), and synthesizes the results into a single coherent output.

**Use cases:**
- Parallel code review (N files → N review tasks → merged report)
- Multi-file refactor or bulk test generation
- Codebase audit (security, style, coverage — run simultaneously)

---

## 2. Module Layout

```
mullet/core/orchestration/
  swarm.py              # Top-level orchestrator: run_swarm()
  swarm_dispatcher.py   # Async fan-out with semaphore + retry
  swarm_types.py        # SwarmTask, SwarmContext, TaskResult, SwarmResult
  static_decomposer.py  # 1:1 task creation from explicit file/item lists
  llm_decomposer.py     # Kimi/Anthropic-based natural-language decomposition
  synthesizer.py        # Final merge call (Kimi → Haiku → Sonnet fallback)

mullet/core/router/
  auto_swarm.py         # Heuristic: should a single task be fanned out?

mullet/core/providers/
  kimi_async.py         # asyncio.to_thread facade over KimiProvider

mullet/server/mcp_handlers.py
  handle_mullet_swarm()  # MCP entry point
```

---

## 3. Pipeline

```
Goal + Context
    │
    ▼
Decomposer (Static or LLM)
    │
    ▼
SwarmDispatcher ──► KimiAsyncProvider ──► KimiProvider
    │                    (asyncio.to_thread)
    ▼
TaskResult[]
    │
    ├──► Cross-task retry (CRITICAL tier) for timeout / rate_limit
    │
    ▼
Synthesizer ──► KimiAsyncProvider (or Anthropic fallback)
    │
    ▼
SwarmResult + optional JSON report
```

### 3.1 Decomposition

| Decomposer | When Used | LLM Call? |
|------------|-----------|-----------|
| **StaticDecomposer** | Explicit `--files` or `--items` provided | No |
| **LLMDecomposer** | Natural-language goal without enumeration | Yes (one call) |

- Auto-selection: Static when `context.files` or `context.items` are present; LLM otherwise.
- LLM decomposer returns a JSON array of `{id, instruction, expected_output_shape, context_refs}`.
- Empty array = "not swarmable"; caller falls back to single-provider dispatch.
- Results are **deduplicated** by normalized instruction text.
- Diversity gate: < 2 unique tasks triggers a warning and caller fallback.

### 3.2 Dispatch

- **Concurrency cap:** `asyncio.Semaphore` (default 20, tunable).
- **Per-task timeout:** 120 s (configurable).
- **Retry:** 1 retry for `timeout` and `rate_limit` errors; semaphore released during backoff.
- **Cancellation:** `asyncio.CancelledError` propagates cleanly; partial results preserved.
- **Never raises:** All failures become `TaskResult(success=False, error=...)`.
- **Model selection:** Delegation-safety D1 — `estimate_tokens()` + `model_registry` picks a tier-appropriate model with sufficient context window.
- **Provider restriction:** `allowed_providers` frozenset can constrain routing. Swarm path requires Kimi today.

### 3.3 Synthesis

- One final LLM call merges N `TaskResult`s into the requested `output_shape`.
- Failed tasks are surfaced explicitly; no hallucinated content for failures.
- **Fallback:** If synthesis fails, raw concatenated results are returned.
- **Provider fallback chain:** kimi → haiku → sonnet (respects `allowed_providers`).

### 3.4 Cross-Task Retry

Transient failures (`timeout`, `rate_limit`) are automatically retried at **CRITICAL tier** once before giving up.

---

## 4. MCP Surface (`mullet_swarm`)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `goal` | string | ✅ | High-level objective for the swarm |
| `files` | string[] | — | File paths to include as context (glob patterns expanded by caller) → StaticDecomposer |
| `items` | string[] | — | Arbitrary string items to process (alternative to `files`) → StaticDecomposer |
| `concurrency` | integer | — | Max parallel Kimi calls (default 20) |
| `output_shape` | string | — | Desired synthesis format (default: "markdown report") |
| `dry_run` | boolean | — | Decompose only — return task plan without dispatching |
| `apply` | boolean | — | If true, apply EDIT/FILE/DELETE blocks from each sub-task directly to disk |
| `target_dir` | string | — | Directory to resolve file paths against when applying changes (defaults to cwd) |

---

## 5. Auto-Swarm Routing

`mullet/core/router/auto_swarm.py` decides whether a routine `mullet_run` task should be fanned out instead of sent as a single call.

**Signals (any one triggers swarm):**
- ≥ 5 files in scope
- `tier=CRITICAL`
- Effective prompt length ≥ 8 000 chars
- Multi-language file set (≥ 2 extensions)
- ≥ 3 documentation files
- Task text mentions ≥ 5 file-path-like tokens (caller forgot `files=[...]`)

The heuristic is intentionally conservative: false-negatives are cheaper than false-positives.

---

## 6. Observability

### Per-run output

`SwarmResult` carries:
- `tasks`, `results`, `synthesis`
- `partial_failure`, `synthesis_fallback`
- `total_tokens` (input + output)
- `wall_clock_seconds`
- `timestamp` (ISO-8601)
- Dry-run: `estimated_input_tokens`, `estimated_cost_usd`

### JSON report

Optional `report_path` argument writes a structured JSON file containing the full result, task list, and per-task outcomes. The MCP handler does not auto-create a `.mullet/swarms/` directory; callers may supply a path.

### Progress callback

`on_progress(result, completed, total)` is called after every task finishes. The MCP handler prints a simple `[N/M] ✓ task-XXX` line to stderr.

---

## 7. Failure Modes

| Scenario | Behaviour |
|----------|-----------|
| Partial failure (k of N fail) | Synthesize with succeeded results; failures listed in output. `partial_failure=True`. |
| All tasks fail | Empty synthesis; `SwarmResult` carries all errors. |
| Synthesis fails | Raw concatenated results returned; `synthesis_fallback=True`. |
| Rate limit mid-swarm | Per-task retry with backoff; semaphore throttles naturally. |
| Decomposer returns empty | Log warning; caller should fall back to single call. |
| Decomposer returns bad JSON | `DecompositionError` raised; no dispatch. |
| Ctrl-C | `CancelledError` propagates; partial results available to caller. |
| No synthesizer provider available under allowlist | Raw concatenation fallback. |

---

## 8. Test Coverage

18+ test files across unit, integration, and E2E layers:

| Area | Test Files |
|------|------------|
| StaticDecomposer | `test_static_decomposer.py` |
| LLMDecomposer | `test_llm_decomposer.py` |
| SwarmDispatcher | `test_swarm_dispatcher.py` |
| Synthesizer | `test_synthesizer.py`, `test_synthesizer_extras.py` |
| Orchestrator | `test_swarm_orchestrator.py` |
| Auto-swarm | `test_auto_swarm.py`, `test_auto_swarm_extras.py` |
| Dedup / context | `test_swarm_dedup.py`, `test_swarm_context_validation.py` |
| Token estimation | `test_token_estimation_dedup.py` |
| Integration | `tests/integration/test_e2e_swarm.py` |
| E2E (live Kimi) | `tests/test_swarm_e2e.py` |
| Delegation safety | `test_delegation_safety_regression.py`, `test_delegation_safety_d1.py`, `test_delegation_safety_d3.py` |

---

## 9. Planned / Future

- **Native async provider:** Replace `asyncio.to_thread` with `httpx.AsyncClient` when concurrency regularly exceeds 50.
- **HUD badge:** Real-time swarm progress in the terminal HUD (`Swarm: 12/20`).
- **YAML spec files:** `--spec task.yaml` for complex multi-step swarm definitions.
- **Multi-provider swarm:** Currently Kimi-only; abstracting `KimiAsyncProvider` to support other async providers.
- **progress.txt integration:** Automatic per-run line appended to `.mullet/progress.txt`.

---

## 10. What This Is NOT

- **Not a replacement for `team.py`** — team runs a staged agent pipeline (plan → PRD → exec → verify → fix). Swarm fans out directly to LLM providers with no agent runner.
- **Not a replacement for `ralph.py`** — ralph is a vertical persistence loop. Swarm is horizontal fan-out.
- **Not for shared mutable state** — tasks are embarrassingly parallel. Cross-file consistency refactors belong in `team.py`.
