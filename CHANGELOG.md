# Changelog

All notable changes to Mullet are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semver](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Licence: the Additional Use Grant now allows production use for any purpose, including
  inside a company or team, on any number of machines, and to build software you sell or
  ship. Offering Mullet, or a service whose value comes substantially from it, to others
  as a hosted, managed or embedded service still needs a commercial licence.
  Contributions need a DCO sign-off (`git commit -s`) and grant the Licensor the right to
  relicense them.

### Fixed
- The file tools accept paths in the user's temp directory, which on macOS is under
  `/var/folders` rather than `/tmp`.
- `@file:` references check a file's size before guessing whether it is binary, so an
  oversized file is always reported as too large.
- Tests that failed only on macOS or without bubblewrap: short key-proxy socket paths
  (macOS allows 104 bytes), a mocked sandbox check, and a spinner test that waits for
  its frame.

## [2.0.0] - 2026-10-25

### Added
- **Gated jobs.** `mullet job plan|show|approve|run|status|tail|stop|resume|report|list|discard|merge`
  runs a large coding job unattended. A planner proposes a task tree, you approve it, and
  each step runs in a private clone. A step passes only when gates G0 to G7 pass. You get a
  branch and an evidence report; `mullet job merge` copies the branch into your repo and
  never pushes.
- **Sandbox.** Workers and gate commands run in bubblewrap on Linux, with no network, a
  hidden home directory and an allowlisted environment. The clone's git directory is out of
  the worker's reach. If the sandbox can't start, the job refuses to run. Jobs refuse to
  start on macOS for now.
- **SQLite ledger and report.** A step's status is set only from gate results. The report
  is built from the ledger, and anything a worker says about itself is labelled as a claim.
- **Claude Code worker.** It runs your own `claude -p`. `ANTHROPIC_API_KEY` is left out
  unless a job is planned with `--billing api`. A usage limit pauses the job until the
  reset.
- **Notifications** when a job blocks, pauses or finishes: desktop, and Slack in working
  hours. Set with `MULLET_NOTIFY`, `MULLET_SLACK_WEBHOOK`, `MULLET_NOTIFY_HOURS` and
  `MULLET_NOTIFY_TZ`.
- **Portable install.** `scripts/install.sh` installs from GitHub with uv or pipx. Config
  goes in `~/.config/mullet` and state, data and caches follow the XDG directories.
  `mullet paths` prints them and `mullet migrate-state` moves an older `~/.mullet`.
- **`mullet setup-claude`** records what it changes, and `--uninstall` undoes exactly that.
  Hooks are opt-in, run with `mullet hook <name>`. `mullet doctor` checks Claude Code, the
  sandbox and a stray `ANTHROPIC_API_KEY`.
- **opencode worker and the delegation ladder.** With a `ladder:` section in the config
  and opencode allowed by a repo's profile, each step goes up a ladder of workers, one
  attempt per rung: an OpenRouter `:free` model (verified-public GitHub repos only), a
  model in LM Studio, Kimi on Moonshot's API, paid OpenRouter models, and Claude. The
  order follows the step's kind and Claude's paced weekly usage; a spending cap covers the
  paid rungs. Workers reach their provider through a key proxy pinned to one model, and
  paid OpenRouter requests always ask for hosts that keep no prompts and, by default,
  serve no 4-bit copies (`ladder.routing` sets more per model). `mullet job ladder`,
  `mullet job retry <job> <step> --rung <rung>` and `--ladder/--no-ladder/--max-spend` on
  `mullet job plan`; the report shows each attempt's rung, model and cost.
- **`mullet_opencode` is a gated single-step ladder job.** Its change reaches your working
  tree only after the gates pass, with `git apply --3way` on clean, unchanged files.
- **Night mode.** `mullet night on [--until HH:MM] | off | status`: local attempts get
  longer, one more try below Claude and a wait for machine headroom; it ends at its time
  or when you come back. `ladder.local_by_day: false` keeps the local rung for night mode.
- **`mullet local`** keeps LM Studio usable: `status`, `restart`, `check --fix`, and
  `watchdog --install` for a two-minute systemd timer that restarts a stuck server.
- **`mullet reap`** stops idle or orphaned Mullet work: pytest-xdist workers, opencode runs
  of finished jobs, idle LM Studio models, orphaned `mullet mcp` servers and stale
  `/tmp/mullet-*` directories. It matches exact command lines and parentage only, and
  `--dry-run` lists what it would do. `--install-timer` adds a three-minute systemd timer
  that only reports for its first 24 hours.
- **`mullet ladder report [--since YYYY-MM-DD | --days N] [--json]`** adds up every job's
  attempts by rung, step kind and model from the ledger alone: passes, escalations,
  provider errors, dollars, tokens and time, an estimate of the Claude work avoided, and a
  short "what to change" list when there is enough data.
- **Machine guard.** One GPU lease across Mullet's processes and limits on VRAM, RAM, GPU
  temperature and load; reindexing waits for headroom, and one full test suite runs at a
  time across jobs.
- Not in 2.0.0: a macOS sandbox. Until then jobs refuse to start on macOS.

### Changed
- Claude workers run one at a time across jobs, as they share a subscription; opencode
  workers have their own slots (`MULLET_API_WORKERS`, default 3).
- Jobs on the same repo, base commit and profile share one G0 baseline for a day.
- The test suite runs in parallel (about 4.5 minutes) and every test
  is cut off from the network, provider keys and the real home directory.
- One loader reads `.env`, from the config directory. Keys in the working directory's
  `.env` are no longer picked up.
- Reindex, evaluate and memory work from an installed copy, not only from a checkout.
- The public README describes jobs first.

### Removed
- `ultrawork`, `ralph`, `autopilot`, `team` and `eco`, and the overnight loop.
- Substring verifiers that read PASS or APPROVAL out of a model's reply.
- The `gemini_cli` provider.
- Unsafe autonomous paths, including automatic commits and pushes to `origin master`.

### Security
- The HUD no longer reads Claude Code's OAuth token. Usage figures come from the rate
  limits Claude Code passes to the statusline command.
- Delegates and opencode no longer receive keys from the current directory's `.env`.
- Reviews before each phase closed gate bypasses (skipped tests counted as passing, a
  committed `.gitattributes` hiding changes from the diff) and a prompt-injection route
  into the G7 reviewer, which now sees the diff fenced with a random marker.

The entries below record the changes since 1.9.0 in the order they were made.

### Added (2026-10-20: switch delegation off)
- **`MULLET_DELEGATION=off`** (environment or `.env`; default `on`) stops Mullet sending
  work to any model provider. `mullet_run`, `mullet_delegate`, `mullet_swarm`,
  `mullet_opencode`, CLI task runs, review, watch, the orchestration modes, the REPL,
  `live-test` and `mullet serve`'s chat endpoints reply that delegation is turned off.
  Trawl, knowledge pools, status, budget and the other read-only tools are unaffected, and
  the delegation hooks stay quiet while it is off.

### Fixed (2026-10-20: withdrawn OpenRouter models)
- **Removed `qwen/qwen3-coder:free` and `tencent/hy3:free`**, which OpenRouter no longer
  offers; LOW-tier work goes to the paid ladder until a vetted free model is added.
- **OpenRouter fallback on a missing model.** A "model unavailable" reply (404, "No endpoints
  found", "not a valid model ID") no longer fails the task: the provider records the model in
  `~/.mullet/unavailable_models.json` for a day, the routers skip it, and the request goes
  to the next candidate.

### Changed (2026-10-20: personal values in local settings)
- **`MULLET_NO_AI_ATTRIBUTION`** (default off) decides whether `mullet init`,
  `mullet setup-claude` and `mullet autopilot init` write a "no Co-Authored-By or AI
  attribution" rule into projects. **`MULLET_OPENROUTER_REFERER`** (default unset) replaces
  the hard-coded OpenRouter `HTTP-Referer`. See `.env.example`.

### Changed (2026-09-17 — delegation gated on real budget pressure)
- **Delegation is now a last resort rather than the default.** The `UserPromptSubmit`
  reminder (`scripts/hooks/delegation_reminder.py`) previously fired on prompt shape alone:
  any prompt matching "plan", "research", "analyse", "review", "mullet" and friends, or
  simply longer than 400 characters, produced a nudge toward a paid provider regardless of
  how much Anthropic allowance was left. The common case was being told to spend real money
  to protect an allowance that was barely touched. In-context work is already covered by the
  subscription; a third-party token is not.
- **New `scripts/hooks/_delegation_gate.py`** holds the thresholds in one place and both
  delegation hooks check it before anything else: paced 5-hour pressure >= 0.80, paced weekly
  pressure >= 0.80, raw weekly utilisation >= 85%, or a `PAUSE` status. Below all of those the
  reminder prints nothing. It fails closed, so an unreadable usage figure means silence rather
  than a guess. `check_budget()` reads a file cache with a 5-minute TTL, so the gate costs
  nothing on the hook path.
- **Every reminder template now leads with the reason, points at the free routes first**
  (a Claude Code subagent, the local LLM, trawl search) **and requires an explicit go-ahead**
  before anything reaches a paid provider. The escalation stages and their countdown survive.
- **`manual_tool_guard.py`'s hard block is gated the same way** and returns 0 while the budget
  is comfortable. It is not currently registered in `settings.json`, but left ungated it would
  have overridden this policy the moment it was.

### Added (2026-09-17 — delegation gate tests)
- **`tests/test_delegation_budget_gate.py`** covers both threshold edges, the fail-closed path,
  raw weekly danger with calm pacing, `PAUSE`, the ask-first requirement, and that the hook
  never blocks a prompt. The nine existing tests in `tests/test_manual_tool_guard.py` that
  encoded the old always-fire behaviour now stub the gate to "under pressure", since what they
  actually check is message content.


### Removed (2026-07-10 — harness layer and frontend retirement)
- **Removed the `mullet/harness/` agent-runtime layer entirely** (118 files) along with everything built on it: the interactive TUI (`mullet shell`/`mullet cli`), the web dashboard (`mullet/server/web/`, `mullet web`), the ACP/Zed integration (`mullet acp`), and ~15 CLI subcommands that had no independent implementation outside harness (`status`, `provider`, `search`, `swarm`, `skill`, `cron`, `demo`, `pr`, `benchmark`, `config` [harness variant], `metrics`, `showcase`, `new`, `share`, `harness-init`). `mullet config` (get/set/list/edit/reset/budget) is unaffected. Mullet is now MCP-server-first: `mullet mcp`/`mullet-mcp` (stdio MCP server) and `mullet serve` (OpenAI-compatible REST+WebSocket API) are the two supported entry points. All MCP tools (`mullet_run`, `mullet_opencode`, `mullet_swarm`, `mullet_status`, etc.) are unaffected — they always routed through `mullet/core/*` directly, never through harness.
- **Deleted the unused `ui/` Next.js admin frontend** — had real started work (auth flow, terminal grid, admin/settings pages) but was never wired up or used.

### Fixed (2026-07-10 — provider routing and swarm dispatch)
- **Fixed a `claude_max` provider-allowlist bug** — `providers="haiku"`/`"sonnet"` allowlist restrictions in `mullet_run`/`mullet_delegate` always failed before this fix.
- **Wired OpenRouter into the direct-dispatch provider resolution path** (`providers="openrouter"` now works in `mullet_run`, previously had no candidate at all in that path).
- **Fixed auto-swarm silently reporting success when decomposition produced zero tasks** — now falls back to single-provider dispatch instead of returning an empty "success".

### Added (2026-07-10 — free-tier OpenRouter routing)
- **Added free-tier OpenRouter routing for LOW-complexity tasks** — `qwen/qwen3-coder:free` and `tencent/hy3:free` are tried ahead of the paid ladder for simple/low-complexity requests, at zero cost.

### Changed (2026-07-10 — value-picker maintenance)
- **Deduped value-picker candidate lists** into a shared `value_picker_constants.py` module (previously hardcoded identically in two files, a silent-drift risk).

### Added (2026-07-05 — routing-v3 Phase 0: quality scoring research)
- **New `mullet/core/providers/quality.py`** — per-model domain quality scores
  (`coding`/`reasoning`/`general`, 0-100 each) for the 23-model candidate set
  reachable via the curated provider ladders, each with a cited benchmark
  source and a `confidence` level (`direct`/`inherited`/`estimated`). First
  data-gathering phase of the routing-v3 value-based picker (replaces the
  single-hardcoded-pick-per-tier approach with cost x complexity x quality
  selection across the full candidate set — see
  `docs/design/routing-v3-value-picker.md`). 8 schema/sanity tests in
  `tests/test_quality_scores.py`.

### Added (2026-07-05 — routing-v3 Phase 1: quality-floor mapping)
- **New `mullet/core/providers/quality_floors.py`** — `TIER_QUALITY_FLOORS` (minimum
  domain score per complexity tier), `meets_quality_floor()`, `meets_reasoning_gate()`.
  `QualityScore` gained an `is_reasoning_tuned` bool field (hard capability gate,
  separate from the numeric reasoning score). 11 new tests in `tests/test_quality_floors.py`.

### Added (2026-07-06 — routing-v3 Phase 2: the value picker)
- **New `mullet/core/providers/value_picker.py`** — `pick_by_value()`: filters a
  candidate set by context fit → reasoning gate → quality floor, then picks the
  cheapest survivor. Tier escalation capped at one level with a logged warning
  (unbounded escalation could silently promote a LOW task to CRITICAL pricing).
  `domain` is a required parameter, no default (a `"coding"` default would
  silently mis-score non-coding tasks). Zero/missing pricing sorts last in the
  cost tie-break, never first. 10 tests in `tests/test_value_picker.py`.

### Changed (2026-07-06 — routing-v3 Phase 3a: value picker wired into Gemini routing)
- **`_best_gemini`/`select_gemini_model` now use `pick_by_value`** for their
  non-live-cache pick, falling back byte-identically to the static ladder on
  any exception or `None` result. Domain is `"reasoning"` when
  `prefer_reasoning`/`needs_reasoning`, else `"coding"` (no `"general"` domain
  — `PreflightResult` doesn't carry lexical signals without much bigger
  cross-codebase plumbing). Validated against all 928 real historical Gemini
  routing decisions in `.mullet/routing_history.jsonl` — zero observed
  behavior change. Kimi/Anthropic/OpenRouter not yet wired (single-provider
  validation pass by design). 9 new tests in
  `tests/test_gemini_value_picker_integration.py`.

### Fixed (2026-06-21 — routing-v2 Phase F: swarm polish)
- **Swarm provider pick is now headroom-aware** — `SwarmDispatcher._pick_task_provider` computed
  with a hardcoded `pressure=0.0`; new `_live_pressure()` helper feeds real `compute_pressure()`
  (fail-open to 0.0) so swarm tasks exploit prepaid Anthropic headroom like the main router.
- **Removed deprecated `asyncio.coroutine`** from the swarm tests (removed in Python 3.11+) —
  replaced with real `async def` coroutines. Suite passes under `-W error::DeprecationWarning`.

### Changed (2026-06-21 — routing-v2 Phase E: cost-class corrections)
- **Gemini is no longer treated as low-cost** — demoted to last-resort in both router allowlist
  orderings (`_ALLOWLIST_FALLBACK_ORDER`, `_ALLOWLIST_TIER_PICK_ORDER`) and the swarm preference;
  it no longer fills the `cheap_cloud` alias (only kimi/openrouter do). `openrouter` added to the
  router orderings (was missing). Matches the standing "Gemini = last resort" policy.
- **Local LLMs are now a candidate at every tier** — `PROVIDER_TIER_CAPABILITIES["local"]` expanded
  to `{LOW, STANDARD, ADVANCED, CRITICAL}`, so a `local-only` preset can serve ADVANCED/CRITICAL work
  (routes to local instead of blocking). Free fallback is always reachable.
- **Anthropic (Claude Code/Max) reachable at every tier, free-first** — verified: at low pressure
  LOW→local (free, not wasted on trivial), STANDARD→haiku, ADVANCED→sonnet; under pressure it drops
  off Anthropic. 436 router/swarm tests green; 7 reconciled to the new cost model.

### Added (2026-06-21 — routing-v2 Phase D: fan-out regression benchmark)
- **`tests/test_routing_fanout.py`** — a regression sentinel guarding against the router collapsing
  to single-provider dispatch. Asserts per-preset fan-out (balanced-mix routes a mixed batch to ≥2
  providers; local-only→local), headroom exploitation via `pick_provider_for_tier` (low-pressure
  ADVANCED→sonnet, STANDARD→haiku; cheap-cloud excludes Anthropic; high-pressure drops Anthropic),
  the reasoning-task tier floor, and that no task is left blocked. Surfaced a residual (cheap-cloud
  collapses to kimi within its single cost tier) is still open.

### Added (2026-06-21 — routing-v2 Phase C2: multi-provider swarm)
- **Swarm dispatch now routes across providers, not just Kimi** — `SwarmDispatcher` dropped its
  hard kimi-only guard. `SWARM_CAPABLE_PROVIDERS` (kimi, gemini, openrouter, anthropic/haiku/sonnet/
  opus, local) are dispatched via the existing duck-typed `KimiAsyncProvider` wrapper (no new async
  classes). Each sub-task picks its provider per tier via `pick_provider_for_tier` (Phase B,
  pressure-aware) ∩ allowlist, falling back to a static kimi-first order; `__init__` requires ≥1
  swarm-capable provider in the allowlist. Kimi remains the default (backward compatible).
- **Fix (review-found):** `_pick_task_provider` unpacked `build_default_router()` wrong (it returns
  a `(Router, ProviderResolver)` tuple), so the Phase B pressure/tier-aware selection silently fell
  through to the static order. Fixed + added a regression test that exercises the real selection path.

### Fixed (2026-06-21 — routing-v2 Phase C1: per-sub-task overrides)
- **Ultrawork no longer collapses per-sub-task routing to one provider** — `ultrawork` routed each
  sub-task via `route_delegation()` but then discarded all but the first task's `_provider_override`
  and applied it batch-wide. Now each sub-task keeps its own override; `delegate_batch` reads
  `task["_provider_override"]` per task (closure-safe submit-time kwarg) and limits concurrency to 1
  if *any* task routes local. Heterogeneous batches now fan out across providers. (`delegation.py`,
  `ultrawork.py`; new `tests/test_delegation_per_task_override.py`.)

### Changed (2026-06-21 — routing-v2 Phase B: within-tier provider pick)
- **Budget pressure now biases *within* a tier+allowlist instead of bypassing tier routing** —
  new `Router.pick_provider_for_tier(tier, allowed, pressure)` is a thin wrapper over
  `recommend_provider` constrained to the preset allowlist and `PROVIDER_TIER_CAPABILITIES`; it
  preserves the headroom pick (Haiku@STANDARD<0.6, Sonnet@ADVANCED<0.3) and falls back to the
  best tier-capable allowed provider. The budget-pressure branch now uses it and **falls through**
  to tier_map/quality/fallback when nothing fits (no more premature null decision).
- **Reasoning-task tier floor** — `is_reasoning_task(prompt)` + LOW tier now floors to STANDARD,
  so genuine analysis/critique/review work routes to at least cheap-cloud/Haiku (and Anthropic at
  low pressure), not just local. Trivial keyword-bearing prompts ("list the status") stay LOW.
- **Capability table fix** — added `opus-4-8` (`claude-only` allows it) and `openrouter` to
  `PROVIDER_TIER_CAPABILITIES` (were silently dropped). Verified: balanced-mix routes STANDARD→haiku
  and reasoning→sonnet at low pressure; cheap-cloud→kimi; local-only→local. 363 router tests green.

### Changed (2026-06-21 — routing-v2 Phase A: complexity scoring)
- **Reasoning/prose tasks no longer under-scored to LOW** — `score_complexity` now adds `+15` for
  reasoning keywords (analyse/review/critique/evaluate/assess/audit/…) and `+10` for long plain-prose
  (120–200 word band, no code/file-paths, de-overlapped from the existing >200/>500 word_count signals).
  `extract_signals` populates a new `LexicalSignals.has_reasoning_keywords` (default False, backward
  compatible); `confidence_breakdown` kept in lockstep. A full plan-critique prompt now scores STANDARD
  (was 0.0→LOW). First phase of the within-preset multi-provider routing work (`docs/design/routing-v2.md`).
  236 complexity tests green.

### Added (2026-06-21 — routing simulate)
- **`mullet routing simulate`** — replay historical routing outcomes (`.mullet/routing_history.jsonl`) through the CURRENT router policy + active preset and diff how each prompt *would* route now vs what actually happened. Flags: `--limit N`, `--provider P` (filter; over-fetches 5× then trims to limit), `--changed-only`, `--format text|json`, `--no-preset` (ignore the active preset's allowlist). Text shows `WAS provider/model/tier → NOW …` rows + a summary footer (`N replayed, M changed (X%), K unchanged`) + per-transition counts; JSON emits `{summary, rows}` with the routing reasons. Routes on the stored `task_preview` (≤120 chars), so complexity scoring is approximate; `none` in the NOW column = no eligible provider under current policy. 15 new tests. (`mullet/cli/routing_cmd.py`)

### Fixed (2026-06-21 — local routing)
- **`routing_matrix.select_local_model` deprioritises thinking models for code** — when a task is not `needs_reasoning`, `reasoning`-tagged local models (e.g. `*-thinking`, `r1`) are dropped from the candidate set if any non-reasoning model qualifies, so they're no longer selected for plain code tasks (they over-think and truncate). Reasoning tasks are unaffected.
- **`_local_param_count` known-model map** — `deepseek-coder-v2-lite` (16B MoE, no size token in its name) now resolves to 16.0 instead of the 7.0 default, so it ranks correctly in tier selection.

### Added (2026-06-21 — Provider preset system)
- **Provider presets** (`mullet/core/presets.py`) — a named bundle of `{allowed_providers, speed_mode, opencode_model, native_only}` persisted to `~/.config/mullet/presets.yaml`, read by BOTH the MCP delegation path (`mullet_run`/`mullet_delegate`/`mullet_opencode`) and the harness engine. Four built-ins: `balanced-mix` (default), `local-only`, `claude-only`, `cheap-cloud`. Built-ins are always sourced from code (self-healing on load); user presets persist.
- **`mullet preset` CLI** — `list|use|show|add|remove` (`mullet/cli/preset_cli.py`).
- **`/preset` slash command** — switch the active preset inside `mullet shell`; applies the allowlist + speed_mode to the live session.
- **Preset-aware routing** — when a caller omits `providers`, MCP delegation defaults to the active preset's allowlist and threads its `speed_mode` (closing the previously hardcoded-`balanced` gap). `mullet_opencode` uses `preset.opencode_model`.
- **`claude-only` native mode** — `native_only=true` makes `mullet_run`/`mullet_delegate`/`mullet_opencode` refuse delegation and defer to Claude Code natively; explicit caller overrides still win. Hard invariant: no preset's `opencode_model` may be an Anthropic model.
- **Session-start prompting** — `scripts/hooks/preset_prompt.py` (SessionStart hook, registered in `.claude/settings.json`) reports the active preset and nudges Claude Code to ask the user; `mullet shell` shows a TTY-gated keep/switch picker at startup.
- Docs: README "Provider Presets" + `docs/HARNESS.md` §5.5. ~80 new tests.

### Added (2026-06-17 to 2026-06-19 — coverage sweep)
- **Test suite: 12,700 → 22,000+ tests** — systematic gap-filling across all subsystems. Harness layer brought to near-100% coverage across all modules. Knowledge layer (graph, engine, sources) taken from 0% to 100%. Core layer and CLI completed in earlier passes.
- **`mullet/knowledge/` coverage sweep** — 14 new test files covering: `reference_config.py`, `sources/base.py`, `sources/github_models.py`, `sources/github_client.py` (98%), `sources/github_connector.py`, `sources/web_cache.py`, `sources/web_crawler.py`, `graph/export.py`, `graph/extractor.py` (95%+), `graph/search.py`, `engine/prompts.py`, `engine/evaluation.py`, `engine/code_extractor.py`, `library_pool.py` — all 0%→100% (or near).
- **`mullet/knowledge/sources/web_cache.py` bug fix** — `UnboundLocalError` in `save()` cleanup handler when `os.makedirs` raised before `tmp_path` assignment; moved assignment before the try block.
- **Harness layer coverage** — all `harness/commands/*`, `harness/engine/*`, `harness/mcp_client/*`, `harness/coordinator/*`, `harness/tui/*`, `harness/sessions/*`, `harness/profiles/*`, `harness/cron/*` at 97–100% coverage.
- **Server layer coverage** — `server/app.py` (100%), `server/db.py` (100%), `server/web/app.py` (94%), `server/api.py` (83%), `server/mcp.py` (99%), `server/mcp_handlers.py` (89%), `server/acp_server.py` (94%).
- **Core layer coverage** — `core/router/routing_matrix.py` (100%), `core/agent/context_refs.py` (100%), all `core/tools/*` (94–100%), `hud/*` (100%).

### Fixed (2026-06-16 — opus-4-8 harness sweep)
- **`agents.yaml` sisyphus/oracle** — primary model and fallback chain updated to claude-opus-4-8 (was 4-7).
- **Dispatcher `_VALID_ASK_PROVIDERS`** — opus-4-8 added so `/ask opus-4-8` is accepted.
- **TUI `_try_failover` `_anthropic` set** — opus-4-8 added for correct Anthropic rate-limit detection.
- **`resolver.py` alias guard** — opus-4-8 included so `force_provider="opus-4-8"` routes through Anthropic credentials.
- **`openai_compat.py` `_CLOUD_ESCALATION_TARGETS`** — opus-4-8 added; escalation logic now recognises it as a cloud target.
- **`override_cmd.py` default** — `--provider` default changed from opus-4-7 to opus-4-8; docstring updated.
- **`main.py` throttle guards** — opus-4-8 included in auto-downgrade and THROTTLE detection sets.
- **`model_registry.py` Anthropic fallback** — ADVANCED tier static fallback → claude-opus-4-8 (was 4-7); CRITICAL was already 4-8.
- **README model examples** — kimi-k2.6 references updated to kimi-k2.7-code throughout.

### Added (2026-06-16)
- **`/doctor` routing quality section** — shows per-provider success rate from routing_history.jsonl; flags providers below 60% quality threshold; notes when sample count is insufficient for the filter to apply. 3 new lock-in tests.
- **`mullet doctor` routing quality summary** — CLI doctor (`mullet doctor`) now also shows per-provider success rates after the main checks table. Informational only; never fails the doctor run.
- **`mullet routing compare <p1> <p2>`** — side-by-side provider comparison showing success rate, avg/total cost, latency, and p50/p95 tails. Highlights the better value with ◀. 2 new tests.
- **`mullet routing reset [--yes]`** — clears all routing history with confirmation prompt.
- **`mullet routing purge-provider <name> [--yes]`** — removes history for a single provider while preserving others. 8 new tests.
- **`mullet routing export`** — exports routing_history.jsonl to JSON or CSV for external analysis. Supports `--format json|csv`, `-o FILE`, and `--limit N`. 4 new tests.

## [1.9.0] — 2026-06-16

### Fixed (2026-06-16 — full model default sweep)
- **OpenRouter default `deepseek-chat` → `deepseek-v4-flash`** — `model_registry.py` `PREFERRED_DEFAULTS["openrouter"]` and `_best_openrouter()` (LOW/STANDARD/ADVANCED fallbacks) updated; `init_project.py` new-project scaffold and migration template updated.
- **`kimi` / `kimi-k2` plain aliases** — `model_info.py` `_ALIASES` entries now resolve to `kimi-k2.7-code` (was k2.6). Backward-compat `kimi-k2.6` explicit alias preserved.
- **`"use opus"` hook routes to opus-4-8** — `user_prompt_detect_override.py` split into three patterns: generic "use opus" → claude-opus-4-8 (current best); explicit "use opus 4.7" → claude-opus-4-7; explicit "use opus 4.6" → claude-opus-4-6. Test renamed and updated.
- **Generic `"opus"` alias in mcp_handlers fallback chains → version-pinned `"opus-4-6"`** — prevents ambiguity; the waterfall is now explicit: opus-4-8 → opus-4-7 → opus-4-6.
- **`engine.py` opus-4-8 gaps** — added to `_claude_code_providers`, cost rate table, `get_context_limit()` large-cloud set, and `_is_claude` identity list.
- **`main.py` opus picker** — removed speed_mode conditional (premium→4-8, else→4-7); always uses claude-opus-4-8 since price parity makes 4-7 redundant. Kimi model catalog now lists k2.7-code before k2.6.
- **Docs** — `API.md`, `opencode_integration.md` (config example, HUD display, troubleshooting), `LOOP_AGREEMENT.md` (OpenRouter rung label "DeepSeek V3" → "DeepSeek V4 Flash"), `kimi.py` docstring (recommends k2.5 for non-coding, not k2.6 which is now same price as k2.7-code).

### Fixed (2026-06-16 — k2.7-code routing sweep)
- **Stale kimi-k2.6 routing defaults** — swarm_dispatcher `_FALLBACK`, resolver `cheap_cloud` default and kimi named-alias, `agents.yaml` sisyphus fallback chain, dispatcher `/think` help text, and `showcase_cli` benchmark fixture all pointed to k2.6 after the ADVANCED-tier upgrade. All updated to k2.7-code.
- **KimiProvider default model test** — expected k2.6; `_KIMI_DEFAULT_MODEL` had already been set to k2.7-code. Test updated to match.
- **`"kimi"` / `"kimi-k2"` alias pricing tests** — expected 0.0006/1K (old k2.6 price). k2.6 was repriced to 0.00095 earlier; test expectations updated to match current model_info.py.

### Added (2026-06-16)
- **Cost + latency percentiles in routing history** — `summary_by_provider()` now returns `p50_cost`, `p95_cost`, `p50_wall_ms`, `p95_wall_ms`, and `total_cost`. Section 2 of insights shows TotalCost; Section 5 shows cost and latency tail with ⚠ spike when p95 ≥ 5× p50. 3 new tests.
- **`mullet routing show` Total $ column** — shows cumulative spend per provider alongside avg_cost.
- **`mullet routing recent` filters** — `--provider <name>` and `--failed` flags narrow the output. Both filters compose with `--limit`. 2 new tests.

### Refactored (2026-06-16)
- **Insights Section 5 uses outcomes directly** — no longer calls `summary_by_provider()` for a second file read; computes cost/latency percentiles from the already-loaded `iter_recent` results. Consistent data across all 5 sections and fully mockable in tests.

### Fixed (2026-06-16)
- **Hardcoded dev path in opencode .env resolution** — `_resolve_forwarded_keys()` searched a hardcoded developer checkout path before `~/.mullet/.env`. Removed the hardcoded path; `~/.mullet/.env` and `$PWD/.env` cover all real installs.

### Changed (2026-06-16)
- **`mullet routing insights` Section 4** — Anti-recommendations thresholds now use `QUALITY_THRESHOLD`/`QUALITY_MIN_SAMPLES` imported from `decision.py` instead of hardcoded 50%/5. The "none" fallback message is also dynamic.
- **Kimi ADVANCED tier: K2.6 → K2.7-Code** — K2.7-Code is identically priced to K2.6 ($0.95/$4.00/1K) but has thinking always ON; strictly better for complex tasks. K2.5 stays for STANDARD and large-context. Now all non-trivial tiers (ADVANCED, CRITICAL, reasoning, max-speed, high-output) use K2.7-Code.

### Added (2026-06-15)
- **Quality-feedback routing** — `decision.py` skips providers with ≥3 samples and <60% success rate from `routing_history.jsonl`. Least-bad fallback when all candidates are below threshold. `_quality_history_path()` is patch-injectable for tests. 6 new tests.
- **Cost distribution tracking** — `cost_history.py` appends per-session USD to `~/.mullet/cost_history.jsonl` (JSONL, fsync). `get_percentiles(window_days=30)` returns p50/p95/mean/max/n. `mullet budget` displays 30-day distribution when ≥2 data points. 7 new tests.
- **OpenRouter free-tier 429 fast-fail** — `:free` model suffix detected in `_call_with_retry`; 429 errors bypass all retries and escalate immediately. 5xx still retries. 4 new tests.
- **Opencode delegation outcome recording** — `_provider_from_model()` maps model IDs to provider names; `opencode_run()` records `RoutingOutcome` on success and failure, tagged `["opencode"]`. Quality routing now has data from opencode calls.
- **OpenRouter live pricing refresh script** — `scripts/refresh_pricing.py` fetches live prices from OpenRouter API and updates `pricing.yaml` (>0.1% threshold, `--dry-run` flag). 5 new tests. 7 models refreshed on 2026-06-15.
- **Quality flag in `mullet routing show`** — providers below the 60% quality threshold (≥3 samples) display "⚠ skip" in the Flag column with a legend line.

### Changed (2026-06-15)
- **Kimi ADVANCED routing**: `select_kimi_model()` now routes ADVANCED complexity to `kimi-k2.6` (was `kimi-k2.5`). K2.5 stays for STANDARD, large-context, and default tasks.
- **`handle_mullet_opencode` model routing**: LOW tier uses `_resolve_configured_model()` (was hardcoded `qwen2.5-coder-14b`); STANDARD→`deepseek-v4-flash`; ADVANCED→`deepseek-v4-pro`; budget fallback→`kimi-k2.7-code`.
- **`opencode_run` retry fallback**: local→`deepseek-v4-flash` (was `deepseek-v3.2-exp`); openrouter→`kimi-k2.7-code` (was `kimi-k2.6`).

### Fixed (2026-06-15)
- **`test_premium_costs` regression** — `_get_rates("anthropic", "sonnet")` was returning Opus 4.1 pricing ($15/1M) instead of Sonnet ($3/1M) due to dict insertion-order matching after the June-13 model catalog added `anthropic/claude-opus-4.1` before `anthropic/claude-sonnet-4.5`. Fixed: combined model+provider lookup before provider-only.
- **MODEL_DATABASE price drift** — 37 models were missing from `pricing.yaml`; 9 had stale prices in `MODEL_DATABASE`. Fixed by `_apply_pricing_from_yaml()` override at module load + new lock-in test `test_pricing_model_info_parity.py`.

### Added
- **Pricing single-source-of-truth** — `mullet/core/providers/pricing.yaml` centralises all model pricing (31 models: Anthropic, Kimi, Gemini, OpenRouter). `pricing.py` loader exposes typed accessors; `model_registry.py` now reads from YAML instead of hardcoded dicts. 9 new tests.
- **Kimi K2.7-Code HighSpeed** — `kimi-k2.7-code-highspeed` variant registered; aliases `kimi-k2.7-speed`, `k2.7-speed`. `ModelConstraints.thinking_required` field added; enforced in `shape_request()` for K2.7 models.

### Changed
- `kimi.py` default model bumped to `kimi-k2.7-code`.

### Added
- **Gemini tier detection** — `probe_gemini` now reports the active tier (Free / Pro / Ultra) and the matching rate-limit window. Tier resolves from `GEMINI_TIER` / `MULLET_GEMINI_TIER` env var, then a Vertex-AI heuristic (`GOOGLE_GENAI_USE_VERTEXAI=1` + `GOOGLE_CLOUD_PROJECT`), defaulting to "free". Matches the v1.8.0 Claude Max ×1/×5 pattern.
- **Extension-less doc filenames** — `documents.filenames` config (e.g. `[Jenkinsfile, Dockerfile, Makefile]`) so pools can ingest files without a recognised extension. Wired through `ingest_directory`, `build_file_manifest`, and the four call sites in `scripts/reindex.py`. New `Client.get_document_filenames()` + `DEFAULT_DOCUMENT_FILENAMES` constant.
- **OpenRouter cloud-key forwarding** — `_resolve_forwarded_keys()` in `mullet/tools/opencode_tool.py` reads `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `QWEN_API_KEY` from `.env` files (`~/.mullet/.env`, `$PWD/.env`) and forwards them to the opencode subprocess. Lets `mullet_opencode` use any cloud model without polluting the MCP server's env.
- **2026 frontier model catalog** — `mullet/core/providers/model_info.py` registered with verified live pricing for `anthropic/claude-opus-4.1`, `anthropic/claude-sonnet-4.5`, `openai/gpt-5`, `google/gemini-2.5-pro`, `google/gemini-2.5-flash`, `x-ai/grok-4.20`, `deepseek/deepseek-v3.2-exp`, `deepseek/deepseek-r1`, `qwen/qwen3-coder`, `z-ai/glm-4.6`.

### Changed
- **Tier-aware OpenRouter routing** — `select_openrouter_model` rewritten. Premium mode (`speed_mode in {"max","premium"}`) routes CRITICAL → claude-opus-4.1; ADVANCED → claude-sonnet-4.5; reasoning → gpt-5. Default routing picks deepseek-v3.2-exp for standard/advanced code work, qwen3-coder for >100K-token contexts. Old picks (`meta-llama/llama-4-*`, `mistralai/devstral-small`, `deepseek/deepseek-chat` [V3]) retained in model_info.py as legacy entries.

### Fixed
- **`--rebuild` no longer wipes doc chunks** — `ingest_code(rebuild=True)` was deleting ALL chunks before re-adding code; now scoped to `where={"chunk_type": "code"}` so document chunks (README, Jenkinsfiles via the new `documents.filenames` support, etc.) survive a full rebuild.
- **`mullet budget --refresh/-r` flag was dead** — `cli/main.py` parsed the flag into a local then hard-coded `force_refresh=True`, so every call hit the cloud probes regardless. The flag is now plumbed through to `probe_all`; without it, cached balances (5-min TTL) are returned.
- **`scripts/onboard.py` ignored Anthropic-only setups** — `has_claude` was collected from the user but never used; the "no non-Anthropic provider" warning never fired for Claude-only setups. Now the warning runs (with tailored copy explaining the 5h Max quota) when Kimi / Gemini / LM Studio are all absent.
- **3 NameError landmines** — `mullet/cli/voice-record`, `swarm` slash command, and `git_ops.py` referenced names that would have raised at runtime. Surfaced by enabling ruff F821 on the whole tree.
- **3 shadowed tests in `test_ralph_integration.py`** — `TestStoryFormatting` was defined twice; pytest silently kept only the second class, so `test_next_story_returns_lowest_priority` / `test_next_story_skips_passed` / `test_next_story_none_when_all_pass` had not been running. First class renamed to `TestNextStory`. Surfaced by ruff F811.
- **`test_oldest_removed` had a missing assertion** — captured `ts2` but never asserted it survived. Added `assert ts2 in ids` so the `max_snapshots=2` retention claim is actually verified.

### Removed
- **All F-class ruff warnings cleared** — `mullet/` + `scripts/` (493 modules) is now F-clean (F401 unused-import, F541 empty f-string, F811 redefinition, F821 undefined-name, F841 unused-variable). Session start: 1201. End: 0. CI guard `tests/test_package_imports.py` smoke-imports every module on every run.
- **All E-class ruff warnings cleared** except `E501 line-too-long` (cosmetic, 3182 — mostly long URLs / dataclass defaults / error strings). Cleared: E401 (multi-imports), E701 (`if x: return y`), E702 (`a; b; c`), E731 + E741 + E402 (per-file ignores for tests/ and 11 entry-point modules).
- **Other ruff cleanups:** B007 (17 unused loop vars), B904 (10 exception-chain preserves), SIM114 (6 duplicate-branch merges), SIM118 (1× ``.keys()`` drop), UP015 (24 redundant ``'r'`` open-modes), UP024 (4 ``IOError`` aliases), UP037 (83 forward-ref quotes).
- **Public-def docstring sweep — now 100%.** Gaps 413 (session start) → 0. Mostly delegated to Kimi K2.6, DeepSeek V3.2-exp, GLM 4.6, and Qwen3-Coder via `mullet_opencode` to keep Anthropic quota free for orchestration; the final 40 were knocked out in-context.

## [1.8.0] - 2026-05-11

### Removed
- **Harness v1 retired** — `mullet/harness/` (129 Python files) deleted. `mullet shell` and all CLI entry points now use `mullet/harness/` exclusively.
- `mullet/cli/harness_cli.py` and `mullet/cli/harness_dry_run.py` deleted (v1-only shell runners).
- `tests/harness/` (991 tests, 114 files) moved to `tests/_archive/harness_v1/` — preserved for reference.

### Added
- **`/budget` shows spend caps** — `_budget_status()` now loads configured per-provider caps from `.mullet/budget.yaml` and appends `cap=$N` to each provider line.
- **Update checker** — `mullet/cli/update_checker.py` queries PyPI silently on `mullet doctor`; prints update hint to stderr if a newer version is available.
- **`HarnessV2Engine.arun()`** — convenience `async def arun(prompt, *, provider, model, session_id)` wrapper for swarm coordinator use.

### Fixed
- All CLI files (`benchmark_cli`, `cron_cli`, `demo_cli`, `doctor_harness`, `help_cli`, `pr_cli`, `provider_cli`, `search_cli`, `shootout_cli`, `showcase_cli`, `new_cli`, `main`, `swarm_cli`) migrated from `mullet.harness.*` to `mullet.harness.*` imports.
- `mullet/server/acp_server.py` and `mullet/server/web/app.py` likewise migrated.

---

## [1.7.1] - 2026-05-11

### Added
- **Per-provider spend caps** — `.mullet/budget.yaml` lets users set USD caps per provider (`openrouter: {cap_usd: 10.0}`). `check_spend_cap()` is called in `_apply_probe_filter()` alongside `has_budget`; over-cap providers are skipped in the routing fallback chain.
- **`mullet config budget` subcommand** — `show`, `set <provider> <cap>`, `monthly <total>`, `reset`. Reads/writes `.mullet/budget.yaml`, detected by walking up from cwd. Rich table output with plain-text fallback.
- **`CostTracker`** — thread-safe in-flight + committed spend tracker for the swarm coordinator (`mullet/harness/coordinator/cost_tracker.py`). `can_route()` blocks over-cap providers; `estimate_cost_usd()` provides pre-run cost estimation from KNOWN_PRICING. Integrated into `Coordinator.execute()` and `_run_one()`.
- **Low-balance HUD warning** — HUD line 4 shows `⚠ Kimi low` / `⚠ OR low` when API balance < $0.50.
- **`mullet budget --watch`** — live 60-second refresh loop; Ctrl+C to exit. `--json` output now includes `window_label` and `window_resets_at`.
- **`has_budget` dual-window check for Anthropic** — `has_budget` returns `False` when either the 5h raw percentage ≥ 90% OR the weekly raw percentage ≥ 90%, preventing over-spend near window resets even when paced pressure is low.
- **Multi-provider swarm** — `/swarm --providers kimi,local,openrouter` distributes subtasks round-robin across providers weighted by `priority_score`; implemented via `parallel_provider_pairs()`.

### Fixed
- **Spend cap comparison** — `_provider_allowed()` now passes `credits_used_usd` (spent amount) rather than `credits_remaining_usd` to `check_spend_cap()`, so cap comparisons are correct for providers with large credit limits.
- **Budget advisor model→provider mapping** — `_LANE_PROVIDERS` uses model names (opus/sonnet/haiku) but `budget_probe` cache uses provider keys (anthropic/kimi/local). Added `_MODEL_TO_PROVIDER_KEY` mapping so auto-switch logic actually fires when primary provider is out of budget.

---

## [1.7.0] - 2026-05-11

### Added
- **Multi-provider budget awareness** — `budget_probe.py` queries all 5 configured providers concurrently: OpenRouter (real $ via `/auth/key`), Kimi/Moonshot (real $ via `/users/me/balance`), Anthropic (HUD token pressure), Gemini (subscription detection), Local LLM (reachability). Results cached 5 min in-process.
- **ProviderBalance dataclass** — `billing_model` field (`"free"` | `"subscription_cli"` | `"api"`), `priority_score` (0.0–1.0 routing weight), `has_budget`, `is_low`, `window_label`, `window_resets_at`. Auto-detects subscription vs API billing.
- **`mullet budget` command** — shows all 5 providers with balance/quota/score in a Rich table; `--json` and `--refresh` flags.
- **`/budget` slash command** — full multi-provider view in harness CLI; shows Anthropic 5h/weekly alongside real API balances for OpenRouter/Kimi and subscription status for Gemini/Local.
- **HUD line 4** — persistent status bar now shows compact per-provider balance icons: `free`, `pct%` (subscription), or `$amount` (API). Updated dynamically from probe cache.
- **Budget-probe routing filter** — `_apply_probe_filter()` in `decision.py` skips providers with zero cached balance during HUD-pressure routing; never blocks (fail-open when cache is empty).
- **Claude Code tool parity** — harness tools now match Claude Code exact names and schemas: `Read`, `Write`, `Edit`, `MultiEdit`, `Bash`, `Glob`, `Grep`, `TodoWrite`, `TodoRead`.
- **MultiEdit tool** — apply multiple sequential replacements to one file in a single call; later edits see results of earlier ones.
- **TodoWrite / TodoRead tools** — in-session task list keyed by `session_id`; status validation (pending/in_progress/completed).
- **System prompt tool hints** — cloud providers receive full `## Available Tools` section (9 tools, usage patterns, workflow guidance); local models receive a compact single-line list.
- **Plan mode visual diff** — Write/Edit in plan mode renders a colour unified diff (green +, red -) instead of a hard block.
- **`/permissions` command** — view and modify the runtime permission policy: mode (default/auto/plan), denied_commands, denied_tools, allowed_tools, path_rules.
- **`mullet shell` → harness** — `mullet shell` now invokes `HarnessV2Engine` (same as `mullet cli`). Harness v1 shell path retired.

### Fixed
- **Kimi balance currency** — `probe_kimi()` defaulted to CNY→USD conversion; international `api.moonshot.ai` returns USD natively. Now only converts if response explicitly declares `currency == "CNY"`.
- **Budget `has_budget` with missing API key** — providers with "not set" or "not found" errors now return `has_budget=False`, preventing broke providers from appearing available.
- **Gemini priority_score** — subscription Gemini with a missing API key previously scored 1.00; now scores 0.0 when `has_budget=False`.
- **Snapshot multiline breaking banner** — budget preflight banner snapshot field stayed single-line; provider detail moved to dedicated section.

---

## [1.6.0] - 2026-05-10

### Added
- **OpenRouter provider** — 8 curated coding/reasoning models (deepseek-chat, deepseek-r1, llama-4-scout, llama-4-maverick, qwen-2.5-coder-32b, qwen3-30b-a3b free tier, devstral-small, grok-3-mini). Full routing pipeline integration (KNOWN_PRICING, model_registry, routing_matrix, resolver, /model test).
- **select_openrouter_model()** — routing matrix tier-to-model mapping with reasoning, speed, large-context, and code-task branches.
- **OpenRouter doctor check** — OPENROUTER_API_KEY presence + live /models endpoint probe in --full mode.
- **HUD persistence** — DECSTBM scroll region with 4Hz background refresh thread for stable terminal output.
- **Rust language extractor** — knowledge/code indexing now handles .rs files.
- **/model test improvements** — live streaming, grouped by provider, preview/old model filtering, full error messages.

### Fixed
- **Deprecated Kimi models** — routing_matrix.py replaced kimi-k2-0711-preview and kimi-k2-0905-preview (EOL 2026-05-25) with kimi-k2.5. Static model_registry fallback ladder updated. Tests updated.
- **Trawl --rebuild-code** — ingest_client_code() now scans sources/ in addition to documents/ when code.paths is unset, fixing silent no-ops on source-based pools.

---

## [1.5.1-harness] - 2026-05-10

### Fixed
- **MCP tool name sanitisation** — replace `.` in `server.tool` qualified names with `__` before sending to LLMs; Kimi (and strict providers) reject dots in function names. Reverse map maintained for tool-call routing.
- **Tool dispatch missing-args guard** — validate `required[]` from schema before calling; return clear error JSON so the LLM can self-correct. Narrowed `TypeError` fallback to only suppress `task_id` injection errors.
- **Version banner** — `mullet/__init__.py` bumped to `1.5.0` to match `pyproject.toml`.

---

## [1.5.0-harness] - 2026-05-09

### Added — Phase 9: Beyond Claude Code Parity

**New commands and UX:**
- `/compare p1 p2 -- prompt` — runs same prompt against multiple specified providers concurrently, shows full responses with timing
- `/model temperature N` · `/model max-tokens N` · `/model top-p N` · `/model reset` — per-session model generation parameter overrides
- `/add-dir <path>` — recursively add all files in a directory to context
- `/session diff N M` — unified diff of two sessions' message threads
- `/session branch save|list|switch N` — session branching with explicit subcommands (list branches, switch by index)
- `/think budget N` — set extended thinking token budget for Claude models; `/think off` disables
- `/review pr <N>` — fetch a GitHub PR diff via `gh` CLI and send for AI code review
- Custom slash commands: `.mullet/commands/*.md` or `.claude/commands/*.md` → `/command-name` slash commands
- Custom tool registration: `.mullet/tools/*.py` loaded at startup via `register(registry)` entry-point
- `--metrics` flag: Prometheus HTTP endpoint on port 9090; `MetricsRegistry.to_prometheus()` text format
- Stdin pipe auto-detection: `echo "task" | mullet cli` → one-shot non-interactive run
- Smart session auto-title: strips filler prefixes (can you/please/hi) and capitalises
- Multiline brace detection: Enter waits for matching `}` `]` `)` before submitting
- Cost estimation warning: requests >20K tokens and >$0.05 estimated show inline alert
- Auto-suggest cheaper provider: when budget pressure >60%, suggests downgrade path inline
- MCP client auto-connect at startup: if `.mcp.json` exists, `MCPManager.connect_all()` fires on first turn
- Interactive permission modal: `[A]llow / [D]eny / [!] Always Allow` for `Prompt`-mode tool calls
- CLAUDE.md auto-injection: walks cwd + 5 parents + `~/.claude/CLAUDE.md`
- `@image.png` @mention: detects image extensions, base64-encodes, attaches as multimodal vision content

**Shell completions:** bash/zsh scripts updated with `cli` subcommand + all slash commands

- `/rate up|down|show` — rate last provider response; `QualityTracker` persists to `~/.config/mullet/provider_quality.json`; engine auto-records responses; `best_provider()` for quality-aware routing
- 3 built-in skills auto-loaded at global scope: `delegate`, `commit`, `review`
- 8-entry `registry.json` for `/skill install` marketplace

**Tests:** 630+ harness tests (was 362); 12,900+ project-wide

---

## [1.4.0-harness] - 2026-05-09

### Added — Full subsystem parity + Claude Code integration wiring

**Security:**
- Secret scanner with 12 regex patterns (AWS, GitHub, OpenAI, Anthropic, JWT, Slack, private keys, etc.)
- `PreToolUse` redactor hook — automatic redaction in tool inputs, outputs, and session exports

**Permissions:**
- `PermissionChecker` with `default` / `auto` / `plan` modes
- `.mullet/permissions.yaml` project-level config with `path_rules`, `denied_commands`, `denied_tools`, `allowed_tools` whitelist

**Hooks:**
- 6-event `HookRegistry` — PreToolUse, PostToolUse, Message, Event, Transform, Params
- Secret redaction guard wired as PreToolUse hook
- Auto-compaction via Transform hook (70% threshold)
- IntentGate keyword classifier via Message hook
- Per-session fallback via Event hook

**Skills:**
- `SKILL.md` 3-scope auto-loader (project > user > global)
- `/skill list|show|use` commands
- `mullet skill` marketplace: search / install / remove / info / publish / new

**Profiles:**
- `ProfileManager` with 10 built-in provider profiles
- `/provider list|add|use|show|remove` commands

**Plugins:**
- `plugin.json` manifest loader with commands/agents/skills/hooks/mcpServers/lspServers
- `/plugin` commands
- Error-isolated loader with Claude Code plugin compatibility

**Coordinator:**
- Swarm dispatch with git worktree isolation
- `ConcurrencyManager` per-provider/per-model limits
- `/swarm` command

**Voice:**
- Whisper backends (whisper.cpp / OpenAI / Groq / mock)
- `/voice` command and `--voice` CLI flag

**Workflows:**
- Durable SQLite 3-table state (`WorkflowRun` + `WorkflowEvent` + `WorkflowMessage`)
- `replay_events()` for corruption recovery
- `/workflow` commands

**Observability:**
- `MetricsRegistry` — counters, gauges, histograms with dimensional tags (thread-safe)
- `LogAggregator` — structured log collection, filtering by level/source/since, FIFO eviction
- `/metrics` command and Prometheus `/metrics` HTTP endpoint

**Fallback:**
- Provider error classification (6 categories)
- Auto-retry with cooldown map
- Fallback hooks wired into engine Event pipeline

**Cron:**
- `CronScheduler` + `CronStore` with cron-expression parsing
- `/cron list|add|remove|run` commands

**MCP Client:**
- `MCPClient` + `MCPManager` — JSON-RPC 2.0 over stdio + HTTP/SSE
- Multi-server manager via `.mcp.json`
- `/mcp add|list|remove` commands

**Sessions Extended:**
- Export (Markdown / HTML / JSON), branch (fork with `parent_session_id`), tags, fuzzy resume
- Project isolation via `project_hash`
- `/session export|branch|tag|resume` commands

**Tools Extended:**
- `TodoWrite` — structured todo lists with `replace_all` semantics
- `RunPython` — sandboxed subprocess execution
- `BrowserTool` — Playwright multi-session automation
- `ReadPDF` — pypdf-backed PDF extraction

**Engine Wiring:**
- `HarnessV2Engine` async/sync loop with full tool dispatch, streaming, and hook fire-points
- Parallel tool execution via `asyncio.gather`
- `StreamingProviderAdapter` with token-by-token rendering
- Auto-compaction and auto-context injection

**UX Polish:**
- Rich `bottom_toolbar` with budget, ctx%, 5h%, git branch, permission mode
- Context pressure warnings at 70%/85%
- Spinner shows active provider name and elapsed time
- Relative timestamps in session listings
- `@mention` enhancements: line ranges, URLs, globs

**Tests:**
- 603 harness tests passing (was 362)

---

## [1.3.0-harness] - 2026-05-09

### Added: harness overnight UX polish

**New commands:**
- `!` shell shorthand — alias for `/run`, identical to Claude Code behaviour
- `/find <glob>` — file finder with glob expansion
- `/open <file>` / `/edit <file>` — open file in `$EDITOR`, reload into context on save
- `/copy` / `/cp` — re-copy last assistant response to clipboard
- `/reset` — alias for `/clear history` (matches common user expectation)
- `/version` — show Mullet version + Python version
- `/g` / `/gemini` — quick shortcut to switch to Gemini Flash provider
- `/haiku` — quick shortcut to switch to Claude Haiku provider
- `/session rename <id> <title>` — rename saved sessions
- `/pin remove N` — delete individual pins without clearing all
- `/note delete N` — delete individual notes by numbered index
- `/todo remove N` — remove individual todo items by display index
- `?` — shorthand for `/help`
- `/config get <key>` — read dot-path config values (e.g. `/config get defaults.provider`)

**@mention enhancements:**
- `@file:N-M` — inject only a specific line range from a file
- `@https://url` — fetch URL and inject content inline
- `@glob/**/*.py` — glob expansion: inject all matching files in one mention
- Tab completion for `@file` now works in subdirectory paths (`@src/util<Tab>`)

**Context / command improvements:**
- `/context add 'src/**/*.py'` — glob patterns supported in context add
- `/context add file.py:10-50` — inject only a line range
- `/peek file.py:5-10` — display and inject specific line range
- `/context list` now shows estimated token counts alongside char counts
- `/diff` shows file-change stat summary (`2 files changed, 15 insertions(+)`) at top
- `/benchmark` now shows tok/s throughput column
- `/search` grep fallback extended to cover `.rs`, `.go`, `.rb`, `.java`, `.c`, `.cpp`, `.h`, `.md`, `.yaml`, `.toml`
- `/compact` router tuple unpacking bug fixed (was breaking AI summarisation)
- `/run` preserves quoted args — `/run git log --format="%H %s"` now works correctly
- `/run` caps display at 60 lines, full output still injected to context
- Spinner now shows active provider name: `⠸ kimi (2.3s)` instead of `⠸ Thinking...`
- Context window pressure warning shown at 70%/85% with `/compact` suggestion

**TUI polish:**
- Git branch shown in persistent bottom toolbar (cached 30s)
- `/session list` and `/session search` results show relative timestamps (`2h ago`, `3d ago`)
- `/session search` results include title and provider (consistent with `/session list`)
- `/note list` shows relative timestamps (`2m ago`, `3h ago`)
- `/commit` shows staged diff `--stat` preview before asking y/N confirmation
- `/history` improved format — header with turn count, `▶`/`←` prefixes, blank-line separation
- `/ask` streams tokens in real-time (was sentinel/delayed return)
- Markdown rendering for AI-generated command outputs (`/review`, `/summarize`, etc.)

**Bug fixes / performance:**
- Post-response delay eliminated — 3 sources fixed:
  1. `ClaudeCodeProvider` stderr thread moved to background daemon on success path
  2. Spinner `Event.wait(0.1)` pattern — exits within 0.1s (was sleeping 0.5s)
  3. Clipboard copy now non-blocking via `asyncio.ensure_future(to_thread(...))`
- Status bar no longer cycles on empty Enter presses
- `/todo done/remove` now operate on checkbox display lines only, not raw file line numbers
- `/grep` handles unquoted patterns with special chars (parens, brackets)

**Tests:**
- 362 tests total in harness suite (was 316); 12,600+ tests project-wide

---

## [1.2.0-harness] - 2026-05-09

### Added — harness round 2 (`mullet cli`)

**New commands (round 2):**
- `/commit [-a] [-n] [-p]` — AI-generated git commit message from staged diff; flags for stage-all, dry-run, push
- `/tokens` / `/usage` — session token breakdown with per-type cost estimates, provider, context pressure
- `/paste` — inject clipboard content into context (pyperclip → xclip → xsel → pbpaste)
- `/init` now generates CLAUDE.md via AI when none exists — analyzes project structure, writes concise context file
- `/config set <key> <value>` — update providers.yaml via dot-path (e.g. `/config set defaults.provider kimi`)
- `/config edit` — open providers.yaml in `$EDITOR` (defaults to nano)

**New CLI flags:**
- `mullet cli --speed economy|balanced|fast|max|premium` — maps to provider overrides

**Engine improvements:**
- Parallel tool execution: multiple tool calls in same LLM batch now run concurrently via `asyncio.gather`
- Thinking/reasoning display: `thinking_delta` events from reasoning models (deepseek-r1, kimi-thinking) rendered dimmed
- Persistent `bottom_toolbar` in prompt_toolkit: always-visible provider · cost · turns · ctx% while typing

**Test improvements:**
- Knowledge rebuild regression tests (3 tests) — verifies `collection.delete` fires on `--rebuild-code`
- Engine test: parallel tool execution order preserved (test_engine.py)
- 316 tests total in harness suite (was 308); 1305+ tests project-wide

**Bug fixes:**
- `test_guard_internals`: stale routing assertions updated to reflect local-first token economy
- Coroutine warnings in /commit and /pr tests are test-framework artifacts (production unaffected)

---

## [1.1.0-harness] - 2026-05-09

### Added — harness (`mullet cli`)

The next-generation CLI shell is now feature-complete and daily-driver ready (298+ tests).

**New commands:**
- `/review [last]` — AI code review of current diff or last commit via cheapest provider (local→kimi)
- `/test` — auto-detect test runner (pytest/npm/cargo/make) and inject output to context
- `/grep <pattern> [path] [-C N]` — regex search with line numbers and context injection
- `/stats` — cross-session analytics: total sessions, turns, provider frequency
- `/vim` — toggle vi keybindings (Esc=normal, i=insert, hjkl=move) via EditingMode.VI
- `/pr [--dry-run]` — create GitHub PR with AI-generated title/body via `gh`
- `/session show N` — preview a saved session without loading it
- `/session delete N` — delete session by numbered index from `/session list`
- `/summarize` — AI-powered conversation summary injected to context
- `/stats` — cross-session statistics dashboard

**New CLI flags:**
- `mullet cli --system-prompt "..."` — inject custom system prompt at startup
- `mullet cli --max-turns N` — limit autonomous tool loop iterations

**Improvements:**
- Auto-save every 5 turns (silent, in addition to exit auto-save)
- `/session list` shows numbered index, provider, cost, and title
- `/session load N` / `/session delete N` support numeric index
- `/session load` restores provider from session metadata
- `/session save` stores `cost_usd` in metadata
- `/compact` shows compression stats: `47 → 12 messages (~8400 → ~1200 tokens)`
- `/compact` injects AI summary into context for future-turn awareness
- `/benchmark` shows cost per provider in comparison table
- Tab-completion: session filenames for `/session load`, file paths for `/context add`
- Tool output preview (60-char snippet) shown after each tool call in TUI
- `/doctor` shows harness test suite count
- Persistent pins saved to `.mullet/pins.md`, loaded on startup
- Provider availability indicators (✓/✗) in `/model list`

**Bug fixes:**
- `knowledge rebuild-code`: Chroma collection now cleared before re-inserting chunks on `--rebuild-code`
- Provider correctly restored when resuming sessions via `--resume`

## [1.0.0-harness] - 2026-05-08

### Added — Engine
- 🚀 **Standalone harness layer** at `mullet/harness/` — Mullet now owns its own agent loop independently of Claude Code (`mullet shell`, `mullet -p`).
- 🤖 **HarnessEngine** — full async/sync agent loop with tool execution wired into the engine (R2.A1), hook fire-points (PreToolUse / PostToolUse / Message / Event / Transform / Params), permission check, and fallback retry.
- ⚡ **Streaming** — token-by-token via `assistant_chunk` events; tool-call deltas; `thinking_delta` for Claude; `StreamingProviderAdapter` with chat-screen incremental render (R2.A2).
- 🐝 **Swarm** — `mullet swarm "<goal>"` parallel agents with git worktree isolation and `ConcurrencyManager` limits per provider/model (R2.A3).
- 🔄 **Auto-compaction** — Transform hook summarises older messages when context window exceeds 70 % (R2.B1).
- 📦 **Workflows** — durable SQLite 3-table state (`WorkflowRun` + `WorkflowEvent` + `WorkflowMessage`); `replay_events()` for corruption recovery.
- 🚨 **Fallback** — per-session cooldown map; 6-category error classifier; auto-retry signal extraction.
- 🧠 **Auto-context** — Trawl-backed `AutoContextResolver` injects relevant code chunks via the Message hook (R2.C5).
- 🎯 **IntentGate** — keyword classifier (`research` / `implementation` / `investigation` / `evaluation` / `fix`) biases routing decisions.
- 🤝 **Named subagents** — Sisyphus, Hephaestus, Oracle, Librarian with primary+fallback provider chains.
- 📖 **Memory auto-discovery** — walks up to git root collecting `CLAUDE.md` / `AGENTS.md` / `MEMORY.md`.
- 🖼 **Multimodal / image input** — `ImageAttachment` + `MultimodalMessageBuilder` via `--image PATH` / `--image-url URL` / `/image add` (R2.C14).

### Added — CLI
- 🎬 **`mullet demo`** — 9-step interactive walkthrough with provider auto-detect, skill discovery, mini-session, tool/hook showcase, and GitHub nudge (R2.C1).
- 📊 **`mullet status`** — single-pane dashboard showing budget, active profile, permissions, skills, sessions, workflows, MCP servers, and watchdog health (R2.C7).
- 📤 **`mullet share` / `/share`** — export sessions to Markdown, HTML, or JSON with optional anonymisation (R2.B3).
- 🔍 **Conversation search** — full-text `/search` slash command and `mullet search` CLI for querying session history (R2.C23).
- 🚀 **`mullet pr "<task>"`** — autonomous PR workflow: branch → `engine.arun` → commit → push → `gh pr create` (R2.C15).
- 🪄 **`mullet harness-init`** — auto-detects project type and scaffolds `.mullet/` config + `MEMORY.md` from README, git log, `permissions.yaml`, and MCP template (R2.B2).
- ⚙️ **`mullet config`** + **`mullet new`** — `config get/set/list/edit/reset`; scaffold new skill, plugin, workflow, or project (R2.C25).
- ⏰ **Cron-scheduled tasks** — `CronStore` + `CronScheduler` with `mullet cron list/add/remove/run/crontab` (R2.C17).
- 📈 **Benchmark suite** — 5-task default suite (`hello` / `reasoning` / `code` / `json` / `instruction`), substring + regex grading, `mullet benchmark` CLI (R2.C27).
- 🎯 **`mullet shootout`** — multi-provider comparison with parallel execution and side-by-side results table (R2.C28).
- 🎥 **`mullet showcase`** — 10-section visual demo for asciinema / screencasts (R2.C30).
- 📡 **Opt-in telemetry** — privacy-first collector (no paths, code, or PII), session UUID, jcode-style disclosure, `mullet telemetry on/off/status/clear` (R2.C18).
- 📊 **`mullet metrics`** — CLI for counters, gauges, and histograms (R2.C29).
- 🔌 **Provider profiles** — named profiles with per-profile credentials (`mullet provider {add|list|use|remove|show}`); 10 built-in profiles (Kimi, Anthropic, OpenAI, Gemini, Bedrock, Ollama, LM Studio, OpenRouter, Claude CLI, Gemini CLI).
- 📚 **Skills system** — 3-scope `SKILL.md` loader (project > user > global); 8 built-in skills; `/skill list/show/use`; `mullet skill search/install/remove/info/publish` marketplace (R2.C13).
- 🔌 **Plugin system** — `plugin.json` manifest with commands/agents/skills/hooks/mcpServers/lspServers; error-isolated loader; Claude Code plugin compatibility.
- 🛡️ **Permissions** — `default` / `auto` / `plan` modes; `path_rules`; `denied_commands`; `denied_tools`; `allowed_tools` whitelist; `/permissions list/mode/allow/deny`.
- 🌐 **MCP client** — JSON-RPC 2.0 over stdio + HTTP/SSE; multi-server manager via `.mcp.json`; `/mcp` slash commands.
- 🌐 **One-line install** — `curl -fsSL .../install.sh | bash` with `--install-help` (R2.C6).

### Added — Tools
- 🐍 **RunPython tool** — sandboxed subprocess execution with configurable timeout (R2.C31).
- 🖥 **RunBash tool** — sandboxed subprocess execution with configurable timeout (R2.C31).
- 📄 **ReadPDF tool** — extract text and metadata from PDF documents (R2.C22).
- 🌐 **Browse tool** — Playwright integration with multi-session contexts: `goto` / `extract` / `click` / `fill` / `screenshot` / `links` / `text` (R2.C24).
- ✅ **TodoWrite tool** + `/todos` — per-session todo list with `replace_all` semantics, atomic jsonl writes, and rich table render (R2.C10).
- 🖼 **Image tool** — attach and process image inputs for multimodal conversations (R2.C14).

### Added — Slash commands
- `/help` + `mullet help` — discoverable docs for commands, skills, shortcuts, and CLI usage (R2.C9).
- `/clear`, `/export`, `/compact`, `/cost` — quick aliases for common operations (R2.C11).
- `/todos` — view and manage per-session todo list (R2.C10).
- `/image add` — attach an image to the current conversation (R2.C14).
- `/voice` — record and transcribe voice input via `VoiceRecorder` + `VoiceTranscriber` (whisper.cpp / OpenAI / Groq / mock); also `--voice` CLI flag (R2.C20).
- `/theme` — switch between 5 TUI themes: dark, light, monokai, solarized, synthwave (R2.C26).
- `/search` — full-text search across conversation history (R2.C23).
- `/tag pin|unpin|add|remove` — tag and pin messages for quick retrieval (R2.C23).
- `/undo` — rewind session history to any prior point (R2.C8).
- `/branch` — fork session at any point with `parent_session_id` genealogy (R2.C8).
- `/browse` — open a browser tool session (R2.C24).
- `/mcp` — list, start, stop, and configure MCP servers.
- `/context` — trigger auto-context injection.
- `/skill list|show|use` — manage loaded skills.
- `/permissions list|mode|allow|deny` — manage permission settings.
- `/resume` — resume a previous session from the resume picker.

### Added — Hooks
- 🛠 **5 tool guards** — `writeExistingFileGuard`, `hashlineReadEnhancer`, `directoryAgentsInjector`, `bashFileReadGuard`, `toolOutputTruncator` (R2.A4).
- 🔒 **Secret redaction hook** — `PreToolUse` guard with 11 patterns (AWS, GitHub, OpenAI, Anthropic, JWT, etc.) plus exporter redaction (R2.C32).
- 🎯 **IntentGate** — keyword classifier biasing routing decisions via the Message hook.
- 📖 **Memory injection** — `directoryAgentsInjector` injects `AGENTS.md` / `CLAUDE.md` context into prompts.
- 🔄 **Auto-compaction** — Transform hook summarises older messages when context > 70 %.

### Added — Sessions
- 🔍 **Conversation search** — full-text search across session history (R2.C23).
- 🏷 **Tags + Pin** — tag messages and pin important context for quick recall (R2.C23).
- ⏪ **Undo** — rewind sessions to any prior point, restoring previous state (R2.C8).
- 🌿 **Branch** — fork at any point with `parent_session_id` genealogy tracking (R2.C8).
- 💾 **Session storage** — append-only event log with `project_hash` isolation at `~/.config/mullet/sessions/{hash}-{cwd}/`.
- 🔁 **`/resume`** — resume previous sessions from the interactive picker.

### Added — Servers
- 🤖 **ACP server** — JSON-RPC 2.0 stdio Agent Client Protocol so Zed and external clients can drive Mullet (R2.C3).
- 🌐 **Web dashboard** — FastAPI server with project/session browser, JSON API, dark-mode HTML, and `mullet web` CLI (R2.C4).

### Added — Polish
- 🎓 **First-run onboarding** — `mullet shell` greets new users, flags setup issues, and guides through provider config (R2.C12).
- 🎨 **ASCII banner** — startup banner with version info and quick hints (R2.C21).
- 🎨 **5 TUI themes** — dark, light, monokai, solarized, synthwave via `/theme` (R2.C26).
- 🔄 **Auto-update checker** — 24-hour cache comparing latest PyPI/GitHub release; integrates into `mullet status` (R2.C21).
- 🩺 **`mullet doctor --fix`** — diagnostics with automatic repair suggestions and install hints (R2.C21).
- ✨ **TUI polish** — markdown rendering for assistant messages, tool-call cards with syntax highlighting, real-time cost+latency in the status bar (R2.C2).
- 🧠 **Thinking blocks** — Anthropic reasoning stream display with `Ctrl+T` toggle (R2.C12).
- 💬 **FriendlyError** — actionable error messages with install hints and documentation links (R2.C22).

### Added — Security
- 🔒 **Secret scanner** — 11 regex patterns covering AWS keys, GitHub tokens, OpenAI/Anthropic keys, JWT, Slack webhooks, private keys, and more (R2.C32).
- 🛡️ **Redactor** — automatic secret redaction in tool inputs, outputs, and session exports (R2.C32).
- 🚫 **PreToolUse guard** — blocks or masks secrets before they reach tool execution (R2.C32).

### Added — Observability
- 📊 **MetricsRegistry** — counters, gauges, histograms with dimensional tags (R2.C29).
- 📝 **LogAggregator** — structured log collection, filtering, and querying (R2.C29).
- 📈 **Prometheus export** — `/metrics` endpoint for external monitoring stacks (R2.C29).
- 📡 **Opt-in telemetry** — privacy-first session analytics, zero PII/paths/code, jcode-style disclosure banner (R2.C18).

### Added — Release prep
- 🌐 **One-line install script** — `curl -fsSL .../install.sh | bash` with `--install-help` (R2.C6).
- ⚙️ **GitHub Actions CI** — lint and test workflows on every push/PR (R2.C19).
- ✨ **Skill author wizard** — `mullet skill new` interactive scaffold for skill creators (R2.C19).
- 🏷 **README / CHANGELOG / CONTRIBUTING badges** — release documentation polish (R2.C16).
- 📦 **PyPI prep** — metadata polish, `MANIFEST.in`, `scripts/release.sh` (R2.C26).

**Tests:** 991 harness tests passing across 111 test files.

## [Unreleased] — Harness Layer

### Added — Engine
- Tool execution wired (22 tools auto-discovered from `core/tools`)
- Streaming via `StreamingProviderAdapter` (`text_delta` + `tool_call_delta` + `thinking_delta` for Claude)
- Auto-compaction Transform hook (70% threshold, keep-recent-N-turns, summarize via Kimi)
- Multimodal/image input (`--image PATH`, `--image-url URL`, `/image add`)
- Async `arun()` + sync `run()` with full agent loop and `max_iterations` cap

### Added — Coordinator / Workflows
- Swarm with git worktree isolation + `ConcurrencyManager` (per-provider/per-model limits)
- Autonomous PR workflow (`mullet pr "<task>"` → branch + agent + commit + push + `gh pr create`)
- Cron-scheduled tasks (`mullet cron list/add/remove/run/crontab`)
- SQLite 3-table workflow store (LiteLLM-compatible)

### Added — CLI commands
- `mullet demo` — 9-step interactive walkthrough
- `mullet showcase` — 10-section visual demo for asciinema
- `mullet status` — single-pane dashboard (budget+profile+permissions+skills+sessions+workflows+MCP+watchdog)
- `mullet web` — FastAPI dashboard for browsing sessions
- `mullet acp` — ACP server (JSON-RPC stdio for Zed)
- `mullet share` — markdown/html/json export with anonymize
- `mullet search` — full-text session search
- `mullet harness-init` — auto-detect project type, scaffold `.mullet/`
- `mullet skill` — list/search/install/remove/info/publish/new marketplace
- `mullet pr` — autonomous PR workflow
- `mullet benchmark` — 5-task default suite (hello/reasoning/code/json/instruction)
- `mullet shootout` — multi-provider comparison with parallel execution
- `mullet cron` — scheduled tasks
- `mullet telemetry` — opt-in usage stats (jcode-style privacy)
- `mullet metrics` — list/reset/export (Prometheus + JSON)
- `mullet config` — get/set/list/edit/reset
- `mullet new` — scaffold skill/plugin/workflow/project
- `mullet doctor --fix` — auto-remediation
- `mullet help` / `--install-help`

### Added — Tools
- `RunPython`, `RunBash` — sandboxed subprocess execution with timeout
- `ReadPDF` — `pypdf`-backed PDF extraction
- `Browse` — Playwright multi-session browser (goto/extract/click/fill/screenshot/links/text)
- `TodoWrite` — replace_all todo list, jsonl persisted

### Added — Slash commands
`/help`, `/clear`, `/export`, `/compact`, `/cost`, `/todos`, `/image`, `/voice`, `/theme`, `/search`, `/tag`, `/undo`, `/branch`, `/browse`, `/mcp`, `/context` — all wired through dispatcher with signature-aware kwargs.

### Added — Hooks
- 5 tool guards (`writeExistingFileGuard`, `hashlineReadEnhancer`, `directoryAgentsInjector`, `bashFileReadGuard`, `toolOutputTruncator`)
- Secret redaction guard (`PreToolUse`, 11 patterns: AWS, GitHub, OpenAI, Anthropic, JWT, etc)
- IntentGate (Message hook, keyword classifier)
- Memory file auto-discovery (Transform hook, walks to git root)
- Auto-context Trawl injection (Message hook)
- Per-session fallback Event hook (cooldown map, error pattern classifier, retry signal extraction)

### Added — Sessions
- `search.py` — full-text grep across `messages.jsonl` with role filter + recency scoring
- `tags.py` — `/tag pin/unpin/add/remove`, `list_pinned`, `list_by_tag`
- `branch.py` — `fork_session` at any index, `undo_last_turn`, `rewind_to_index`, `parent_session_id` genealogy

### Added — Server
- `mullet/server/acp_server.py` — ACP JSON-RPC over stdio (`initialize`/`session/new`/`load`/`prompt`/`cancel`)
- `mullet/server/web/` — FastAPI dashboard with project/session browser, JSON API, dark-mode HTML

### Added — Polish
- First-run onboarding (banner + setup-issue detection on `mullet shell`)
- ASCII banner (`mullet/cli/banner.py` with Rich color)
- 5 TUI themes (dark/light/monokai/solarized/synthwave) + `/theme use`
- Thinking-block widget (`Ctrl+T` toggle in TUI)
- Auto-update checker (24h cached GitHub Releases poll)
- TUI markdown rendering for assistant + tool call cards with syntax highlighting
- Real-time cost+latency in status bar

### Added — Security
- Secret scanner (`mullet/harness/security/secret_scanner.py`) — 11 patterns
- Secret redactor — replaces with `[REDACTED]` in exports + tool args via `PreToolUse` guard
- `FriendlyError` diagnose system (`mullet/harness/errors.py`) — actionable messages with install hints + docs links

### Added — Observability
- `MetricsRegistry` (counters/gauges/histograms with tags, thread-safe)
- `LogAggregator` (filtering by level/source/since, FIFO eviction)
- Prometheus export format
- Opt-in telemetry collector (jcode-style privacy: no paths/code/PII)

### Added — Release prep
- One-line install: `curl -fsSL .../install.sh | bash`
- GitHub Actions CI workflow (`.github/workflows/test.yml`) — matrix py3.10/3.11/3.12
- GitHub Actions lint workflow (`ruff` + `black` + `mypy`)
- PyPI metadata polish (classifiers, `project.urls`, `MANIFEST.in`)
- `scripts/release.sh` — version bump + tag + build helper
- `CONTRIBUTING.md`
- README badges (test count, license, Python, GitHub stars)

### Tests
- 991 harness tests passing across 111 test files (was 530 at end of Round 1)
- ~28,400 LOC of harness code added

### Performance
- `ToolDispatcher` reuses provider connections via async/sync registry
- `HookRegistry` has hard 5-iteration cap to prevent loops
- Session storage uses atomic tmp+rename writes for meta updates

## [1.0.0] — Pre-harness baseline (preserved)
Original router + delegation + knowledge engine + 25 tools + MCP server.
