# Mullet: context for Claude

Mullet runs large coding jobs unattended and checks every step itself. A job is
a task tree that you approve; each step runs with the user's own logged-in
`claude -p` in a sandboxed private clone, and passes only when Mullet's own
gates pass. The user gets back a branch and an evidence report, and their
checkout is untouched until they merge. Mullet also serves an MCP server
(`mullet mcp`) with Trawl knowledge-search tools and older delegation tools.

The design, decisions and phase plan are in `docs/MULLET2.md`. Read it before
changing anything under `mullet/jobs/`.

## Layout

- `mullet/jobs/`: the job runner. `planner.py` (task tree), `runner.py`
  (detached runner, resume), `clone.py` (private clone), `sandbox.py`
  (bubblewrap; no unsandboxed fallback), `gates.py` and `review.py` (gates G0 to
  G7), `ledger.py` (SQLite; a node's status comes from gate results only),
  `profile.py` (per-repo setup and check commands), `ladder.py` (which
  worker attempts each step: free, local, Kimi, OpenRouter, Claude),
  `report.py`, and `executors/` (`claude_code.py`, `opencode.py` behind the
  `keyproxy.py` key proxy). CLI: `mullet job plan|show|approve|run|status|tail|stop|resume|report|list|discard|merge|ladder|retry`.
- `mullet/hooks/`: Claude Code hooks, run as `mullet hook <name>`. None are
  registered by default; `mullet setup-claude --hooks <names>` adds them.
- `mullet/cli/`: the `mullet` command. `setup_claude.py` registers the MCP
  server and skills and records every change for `--uninstall`;
  `doctor_checks.py` holds the toolchain checks behind `mullet doctor`.
- `mullet/server/`: the MCP server (`mcp.py`, `mcp_handlers.py`,
  `mcp_knowledge.py`) and the REST API (`mullet serve`).
- `mullet/core/`: providers, routing, tools and skills used by the delegation
  tools; `mullet/knowledge/`: the Trawl engine.
- `scripts/hooks/`: thin shims that run the packaged hooks for this repo's own
  `.claude/settings.json`.

## Advice for users

1. **Provider keys go in `~/.config/mullet/.env`** (`$MULLET_CONFIG_DIR/.env` if
   that is set), not in `~/.zshrc` or `~/.bashrc`. A source checkout's `.env` is
   still read, for development.
2. **`ANTHROPIC_API_KEY` is optional and billed per token.** In `claude -p` it
   silently replaces subscription login, so Mullet jobs leave it out unless the
   job was planned with `--billing api`, and `mullet doctor` warns when it is
   set. Mullet never reads or reuses the subscription's own credentials.
3. **Install from GitHub, never `pip install mullet`**: the PyPI package of that
   name is someone else's. Use `scripts/install.sh` (uv or pipx).
4. **Don't suggest 7B+ local models on machines with 10GB of RAM or less**, and
   don't suggest the `[gpu]` extra on a Mac (it pulls NVIDIA-only `pynvml`).

## Testing

- While working, run only the test files you added or changed:
  `python3 -m pytest tests/test_foo.py -q`.
- The full suite runs in parallel in about 4 minutes:
  `python3 -m pytest tests -q -n 16 --timeout 300`. Run it before committing a
  change that touches several modules.
- Tests run with no network, no provider keys and a throwaway `HOME`
  (`tests/conftest.py`). Use mock providers, never real LLM calls; tests that
  need a real model are marked `live` and excluded by default. Use the
  `fake_claude` fixture for anything that runs the `claude` CLI.

## Code standards

- Python 3.10+, type hints on public APIs, docstrings on public classes and
  functions only.
- Every new module gets tests.
- No placeholder TODOs, `pass` stubs or commented-out code in production code.
- A job's status comes from evidence (gate results, exit codes), never from
  words in a model's reply.
