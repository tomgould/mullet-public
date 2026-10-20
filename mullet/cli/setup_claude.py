"""`mullet setup-claude` — install Mullet delegation hooks + CLAUDE.md into ~/.claude/."""
import json
import shutil
import subprocess

from pathlib import Path


def _get_claude_dir():
    return Path.home() / ".claude"


def _get_claude_md_path():
    return _get_claude_dir() / "CLAUDE.md"


def _get_claude_settings_path():
    return _get_claude_dir() / "settings.json"


def _get_claude_skills_dir():
    return _get_claude_dir() / "skills"


def _append_claude_md_section(dry_run: bool):
    claude_md_path = _get_claude_md_path()
    section_to_append = """
## Delegation (Mullet)

**Hard rule: delegate before reading or generating.** Mullet provides cheap/free LLMs via MCP — use them.

### Provider priority (default order — used when user has not specified a provider)

> **User overrides always win.** If the user says "no local LLM", "no Gemini", or names a specific provider, follow that for the rest of the session. The table below is the fallback order when no preference is given.
>
> **Key rule: OpenRouter is the default workhorse. Before each delegation, check `check_budget()` — if Anthropic pressure <60%, substitute Haiku/Sonnet instead of paying OpenRouter/Kimi tokens (Max plan is already paid for).**

| Priority | Provider | Cost | Use for |
|----------|----------|------|---------|
| 1st | Local LLM (`--provider local`) | Free | Always first — reads, greps, bulk/doc summaries |
| 2nd | OpenRouter (`--provider openrouter`) | Pay-per-token ~$0.27/M | Default cloud — cheap, reliable workhorse |
| 3rd | Haiku (`--provider haiku`) | Prepaid (Max plan) | Substitute OpenRouter when pressure <60% — saves pay-per-token spend |
| 4th | Sonnet (`--provider sonnet`) | Prepaid (Max plan) | Complex reasoning; substitute when pressure <60% |
| 5th | Kimi K2 (`--provider kimi`) | Pay-per-token ~$0.60/M | Large context; OpenRouter fallback |
| 6th | Gemini (`--provider gemini`) | Pay-per-token | Last resort only |
| Orchestrate only | Opus (you) | Prepaid (Max plan) | Never delegate here — this is you |

**Enforcing a provider override:** pass `providers=["kimi", "haiku"]` to `mullet_run` to hard-block excluded providers at the tool level.

**Speed overrides** — user phrases that change the active speed mode for the session:

| User says | Speed mode | Effect |
|-----------|------------|--------|
| "go fast", "I'm in a rush", `--speed fast` | `fast` | Skip local LLM; OpenRouter first, Haiku/Kimi as alternatives |
| "max speed", "don't worry about cost", `--speed max` | `max` | Sonnet/Opus first; Anthropic-first for quality |
| "save tokens", "go slow", `--speed economy` | `economy` | Local LLM first, then OpenRouter/Kimi; Haiku when headroom |
| "balanced", `--speed balanced` | `balanced` | OpenRouter default; substitute Haiku when Anthropic pressure <60% |

When a speed override is active, pass `--speed <mode>` on `mullet` CLI calls and set the equivalent tier in `mullet_run` tool calls. Default is `economy`.

### How to delegate via Mullet MCP
```
mcp__mullet__mullet_opencode — delegate file edits (has real filesystem access — PREFERRED)
mcp__mullet__mullet_run      — research/analysis only (no filesystem access)
mcp__mullet__trawl_search — semantic search across indexed codebase
mcp__mullet__trawl_search_code — code-specific search
mcp__mullet__trawl_find_symbol — find a class/function/method by name
```

### Triggers (no exceptions)
- Grep/scan across >3 files → use `trawl_search_code` first
- File rewrite >200 lines → delegate via `mullet_opencode`
- Doc generation, README, summary → delegate via `mullet_opencode`
- Bulk multi-file edits → delegate via `mullet_opencode`

### Default fallback chain (override as directed by user)
1. `mullet_opencode` (file edits — filesystem access; uses OpenRouter by default, Haiku when pressure <60%)
2. `mullet_run --provider openrouter` (research/analysis — default pay-per-token cloud)
3. `mullet_run --provider haiku` (substitute for step 2 when Anthropic pressure <60%)
4. `mullet_run --provider sonnet` (complex reasoning when pressure <60%)
5. `mullet_run --provider kimi` (large context or OpenRouter fallback)
6. `mullet_run --provider gemini` (last resort only)
7. Direct Opus — only with explicit user approval

### Delegating code edits — use mullet_opencode, NOT mullet_run

`mullet_opencode` has real filesystem access and reads/edits files directly:

```
mcp__mullet__mullet_opencode(
    task="Fix the calculate_total function in src/utils.py to handle tax"
)
```

**Never use `mullet_run apply=true` for file edits.** `mullet_run` has no filesystem
access — it is for research and analysis tasks that return text only.

If opencode is unavailable, fall back to an `Agent` subagent (has filesystem access),
not `mullet_run`.

### API LLM delegates have NO filesystem access — MANDATORY

`mullet_run` delegates (local LLM, Kimi K2, Gemini, Haiku) are **remote endpoints
with zero filesystem access**. Use them only for research/analysis tasks that return
text. For any task that involves reading or editing files, use `mullet_opencode`.
4. State explicitly which file and line range to modify

Only Claude Code subagents (spawned via the `Agent` tool) have real filesystem tool access and can read files themselves.

## Schema & Assumptions

- Never assume database schema, column names, or table structures — query the DB or read migration/schema files first
- Never assume API response shapes, config defaults, or file paths — read the source before acting
- State every assumption explicitly before making an edit that depends on it
- If a fix requires a schema assumption and you cannot verify it, say so and ask rather than guess
"""

    if not claude_md_path.exists():
        if not dry_run:
            claude_md_path.parent.mkdir(parents=True, exist_ok=True)
            claude_md_path.write_text(section_to_append.strip() + "\n")
        print(f"  Created {claude_md_path} with Mullet delegation section.")
        return True
    else:
        content = claude_md_path.read_text()
        if "## Delegation (Mullet)" not in content and "## Mullet" not in content:
            if not dry_run:
                claude_md_path.write_text(content.strip() + "\n" + section_to_append.strip() + "\n")
            print(f"  Appended Mullet delegation section to {claude_md_path}.")
            return True
        else:
            print(f"  Mullet delegation section already present in {claude_md_path}. Skipped.")
            additions = []
            if "## Schema & Assumptions" not in content:
                additions.append("\n## Schema & Assumptions\n\n- Never assume database schema, column names, or table structures — query the DB or read migration/schema files first\n- Never assume API response shapes, config defaults, or file paths — read the source before acting\n- State every assumption explicitly before making an edit that depends on it\n- If a fix requires a schema assumption and you cannot verify it, say so and ask rather than guess\n")
                print(f"  Will append Schema & Assumptions section to {claude_md_path}.")
            if "API LLM delegates have NO filesystem access" not in content:
                additions.append("\n### API LLM delegates have NO filesystem access — MANDATORY\n\nAPI delegates (local LLM via LM Studio, Kimi K2, Gemini, Haiku) are **remote endpoints with zero filesystem access**. They can only work with what is pasted into the prompt. Asking them to modify a file they have never seen produces empty or hallucinated output.\n\n**Before every delegation that involves existing code:**\n1. Read the target file(s) with the Read tool first\n2. Paste the relevant sections verbatim into the delegation prompt — full file if <100 lines, the target function/class ±20 lines context if larger\n3. Include imports and type signatures so the delegate understands the surrounding API\n4. State explicitly which file and line range to modify\n\nOnly Claude Code subagents (spawned via the `Agent` tool) have real filesystem tool access and can read files themselves.\n")
                print(f"  Will append API LLM filesystem rule to {claude_md_path}.")
            if "AI-Bridge changes MUST be validated" not in content:
                additions.append("\n### AI-Bridge changes MUST be validated before committing — MANDATORY\n\nWhen `apply=true` writes EDIT/FILE/DELETE blocks to disk, validate before committing:\n1. Syntax check: `python3 -m py_compile <changed_file>`\n2. Run tests: `python3 -m pytest tests/ -q --tb=short -x`\n3. Read the diff: `git diff` — confirm intent matches output\n\nNever commit AI-Bridge applied changes without this validation.\n")
                print(f"  Will append AI-Bridge validation rule to {claude_md_path}.")
            if additions:
                if not dry_run:
                    claude_md_path.write_text(content.rstrip() + "".join(additions))
                return True
            return False


def _get_hooks_dir() -> Path | None:
    """Return the absolute path to Mullet's scripts/hooks directory.

    Works for editable installs (pip install -e .) where __file__ lives in
    the source tree.  Returns None for non-editable pip installs where the
    scripts/ directory is not shipped — callers should skip hook installation
    and print a warning in that case.
    """
    candidate = Path(__file__).resolve().parent.parent.parent / "scripts" / "hooks"
    if candidate.exists() and (candidate / "delegation_reminder.py").exists():
        return candidate
    return None


def _merge_claude_settings_hooks(dry_run: bool):
    claude_settings_path = _get_claude_settings_path()

    # Static hooks that don't need the scripts directory
    static_pretooluse = [
        {
            "matcher": "Grep",
            "hooks": [
                {
                    "type": "command",
                    "command": "echo '{\"type\":\"warning\",\"message\":\"DELEGATION REMINDER: For broad codebase searches use mcp__mullet__trawl_search_code or trawl_find_symbol instead of Grep. Only use Grep for targeted single-file searches.\"}'"
                }
            ]
        },
        {
            "matcher": "Edit|Write",
            "hooks": [
                {
                    "type": "command",
                    "command": "echo \"{\\\"type\\\":\\\"info\\\",\\\"message\\\":\\\"REPO CHECK: $(pwd) | branch: $(git branch --show-current 2>/dev/null || echo not-git)\\\"}\""
                }
            ]
        },
    ]

    # Dynamic hooks that require the scripts/hooks directory (editable install only)
    hooks_dir = _get_hooks_dir()
    if hooks_dir is None:
        print(
            "  WARNING: scripts/hooks not found — hook scripts require an editable install "
            "(pip install -e .). Skipping UserPromptSubmit/mullet_pretooluse hooks."
        )

    hooks_to_merge: dict = {"PreToolUse": static_pretooluse}

    if hooks_dir:
        def _h(name: str) -> str:
            return f"python3 {hooks_dir / name}"

        hooks_to_merge["UserPromptSubmit"] = [
            {
                "matcher": "",
                "hooks": [
                    {"type": "command", "command": _h("compress_whitespace.py")},
                    {"type": "command", "command": _h("delegation_reminder.py")},
                    {"type": "command", "command": _h("skill_activator.py")},
                    {"type": "command", "command": _h("session_context.py")},
                    {"type": "command", "command": _h("user_prompt_detect_override.py")},
                ],
            }
        ]
        hooks_to_merge["PreToolUse"].append(
            {
                "matcher": "mcp__mullet__.*",
                "hooks": [{"type": "command", "command": _h("mullet_pretooluse_rewriter.py")}],
            }
        )
        hooks_to_merge["PostToolUse"] = [
            {
                "matcher": "Read|Write|Edit|Bash",
                "hooks": [{"type": "command", "command": _h("mullet_observation_capture.py")}],
            }
        ]

    if not claude_settings_path.exists():
        if not dry_run:
            claude_settings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(claude_settings_path, "w") as f:
                json.dump({"hooks": hooks_to_merge}, f, indent=2)
        print(f"  Created {claude_settings_path} with Mullet hooks.")
        return True

    with open(claude_settings_path) as f:
        current_settings = json.load(f)

    if "hooks" not in current_settings:
        current_settings["hooks"] = {}

    changed = False
    for event, new_groups in hooks_to_merge.items():
        existing_groups = current_settings["hooks"].setdefault(event, [])
        for new_group in new_groups:
            matcher = new_group.get("matcher", "")
            existing_group = next(
                (g for g in existing_groups if g.get("matcher", "") == matcher), None
            )
            if existing_group is None:
                existing_groups.append(new_group)
                changed = True
            else:
                # Group exists — merge individual hooks that are missing.
                existing_cmds = {h.get("command", "") for h in existing_group.get("hooks", [])}
                for new_hook in new_group.get("hooks", []):
                    cmd = new_hook.get("command", "")
                    if cmd not in existing_cmds:
                        existing_group.setdefault("hooks", []).insert(0, new_hook)
                        existing_cmds.add(cmd)
                        changed = True

    if changed:
        if not dry_run:
            with open(claude_settings_path, "w") as f:
                json.dump(current_settings, f, indent=2)
        print(f"  Merged Mullet hooks into {claude_settings_path}.")
        return True
    else:
        print(f"  Mullet hooks already present in {claude_settings_path}. Skipped.")
        return False


# Step 5 of the /commit skill. With MULLET_NO_AI_ATTRIBUTION on, the personal
# no-attribution rule replaces the neutral wording.
_COMMIT_STEP_NEUTRAL = (
    "Commit using a HEREDOC so the message keeps its formatting; "
    "follow the repository's own conventions for trailers"
)
_COMMIT_STEP_NO_AI_ATTRIBUTION = (
    "Commit using HEREDOC — never append Co-Authored-By or attribution lines"
)


def _copy_claude_skill_files(dry_run: bool, force: bool):
    claude_skills_dir = _get_claude_skills_dir()
    claude_skills_dir.mkdir(parents=True, exist_ok=True)

    skills_to_copy = {
        "delegate.md": """# /delegate — Delegate Task to Cheaper Model

When this skill is invoked, follow these steps exactly:

## Step 1: Classify the task
- **File edit (any size)** → `mullet_opencode` (has real filesystem access)
- **Research/analysis (text output only)** → `mullet_run --provider kimi`
- **Edit < 5 lines** → use `Edit` tool directly, no delegation needed
- **Complex multi-file reasoning** → `Agent` subagent

## Step 2: Construct the delegation

**For file edits — always use `mullet_opencode`:**
```
mcp__mullet__mullet_opencode(
    task="<describe the edit with file path and what to change>"
)
```
The delegate has filesystem access and reads files itself — do NOT paste file content.

**For research/analysis — use `mullet_run` (text output only, no file writes):**
- Write a self-contained prompt with all necessary context
- `mullet_run` has NO filesystem access — include any relevant excerpts in the prompt

## Step 3: Execute the fallback chain (do NOT skip steps)
1. File edits → `mullet_opencode` (OpenRouter default; substitute Haiku when pressure <60%)
2. Research → `mullet_run --provider openrouter` (default pay-per-token)
3. If Anthropic pressure <60% → substitute `mullet_run --provider haiku` or `--provider sonnet`
4. If OpenRouter fails → `mullet_run --provider kimi`
5. Gemini → last resort only: `mullet_run --provider gemini`
6. If opencode unavailable → `Agent` subagent (has filesystem access)
7. Falling back to direct Opus **requires explicit user approval**

## Step 4: Return results
- Report back with a diff summary only — no inline full-file content
- Note which provider succeeded and approximate token use
- If all delegates failed, report clearly and ask before doing the work yourself

## Rules
- NEVER use `mullet_run apply=true` — opencode handles all file edits
- Do NOT do exploratory Bash/grep in Opus to "prepare" the delegation prompt
- Do NOT fall back to Opus silently because delegation is slower
- Do NOT retry the same failing provider more than once without changing the approach
""",
        "pool-config.md": """# /pool-config — Audit and Improve a Mullet Pool Config

Use this skill when the user wants to review, improve, or create a `config.yaml` for a Mullet pool. Triggered proactively when search quality is poor or a new project scope is identified.

---

## Step 1: Identify the pool

If no pool name is given, ask. Pool configs live at:
```
~/.mullet/pools/<pool-name>/config.yaml
```

Read the existing config. If none exists, proceed to Step 4.

## Step 2: Inventory the documents directory

List files in the pool's `documents/` directory:
```
~/.mullet/pools/<pool-name>/documents/
```

Group by type: markdown, code, PDF, HTML, YAML, other.

## Step 3: Audit the current config

Check:
- `name` and `machine_name` — must be set, machine_name lowercase-hyphenated
- `documents.path` — must point to an existing directory
- `documents.extensions` — must cover the file types actually present
- `documents.exclude_patterns` — reasonable? Any build artifacts being indexed?
- `chunking.strategy` — `semantic` for docs, `code` for code repos
- `chunking.chunk_size` / `chunk_overlap` — 512/64 is a good default for docs
- `embedding.model` — should match what's actually running (LM Studio or remote)
- `search.top_k` — 10 is a good default
- `entity_extraction.enabled` — true if the pool has structured content worth graphing

Report any gaps or mismatches.

## Step 4: Generate or improve the config

Output a complete improved `config.yaml`. Use this schema:

```yaml
name: "Human Readable Pool Name"
machine_name: "pool-name-slug"

documents:
  path: "documents"
  extensions:
    - .md
    - .txt
    - .pdf
    - .html
  exclude_patterns:
    - "*.log"
    - "__pycache__"
    - ".git"

chunking:
  strategy: semantic      # semantic | code | fixed
  chunk_size: 512
  chunk_overlap: 64

embedding:
  model: "text-embedding-nomic-embed-text-v1.5"   # or remote model

search:
  top_k: 10
  score_threshold: 0.3

entity_extraction:
  enabled: true
  types:
    - Person
    - Organization
    - Technology
    - Concept
```

## Step 5: Apply and reindex

After writing the config:
1. Confirm with the user before overwriting
2. Run: `mullet reindex <pool-name>` to apply
3. Run a test search to verify recall improved
""",
        "commit.md": """# /commit — Disciplined Commit Workflow

## Steps

1. Run `git status` (never -uall) and `git diff --staged`
2. Run `git log --oneline -5` to match commit style
3. Draft commit message: imperative mood, under 72 chars, focus on "why" not "what"
4. Stage files by name only — NEVER `git add -A` or `git add .`
5. Commit using a HEREDOC so the message keeps its formatting; follow the repository's own conventions for trailers
6. Run `git status` after to verify
7. Do NOT push unless user explicitly asks

## Anti-Patterns
- Never `--no-verify` — fix the hook cause instead
- Never `--amend` published commits
- Never batch unrelated changes in one commit
- If pre-commit hook fails: fix, re-stage, NEW commit (never --amend)
""",
        "review.md": """# /review — Code Review

## Trigger
- `/review` — review staged changes (`git diff --cached`)
- `/review HEAD` — all changes since main
- `/review path/to/file.py` — specific files

## Behavior
1. Read the diff (Read tool or `git diff`) — do NOT grep the whole codebase
2. Check: bugs, security (SQLi/XSS/injection/hardcoded creds), test regressions, schema assumptions, scope creep
3. Python-specific: type hints on public APIs, no TODO stubs, no commented-out code
4. Report in Must Fix / Should Fix / Optional format
5. Do NOT rewrite code — report only, user decides

## Delegation
If >5 files: delegate to `mullet_run --provider local` with Must/Should/Optional format instruction.

## Notes
- Run `pwd && git branch` first to confirm correct repo
- For DB-touching changes: read schema file before assuming field names
"""
    }

    from mullet.core.env import no_ai_attribution
    if no_ai_attribution():
        skills_to_copy["commit.md"] = skills_to_copy["commit.md"].replace(
            _COMMIT_STEP_NEUTRAL, _COMMIT_STEP_NO_AI_ATTRIBUTION,
        )

    results = {}
    for filename, content in skills_to_copy.items():
        dest_path = claude_skills_dir / filename
        if dest_path.exists() and not force:
            results[filename] = f"  Skill file {filename} already exists. Skipped (use --force to overwrite)."
        else:
            if not dry_run:
                dest_path.write_text(content)
            results[filename] = f"  {'Overwrote' if dest_path.exists() else 'Copied'} skill file {filename} to {claude_skills_dir}."

    for message in results.values():
        print(message)
    return any("Copied" in msg or "Overwrote" in msg for msg in results.values())


def _install_claude_agents(dry_run: bool, force: bool):
    claude_agents_dir = _get_claude_dir() / "agents"
    claude_agents_dir.mkdir(parents=True, exist_ok=True)

    agents_to_create = {
        "patch-writer.md": """---
name: patch-writer
description: Generates unified diff patches for file changes. Use when a file needs editing but you want patch output rather than full file rewrite. Specialist for large file changes (>200 lines).
tools: Read, Glob, Grep
---

You generate unified diff patches only — never full file rewrites.

1. Read the target file fully before generating any patch
2. Output unified diff format: `--- a/file`, `+++ b/file`, `@@ hunks @@`
3. 3 lines of context around each change, minimal hunks
4. Never output full file content
5. Include a one-line patch description at the top
6. Decline if: file unreadable, >3 files affected, or structure unclear
""",
        "test-runner.md": """---
name: test-runner
description: Runs the test suite and reports failures. Returns pass/fail counts and failing test names only. Does not fix failures.
tools: Bash
---

Run `python -m pytest --tb=short -q --no-header` from project root.
For targeted runs: `pytest --tb=short -q path/to/test_file.py`

Report format:
- Passed / Failed / Error counts
- Each failing test: `module::test_name — one-line reason`

Rules: report only, never fix. If pytest missing or no tests found, say so.
""",
        "verifier.md": """---
name: verifier
description: Verifies a patch or code change is safe to apply. Checks syntax, schema assumptions, security, and test regressions. Returns PASS or FAIL with specifics.
tools: Read, Bash, Grep
---

Before a patch is committed, verify:
1. Syntax: `python -m py_compile` on all affected Python files
2. Schema: confirm any DB table/column names exist in migration or schema files
3. Security: no hardcoded secrets, no SQL interpolation, no shell injection
4. Tests: `pytest --tb=short -q` — report any newly failing tests
5. Logic: imports present, no obvious None dereference

Output: `Status: PASS` or `Status: FAIL` with one line per issue.
Never apply fixes — report only.
"""
    }

    results = {}
    for filename, content in agents_to_create.items():
        dest_path = claude_agents_dir / filename
        if dest_path.exists() and not force:
            results[filename] = f"  Agent file {filename} already exists. Skipped (use --force to overwrite)."
        else:
            if not dry_run:
                dest_path.write_text(content)
            results[filename] = f"  {'Overwrote' if dest_path.exists() else 'Created'} agent file {filename} in {claude_agents_dir}."

    for message in results.values():
        print(message)
    return any("Created" in msg or "Overwrote" in msg for msg in results.values())


def _install_project_hooks(project_dir: Path, dry_run: bool):
    project_settings_path = project_dir / ".claude" / "settings.json"
    project_settings_path.parent.mkdir(parents=True, exist_ok=True)

    post_tool_use_hook = {
        "matcher": "Edit|Write",
        "hooks": [
            {
                "type": "command",
                "command": "cd \"${CLAUDE_PROJECT_DIR}\" && python3 -m pytest --tb=short -q --no-header 2>&1 | tail -30"
            }
        ]
    }

    if not project_settings_path.exists():
        settings = {"hooks": {"PostToolUse": [post_tool_use_hook]}}
        if not dry_run:
            with open(project_settings_path, "w") as f:
                json.dump(settings, f, indent=2)
        print(f"  Created {project_settings_path} with PostToolUse pytest hook.")
        return True
    else:
        with open(project_settings_path) as f:
            current_settings = json.load(f)

        if "hooks" not in current_settings:
            current_settings["hooks"] = {}
        if "PostToolUse" not in current_settings["hooks"]:
            current_settings["hooks"]["PostToolUse"] = []

        hook_exists = False
        for existing_hook in current_settings["hooks"]["PostToolUse"]:
            if existing_hook.get("matcher") == "Edit|Write" and "pytest" in str(existing_hook.get("hooks", [])):
                hook_exists = True
                break

        if not hook_exists:
            current_settings["hooks"]["PostToolUse"].append(post_tool_use_hook)
            if not dry_run:
                with open(project_settings_path, "w") as f:
                    json.dump(current_settings, f, indent=2)
            print(f"  Added PostToolUse pytest hook to {project_settings_path}.")
            return True
        else:
            print(f"  PostToolUse pytest hook already present in {project_settings_path}. Skipped.")
            return False


_OPENCODE_CONFIG_PATH = Path.home() / ".config" / "opencode" / "opencode.json"
_GLOBAL_MCP_PATH = Path.home() / ".claude" / "mcp.json"
_OPENCODE_DEFAULT_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "model": "moonshot/kimi-k2.7-code",
    "provider": {
        "moonshot": {
            "name": "Moonshot",
            "api": "https://api.moonshot.ai/v1",
            "env": ["KIMI_API_KEY"],
            "models": {
                "kimi-k2.7-code": {
                    "name": "Kimi K2.7-Code",
                    "tool_call": True,
                    "limit": {"context": 262144, "output": 32768},
                    "cost": {"input": 0.95, "output": 4.0},
                },
                "kimi-k2.6": {
                    "name": "Kimi K2.6",
                    "tool_call": True,
                    "limit": {"context": 262144, "output": 32768},
                    "cost": {"input": 0.60, "output": 2.50},
                },
                "kimi-k2.5": {
                    "name": "Kimi K2.5",
                    "tool_call": True,
                    "limit": {"context": 262144, "output": 32768},
                    "cost": {"input": 0.6, "output": 3.0},
                },
            },
        }
    },
}


def _install_opencode(dry_run: bool, yes: bool = False) -> bool:
    """Check for opencode, offer to install it, and write Moonshot config.

    opencode is a zero-Anthropic-token coding agent with native filesystem
    access. Mullet uses it via mullet_opencode (ACP JSON-RPC over stdio).
    """
    already_installed = shutil.which("opencode") is not None
    config_exists = _OPENCODE_CONFIG_PATH.exists()

    if already_installed and config_exists:
        print("  opencode already installed and configured. Skipped.")
        return False

    if not already_installed:
        if not yes:
            try:
                answer = input(
                    "  Install opencode? (Zero-Anthropic-token coding agent — uses Kimi/local.) [y/N] "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer not in ("y", "yes"):
                print("  Skipped.")
                return False

        if dry_run:
            print("  [dry-run] Would run: npm i -g opencode-ai@latest")
        else:
            print("  Installing opencode via npm...")
            result = subprocess.run(
                ["npm", "i", "-g", "opencode-ai@latest"],
                capture_output=False,
            )
            if result.returncode != 0:
                print("  opencode install failed — you can install manually: npm i -g opencode-ai@latest")
                return False
            print("  opencode installed.")

    if not config_exists:
        if dry_run:
            print(f"  [dry-run] Would write opencode Moonshot config to {_OPENCODE_CONFIG_PATH}")
        else:
            _OPENCODE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            _OPENCODE_CONFIG_PATH.write_text(json.dumps(_OPENCODE_DEFAULT_CONFIG, indent=2) + "\n")
            print(f"  Wrote opencode Moonshot/Kimi config to {_OPENCODE_CONFIG_PATH}")
            print("  Set KIMI_API_KEY in your environment for opencode to use Kimi K2.")

    return True


_KARPATHY_PLUGIN_URL = "https://github.com/forrestchang/andrej-karpathy-skills"


def _install_statusline(dry_run: bool) -> bool:
    """Write the statusLine config into ~/.claude/settings.json.

    Uses shutil.which to resolve mullet-hud at install time so the stored path
    works on any system regardless of the installing user's HOME or PATH layout.
    """
    hud_path = shutil.which("mullet-hud")
    if hud_path is None:
        print(
            "  WARNING: mullet-hud not found on PATH — statusLine not configured. "
            "Re-run after installing Mullet."
        )
        return False

    settings_path = _get_claude_settings_path()
    try:
        settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    except json.JSONDecodeError:
        settings = {}

    existing = settings.get("statusLine", {})
    current_cmd = existing.get("command", "") if isinstance(existing, dict) else ""

    # Already correct — absolute path pointing to the same binary.
    if current_cmd == hud_path:
        print(f"  statusLine already configured correctly ({hud_path}). Skipped.")
        return False

    settings["statusLine"] = {
        "type": "command",
        "command": hud_path,
        "refreshInterval": 30,
    }

    if dry_run:
        print(f"  [dry-run] Would set statusLine command to {hud_path}")
        return True

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"  Configured statusLine → {hud_path}")
    return True


def _install_global_mcp(dry_run: bool) -> bool:
    """Write the mullet MCP server config to ~/.claude/mcp.json.

    This makes mullet tools available in every project without a per-project .mcp.json.
    """
    source = Path(__file__).resolve().parent.parent.parent / ".mcp.json"
    if not source.exists():
        print("  WARNING: .mcp.json not found in mullet source tree — skipping global MCP install.")
        return False

    import json as _json
    new_config = _json.loads(source.read_text())

    if _GLOBAL_MCP_PATH.exists():
        existing = _json.loads(_GLOBAL_MCP_PATH.read_text())
        if existing.get("mcpServers", {}).get("mullet"):
            print(f"  mullet MCP already present in {_GLOBAL_MCP_PATH}. Skipped.")
            return False
        # Merge — preserve any other servers the user has configured.
        existing.setdefault("mcpServers", {}).update(new_config.get("mcpServers", {}))
        merged = existing
    else:
        merged = new_config

    if dry_run:
        print(f"  [dry-run] Would write mullet MCP config to {_GLOBAL_MCP_PATH}")
        return True

    _GLOBAL_MCP_PATH.parent.mkdir(parents=True, exist_ok=True)
    _GLOBAL_MCP_PATH.write_text(_json.dumps(merged, indent=4) + "\n")
    print(f"  Wrote mullet MCP config to {_GLOBAL_MCP_PATH}")
    return True


def _offer_karpathy_plugin(dry_run: bool, yes: bool = False) -> bool:
    """Offer to install the andrej-karpathy-skills CC plugin.

    Installs by appending the plugin URL to ~/.claude/settings.json plugins list.
    Only prompts if not already installed and not in --yes mode.
    """
    settings_path = _get_claude_settings_path()

    # Load existing settings
    try:
        with open(settings_path) as f:
            settings = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        settings = {}

    plugins = settings.get("plugins", [])
    if any(_KARPATHY_PLUGIN_URL in str(p) for p in plugins):
        print("  andrej-karpathy-skills plugin already installed. Skipped.")
        return False

    # Prompt unless --yes
    if not yes:
        try:
            answer = input(
                "  Install andrej-karpathy-skills CC plugin? (Adds Think Before Coding, Simplicity First, "
                "Surgical Changes guidelines to CC.) [y/N] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer not in ("y", "yes"):
            print("  Skipped.")
            return False

    if dry_run:
        print(f"  [dry-run] Would add {_KARPATHY_PLUGIN_URL} to plugins list in {settings_path}")
        return True

    plugins.append(_KARPATHY_PLUGIN_URL)
    settings["plugins"] = plugins
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"  Added andrej-karpathy-skills plugin to {settings_path}")
    print(f"  Restart Claude Code to activate. Plugin URL: {_KARPATHY_PLUGIN_URL}")
    return True


def run_setup_claude(dry_run: bool = False, yes: bool = False, force: bool = False):
    """Set up Claude Code global configuration for Mullet."""
    print("Setting up Claude Code global configuration for Mullet users...")

    if dry_run:
        print("\n*** DRY RUN: No changes will be applied. ***\n")

    changes_made = []

    print("\n--- CLAUDE.md ---")
    if _append_claude_md_section(dry_run):
        changes_made.append("CLAUDE.md updated")

    print("\n--- settings.json ---")
    if _merge_claude_settings_hooks(dry_run):
        changes_made.append("settings.json updated")

    print("\n--- Global MCP config ---")
    if _install_global_mcp(dry_run):
        changes_made.append("Global MCP config written")

    print("\n--- HUD statusLine ---")
    if _install_statusline(dry_run):
        changes_made.append("statusLine configured")

    print("\n--- Skill Files ---")
    if _copy_claude_skill_files(dry_run, force):
        changes_made.append("Skill files copied")

    print("\n--- Agent Files ---")
    if _install_claude_agents(dry_run, force):
        changes_made.append("Agent files installed")

    print("\n--- Project Hooks ---")
    project_dir = Path.cwd()
    if (project_dir / ".claude" / "settings.json").exists() or (project_dir / ".git").exists():
        if _install_project_hooks(project_dir, dry_run):
            changes_made.append("Project hooks installed")

    print("\n--- opencode (zero-Anthropic coding agent) ---")
    if _install_opencode(dry_run, yes):
        changes_made.append("opencode installed/configured")

    print("\n--- Karpathy Skills Plugin ---")
    if _offer_karpathy_plugin(dry_run, yes):
        changes_made.append("andrej-karpathy-skills plugin installed")

    print("\n--- Pool Hardware Tune ---")
    try:
        from mullet.server.mcp_knowledge import _get_pools_dir
        from mullet.cli.pool_tune import run_pool_tune
        pools_dir = _get_pools_dir()
        if pools_dir.exists() and any(pools_dir.iterdir()):
            if not dry_run:
                run_pool_tune(pools_dir, dry_run=False)
            else:
                run_pool_tune(pools_dir, dry_run=True)
            changes_made.append("Pool settings tuned")
        else:
            print("  No pools found — skipped (run 'mullet pool tune' after creating pools).")
    except Exception as e:
        print(f"  Pool tune skipped: {e}")

    print("\n--- Summary ---")
    if changes_made:
        print("Successfully applied the following changes:")
        for change in changes_made:
            print(f"- {change}")
        if dry_run:
            print("\n(Dry run complete. No actual changes were made.)")
        else:
            print("\nSetup complete!")
    else:
        print("No changes were needed. Claude Code config is already up-to-date with Mullet recommendations.")
