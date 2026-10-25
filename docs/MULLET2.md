# Mullet 2: a global, gated job runner

## Context

Mullet 2 exists so that long model-driven work can run unattended and be believed:

- **Status comes from evidence.** A step is done when Mullet's own gates pass: exit codes, test reports and diffs. Words in a model's reply never mark anything passed or done.
- **Workers are isolated.** Each job runs in a private clone with a sandbox: no write access outside the clone, no network, no loaded keys. Your checkout is never touched.
- **It runs from any machine.** Paths, settings and keys come from the config directory, never from a source checkout, and Mullet installs from the public repo with `scripts/install.sh`.
- **Changes are quick to verify.** The suite is about 18,000 tests, run in parallel in about 5 minutes.

**Outcome:**
1. On a Linux machine: `uv tool install`, then `mullet job plan "<goal>"` in any repo, Mullet included.
2. You approve a task tree.
3. Mullet runs it unattended in a sandboxed private clone, checking every step with tests and checks it runs itself.
4. It hands back a branch and an evidence report. Your checkout, including `.git`, is untouched until you merge.

## Decisions

- **Recursion is a task tree with gated nodes.** The planner breaks a goal down until each leaf fits one pass. A parent passes only when its children and its own integration checks pass. Improving Mullet is just a job on the Mullet repo.
- **Approvals happen twice:** the plan before work starts and the merge at the end. In v1.1, a runtime split inside the approved scope runs without asking, and a split that widens the scope pauses.
- **Jobs run on Linux**, and on Windows through WSL. On macOS the rest of Mullet works, but jobs refuse to start until a macOS sandbox exists.
- **Agents spawning agents (recommendation):**
  - Workers never spawn Mullet workers. Only the orchestrator creates nodes, so budgets, depth and checks live in one place.
  - Claude Code's own subagents inside a node count against that node's budget.
- **Free tiers (recommendation): v1.1, routed by how checkable the work is.**
  - Small mechanical leaves on repos marked public may try opencode with a free model or freellmapi first.
  - Two failed attempts escalate to Claude.
  - Private repos only ever see Claude or the local LLM.
  - This waits for v1.1 because opencode has no OS sandbox of its own and needs Mullet's sandbox wrapper first.
- **Trawl (recommendation): v1.1, one repo and one install story.**
  - Trawl becomes the `mullet[trawl]` extra with its own MCP server registration.
  - Core no longer pulls in chromadb, sentence-transformers or torch (which arrives through sentence-transformers), and sessions don't load 43 extra tools.
- **The worker is your own logged-in `claude -p`.** The docs allow this for your own unmodified binary. In return, Mullet:
  - never reads, stores or passes Claude credentials;
  - never uses the Agent SDK with subscription login;
  - strips `ANTHROPIC_API_KEY` from the worker's environment unless `billing: api` is set, because in `-p` mode the key silently wins over the subscription;
  - runs one Claude worker at a time across all jobs, since they share your subscription quota. opencode workers on paid or local models have their own slots (`MULLET_API_WORKERS`, default 3) and run alongside it.

## Architecture

```
goal ─► Planner (claude -p, read-only, --json-schema task tree) ─► plan.md ─► you approve (base SHA pinned)
                                     │
private clone of the pinned SHA (git dir outside the worker's reach) on branch mullet/<job>
                                     │
for each node:  Executor (claude -p, sandboxed)  ─►  Gates (run by Mullet, also sandboxed)
     pass ─► Mullet commits the node
     fail ─► reset to node start, retry with gate output, up to 3 times ─► BLOCKED (never "done")
                                     │
Ledger (SQLite + transcripts) ─► report from the ledger ─► `mullet job merge` fetches the branch into your repo
```

The new package is `mullet/jobs/`. It is built alongside the old code and doesn't depend on it.

### Isolation (`clone.py`)
- One private clone per job holding the base commit and its history and nothing else: `git init --separate-git-dir=$DATA/jobs/<id>/git`, then `git fetch --no-tags <repo> <base_sha>` (protocol v2), with `gc.auto=0`. Your other branches, unpushed commits, stashes and tags never reach the clone or a worker, and the objects arrive as a pack, never hard links, so the jobs directory can be on any filesystem.
- The git dir sits outside every sandbox's writable area, so a worker can't move refs, drop stashes or prune objects. A git worktree would leave the shared `.git` writable (Claude sandbox docs).
- Every Mullet git command runs with `-c core.hooksPath=/dev/null`, because a tracked `.husky/` could be edited by the worker.
- Merge means fetching `mullet/<job>` from the clone into your repo, and only on `mullet job merge`.

### Sandbox (`sandbox.py`)
- One wrapper for the worker's Bash, `setup`, and every gate command: bubblewrap on Linux, Seatbelt on macOS.
- It allows writes to the clone's working tree only, gives no network, and passes an allowlisted environment, so none of Mullet's own loaded keys get through (the MCP server loads `.env` into `os.environ`).
- The one exception is `setup`: its commands run with the host's network (no `--unshare-net`), because package managers need it. They can reach anything the host can, loopback services and the local network included, and nothing filters hosts. They are still confined in every other way (read-only filesystem apart from the clone and the job's caches, allowlisted environment, none of Mullet's keys). Setup commands come from the stored profile or from `.mullet/project.yaml` as committed at the job's base commit (an in-repo file may change `setup`; it can't set `setup_writable`, `expose_ro`, `copy_in` or `opencode_lsp`). A worker's edits never change them. Unshared-network setup would need a proxy or pre-fetched dependencies, which isn't built.
- Gates run the code the worker just wrote, so they need the sandbox as much as the worker does.
- The worker's Claude settings are fail-closed:
  - `sandbox.enabled`
  - `failIfUnavailable: true`
  - `allowUnsandboxedCommands: false`
  - `autoAllowBashIfSandboxed: false`
  - `denyRead: ["~/"]`, with the clone and toolchain directories allowed back in
  - `allowedDomains: []`
- If the sandbox self-test (`bwrap --ro-bind / / true`, which passes on this box) fails, the worker gets no Bash. Gates still run, but only inside a working sandbox, or the job refuses to start.

### Dependencies
- Install once per job from the base lockfile. Dependency directories (`deps_paths`, such as `.venv` or `node_modules`) stay in the tree but are mounted read-only for the worker and gates, and a dependency path that has been swapped for a symlink fails the gates.
- Package caches (pip, uv, npm, yarn, composer, Go, cargo) live in the job directory, so setup never writes to the user's own caches.
- The worker can't write to dependency directories: its Claude settings deny Edit and Write there, and its sandbox lists them in `denyWrite`. As a backstop, the runner fingerprints them after setup (path, mode, size, inode and ctime of every entry) and every check gate fails if the fingerprint has changed.
- Reinstall only when a lockfile changes inside the approved scope.
- Resets therefore don't reinstall everything, and the worker can't tamper with `node_modules/.bin`.

### Ledger (`ledger.py`)
- SQLite tables for jobs, nodes, attempts, gate_results and events.
- A node's status is set from gate results only.
- The schema is copied from `knowledge/engine/workflow.py` rather than imported, since Trawl becomes optional.

### Project profile (`profile.py`)
- Detection extends `core/agent/project_detect.py` with composer/phpunit, make and ddev, and with lint and typecheck commands.
- Profiles are stored in `~/.config/mullet/projects/<repo-id>.yaml`. An optional in-repo `.mullet/project.yaml` overrides it.
- Fields:
  - `setup`;
  - `checks`: lint, typecheck, `test_targeted`, `test_full`, with junit XML output where the tool supports it;
  - timeouts;
  - `protected_paths`;
  - `data_class`: private by default;
  - `copy_in`.
- Default protected paths: `.git`, `.gitmodules`, `.gitattributes`, `.lfsconfig`, `.husky`, `CLAUDE.md`, `.claude/`, `.mcp.json`, `opencode.json`, `.mullet/`, `.env*`, CI workflows (`.github/workflows`, `.github/actions`, `.gitlab-ci.yml`), `.pre-commit-config.yaml`, files your own tools run when you open the repo (`.envrc`, `.vscode`, `.devcontainer`, `.npmrc`, `.yarnrc`, `.pypirc`), `AGENTS.md`, lockfiles, and test runner config (`pytest.ini`, `tox.ini`, `jest.config.*`, `phpunit.xml*` and similar). The list is `DEFAULT_PROTECTED` in `mullet/jobs/profile.py`. The profile's `copy_in` paths are protected too. Changing any of them needs node-level permission in the approved plan. `.gitignore` is not protected, because anything it hides is removed before gates run.
- Mullet's git commands ignore the user's global and system git config and attributes files, so a `.gitattributes` in the work tree can't switch on filter or diff drivers. Diffs are always read as text.
- `copy_in` files (for example a test `.env`):
  - are copied only into the gate sandbox, after the executor exits;
  - are opened without following symlinks;
  - are never staged;
  - are never shown to the reviewer.

### Gates (`gates.py`)
- A new gate runner with profile timeouts and full logs, and per-test results from junit XML.
- Each gate command runs with the profile's timeout, and its full output is kept in the job's logs.

| Gate | Check | Outcome |
|---|---|---|
| G0 baseline | Profile checks on the pinned base, once per job | A red or broken baseline doesn't block the job. G5 becomes "not run", nodes become "unverified", and the plan says so. |
| G1 real change | Diff from git, untracked files included | Fails when an edit task leaves an empty diff |
| G2 scope | Files against the node scope, protected paths, size limits, symlinks pointing out of the repo | Fails when anything falls outside them |
| G3 static | `diff_risk.py`, secret scan (`gitleaks --redact`, or a regex fallback), deletion count | Fails when flags or secrets are over the limit |
| G4 build | Lint, typecheck and build | Fails on a non-zero exit |
| G5 tests | Targeted tests per node: the planner's acceptance commands, tests touched in the diff, and tests mapped by name. The full suite at parent nodes and at job end. | Fails when a test that passed on base no longer passes (a skip counts as a loss), when a test newly fails, when no report is produced, or when no tests run. In a targeted run only the tests that ran are compared, and the full run catches tests that vanished. A failure is rerun up to twice on the branch, which filters out flaky tests: a test is excused only if it passed in one of the runs. A test that failed and never passed (skipped or missing in the other runs) is a regression, whichever run it failed in. |
| G6 test weakening | Diff of test files | Fails when assertions are removed, skip/xfail is added or the assertion count drops, unless the node allows it |
| G7 review | A fresh `claude -p` reviewer session, run after G1 to G6 pass, with no author transcript and no `copy_in` files, gives a schema-checked JSON verdict | The reviewer can block a node. It can never pass one. A review with no verdict is never read as unverified: the job pauses (see the runner below). |

### Executor (`executors/claude_code.py`)
- Command: `claude -p --restricted --strict-mcp-config --disable-slash-commands --tools Read,Edit,Write,Glob,Grep,Bash --settings <job.json> --permission-mode dontAsk --permission-prompts none --output-format stream-json --json-schema <result>`, with cwd set to the clone.
- Checked against Claude Code 2.1.302 on 2026-10-24:
  - `--max-turns` works, though `--help` doesn't list it. The run ends with `terminal_reason: max_turns`.
  - `--restricted` keeps subscription login (`apiKeySource: none`).
  - `dontAsk` denies every tool that isn't explicitly allowed, so `job.json` must allow `Edit(./**)`, `Write(./**)` and the profile's Bash commands.
  - With `failIfUnavailable`, Claude Code refuses to start when its sandbox dependencies are missing. On Linux they are bubblewrap and socat. Doctor must check both.
  - Pass stdin explicitly (Mullet sends the prompt on stdin), or the CLI waits 3s for input.
  - Claude Code's sandbox lets Bash write to its per-user temp directory, `/tmp/claude-<uid>`, which every session shares. A real probe wrote outside the repo that way. Each attempt therefore gets its own short `TMPDIR` (`/tmp/mullet-XXXX`, removed afterwards). It must be short because the sandbox puts Unix sockets in it. With it, Bash writes outside the repo and to `/tmp/claude-<uid>` are refused.
  - Workers may only run single Bash commands that start with an allowed prefix (read-only tools plus the profile's checks). Pipes, redirects and `&&` are refused, so the prompt says so.
  - Probed on 2026-10-24, with the user in the docker group:
    - The worker's Bash can't connect to `/var/run/docker.sock` ("Operation not permitted"), and Mullet's gate sandbox can't even see it.
    - Hooks planted in the clone's `.claude/settings.json` and `settings.local.json` don't fire under `--restricted`.
    - `/run`, `/var/run` and `/var/snap` are denied anyway, and `XDG_RUNTIME_DIR` isn't passed.
- Usage limits:
  - The stream carries `rate_limit_event` with `utilization` and `resetsAt` for the five-hour and seven-day windows, so the runner can pause before a limit as well as after one.
  - A usage-limit exit pauses the job until the limit resets.
- The stream is parsed the way `tools/opencode_tool.py` already parses its JSONL.

### Planner, runner, report
- The planner (`planner.py`):
  - A read-only `claude -p` session (Read, Glob and Grep only) on the job's clone answers with a schema-checked flat list of nodes, each naming its parent. Each node has an id, goal, kind, scope, acceptance commands, and any protected paths or test removals it is allowed. It may also name the nodes it waits for (`after`) and mark itself a one-way door (`door: one-way`); the plan may list up to 10 `open_questions`, for things the planner could not check in the code.
  - The prompt gives three tests for `after` (B needs code A adds; A and B touch the same files or modules; B depends on a decision or API shape A settles), asks for `door: one-way` on steps that are hard to walk back (a public API, a default users build on, a removal, a persisted format), says a step that deletes, kills, sends data or changes a public API or persisted format is never mechanical, and asks for new test files to be listed in scope. See [ADR 0008](adr/0008-planner-dependencies-and-risk.md).
  - Mullet checks the shape before showing it: unique ids, parents before children, depth 3 at most, 40 nodes at most, every node valid, and every `after` names another node, has no cycle and is finished first in the runner's order (list order, children before their parent).
  - A one-way step is risky (`is_risky` in `ladder.py`): it is Claude-first, and it gets a forced G7 review in every job, ladder or not, even with `--no-review`. That review must pass with a verdict of its own: an approval given only when the reviewer was asked again pauses the job for a person. The planner can raise a step's risk this way, never lower it: the risky words still apply in ladder jobs.
  - A reply with no usable plan (no structured answer, or one that fails the checks) is asked for again in the same session, with the reason, up to `jobs.reask_attempts` times (default 1, 0 turns it off). The planner session is kept only while a re-ask is allowed.
  - `mullet job plan` and `mullet job show` list one-way steps and open questions under the tree, before approval. Open questions are kept with the plan notes (the `plan_notes` event).
  - The clone is made at plan time, so the planner reads exactly what the runner will work on.
  - To change a plan: `mullet job show <id> --yaml > plan.yaml`, edit it, then `mullet job plan "<goal>" --plan-file plan.yaml`. A plan file goes through the same checks as the planner's answer. `--single` skips the planner.
  - A plan may set only a node's id, parent, goal, kind, door, after, scope, acceptance commands, allowed protected paths and test removal. It can't raise size limits. `mullet job show` prints any limit that isn't the default.
  - Feedback sent to the next attempt is scrubbed of values from the profile's copy_in files, because test output can print them.
- Integration nodes (a node with children):
  - They run after their children.
  - They are judged on everything since the first child started, using the full test suite. Their size limits are the sum of their children's.
  - The first attempt runs the gates only. If that fails, later attempts give the executor the gate feedback to fix the integration.
  - An integration check that changes nothing makes no commit.
- G7 (`review.py`) is on unless `--no-review`:
  - When on, it is a required gate, so a node can't pass without it. A review that finished on its own but gave no verdict is first asked again in the same session, only for the JSON verdict, up to `jobs.reask_attempts` times (default 1; 0 turns it off). One that timed out or ran out of turns is not asked again, since it may not have read everything. Reviewer sessions are kept for that while it is on. A review that still has no verdict, or that crashed or hit a usage limit, is treated like a crashed gate: the job pauses, nothing is committed, and a resume runs the review again on the same work.
  - The reviewer sees the node, the diff and the clone, never the author's transcript or the copied-in files.
  - It approves only if the change does the whole step, handles the edge cases the goal or acceptance checks name, has a test that would fail without any new or changed behaviour (a behaviour-keeping refactor or a no-code step needs none), does nothing it wasn't asked to, contains nothing harmful, and adds no secrets or injection. It blocks only for a specific defect it can name.
  - Its verdict carries `findings` (`path`, optional `line`, `problem`) beside the older free-text `reasons`. A finding's path may be absolute in the clone or start with `./`; one that leaves the clone is dropped. Findings on files the attempt didn't change are counted in the gate details (`findings_dropped`) and passed on as reasons marked "(file not changed)". The next attempt's feedback carries the reasons first, then the findings as one `path:line: problem` line each, each under a label saying they are the reviewer's words.
  - The profile's `review_standards` (a repo-relative Markdown file, default none) is shown to the reviewer only, fenced as data, read from the job's base commit and capped at `MAX_STANDARDS_CHARS` (one file's diff cap). A missing file is ignored with a note in the gate details. See [ADR 0009](adr/0009-reviewer-findings-and-standards.md).
- The runner (`runner.py`):
  - A detached process (`setsid`) with a lock file per job.
  - Stop and resume kill the whole process group. The signal handler only sets a flag; a watcher thread does the killing, and once a stop is requested no new sandboxed command or executor run may start.
  - Waiting for the machine-wide worker lock can be interrupted by a stop.
  - On start, the branch is reset to the last commit the ledger recorded, so a commit made just before a crash, and never recorded, is dropped and its node runs again.
  - An executor that ends without a result (a crash, a lost login) doesn't use up an attempt, but three in a row block the job.
  - A gate that crashes (G0 to G7, the check of a parent's own fix and the final check) is a bug or outage in Mullet's checks, not a verdict on the work, and so is a required review with no verdict. The checks run again on the same work, up to three times in a row; then the attempt is set aside as paused (not counted, so the worker's model isn't marked as having failed the step), nothing is committed and the job pauses with the reason. `mullet job resume` runs the checks again on that same work without asking the worker for it anew. Only work the gates refuse to run against (an unsafe work tree or `copy_in` path) still fails the gate.
  - `mullet job run` accepts a job left RUNNING by a dead runner. It tells whether a runner is alive from the job's lock, not a pid, so foreground runners count too, and `stop` checks the pid's start time before signalling.
  - A partial attempt is thrown away. Neither that nor a usage-limit pause counts as an attempt.
  - If your base branch moves during a job, the job carries on from the pinned base. Reconciling the two is left to you at merge time, which keeps the job's evidence about exactly the commits it checked.
- The report (`report.py`) is built from the ledger only. Executor summaries are labelled as claims.
- CLI: `mullet job plan|show|approve|run|status|tail|stop|resume|report|list|discard|merge|ladder|retry`. `merge` shows the diffstat, asks first, and copies the job branch into your repo as a new branch. You then merge it yourself. It never touches a checked-out branch and never pushes.
- A `/mullet-job` skill drives the CLI from Claude Code.

### Machine guard (`machine.py`)
Mullet's local GPU work shares one laptop with LM Studio and test suites. On a single 12 GB GPU, a Trawl reindex, a loaded LM Studio model and two 16-worker test suites at once made the reindex crawl. So:
- `gpu_lease(purpose)` is one lease across all of a user's Mullet processes: an `flock` on `state_dir()/gpu.lock`, with the holder's pid, purpose and start time in `gpu.lock.json` beside it. A waiter can then say "waiting for the GPU: reindex of pool X (pid N, since 22:41)". The kernel drops the lock when its holder dies, and a record from a dead or reused pid is ignored. `wait=False` and a timeout raise `GpuBusy`, which names the holder.
- `wait_for_headroom()` waits for free VRAM (`MULLET_GPU_MIN_FREE_MB`, default 2048 MiB), available RAM (`MULLET_RAM_MIN_FREE_MB`, 8192), a cool enough GPU (`MULLET_GPU_MAX_TEMP_C`, 83) and a low enough 1-minute load (`MULLET_MAX_LOAD`, the CPU count). The readings come from `nvidia-smi`, `psutil` and `os.getloadavg()`, and the settings from the environment or Mullet's `.env`. VRAM held by the caller, its children, or its siblings in the same process group (idle `--processes` workers) counts as free, so a process never waits on itself.
- Nothing waits for long. Both waits stop after `MULLET_GPU_WAIT_MAX_S` (default 900 s; LM Studio unloads an idle model after 600 s). Without an NVIDIA GPU there is no lease and no GPU check, and RAM and load are waited on for 60 s at most before the work goes ahead.
- For each pool, the reindexer waits for headroom before taking the lease, then takes the lease and re-checks for 30 s at most. If the GPU is still busy when the time is up, that pool embeds on the CPU instead, without the lease (`model_cache.use_device("cpu")`, which the ingestor and searcher follow). It releases its cached models before letting go of the lease. It prints a line when it starts waiting, naming the holder or the setting and `--no-wait`, and another when it starts work. With a GPU, `--workers` and `--processes` pools take turns on it; without one they run side by side. `--no-wait` skips a busy pool and names the holder, and the run exits 75 if any pool was skipped.
- `mullet doctor` shows free VRAM, temperature, the lease holder and LM Studio's processes on the GPU, from two nvidia-smi calls of 3 s at most. When the power limit is at the default, it suggests `sudo nvidia-smi -pl 100` (it never runs sudo).
- The job runner doesn't take the lease for Claude or cloud workers, which don't use the local GPU. A ladder job's `local` rung takes it for the attempt without waiting (P5.6); a busy GPU moves the step up the ladder.

## Phases and Claude Code time

### v1: a trustworthy runner, Claude only

| Phase | Work | Claude Code time |
|---|---|---|
| P0 Stop the worst | Disable `overnight_loop` (ungated, pushes master). Make swarm's `applied` honest. Delete `CheckpointManager` (destructive restore, unused). Nothing else, because `MULLET_DELEGATION=off` already blocks the four delegation tools. | 1h |
| P1 Runner foundation | Check the `claude -p` flags; clone manager; sandbox wrapper with self-test; dependency handling; ledger; profile and detection; gate runner with junit parsing; gates G0 to G6; fake-executor test harness | 9–13h |
| P2 Runner execution | Claude Code executor; detached runner with resume and process-group handling; rebase on base moves; report; CLI; `/mullet-job` skill. **Milestone:** a 3-node job on a sample Python repo runs unattended, a planted failure is reset and retried, and a `kill -9` followed by a resume continues correctly. | 7–11h |
| P3 Planner and review | Task-tree planner with schema validation and approval flow; parent and integration gates; G7 reviewer | 4–6h |
| P4 Remove the old machinery | By hand, since 72 test files import it: `overnight_loop`, ultrawork, team, autopilot, ralph, the substring verifiers, the unused modules (team_dispatch, batch, decomposer, eco, autopilot_goals, `agent/recovery.py`, `agent/continuation.py`), duplicate `core/knowledge` pieces, and `gemini_cli` (broken by the Antigravity switch). Correct ARCHITECTURE, NEXT_STAGES, LOOP_AGREEMENT and TODO. | 3–5h |
| P5 Fast suite and portable install | **Tests, on the smaller suite:** `--durations`, pytest-xdist, pytest-timeout, injectable back-off sleeps, and folding in public patches 10 to 20 (test fixes). **Install:** fold in patches 02, 03, 05 and 21; layered XDG config and state with migration from `~/.mullet` (57 files use `Path.home`/`expanduser`); hooks as `mullet hook <name>`, opt-in; `setup-claude` via `claude mcp add --scope user` with an uninstall manifest; `install.sh` uses `uv tool install git+…` (non-editable, so Mullet never runs from the tree it edits); doctor checks claude, git, the sandbox and auth mode; rewrite both CLAUDE.md files. **CI:** Linux and macOS. | 12–18h |
| P6 Dogfood and harden | Three real jobs: (1) a Mullet job on this repo; (2) a refactor of a public repo; (3) a medium refactor of a private repo. Seatbelt wrapper and a job on macOS. A fresh-container install as a different user. Fix whatever breaks. **Characterisation tests:** for code with weak tests, the planner can open a refactor with a node that writes tests pinning down today's behaviour; they must pass on the base commit, so later nodes are checked against real behaviour. **Blocked-job notifications:** a job that blocks, pauses for a usage limit or finishes tells the owner (a desktop notification, or a Slack DM within office hours only), so an unattended job never sits idle. | 10–14h |
| **v1 total** | | **about 44–66h**, or 9 to 13 sessions of 5h |

### P5.5 External executors with full context (about 8 to 10h estimated)

The job runner has an opencode executor that is at least as well informed as an interactive session: it gets the repo's files, a repo map and the gate feedback, runs on a model you name, and never loads the user's own Claude files or skills. Built: everything below except `mullet job bench`, which is still to do.

- **Choosing it:** `mullet job plan "<goal>" --executor opencode --executor-model <provider/model>`. The planner and the G7 reviewer stay on Claude. `--preload` also pre-loads files into a Claude worker's prompt.
- **Privacy:** the profile's `allowed_executors` is `[claude-code]` by default. Only the user's stored profile can add `opencode`; the in-repo file can only narrow the list. `plan`, preflight and the runner all refuse an executor the profile doesn't list, and the message names the stored profile to edit. `data_class` appears in that message but doesn't gate on its own.
- **Invocation:** `opencode run "<short message>" --pure --dir <clone> --agent mullet-worker --model <id> --format json --title "mullet <job>:<node>" --file <context file>...`. A resumed attempt passes `--session <id>` instead of `--title` and the files. The message comes first because `--file` takes every word after it, and `--pure` drops the user's opencode plugins.
- **Model ids:** checked against `opencode models` (run with the worker's own environment and home, cached per process) at plan time, at preflight and before each attempt. A wrong id gets the nearest valid ones; a provider with no key gets the variable to set.
- **Per-attempt config through `OPENCODE_CONFIG`:**
  - `instructions` lists the brief, then the repo's AGENTS.md, or its CLAUDE.md when there's no AGENTS.md. The brief rides in the system prompt, so it survives compaction and is rewritten with the new feedback on a resumed attempt. It isn't attached with `-f` as well.
  - The `mullet-worker` agent starts from `"*": "deny"` and allows read (not `.env*`, `*.pem`, `*.key`), edit, glob, grep, list and todo. Bash allows the profile's checks and read-only commands, and denies `;`, `&`, pipes, redirects, `$(...)`, backticks, newlines and carriage returns, `find -exec` and `rg --pre`. The profile's whole check commands (lint, typecheck, build, full tests) come after the denies as exact rules, so one that contains `;` or `&&` still runs as written while anything added to it is refused; the shell wrapper matches them the same way. A check command with `*` or `?` can't be an exact rule (opencode treats both as wildcards and can't escape them), so it stays refused if it also chains. Web, subagents, skills, questions and external directories are denied. `steps` is the turn budget. opencode's own system prompt is kept.
  - `formatter: false` unless the profile sets `formatter: true`, `snapshot: false`, `share: disabled` and `autoupdate: false`.
  - It is the whole config: `XDG_CONFIG_HOME` points at the per-job home, so the user's `~/.config/opencode` (which can hold a literal key for another provider, plugins and MCP servers) isn't loaded or visible. Only the chosen provider's definition is copied in, through a whitelist (package, name, API, model filters, and per model its metadata and a few generation options), so no `apiKey`, `headers`, `env` or unknown field gets through, and its base URL is set to the proxy. `model` and `small_model` are both the attempt's model, so every request names the model the proxy is pinned to.
  - Language servers are off (`lsp: false`, the `lsp` tool denied). They would run as opencode's children, outside the worker's nested shell, with network and the key, and language servers execute repo code (build scripts, plugins). The user's stored profile can set `opencode_lsp: true` to get edit diagnostics back at that cost; the in-repo profile can't. Wrapping each server in the nested sandbox through opencode's custom `lsp` commands would remove the trade-off, but it hasn't been checked against the real binary.
  - Environment switches: `OPENCODE_DISABLE_PROJECT_CONFIG` (checked: without it a repo's `opencode.json` or `.opencode/agent/*.md` replaces the agent and can allow every command), `OPENCODE_DISABLE_CLAUDE_CODE`, `OPENCODE_DISABLE_EXTERNAL_SKILLS`, `OPENCODE_DISABLE_AUTOUPDATE` and `OPENCODE_DISABLE_SHARE`.
  - The key never enters the sandbox (see the key proxy below). Its variable names come from the user's own provider config and model catalogue (never the job's copy, which a worker can write), and only names that belong to the provider count: a fixed map, or `<PROVIDER>_..._API_KEY`. The value comes from the environment or Mullet's `.env`, or from a key the user wrote into their own opencode config for that provider.
  - Nothing from the repo's profile reaches opencode itself. The profile's `env` applies only to the worker's nested shell and to gate commands, and never with `LD_*`, `DYLD_*`, `*_PROXY`, `NODE_OPTIONS`, `BUN_OPTIONS`, `OPENCODE_*`, `ANTHROPIC_*` or `CLAUDE_*`. Mullet's own variables are set last everywhere. The Claude worker never gets the profile's env.
- **The context pack** (`mullet/jobs/context.py`): the runner gathers the job and parent goals, the planner's notes (a `plan_notes` ledger event, also kept in `show --yaml` plan files), what earlier nodes changed (`git log --stat`, trimmed), the tracked files and the profile's checks. The executor turns them into a markdown brief with the gate feedback, a symbol outline of the files in scope and a repo map grouped by directory. `select_context_files` picks the scope (for a broad scope, the files whose paths share words with the goal), their mapped and similarly named tests, and their direct imports (Python, JS/TS relative, PHP `use` and `require`), cut to 35% of the model's context window (opencode's figure, else the model registry, measured with `estimate_tokens`). It never picks protected paths, copy_in files, secret-looking files, binaries or anything through a symlink.
- **Fix, don't restart:** after a gate failure (not a crash or pause), if the executor can resume and it isn't the node's last attempt, the runner puts back the failed attempt's tree, with the index at the node's start so `git diff` shows the whole change, and resumes the session with the newest feedback. The last attempt always starts fresh, and gates still judge against the node's start. A restarted runner finds the session and tree in the ledger. Claude worker sessions are kept and resumed with `--resume`; planner and reviewer sessions use `--no-session-persistence` unless `jobs.reask_attempts` is above 0, which keeps them for one re-ask (see G7).
- **Sandbox:** opencode runs inside Mullet's bubblewrap wrapper with no network at all. Writable: the clone and a per-job opencode home (data, state, cache, config), never the user's own, which hold `auth.json` and every other session. Read-only: toolchains, the opencode install, the attempt's brief and config, the proxy's socket and the clone's git directory; dependency directories stay read-only inside the clone.
  - **The key proxy** (`mullet/jobs/executors/keyproxy.py`) is the only way out. The provider's base URL in the generated config is `http://127.0.0.1:4141/<the provider's own path>`, with a dummy key. Inside the sandbox, socat listens on that loopback port and passes each connection to a Unix socket; on the host, a proxy thread owned by the attempt serves the socket. It forwards only to that provider's one host (TLS, certificates checked; plain http only to a local server), keeps the path under the provider's prefix, refuses CONNECT and absolute-form targets and paths with `..`, `.`, `%2e`, `%2f`, `%5c`, `%00` or a backslash, caps a request at 32 MB (read in 64 KB pieces; negative, signed or malformed sizes and truncated chunks are refused), times out a stalled sender after 60 s, serves 16 connections at a time and refuses the rest with a 503, caps each attempt's requests (four per turn plus 20, at least 50) and bytes (512 MB) with a 429, refuses a JSON body that names any model but the attempt's own, drops whatever credentials the request carries and adds the real key in the provider's own header (`Authorization: Bearer`, `x-api-key` for Anthropic, `x-goog-api-key` for Google). Responses stream back as they arrive. Its log (`proxy.log`) has method, path without the query, status and size, never headers. The upstream comes from the user's provider config (`options.baseURL` or `api`), then the catalogue's `api`, then the SDK defaults; for the providers in use: OpenRouter is `https://openrouter.ai/api/v1` (`OPENROUTER_API_KEY`, paid and `:free` models alike), and Kimi through Moonshot is the catalogue's `moonshotai` provider at `https://api.moonshot.ai/v1` (`MOONSHOT_API_KEY`, or `KIMI_API_KEY` in Mullet's `.env`), or the user's own `moonshot` provider.
  - **The worker's shell** is a nested bwrap started from a cleared environment (bwrap's own init process keeps whatever it starts with), with no network, its own session, pid, IPC and UTS namespaces and `/proc`, opencode's home read-only and the proxy's socket covered by an empty tmpfs. The wrapper also refuses `;`, `&`, pipes, redirects, `$(...)`, backticks and newlines itself, whatever opencode's matcher makes of them. If a nested sandbox can't start, or socat is missing, opencode isn't run at all.
  - **opencode's home** is rebuilt before each attempt: a fresh cache (provider packages, model catalogue, language servers), state and config, and a data directory that keeps its sessions but loses credential files (an auth entry can pull in remote config), symlinks, `bin` directories and anything executable. `opencode models` runs on the host with a throwaway home that no worker ever saw.
  - **Symlinks:** opencode's file tools check paths as written, so a symlink in the clone is followed wherever it goes. Before each run Mullet refuses any link whose target is outside the clone, `/usr` (a virtualenv's python) and the toolchain directories: `/proc`, `/etc`, `/mnt`, another user's home and the job's own files all count. One found blocks the job with the paths; the Claude executor doesn't do this check, since Claude Code's sandbox resolves paths itself. A link made during the run, or the key turning up in the transcript, the reply or a changed file (as written, JSON-escaped or base64, whole files scanned), throws the attempt away (transcript redacted). With the key outside the sandbox these are defence in depth.
  - **Usage:** each attempt's `usage` has tokens (input, output, reasoning, cache read and write) and `total_cost_usd`, summed from opencode's `step_finish` events, with `cost_reported` saying whether any step carried a price (opencode prices steps from the catalogue; a model with no price there reports none), plus `provider_requests` and `provider_host` from the proxy. The recorded cost is never below what the provider's own responses say, and the key proxy's figure is what the spending caps read (see Spending below).
  - Language servers, if the user's profile turns them on, run beside opencode outside the nested shell; they can reach the proxy (not the key) and execute repo code. Downloading a missing server needs network, which the sandbox doesn't have.
  - No sandbox, no run.
- **Checked against opencode 1.18.30** with a local fake OpenAI-compatible server, no real model: the brief and AGENTS.md reach the system prompt, `--file` attachments arrive as Read results, denied and chained commands are refused, the shell has no network or key, a 401 comes back as an `error` event, `--session` continues the conversation, and a whole job passes through the runner. The key proxy is checked the same way: the real binary, with no network, ran a job for `openrouter/moonshotai/kimi-k2.7-code` and for `moonshotai/kimi-k2.7-code` through socat and the proxy to a fake upstream. The provider SDKs are built into the binary, so nothing was downloaded. The upstream saw only the real key (the dummy never left the sandbox), chunked SSE replies streamed through, and cost was reported. A dead upstream came back as a 502, opencode retried, then gave an `error` event and the attempt was aborted. Not checked yet: a real provider (TLS to openrouter.ai and api.moonshot.ai), that opencode passes the shell exactly the command it was given (the wrapper's own refusal relies on it), and the other security fixes (cleared nested environment, rebuilt home, symlink checks, LSP off) with the real binary; those are tested with the fake CLI only.
- **Not built:** `mullet job bench` runs one plan file once per executor and compares, from the ledgers, first-attempt pass rate, attempts, time and cost.
- **One-off delegations:** `mullet_opencode` (the MCP tool) runs the task as a single-node gated job on the ladder. The result is applied to the caller's working tree only after its gates pass.

### P5.6 The delegation ladder (`ladder.py`)

Goal (the owner's "balanced performance"): Claude when the work needs it and the weekly window has room; free models where the task and repo allow; local models where the task allows and the machine has room; Kimi (Moonshot, more credit) then OpenRouter as the paid workers. The gates judge every attempt the same way, whoever made it. Built: everything below, reworked after two independent reviews. Still to do: a secret scan on free prompts, and bench numbers for the shipped defaults (a first bench ran on 25 Oct).

- **Rungs:** `free` (an OpenRouter `:free` model), `local` (an LM Studio model through opencode's `lmstudio` provider on `127.0.0.1:1234`), `kimi` (Moonshot direct: opencode's `moonshotai` provider, `MOONSHOT_API_KEY` or `KIMI_API_KEY`), `openrouter` (one rung per configured model: a cheap paid coder, then OpenRouter's Kimi) and `claude` (the user's own Claude Code). Every rung but Claude is the opencode executor behind the key proxy.
- **Config:** the `ladder:` key of the user's own `config.yaml` (`config_file("config.yaml")`), never a file in the repo or the working directory. Keys and defaults: `kimi` `moonshotai/kimi-k2.7-code`; `openrouter` `[openrouter/qwen/qwen3-coder, openrouter/moonshotai/kimi-k2.7-code]`; `free` and `local` unset (the rung is skipped); `claude_low` 0.5 and `claude_high` 0.8; `max_spend` $2; `prices` per model in $/M tokens, with a deliberately high fallback ($2 in, $8 out); `start` (the cheapest rung per kind: mechanical and design `free`, analysis `kimi`); `narrow_globs` 4; `risky_words` (delete, remove, rm, rmtree, unlink, kill, terminate, signal, wipe, truncate, drop, push, upload, send); `max_below_claude` 3; `cheap_minutes` 10. A value it can't use is dropped with a note in `mullet job ladder`, never guessed.
- **Groups of steps:** mechanical; checkable (a design step with acceptance commands, at most `narrow_globs` scope globs with no `**` or `*`, and nothing risky); claude-first (analysis, any other design step, a parent's integration fix). The planner prompt asks for `mechanical` whenever goal, scope and acceptance leave no real design choice: a real planner marked all 8 steps of the `mullet reap` job `design`.
- **The order (`ORDERS`), by Claude's paced weekly pressure from `fetch_usage()`:**

  | Level | Mechanical | Checkable design | Claude-first |
  |---|---|---|---|
  | low (under `claude_low`) | free, local, claude, kimi, openrouter | claude, kimi | claude, kimi |
  | medium | free, local, kimi, openrouter, claude | kimi, openrouter, claude | claude, kimi |
  | high (over `claude_high`) | free, local, kimi, openrouter, claude | free, local, kimi, openrouter, claude | kimi, openrouter, claude |

  When the week has room, Claude comes before the paid rungs because it is paid for already; when it is tight, Claude is the last resort with one try. The planner, the G7 reviewer and `ladder: claude-only` profiles (a repo's own file may set that, never loosen it) are always Claude. Without fresh figures the last known level is used if it is under 6 hours old (`state_dir()/ladder/budget-level.json`), else medium. The runner keeps the figures fresh itself: every `claude -p` stream it sees (worker, planner, reviewer) has its `rate_limit_event` recorded through `mullet.hud.usage.record_stream_limits`.
- **Per-rung checks** (`plan_rungs`), each skip recorded with its reason in a `ladder_choice` event:
  - `free` needs the public check: the stored profile's `data_class: public`, no `copy_in`, no `.gitmodules` or gitlinks at the base commit, `origin` on GitHub (owner and repo never `.` or `..`), GitHub's API saying `private: false`, public visibility and `fork: false`, and its compare API putting the base commit on the default branch. The result is cached per repo and base for an hour in the state dir and asked again only when older; a GitHub outage (not a 404) is never cached and skips the free rung for that attempt only.
  - `local` needs a model LM Studio lists (`GET /api/v0/models`, which reports each model's state and never runs a completion), no other process holding the GPU lease, and machine-guard headroom (VRAM a loaded model holds counts as free). The attempt takes the GPU lease without waiting.
  - the paid rungs stop at the job's spending cap; a rung whose executor isn't in `allowed_executors`, or whose provider has no key, is skipped; `start` removes rungs cheaper than a kind may use.
- **Walking the ladder** (`next_rung`, `executor_for`): one attempt per rung; Claude gets what is left of the job's attempts (at least one), or one when its week is tight. A model that failed a step's gates is never tried again on another provider (OpenRouter's Kimi after Moonshot's); only a provider error lets the same model run elsewhere. A step gets at most `max_below_claude` counted attempts (passed, unverified or failed) on the rungs below Claude, before or after Claude's own; then only Claude may take it, and without Claude it has run out of rungs. Night mode adds `extra_below_claude`. Every rung whose requests go to OpenRouter, whatever its name, gets `data_collection: deny` and the default quantisations fp8, fp16, bf16 and fp32 (no fp4, int4, int8 or `unknown`; `ladder.routing.<model>.quantizations` replaces the list, so a user can add `unknown`); the free rung on its `:free` model carries no filter, since it only sees code the public check found public. The position comes from the ledger, so a resumed job carries on where it was, and the rungs are chosen afresh each attempt. Free and local attempts get `cheap_minutes` of wall clock. A provider error (an opencode `error` event, a proxy cap or refused request, an executor that can't start, a busy GPU, the spend limit) or a broken run on any rung but Claude is an aborted attempt with `moved_up`: no attempt used, no crash counted. A session is resumed only on the rung that started it. Risky steps always get G7, even with review off, and there it must pass: a block fails the attempt and a review with no verdict pauses the job.
- **Spending:** each attempt's cost is what the provider's own responses say (the key proxy reads the usage in each one; OpenRouter states the cost), or its tokens times the `ladder.prices` price when the provider states none or $0 (a paid model reporting $0 for real tokens has no catalogue price, and the cap would otherwise never fire). The recorded cost never falls below what the responses say. The proxy refuses an attempt's requests with HTTP 402 once its allowance is spent. The allowance is the least of what is left of the job's cap (lowered by `budget.envelope`) and of the rung's `budget.per_rung` limit; see [CONFIGURATION.md](CONFIGURATION.md). At job end `mullet job report` sets the ledger's spend beside each paid provider's own figure. opencode passes its usage on after every `step_finish`, and the runner writes it to the ledger as it comes (`record_progress`), so a killed attempt keeps its cost; an attempt whose running cost passes what is left of the job's cap is stopped and moves up. Over the cap while Claude's level is high, the job pauses with a notification ("spend cap reached and Claude's weekly window is tight") rather than handing every remaining step to Claude; `mullet job resume <job> --max-spend <dollars>` raises the cap. A non-ladder opencode job is held to the same rules: its model counts as a rung (`model_rung`: `:free` is free, LM Studio and ollama are local, OpenRouter is openrouter, any other provider kimi), a paid one gets the allowance and pauses at the cap, and a `:free` one blocks the job unless the public check passes. Costs seen on real jobs: a small edit $0.05 to $0.28 on Kimi, a 2-step chunk with reviews $2.44.
- **Blocked steps:** a step with no rung left blocks ("ran out of rungs") and the owner is notified. `mullet job retry <job> <node> [--rung claude|kimi|openrouter:<model>]` gives it fresh attempts, on the ladder again or on the named rung (the job's `max_attempts` there), always starting fresh; the profile, keys and cap still apply. `mullet job resume` carries on with it and the remaining steps.
- **Security around it:** the clone holds only the base commit's history (see Isolation). The key proxy, pinned to the attempt's model, takes only identity-encoded JSON bodies without duplicate keys that name that model and no other field naming a model (no OpenRouter `models` list), on an allowlist of paths (`chat/completions`, `completions`, `messages`, `responses`, `models`; no batch or file endpoints), and its socket is created under a 0077 umask. The check commands a worker may run exactly as written (with `;`) come only from the user's stored profile or Mullet's detection, recorded at plan time as `trusted_commands`; a repo's `.mullet/project.yaml` can't add one. Slack notifications carry fixed wording only (job id, status, what to run next), never the goal, gate output, block reasons or error text; desktop ones may show the goal's first line.
- **Ledger and report:** each attempt's usage records `rung`, `rung_key`, `model`, `provider_error` and `moved_up`; planner and reviewer sessions are `claude_usage` events (turns, tokens, the CLI's estimate). Old ledgers read as before (no schema change). The report shows each attempt's rung, model, cost and tokens, all of Claude's work, and a line like "4 of 6 steps passed below Claude; spent $0.31: Kimi $0.27, OpenRouter $0.04; Claude used on 2 steps", counting leaf steps only.
- **CLI:** `mullet job plan --ladder` is the default once the config has a `ladder:` key and the profile allows opencode, unless `--executor` is given; `--no-ladder` keeps one executor; `--max-spend` sets the cap. `mullet job ladder` prints the rungs, the orders per level, the limits, whether each rung has a key, whether `opencode models` offers its model, LM Studio's state, the public check for the current repo and Claude's budget (`--offline` skips the network). `mullet job show` prints the budget level and each step's starting rung and group.

### P7 Public release as 2.0.0 (about 2 to 4h)

The private repo stays the only place code is written. The public repo is generated from it by `sync-to-public.sh` and never edited by hand. The script makes a committed-files-only export, overlays `.public/` files and patches, removes private paths, applies text replacements, and runs the banned-terms and gitleaks checks. A public issue or PR is fixed here and then synced.

1. `.public/` holds only what the public copy alone gets: `README.md`, `LICENSE` and `patches/` (the public CI for the macOS and parallel test matrix). Everything else is in this repo.
2. Review the `EXCLUDE` and `REPLACE` lists for the new files, and decide whether this design doc goes public as-is or as a public version. `./sync-to-public.sh --check` must pass. It is the release's acceptance test.
3. Rewrite `.public/README.md` around jobs:
   - install with `uv tool install` from the public repo;
   - requirements: Linux for jobs, Claude Code, bubblewrap and socat;
   - `mullet doctor`, then `mullet setup-claude`.

   Bump the version to 2.0.0, renumbering the parked "v2.0.0 Web UI" roadmap entry, and add a changelog entry.
4. Run the sync, review the diff in the public clone, then commit, tag and push, with the owner's go-ahead.
5. Install from the public repo on a fresh machine (a container, as a different user) and run a real job. This is also P6's fresh-install check.

Branches: `mullet-2` is pushed to the private origin after each phase as a backup. `master` stays as it is until v1 is reviewed, then it merges locally and is pushed.

### v1.1: the rest of the vision

| Phase | Work | Claude Code time |
|---|---|---|
| Trawl extra | Move the resource manager into core (`core/router/resource_guard.py`); make the knowledge-backed delegation paths optional (in `mcp_handlers.py`); `mullet mcp --trawl`; heavy dependencies into `[trawl]`; drop the `../trove` fallback (in `core/knowledge/manager.py`); CI with and without the extra | 3–5h |
| Cheap executors | opencode executor inside Mullet's sandbox, with a per-job config instead of `--auto`; `base_url` and `api_key_env` survive the config loader (in `config/loader.py`); stop labelling localhost as "local" (in `openai_compat.py`); escalation ladder by data class, kind and attempt; OpenRouter `data_collection: deny`; freellmapi as a provider entry | 6–10h |
| Runtime splits | A worker returns a structured split request; splits inside the approved scope run without asking, and splits that widen it pause | 2–4h |
| MCP job tools | `mullet_job_plan`, `mullet_job_status`, `mullet_job_approve` and `mullet_job_report`, none of which block | 2–3h |
| Parallel subtrees | Nodes with disjoint scopes run in separate clones and merge into the job branch | 4–6h |
| **v1.1 total** | | **about 17–28h** |

The whole vision is about **60 to 95 hours of Claude Code time**.

**Assumptions:**
- Opus in Claude Code with subagents.
- Test runs are included.
- Your plan approvals and diff reviews are not included.
- From P3 onward, Mullet can run some of its own phases as jobs, which also tests it.

**Biggest unknowns:**
- getting real repos' setup and tests to run inside the sandbox (Drupal stacks that need a database will need a docker-backed profile and network rules);
- how much of the suite breaks under xdist.

## Risks

- **Usage limits.**
  - Long unattended jobs can hit the 5-hour and weekly limits, which Anthropic says "assume ordinary, individual usage".
  - The runner pauses and resumes, runs one worker at a time, and per-job budgets cap use.
- **Anthropic's terms for a tool that drives a user's own `claude`.** Whether that counts as "offering claude.ai login" is not covered by Anthropic's docs. Mullet never touches credentials and documents an API-key mode. Mullet's own licence is the Business Source License 1.1 (see `LICENSE`): production use is free for any purpose except offering Mullet to others as a hosted service.
- **`--bare` may become the default for `-p`.** Bare mode ignores subscription login, so the executor passes its flags explicitly and doctor reports the auth mode.
- **Weak tests mean weak gates.** The report says which gates actually ran, and an "unverified" node is never presented as passed.

## Verification

- The fake executor plays back hostile behaviour, and a gate or the sandbox must catch every case:
  - writes outside the clone, including through Bash and symlinks;
  - `git branch -f` and `git push` attempts;
  - an empty diff that claims success;
  - deleted assertions;
  - a changed lockfile or CI file;
  - tampering with `node_modules/.bin`;
  - a secret in the diff;
  - reading `~/.ssh`.
- Your repo, `.git` included, is checksummed before and after every job and must be identical until merge.
- A ledger audit test checks that no node passes without gate evidence.
- The P2 milestone, the P6 real jobs, a fresh-container install, and a green full suite on Linux and macOS CI.

## Working on it

- Work happens on the `mullet-2` branch, with one or more commits per phase.
- Phase status is tracked in `docs/TODO.md` under "Mullet 2".
