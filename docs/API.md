# Mullet API Reference

## REST API

### Health

```
GET /health
```

Returns server status, provider availability.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "active_tasks": 0,
  "providers": {"local": {"available": true, "model": "default"}}
}
```

### Models

```
GET /v1/models
```

OpenAI-compatible model listing. Returns available providers.

### Chat Completions

```
POST /v1/chat/completions
```

OpenAI-compatible chat completions. Supports streaming (SSE).

**Request:**
```json
{
  "messages": [{"role": "user", "content": "list files"}],
  "stream": false,
  "mullet": {
    "client": "backend",
    "mode": "standard",
    "force_tier": "LOW",
    "max_iterations": 90
  }
}
```

**Headers:**
- `X-Mullet-Session-Id` — Optional. Enables multi-turn conversation without resending history. Server stores messages per session (last 50 kept).

**Response:**
```json
{
  "id": "chatcmpl-...",
  "model": "mullet/local/default",
  "choices": [{"message": {"role": "assistant", "content": "..."}}],
  "usage": {"prompt_tokens": 100, "completion_tokens": 50},
  "mullet": {"provider_used": "local", "mode": "standard", "session_id": "abc123"}
}
```

**Orchestration modes:** Set `mullet.mode` to `"autopilot"`, `"ralph"`, or `"ultrawork"` to run in full orchestration mode instead of a single agent turn.

### Status

```
GET /api/v1/status
```

Current system state: providers, active orchestration modes, task count.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "active_tasks": 0,
  "providers": {"local": true, "anthropic": false, "openrouter": false},
  "active_modes": ["ralph"],
  "local_provider": {"name": "local", "model": "default"}
}
```

### System Info

```
GET /api/v1/info
```

Comprehensive: version, providers, tools, skills, mode, platform.

### Skills

```
GET /api/v1/skills
```

List available skills with names, descriptions, and modes.

### Explain

```
POST /api/v1/explain
```

Analyze task complexity and routing decision.

```json
{"task": "refactor auth module for OAuth2"}
```

Returns: complexity score, tier, signals, learning suggestion, project context.

### Metrics

```
GET /metrics
```

Prometheus-compatible metrics. No authentication required.

Available metrics:
- `mullet_requests_total` — request count by endpoint
- `mullet_active_tasks` — currently running tasks
- `mullet_uptime_seconds` — server uptime

### Tools

```
GET /api/v1/tools
```

List available tools, toolsets, and profiles.

### Projects

```
GET /api/v1/projects
```

List projects the authenticated user has access to.

### Search

```
POST /api/v1/search
```

Search project knowledge via Trawl.

```json
{"client": "backend", "query": "auth module", "top_k": 10}
```

### Memory

```
POST /api/v1/memory
```

Store a memory.

```json
{"client": "backend", "key": "pattern", "content": "always use early returns"}
```

```
GET /api/v1/memory?client=backend&query=pattern
```

Search memories.

---

## WebSocket

```
ws://localhost:3000/ws
```

### Client → Server

| Event | Payload |
|---|---|
| `authenticate` | `{token: "mllt_xxx"}` |
| `start_task` | `{prompt, mode, project, options}` |
| `kill_task` | `{task_id}` |
| `list_projects` | `{}` |
| `search` | `{project, query}` |
| `ping` | `{}` |

### Server → Client

| Event | Payload |
|---|---|
| `authenticated` | `{id, email, role}` |
| `task_started` | `{task_id, mode}` |
| `token_stream` | `{task_id, content}` |
| `tool_start` | `{task_id, tool_name, args_preview}` |
| `tool_result` | `{task_id, tool_name, preview}` |
| `task_complete` | `{task_id, result, tokens, tools_called}` |
| `task_error` | `{task_id, error}` |
| `projects_list` | `{projects: [...]}` |
| `search_results` | `{results: [...]}` |
| `pong` | `{}` |

---

## MCP Tools

Available when running `mullet mcp`:

| Tool | Description |
|---|---|
| `mullet_run` | Execute a task with routing and knowledge |
| `mullet_delegate` | Spawn a child agent for a subtask |
| `mullet_search` | Search project knowledge via Trawl |
| `mullet_remember` | Store a memory for future sessions |
| `mullet_analyze` | Analyze task complexity and show routing decision |
| `mullet_status` | Check system status |
| `mullet_providers` | List available providers |
| `mullet_budget` | Query spend and limits |
| `mullet_swarm` | Fan-out parallel processing across Kimi K2 calls |
| `mullet_opencode` | Delegate to opencode agent with native filesystem access |

### `mullet_run` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | string | required | The task to execute |
| `tier` | string | auto | Force routing tier: `LOW`, `STANDARD`, `ADVANCED`, `CRITICAL`, `PREMIUM` |
| `mode` | string | `"standard"` | Execution mode: `standard`, `ultrawork`, `ralph` |
| `files` | array | `[]` | File paths to include as context (exported as FILE blocks for the delegate) |
| `apply` | boolean | `false` | Apply EDIT/FILE/DELETE blocks from the delegate response to disk |
| `target_dir` | string | cwd | Base directory to resolve file paths against |
| `providers` | string | auto | Comma-separated provider allowlist: `"kimi,haiku"`. If omitted, the active preset's `allowed_providers` is applied; passing `providers` explicitly overrides it. |
| `force_single` | boolean | `false` | Disable auto-swarm routing; always use a single provider call |
| `max_iterations` | integer | `90` | Maximum agent iterations |

**`force_single`** — By default, `mullet_run` auto-detects when a task warrants fan-out to a swarm (≥5 files, CRITICAL tier, long prompt, mixed languages, or many docs). Set `force_single: true` to always use a single provider call regardless of these signals:

```json
{
  "task": "add type hints to utils.py",
  "files": ["src/utils.py"],
  "apply": true,
  "force_single": true
}
```

### `mullet_opencode` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | string | required | Coding task to perform (natural language) |
| `cwd` | string | cwd | Project directory |
| `model` | string | auto | Model override, e.g. `"moonshot/kimi-k2.7-code"`. If omitted, the active preset's `opencode_model` is used; if the preset is `native_only` (e.g. `claude-only`), `mullet_opencode` refuses and defers to Claude Code. |
| `timeout` | integer | `300` | Max seconds to wait |
| `apply` | boolean | `false` | Git add + commit changes after completion |
| `commit_message` | string | auto | Custom git commit message when `apply=true` |

### `mullet_swarm` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `goal` | string | required | High-level objective for the swarm |
| `files` | array | `[]` | File paths to include as context (glob patterns) |
| `items` | array | `[]` | Arbitrary string items to process (alternative to files) |
| `concurrency` | integer | `20` | Max parallel calls |
| `output_shape` | string | `"markdown report"` | Desired output format |
| `dry_run` | boolean | `false` | Decompose only — return task plan without dispatching |
| `apply` | boolean | `false` | Apply EDIT/FILE/DELETE blocks from sub-tasks to disk |
| `target_dir` | string | cwd | Directory to resolve file paths against when applying |

### Claude Code Configuration

```json
{
  "mcpServers": {
    "mullet": {"command": "mullet", "args": ["mcp"]}
  }
}
```

---

## Admin API

All endpoints require admin role.

| Method | Path | Description |
|---|---|---|
| GET/POST | `/api/v1/admin/users` | List / create users |
| PUT/DELETE | `/api/v1/admin/users/{user_id}` | Update / delete user |
| GET/POST | `/api/v1/admin/projects` | List / create projects |
| DELETE | `/api/v1/admin/projects/{project_id}` | Delete project |
| GET | `/api/v1/admin/access/{project_id}` | List access for project |
| POST/DELETE | `/api/v1/admin/access` | Grant / revoke access |
| GET | `/api/v1/admin/usage` | Usage tracking |
| GET/POST | `/api/v1/tokens` | List / create PAT |
| DELETE | `/api/v1/tokens/{token_id}` | Revoke PAT |
