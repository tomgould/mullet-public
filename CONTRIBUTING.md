# Contributing to Mullet

Thanks for your interest in improving Mullet! This guide covers setup, testing,
code style, and how to open a PR.

## Quick Start

```bash
git clone https://github.com/tomgould/mullet-public.git
cd mullet-public
pip install -e ".[dev]"
mullet doctor
```

## Project Layout

| Directory | What lives there |
|:----------|:----------------|
| `mullet/core/` | Routing, providers, tools, skills, knowledge engine, orchestration |
| `mullet/cli/` | CLI entry points and subcommands |
| `mullet/hud/` | Terminal status bar and budget guard |
| `mullet/server/` | REST API, WebSocket, MCP server |
| `tests/` | Full test suite (17,700+ tests) |

## Running Tests

```bash
# Full suite
pytest tests/ -q --tb=short -x

# Integration tests (needs LM Studio on localhost:1234)
pytest tests/integration/ -v
```

## Code Style

```bash
pip install "ruff==0.17.1"
ruff check --select E9,F63,F7,F82 mullet scripts   # what CI requires to pass
ruff check mullet scripts                          # the full rule set (advisory)
```

CI also runs Black and mypy as advisory checks. Most of the code base isn't
Black-formatted yet, so don't run `black .` over the whole tree; keep
formatting-only changes in a PR of their own.

We target **Python >= 3.10** and require type hints on all public APIs.
No placeholder TODOs or commented-out code in production paths.

## Commit Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Use for |
|:-------|:--------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, missing semi-colons, etc. (no logic change) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or correcting tests |
| `chore` | Build, tooling, dependency updates |

Example:

```bash
git commit -m "feat(providers): add streaming token cache"
```

## Pull Request Process

1. **Fork** the repo and create a feature branch.
2. **Add or update tests** for any changed logic.
3. **Run the relevant tests:** `pytest tests/test_<name>.py -q` and ensure green.
4. **Update `CHANGELOG.md`** if your change is user-facing.
5. **Open a PR** with a clear description of what changed and why.

## Adding a New Provider

1. Create `mullet/core/providers/your_provider.py` extending `Provider`.
2. Implement `create()`, `health_check()`, and `_to_response()`.
3. Wire into `mullet/cli/main.py:resolve_provider()` and
   `mullet/core/router/resolver.py:_create_provider()`.
4. Add model entries to `mullet/core/providers/model_info.py`.
5. Write tests with a mocked SDK — never require a real API key in CI.

## Adding a New Tool

1. Create `mullet/core/tools/your_tool.py` with schema + handler.
2. Add a `register(registry)` function.
3. Import and register in `mullet/cli/main.py:build_tool_registry()`.
4. Add toolset name to default toolsets in `build_agent()`.
5. Add guidance to `mullet/core/agent/prompts.py`.
6. Write tests in `tests/test_your_tool.py`.

## Writing a Plugin

No source modification required. Drop a `.py` file in `~/.mullet/plugins/`
or `.mullet/plugins/`:

```python
def register(tool_registry=None, **kwargs):
    if tool_registry:
        tool_registry.register(
            name="my_custom_tool",
            func=lambda query="", **kw: f"Result for: {query}",
            schema={
                "description": "My custom tool",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
            toolset="plugin",
        )
```

Then `mullet --tools` will surface it automatically.

## Contributor Licence

By submitting a contribution you agree that your contribution is licensed
under the same terms as the rest of the project: the **Business Source
License 1.1** (see [LICENSE](LICENSE)).
