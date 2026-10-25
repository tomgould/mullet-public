# Contributing to Mullet

Thanks for your interest in improving Mullet! This guide covers setup, testing,
code style, and how to open a PR.

## Quick Start

```bash
git clone https://github.com/tomgould/mullet-public.git
cd mullet-public
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
mullet doctor
```

## Project Layout

| Directory | What lives there |
|:----------|:----------------|
| `mullet/jobs/` | `mullet job`: planner, private clone, sandbox, gates, ledger, report, notifications |
| `mullet/core/` | Routing, providers, tools, skills, knowledge engine, orchestration |
| `mullet/cli/` | CLI entry points and subcommands |
| `mullet/hud/` | Terminal status bar and budget guard |
| `mullet/server/` | REST API, WebSocket, MCP server |
| `tests/` | Full test suite (about 18,000 tests) |

## Running Tests

```bash
# While working: only the files you changed
python3 -m pytest tests/test_foo.py -q

# Full suite, in parallel (about 5 minutes)
python3 -m pytest tests -q -n 8 --timeout 300

# Live tests (a real model; excluded by default)
python3 -m pytest -m live tests/integration/ -v
```

## Code Style

```bash
pip install "ruff==0.17.1"
ruff check --select E9,F63,F7,F82 mullet scripts tests   # what CI requires to pass
ruff check mullet scripts                                 # the full rule set (advisory)
```

CI installs the dependencies with `pip install -c ci/constraints.txt -e ".[dev]"`,
so a new release of a dependency can't change a result unannounced. When you
bump a dependency, update `ci/constraints.txt` to a version the suite passes
with. Third-party GitHub Actions are pinned to a commit SHA with the tag in a
comment; to move one, look up the tag's commit with
`git ls-remote https://github.com/<owner>/<repo> refs/tags/<tag>` (add `^{}`
for an annotated tag). A leak found by `gitleaks detect` that is a checked fake
goes in `.gitleaksignore` as `commit:path:rule:line`, with a comment.

CI also runs Black and mypy as advisory checks. Most of the code base isn't
Black-formatted yet, so don't run `black .` over the whole tree; keep
formatting-only changes in a PR of their own.

We target **Python 3.10 to 3.12** (the versions CI tests) and require type hints on all public APIs.
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

No source modification required. Drop a `.py` file in `plugins/` in the data directory
(`mullet paths data`) or in `.mullet/plugins/` in the project:

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

## Contributor Licence and sign-off

Contributions are welcome. Each commit needs a `Signed-off-by:` line, which you add
with `git commit -s`. It says you wrote the change or have the right to submit it, as
set out in the [Developer Certificate of Origin](https://developercertificate.org/).
A check on pull requests fails any commit without one.

Your contribution is licensed under the same terms as the rest of the project: the
**Business Source License 1.1** (see [LICENSE](LICENSE)). In addition, by contributing
you grant the Licensor, Tom Gould, a perpetual, worldwide, non-exclusive, royalty-free
right to use, modify and sublicense your contribution under that licence and under any
other licence the Licensor chooses, including the Change License. That is what lets the
project change its licence later, or offer a commercial licence, without asking every
contributor again.
