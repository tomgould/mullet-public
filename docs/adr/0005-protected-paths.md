# 0005. Editor, dev container, shell hook and package-manager files are protected paths

Status: Accepted

Context: A worker's branch is merged into the user's checkout. Some files are run or read by the user's own tools as soon as the repo is opened or the branch is merged, so a change there executes code outside the sandbox.

Decision: `DEFAULT_PROTECTED` in `mullet/jobs/profile.py` lists paths a step may not change unless the approved plan allows it for that step, and gate G2 enforces it. Besides git files (`.git/**`, `.gitattributes`, `.gitmodules`, `.lfsconfig`), CI (`.github/workflows/**`, `.github/actions/**`, `.gitlab-ci.yml`, `.husky/**`, `.pre-commit-config.yaml`) and agent files (`AGENTS.md`, `CLAUDE.md`, `.claude/**`, `.mcp.json`, `opencode.json`, `.mullet/**`), it covers `.envrc`, `.vscode/**`, `.devcontainer/**`, `.npmrc`, `.yarnrc`, `.yarnrc.yml` and `.pypirc`. `.gitignore` is left out on purpose, because editing it is routine.

Consequences: A step that needs one of these files fails G2 unless the plan names it. A profile's `protected_paths` adds to this list; it does not replace the defaults.
