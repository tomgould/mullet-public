# ADR 0007: Sandbox backends

Status: Accepted. The `docker` and `podman` backends reserved below are built: see 0011.

## Context

Mullet runs code a model has just written: setup, checks, gates and the
opencode worker. Bubblewrap confines all of it on Linux. macOS, and Linux
machines without working user namespaces, need another way (a Docker or
Podman container, or Seatbelt), and every way must keep the same guarantees.

What each caller needs from a sandbox:

| Caller | Entry point | Needs |
|---|---|---|
| `GateRunner.run_command` (`gates.py`): checks and gates | `sandbox.run_shell` | cwd the clone; writable: clone, the command's log dir, the job cache; dependency dirs read-only on top; toolchains and the clone's git dir read-only; no network; the profile's safe env plus cache variables and `MULLET_JUNIT`; a timeout that kills the whole tree; output to a log file |
| `GateRunner.setup` (`gates.py`) | `sandbox.run_shell` | as above, plus the host's network, dependency dirs writable and the profile's `setup_writable` paths |
| opencode worker (`executors/opencode.py`) | `sandbox.spawn` | a long-lived process with stdout piped (streamed JSON) and stderr to a file, text mode, its own process group so the executor can kill it on wall clock or spend; writable: clone and opencode's own dirs; read-only: toolchains, the opencode install, the attempt dir, the git dir and the key proxy's socket dir; no network (socat inside bridges a loopback port to the proxy's Unix socket); an env of Mullet's own settings only |
| opencode worker's shell (`shell_wrapper`) | `sandbox.nested_argv`, `sandbox.nested_check` | a second sandbox inside the first, for each shell command: no network at all (not even the proxy port), its own pid, IPC and UTS namespaces and /proc, the proxy's socket dir covered, opencode's home read-only, started from an empty environment |
| `mullet job run` preflight, `mullet doctor`, opencode's `check_runnable` | `sandbox.check` | whether the sandbox works, and why not |

The Claude Code worker is not in this table: it needs Anthropic's API, so it
runs `claude -p` under Claude Code's own sandbox for Bash, fail closed (see
`executors/claude_code.py`).

## Decision

`mullet/jobs/sandbox_backends/` holds a `SandboxBackend` Protocol (`name`,
`check`, `build_argv`, `spawn`, `run`, `run_shell`, `nested_argv`,
`nested_check`) and one implementation, `BubblewrapBackend`.
`mullet.jobs.sandbox` keeps the names every caller uses and delegates to one
backend per process, picked by `sandbox.backend` in the config file:
`bubblewrap` (the default), `auto` (the first backend that works here), or
`docker`, `podman` and `seatbelt`, reserved for backends still to come.

Every backend satisfies this contract:

- No network for workers, checks and gates. Setup commands are the only
  network users, and only when the caller asks (`SandboxSpec.network`).
- The root filesystem is read-only. Only the spec's `writable` paths take
  writes, and `readonly_overlays` stay read-only inside them. Home, /tmp and
  /run, with the sockets in them, are hidden unless the spec exposes a path.
- The environment is cleared and rebuilt from an allowlist (`build_env`):
  a few locale and path variables, the spec's own values, then Mullet's.
  Mullet's own environment, which holds provider keys, never enters.
- No provider key is inside the sandbox. The opencode worker reaches its
  provider only through the key proxy's Unix socket, and that socket is
  reachable from inside when its directory is in `expose_ro`.
- A timeout kills the whole process tree.
- No fallback to unsandboxed work. An unknown, reserved or broken backend
  makes `check()` fail and every entry point raise `SandboxUnavailable`;
  `auto` raises when no backend works. Nothing runs instead.

The contract is executable: the `test_contract_*` tests in
`tests/test_sandbox_backends.py` take a `backend` fixture, and a backend
joins them by adding its name to `CONTRACT_BACKENDS`.

## Consequences

- Gates and executors need no change to run under a new backend; a new
  backend is one class, one entry in `BACKENDS` and its name moved from
  `RESERVED` to `AVAILABLE` in `sandbox_backends/settings.py`.
- A backend is ready for jobs only when it passes the contract tests on the
  platforms it claims. Until one does, macOS jobs stay refused.
- `nested_argv` is the hardest part for a container backend: it needs a
  second, network-less confinement inside the container for the opencode
  worker's shell, or opencode jobs stay refused on that backend
  (`nested_check` returns false).
- `mullet doctor` shows the active backend and why, and `mullet config show`
  lists `sandbox.backend` with its default. A bad value is reported and
  bubblewrap kept.
