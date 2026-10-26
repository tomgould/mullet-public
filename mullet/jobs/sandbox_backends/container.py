"""The docker and podman backends: one hardened, throwaway container per command.

Each command runs in a container from a local image (`sandbox.image`, built
by `mullet sandbox build-image` from the Dockerfile beside this module):

- as your own uid and gid, with every capability dropped, no-new-privileges,
  a read-only root, its own IPC namespace, `--pids-limit` and `--memory`
  (`sandbox.pids_limit`, `sandbox.memory`; swap capped at the same size);
- `--network none` unless the spec asks for network (setup only), which
  gets the engine's default network: the internet, not the host's loopback;
- empty tmpfs at /tmp, /run and $HOME, and at any hidden path that would
  otherwise show through a mount; only the spec's paths are bind-mounted,
  at the same place: writable ones read-write, the rest read-only, and a
  symlinked writable path or overlay is refused, as bubblewrap refuses it;
- `--rm`, no stored logs, never a pulled image.

The environment never goes on a command line, where `ps` and `docker inspect`
would show it, nor into the container's configuration. The container's PID 1
is Mullet's own init (`pid1.py`, run by the image's Python under `env -i`
from an empty environment). It reads the variables from its stdin as one
JSON line, starts the command with stdin on /dev/null, and watches the rest
of stdin, the lifeline: Mullet holds the pipe's write end, so when Mullet
dies, or the docker client is killed, the container's stdin closes and PID 1
kills every process in the container and exits. A process inside can't stop
that watch: the kernel won't deliver SIGKILL or SIGSTOP to a namespace's
init from inside it, and PID 1 is non-dumpable, so it can't be traced.

Every container is labelled with the Mullet process that owns it (pid, uid,
start in clock ticks since boot, boot and machine); `reap_orphans` removes
this user's containers whose owner is gone.

There is no nested sandbox here: Docker's default seccomp and AppArmor
profiles refuse new namespaces inside a container, and Mullet won't loosen
them, so the opencode worker (which needs one for its shell) is refused on
these backends. See docs/adr/0011-container-sandbox-backend.md.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import secrets
import select
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import IO

from mullet.jobs import children
from mullet.jobs.sandbox_backends.base import (
    RunResult,
    SandboxCheck,
    SandboxSpec,
    SandboxUnavailable,
    build_env,
    kill_group,
    read_tail,
)
from mullet.jobs.sandbox_backends.settings import AUTO, SandboxSettings, load_settings

BUILD_COMMAND = "mullet sandbox build-image"
ENGINES = ("docker", "podman")
LABEL = "mullet.sandbox=1"
NAME_PREFIX = "mullet-sbx-"
OWNER, OWNER_UID, OWNER_START = "mullet.owner", "mullet.owner.uid", "mullet.owner.start"
BOOT, HOST = "mullet.boot", "mullet.host"
# Without a machine id, the host label ends in this random id, kept in Mullet's state directory.
HOST_TOKEN_FILE = "sandbox-host-id"
# The labels the Dockerfile gives the image; check() refuses an image without them.
IMAGE_TITLE = ("org.opencontainers.image.title", "mullet-sandbox")
IMAGE_VERSION = "mullet.sandbox.version"
# Where the docker client finds its engine. These reach the client on the
# host only; nothing passes them into the container.
LAUNCHER_PASSTHROUGH = ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "DOCKER_CERT_PATH",
                        "DOCKER_TLS_VERIFY", "CONTAINER_HOST", "CONTAINER_CONNECTION", "HOME",
                        "XDG_RUNTIME_DIR")
# The most environment a container gets. It is written once the client has
# started, and a client that hasn't read it all within PAYLOAD_TIMEOUT seconds
# is killed with its container.
MAX_ENV_BYTES = 60_000
PAYLOAD_TIMEOUT = 30.0
# Cleanup runs on `job stop`, a runner's start and `mullet doctor`, so a wedged
# daemon may hold it up only this long: per engine call, and for the whole reap.
REAP_CLI_TIMEOUT = 5.0
REAP_DEADLINE = 15.0
NESTED_REASON = (
    "the {name} backend can't start a nested sandbox: Docker's and Podman's default seccomp and AppArmor "
    "profiles refuse new namespaces inside a container, and Mullet won't loosen them. opencode jobs need "
    "the bubblewrap backend. Checks, gates and setup run under {name}, and the Claude worker uses Claude "
    "Code's own sandbox either way")
# The probe records the uid it ran as, then the confinement the kernel reports for it.
PROBE_SCRIPT = "id -u > probe && grep -E '^(Seccomp|CapEff|NoNewPrivs):' /proc/self/status >> probe"
NO_SECCOMP_REASON = (
    "the {binary} daemon runs containers without seccomp ({evidence}). Without it a command could make system calls Docker's default "
    "profile refuses, such as unshare, mount, keyctl and bpf, which Mullet relies on that profile to stop. "
    "Turn seccomp back on in the daemon (it is on by default) and run mullet doctor again")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_UNMOUNTABLE = (",", '"', "\n", "\r")

# Mullet's PID 1 for every container (see pid1.py). Its source goes on the
# command line, so the image needs only /usr/bin/python3, and an image built
# for an older Mullet still gets this version's init.
PID1 = Path(__file__).with_name("pid1.py").read_text(encoding="utf-8")


def _new_name() -> str:
    return f"{NAME_PREFIX}{os.getpid()}-{secrets.token_hex(4)}"


def _pipe() -> tuple[int, int]:
    """The lifeline pipe: Mullet keeps the write end, the client gets the read end as its stdin."""
    return os.pipe()


def _feed(fd: int, payload: bytes, timeout: float) -> bool:
    """Write all of `payload` to the pipe `fd` within `timeout` seconds; False when it couldn't."""
    deadline = time.monotonic() + timeout
    view = memoryview(payload)
    os.set_blocking(fd, False)
    try:
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if not select.select([], [fd], [], remaining)[1]:
                continue
            try:
                view = view[os.write(fd, view):]
            except BlockingIOError:
                continue
            except OSError:  # the client is gone, so nothing reads the pipe
                return False
        return True
    finally:
        os.set_blocking(fd, True)


def _read_id(*paths: str) -> str:
    for path in paths:
        try:
            text = Path(path).read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if text:
            return text
    return ""


boot_id = children.boot_id


def _install_token() -> str:
    """A random id for this Mullet install, made once in its state directory; else a hash of that directory's path."""
    from mullet.core.paths import state_dir

    directory = state_dir()
    path = directory / HOST_TOKEN_FILE
    try:
        if not path.exists():
            directory.mkdir(parents=True, exist_ok=True)
            fresh = directory / f".{HOST_TOKEN_FILE}.{os.getpid()}.{secrets.token_hex(4)}"
            fresh.write_text(secrets.token_hex(16) + "\n", encoding="ascii")
            try:
                os.link(fresh, path)  # never replaces one another process made first
            except FileExistsError:
                pass
            finally:
                fresh.unlink(missing_ok=True)
        token = path.read_text(encoding="ascii").strip()
        if token:
            return token
    except (OSError, UnicodeDecodeError):
        pass
    return hashlib.sha256(str(directory).encode()).hexdigest()[:32]


@functools.cache
def host_id() -> str:
    """This machine's id, so a daemon shared by several machines only reaps this one's containers.

    Without a machine id, the hostname alone could be shared (WSL2 distros on
    one Docker Desktop daemon all have the Windows host's), so the uid and a
    random id kept in this install's state directory are added.
    """
    machine = _read_id("/etc/machine-id", "/var/lib/dbus/machine-id")
    return machine or f"{socket.gethostname()}/{os.getuid()}/{_install_token()}"


def owner_labels() -> dict[str, str]:
    """The labels that name the Mullet process owning a container: pid, uid, start in clock ticks, boot, machine."""
    return {OWNER: str(os.getpid()), OWNER_UID: str(os.getuid()),
            OWNER_START: str(children.start_ticks(os.getpid())), BOOT: boot_id(), HOST: host_id()}


def owner_gone(labels: dict[str, str]) -> bool:
    """True when the labels name a Mullet process on this machine that no longer runs.

    The owner is the process whose start, in clock ticks since boot, is
    exactly the labelled one: unlike a wall-clock start time, a clock step
    doesn't change it. A container without owner labels, from another
    machine, or of another user, is never called orphaned: another user's
    process can be invisible here (/proc mounted with hidepid), so its pid
    would look gone.
    """
    if (labels.get(OWNER_UID) != str(os.getuid()) or labels.get(HOST) != host_id()
            or OWNER not in labels or OWNER_START not in labels):
        return False
    if labels.get(BOOT) != boot_id():
        return True
    try:
        pid, started = int(labels[OWNER]), int(labels[OWNER_START])
    except ValueError:
        return False
    ticks = children.start_ticks(pid)
    if ticks is None:  # gone, unless it is there but unreadable
        return not os.path.exists(f"/proc/{pid}")
    return ticks != started


def reap_orphans(engines: list[str] | None = None) -> int:
    """Remove sandbox containers whose owning Mullet process is gone; returns how many. Never raises.

    Without `engines`, asks the engine `sandbox.backend` names, or every
    installed one when it is `auto`. An engine that isn't installed is skipped.
    It gives up after REAP_DEADLINE seconds, whatever is left.
    """
    deadline = time.monotonic() + REAP_DEADLINE
    removed = 0
    try:
        if engines is None:
            backend = load_settings().backend
            engines = list(ENGINES) if backend == AUTO else [backend] if backend in ENGINES else []
        settings = load_settings()
        classes = {"docker": DockerBackend, "podman": PodmanBackend}
        for name in engines:
            if name in classes and shutil.which(name) and time.monotonic() < deadline:
                removed += classes[name](settings).reap_orphans(deadline)
    except Exception:  # cleanup must never take a runner, a stop or doctor down with it
        pass
    return removed


def launcher_env() -> dict[str, str]:
    """The docker client's own environment: PATH and where its engine is, nothing else."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    env.update({k: os.environ[k] for k in LAUNCHER_PASSTHROUGH if k in os.environ})
    return env


def env_payload(spec: SandboxSpec) -> bytes:
    """The sandbox's environment as one JSON line, for the container's stdin."""
    env = build_env(spec)
    for key in env:
        if not _ENV_NAME.fullmatch(key):
            raise ValueError(f"refusing to pass {key!r} into the sandbox: not a shell variable name")
    payload = (json.dumps(env, ensure_ascii=True) + "\n").encode()
    if len(payload) > MAX_ENV_BYTES:
        raise ValueError(f"the sandbox environment is {len(payload)} bytes, over {MAX_ENV_BYTES}")
    return payload


def _mountable(path: Path) -> str:
    text = str(path)
    if any(c in text for c in _UNMOUNTABLE):
        raise ValueError(f"refusing to mount {path}: the container backends can't mount a path with a comma, "
                         "quote or line break in it")
    return text


def _sandbox_image(labels_json: str) -> bool:
    """True when an image's labels say it was built from Mullet's Dockerfile, or FROM an image that was.

    Labels are not a signature: they keep an unrelated image that happens to
    carry the tag from being used, not a deliberately forged one.
    """
    try:
        labels = json.loads(labels_json)
    except ValueError:
        return False
    return (isinstance(labels, dict) and labels.get(IMAGE_TITLE[0]) == IMAGE_TITLE[1]
            and isinstance(labels.get(IMAGE_VERSION), str) and bool(labels[IMAGE_VERSION].strip()))


def _unconfined(status_lines: list[str]) -> list[str]:
    """What the probe's /proc/self/status lines show missing: seccomp filtering, no capabilities, no-new-privileges."""
    status = {}
    for line in status_lines:
        key, sep, value = line.partition(":")
        if sep:
            status[key.strip()] = value.strip()
    missing = []
    if status.get("Seccomp") != "2":
        missing.append(f"seccomp filtering (Seccomp: {status.get('Seccomp', 'not shown')})")
    try:
        caps = int(status.get("CapEff", ""), 16)
    except ValueError:
        caps = -1
    if caps != 0:
        missing.append(f"dropping every capability (CapEff: {status.get('CapEff', 'not shown')})")
    if status.get("NoNewPrivs") != "1":
        missing.append(f"no-new-privileges (NoNewPrivs: {status.get('NoNewPrivs', 'not shown')})")
    return missing


def _symlinked_component(path: Path, roots: list[Path]) -> Path | None:
    """The first symlink among `path`'s components at or below any of `roots`, or None.

    Only components inside a writable path matter: those are the ones a
    sandboxed command could have changed.
    """
    for root in roots:
        if path != root and root not in path.parents:
            continue
        current = root
        for part in ("", *path.relative_to(root).parts):
            current = current / part if part else current
            try:
                if stat.S_ISLNK(os.lstat(current).st_mode):
                    return current
            except FileNotFoundError:
                break
    return None


def check_mount_paths(spec: SandboxSpec) -> None:
    """Refuse a spec whose mounts a sandboxed command could have redirected; ValueError says which path.

    Run when the argv is built; `check_binds` checks the strings the argv
    holds again just before the engine starts, since the root daemon resolves
    every source itself, following any symlink.
    """
    exposed = [p for p in spec.expose_ro if p.exists()]
    for path in [*spec.writable, *spec.readonly_overlays, *spec.expose_ro]:
        if ".." in path.parts:
            raise ValueError(f"refusing to mount {path}: not a normalised path")
    for path in [*spec.writable, *spec.readonly_overlays]:
        if path.is_symlink():
            raise ValueError(f"refusing to mount {path}: it is a symlink")
    roots = list(spec.writable)
    for path in [*spec.writable, *spec.readonly_overlays, *exposed, *(p.resolve() for p in exposed)]:
        link = _symlinked_component(path, roots)
        if link is not None:
            raise ValueError(f"refusing to mount {path}: {link} is a symlink inside a writable path")


def _bind_pairs(mounts: list[str]) -> list[tuple[str, str]]:
    """The (source, target) of each bind in `_mounts`' options, exactly as the engine gets them."""
    pairs = []
    for flag, value in zip(mounts[::2], mounts[1::2]):
        if flag == "--mount":
            fields = dict(field.partition("=")[::2] for field in value.split(","))  # no comma in a path
            pairs.append((fields["source"], fields["target"]))
    return pairs


def check_binds(binds: list[tuple[str, str]], writable: list[Path]) -> None:
    """Refuse when a bind's source or target, as handed to the engine, now runs through a symlink.

    The sources are the strings in the argv, so what is checked is exactly
    what the daemon will resolve, not the spec's paths resolved again.
    """
    for source, target in binds:
        for text in (source, target):
            link = _symlinked_component(Path(text), writable)
            if link is not None:
                raise ValueError(f"refusing to mount {text}: {link} is a symlink inside a writable path")
        if os.path.islink(source):
            raise ValueError(f"refusing to mount {source}: it is a symlink")


def _related(path: Path, others: list[Path]) -> bool:
    return any(path == o or path in o.parents or o in path.parents for o in others)


class _Launch:
    """One container about to start: its argv, name, stdin payload, spec, and the binds the argv holds."""

    def __init__(self, argv: list[str], name: str, payload: bytes, spec: SandboxSpec,
                 binds: list[tuple[str, str]] | None = None):
        self.argv = argv
        self.name = name
        self.payload = payload
        self.spec = spec
        self.binds = binds or []

    @property
    def bind_sources(self) -> list[str]:
        """Each bind's source exactly as the argv hands it to the engine."""
        return [source for source, _ in self.binds]


class ContainerBackend:
    """Confine each command in a hardened container. Subclasses name the engine."""

    name = "container"
    binary = "docker"

    def __init__(self, settings: SandboxSettings | None = None):
        self.settings = settings or load_settings()

    # The engine

    def _engine(self) -> str:
        return shutil.which(self.binary) or self.binary

    def _cli(self, args: list[str], timeout: float = 20) -> subprocess.CompletedProcess:
        """Run the engine's CLI on the host with the launcher environment; never raises."""
        try:
            return subprocess.run([self._engine(), *args], capture_output=True, text=True, timeout=timeout,
                                  env=launcher_env(), stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return subprocess.CompletedProcess(args, 127, "", str(exc))

    def _engine_options(self) -> list[str]:
        """Options this engine needs beyond the shared ones."""
        return []

    def _engine_checks(self) -> str:
        """Engine-specific probing before the image check; an error message, or empty."""
        return ""

    def reap_orphans(self, deadline: float | None = None) -> int:
        """Remove this engine's sandbox containers whose owning Mullet process is gone; returns how many.

        Each engine call gets at most REAP_CLI_TIMEOUT seconds, and none starts
        after `deadline` (a time.monotonic() value; REAP_DEADLINE from now by default).

        A container outlives Mullet only when its PID 1 never saw the lifeline
        close (the docker client wedged, the daemon restarted mid-run); this
        is the backstop. Containers of live owners, other users or machines,
        or from before these owner labels existed, are left alone, and so is
        one not named mullet-sbx-*: `docker commit` copies a sandbox
        container's labels into the image, and so into every container
        started from it.
        """
        end = time.monotonic() + REAP_DEADLINE if deadline is None else deadline

        def budget() -> float:
            return min(REAP_CLI_TIMEOUT, end - time.monotonic())

        if budget() <= 0:
            return 0
        listed = self._cli(["ps", "--all", "--quiet", "--no-trunc", "--filter", f"label={LABEL}"], timeout=budget())
        ids = listed.stdout.split() if listed.returncode == 0 else []
        if not ids or budget() <= 0:
            return 0
        found = self._cli(["inspect", "--format", "{{.Id}} {{.Name}} {{json .Config.Labels}}", *ids],
                          timeout=budget())
        gone = []
        for line in found.stdout.splitlines() if found.returncode == 0 else []:
            cid, _, rest = line.partition(" ")
            name, _, raw = rest.partition(" ")
            if not name.lstrip("/").startswith(NAME_PREFIX):  # docker shows "/name", podman "name"
                continue
            try:
                labels = json.loads(raw)
            except ValueError:
                continue
            if isinstance(labels, dict) and owner_gone(labels):
                gone.append(cid)
        if not gone or budget() <= 0:
            return 0
        removed = self._cli(["rm", "--force", *gone], timeout=budget())
        return len(gone) if removed.returncode == 0 else 0

    # The Protocol

    def check(self) -> SandboxCheck:
        """Probe the engine, its daemon, the image, then run a real confined container."""
        if sys.platform != "linux":
            return SandboxCheck(False, self.name, f"the {self.name} backend is verified on Linux only so far; "
                                                  f"on {sys.platform} jobs stay refused")
        if not shutil.which(self.binary):
            return SandboxCheck(False, self.name, f"{self.binary} is not installed")
        version = self._cli(["version", "--format", "{{.Server.Version}}"])
        if version.returncode != 0:
            detail = (version.stderr or version.stdout).strip() or f"exit {version.returncode}"
            return SandboxCheck(False, self.name, f"the {self.binary} daemon is unreachable: {detail}")
        problem = self._engine_checks()
        if problem:
            return SandboxCheck(False, self.name, problem)
        image = self.settings.image
        found = self._cli(["image", "inspect", "--format", "{{json .Config.Labels}}", image])
        if found.returncode != 0:
            return SandboxCheck(False, self.name, f"the sandbox image {image} is not there; build it with: "
                                                  f"{BUILD_COMMAND}")
        if not _sandbox_image(found.stdout):
            return SandboxCheck(False, self.name, f"{image} is not a Mullet sandbox image: it lacks the labels "
                                                  f"{IMAGE_TITLE[0]}={IMAGE_TITLE[1]} and {IMAGE_VERSION}. "
                                                  f"Build it with: {BUILD_COMMAND} (or build yours FROM it)")
        return self._probe()

    def _probe(self) -> SandboxCheck:
        """Write into a bound directory as the host user, and read back the confinement the command got."""
        probe_dir = Path(tempfile.mkdtemp(prefix="mullet-probe-"))
        try:
            spec = SandboxSpec(cwd=probe_dir, writable=[probe_dir])
            result = self.run(["/bin/sh", "-c", PROBE_SCRIPT], spec, timeout=120, log_path=probe_dir / "probe.log")
            made = probe_dir / "probe"
            if result.exit_code != 0 or not made.is_file():
                detail = result.tail.strip() or ("timed out" if result.timed_out else f"exit {result.exit_code}")
                return SandboxCheck(False, self.name, f"{self.binary} can't create a sandbox: {detail}")
            uid, *lines = made.read_text().splitlines() or [""]
            if uid.strip() != str(os.getuid()) or made.stat().st_uid != os.getuid():
                return SandboxCheck(False, self.name, f"{self.binary} doesn't run containers as your own uid "
                                                      "(a rootless or user-namespace-remapped engine?), so "
                                                      "the clone can't be written safely")
            missing = _unconfined(lines)
            if missing:
                return SandboxCheck(False, self.name, f"{self.binary} starts containers without "
                                                      f"{', '.join(missing)}; Mullet runs commands only with "
                                                      "all three (check the daemon's security settings)")
            return SandboxCheck(True, self.name)
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)

    def build_argv(self, command: list[str], spec: SandboxSpec) -> list[str]:
        """The host command line that runs `command` under `spec`.

        The environment is not in it: it goes on the container's stdin
        (`env_payload`), which `run` and `spawn` write.
        """
        return self._launch(command, spec).argv

    def _launch(self, command: list[str], spec: SandboxSpec, *, merged: bool = False) -> _Launch:
        name = _new_name()
        uid, gid = os.getuid(), os.getgid()
        s = self.settings
        owner = [a for key, value in owner_labels().items() for a in ("--label", f"{key}={value}")]
        argv = [self._engine(), "run", "--rm", "--interactive", "--sig-proxy=true",
                "--name", name, "--label", LABEL, *owner, "--pull", "never", "--log-driver", "none",
                "--user", f"{uid}:{gid}", *self._engine_options(),
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only",
                "--ipc", "private", "--pids-limit", str(s.pids_limit),
                "--memory", s.memory, "--memory-swap", s.memory]
        if not spec.network:
            argv += ["--network", "none"]
        mounts = self._mounts(spec, uid, gid)
        argv += mounts
        argv += ["--workdir", str(spec.cwd), "--entrypoint", "/usr/bin/env", s.image,
                 "-i", "/usr/bin/python3", "-I", "-S", "-c", PID1, "merged" if merged else "split", *command]
        return _Launch(argv, name, env_payload(spec), spec, _bind_pairs(mounts))

    def _mounts(self, spec: SandboxSpec, uid: int, gid: int) -> list[str]:
        """tmpfs and bind mounts. The engine mounts parents before children, whatever the order here.

        For one path named twice, the later kind wins, in bubblewrap's order:
        hidden, then exposed, then writable, then the read-only overlays.
        """
        check_mount_paths(spec)
        for path in spec.writable:
            if not path.is_dir():
                raise ValueError(f"writable path {path} is not a directory")
        owned = f"rw,nosuid,nodev,mode=0700,uid={uid},gid={gid}"
        table: dict[str, str] = {
            "/tmp": "tmpfs:rw,nosuid,nodev,mode=1777",
            "/run": f"tmpfs:rw,nosuid,nodev,mode=0755,uid={uid},gid={gid}",
        }
        home = Path(build_env(spec)["HOME"])
        if home.is_absolute() and home != Path("/"):
            table.setdefault(str(home), f"tmpfs:{owned}")
        exposed = [p for p in spec.expose_ro if p.exists()]
        overlays = [p for p in spec.readonly_overlays if p.exists()]
        bound = [*exposed, *spec.writable, *overlays]
        for hidden in spec.hide:
            # Only a hidden path that a mount would show needs covering: nothing else of the host is there.
            if hidden.is_dir() and not hidden.is_symlink() and _related(hidden, bound):
                table.setdefault(str(hidden), f"tmpfs:{owned}")
        for path in exposed:
            table[_mountable(path)] = f"bind:{_mountable(path.resolve())}:ro"
        for path in spec.writable:
            table[_mountable(path)] = f"bind:{path}:rw"
        for path in overlays:
            table[_mountable(path)] = f"bind:{path}:ro"
        tmpfs, binds = [], []
        for target, how in table.items():
            kind, _, rest = how.partition(":")
            if kind == "tmpfs":
                if ":" in target:
                    raise ValueError(f"refusing to cover {target}: the container backends can't put a tmpfs "
                                     "on a path with a colon in it")
                tmpfs += ["--tmpfs", f"{target}:{rest}"]
            else:
                source, _, mode = rest.rpartition(":")
                binds += ["--mount", f"type=bind,source={source},target={target}" + (",readonly" if mode == "ro" else "")]
        return tmpfs + binds

    def _start(self, launch: _Launch, *, stdout: int | IO[bytes] | None, stderr: int | IO[bytes] | None,
               cwd: Path | None = None, text: bool = False) -> tuple[subprocess.Popen, int]:
        """Start the client in its own session; returns it and the lifeline's write end, which the caller closes."""
        check_binds(launch.binds, launch.spec.writable)  # the clone may have changed since the argv was built
        read_end, write_end = _pipe()
        try:
            extra: dict = {"text": True, "encoding": "utf-8", "errors": "replace"} if text else {}
            proc = subprocess.Popen(launch.argv, cwd=cwd, stdin=read_end, stdout=stdout, stderr=stderr,
                                    start_new_session=True, env=launcher_env(), **extra)
        except BaseException:
            os.close(write_end)
            raise
        finally:
            os.close(read_end)
        # Written once the client exists, since a pipe can be smaller than the
        # payload. A client that doesn't take it in time is killed, and so is
        # its container; PID 1 refuses a payload cut short.
        if not _feed(write_end, launch.payload, PAYLOAD_TIMEOUT):
            kill_group(proc, grace=2)
            self._cli(["rm", "--force", launch.name], timeout=60)
        return proc, write_end

    def _finish(self, name: str, write_end: int, returncode: int | None) -> None:
        """Close the lifeline; remove the container if the client didn't see it end."""
        try:
            os.close(write_end)
        except OSError:
            pass
        if returncode is None or returncode < 0:
            self._cli(["rm", "--force", name], timeout=60)

    def spawn(self, command: list[str], spec: SandboxSpec, *, stdout: int | IO[bytes] | None,
              stderr: int | IO[bytes] | None, cwd: Path | None = None,
              text: bool = False) -> subprocess.Popen:
        """Start the container's client in a new session. A watcher thread cleans up after it exits."""
        launch = self._launch(command, spec, merged=stderr == subprocess.STDOUT)
        proc, write_end = self._start(launch, stdout=stdout, stderr=stderr, cwd=cwd, text=text)

        def reap() -> None:
            try:
                proc.wait()
            finally:
                self._finish(launch.name, write_end, proc.returncode)

        threading.Thread(target=reap, name=f"reap-{launch.name}", daemon=True).start()
        return proc

    def run(self, command: list[str], spec: SandboxSpec, *, timeout: float, log_path: Path) -> RunResult:
        """Run `command` in a container, all output to `log_path`.

        The spec is checked before the log is created, so a refused spec leaves
        no log. On timeout the client's group gets TERM (passed on to the
        container), then KILL, which closes the lifeline; the container is
        then removed by name.
        """
        launch = self._launch(command, spec, merged=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        timed_out = False
        with open(log_path, "wb") as log:
            proc, write_end = self._start(launch, stdout=log, stderr=subprocess.STDOUT)
            try:
                with children.track(proc):
                    try:
                        proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        kill_group(proc)
            finally:
                self._finish(launch.name, write_end, None if timed_out else proc.returncode)
        return RunResult(
            exit_code=None if timed_out else proc.returncode,
            timed_out=timed_out,
            duration_s=time.monotonic() - started,
            log_path=log_path,
            tail=read_tail(log_path),
        )

    def run_shell(self, script: str, spec: SandboxSpec, *, timeout: float, log_path: Path) -> RunResult:
        """Run a shell command string (a profile command) in a container."""
        return self.run(["/bin/sh", "-c", script], spec, timeout=timeout, log_path=log_path)

    def nested_argv(self, readonly: list[Path], hidden: list[Path]) -> list[str]:
        """Refused: see NESTED_REASON."""
        raise SandboxUnavailable(NESTED_REASON.format(name=self.name))

    def nested_check(self) -> bool:
        """False: a container here can't start a second, network-less sandbox inside itself."""
        return False


class DockerBackend(ContainerBackend):
    """Confine processes in Docker containers."""

    name = "docker"
    binary = "docker"

    def _engine_checks(self) -> str:
        info = self._cli(["info", "--format", "{{json .SecurityOptions}}"])
        if info.returncode != 0:
            return f"docker info failed: {(info.stderr or info.stdout).strip() or info.returncode}"
        try:
            options = json.loads(info.stdout)
        except ValueError:
            options = None
        seccomp = [o for o in options if isinstance(o, str) and "name=seccomp" in o.split(",")] if isinstance(
            options, list) else []
        if not seccomp or any("profile=unconfined" in o.split(",") for o in seccomp):
            return NO_SECCOMP_REASON.format(
                binary=self.binary, evidence="docker info lists no seccomp profile in SecurityOptions, or an "
                                             "unconfined one")
        return ""


class PodmanBackend(ContainerBackend):
    """Confine processes in Podman containers; rootless Podman keeps your uid with `--userns keep-id`."""

    name = "podman"
    binary = "podman"

    def __init__(self, settings: SandboxSettings | None = None):
        super().__init__(settings)
        self.rootless: bool | None = None

    def _engine_options(self) -> list[str]:
        # Unknown until check() runs: assume rootless, the usual Podman. A rootful
        # engine then refuses to start, which fails closed.
        return [] if self.rootless is False else ["--userns", "keep-id"]

    def _engine_checks(self) -> str:
        info = self._cli(["info", "--format", "{{.Host.Security.Rootless}} {{.Host.Security.SECCOMPEnabled}}"])
        if info.returncode != 0:
            return f"podman info failed: {(info.stderr or info.stdout).strip() or info.returncode}"
        rootless, _, seccomp = info.stdout.strip().lower().partition(" ")
        self.rootless = rootless == "true"
        if seccomp != "true":
            return NO_SECCOMP_REASON.format(binary=self.binary,
                                            evidence="podman info reports Host.Security.SECCOMPEnabled as false")
        return ""
