"""Tests for the docker and podman sandbox backends (mullet.jobs.sandbox_backends.container).

The argv, the environment payload and the checks are unit-tested without
Docker. The tests marked `needs_docker` run real containers; they skip when
Docker, its daemon or the sandbox image is missing (build it with
`mullet sandbox build-image`), and on any platform but Linux. With
MULLET_REQUIRE_SANDBOX=docker (the docker-sandbox CI job) they fail instead.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import psutil
import pytest

from mullet import __version__
from mullet.jobs import sandbox
from mullet.jobs.sandbox_backends import BubblewrapBackend, SandboxBackend, get_backend
from mullet.jobs.sandbox_backends import container as ct
from mullet.jobs.sandbox_backends.base import SandboxCheck, SandboxSpec, SandboxUnavailable, kill_group
from mullet.jobs.sandbox_backends.container import DockerBackend, PodmanBackend
from mullet.jobs.sandbox_backends.settings import SandboxSettings

UID, GID = os.getuid(), os.getgid()
SECRET_PARENT = "sk-parent-must-not-leak-5150"
SECRET_SPEC = "spec-secret-value-4711"


@pytest.fixture(autouse=True)
def fresh_selection():
    sandbox.clear_cache()
    yield
    sandbox.clear_cache()


def _real_docker() -> SandboxCheck:
    if not hasattr(_real_docker, "status"):
        _real_docker.status = DockerBackend().check()  # type: ignore[attr-defined]
    return _real_docker.status  # type: ignore[attr-defined]


@pytest.fixture
def docker_ready():
    """Skip without a working docker sandbox, or fail when MULLET_REQUIRE_SANDBOX names docker (CI)."""
    status = _real_docker()
    if not status.ok:
        if "docker" in os.environ.get("MULLET_REQUIRE_SANDBOX", "").split(","):
            pytest.fail(f"MULLET_REQUIRE_SANDBOX names docker, which is unavailable: {status.reason}")
        pytest.skip(f"docker: {status.reason}")


needs_docker = pytest.mark.usefixtures("docker_ready")


@pytest.fixture
def fixed(monkeypatch):
    """A docker backend with a fixed binary, container name and end marker."""
    monkeypatch.setattr(ct.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(ct, "_new_name", lambda: "mullet-sbx-test")
    return DockerBackend(SandboxSettings())


@pytest.fixture
def layout(tmp_path, monkeypatch):
    home = tmp_path / "home"
    work, deps, tools, hidden = (tmp_path / n for n in ("work", "deps", "tools", "hidden"))
    for path in (home, work, deps, tools, hidden, work / "private"):
        path.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("LANG", "C.UTF-8")
    for name in ("LC_ALL", "LC_CTYPE", "TZ"):
        monkeypatch.delenv(name, raising=False)
    return {"home": home, "work": work, "deps": deps, "tools": tools, "hidden": hidden}


# The argv, without Docker


MY_TICKS = ct.children.start_ticks(os.getpid())
OWNER = {"mullet.owner": str(os.getpid()), "mullet.owner.uid": str(UID), "mullet.owner.start": str(MY_TICKS),
         "mullet.boot": ct.boot_id(), "mullet.host": ct.host_id()}
HEAD = ["/usr/bin/docker", "run", "--rm", "--interactive", "--sig-proxy=true",
        "--name", "mullet-sbx-test", "--label", "mullet.sandbox=1",
        *[a for k, v in OWNER.items() for a in ("--label", f"{k}={v}")], "--pull", "never", "--log-driver", "none",
        "--user", f"{UID}:{GID}", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only",
        "--ipc", "private", "--pids-limit", "4096", "--memory", "8g", "--memory-swap", "8g"]


def test_docker_argv_is_exactly_the_documented_line(fixed, layout):
    home, work, deps, tools = layout["home"], layout["work"], layout["deps"], layout["tools"]
    spec = SandboxSpec(cwd=work, writable=[work], readonly_overlays=[deps], expose_ro=[tools],
                       hide=[layout["hidden"], work / "private", home.parent / "absent"], env={"K": "v"})
    owned = f"rw,nosuid,nodev,mode=0700,uid={UID},gid={GID}"
    tmpfs = ["--tmpfs", "/tmp:rw,nosuid,nodev,mode=1777",
             "--tmpfs", f"/run:rw,nosuid,nodev,mode=0755,uid={UID},gid={GID}",
             "--tmpfs", f"{home}:{owned}",
             "--tmpfs", f"{work / 'private'}:{owned}"]
    mounts = ["--mount", f"type=bind,source={tools.resolve()},target={tools},readonly",
              "--mount", f"type=bind,source={work},target={work}",
              "--mount", f"type=bind,source={deps},target={deps},readonly"]
    tail = ["--workdir", str(work), "--entrypoint", "/usr/bin/env", f"mullet-sandbox:{__version__}",
            "-i", "/usr/bin/python3", "-I", "-S", "-c", ct.PID1, "split", "echo", "hi"]
    assert fixed.build_argv(["echo", "hi"], spec) == [*HEAD, "--network", "none", *tmpfs, *mounts, *tail]
    spec.network = True
    assert fixed.build_argv(["echo", "hi"], spec) == [*HEAD, *tmpfs, *mounts, *tail]


def test_docker_argv_follows_the_settings(monkeypatch, layout):
    monkeypatch.setattr(ct.shutil, "which", lambda name: f"/usr/bin/{name}")
    backend = DockerBackend(SandboxSettings(image="example/tools:1", memory="512m", pids_limit=128))
    argv = backend.build_argv(["true"], SandboxSpec(cwd=layout["work"], writable=[layout["work"]]))
    assert argv[argv.index("--memory") + 1] == "512m" and argv[argv.index("--memory-swap") + 1] == "512m"
    assert argv[argv.index("--pids-limit") + 1] == "128"
    assert argv[argv.index("--entrypoint") + 2] == "example/tools:1"


def test_pid1_is_mullets_own_init_and_refuses_to_run_outside_a_container():
    # The source on the command line is pid1.py's, and it never runs (its last
    # act is kill(-1)) unless it is PID 1.
    assert ct.PID1 == (Path(ct.__file__).parent / "pid1.py").read_text()
    for needed in ("os.getpid() != 1", "PR_SET_DUMPABLE", "os.kill(-1, signal.SIGKILL)", '"/dev/null"'):
        assert needed in ct.PID1
    # No payload on stdin: even without the PID 1 check it would stop before starting anything.
    done = subprocess.run([sys.executable, "-I", "-S", "-c", ct.PID1, "split", "true"], input=b"",
                          capture_output=True, timeout=30)
    assert done.returncode == 125 and b"only as a sandbox container's PID 1" in done.stderr


def _start_and_wait(command: list[str], env: dict) -> int:
    """pid1.start in this process, as PID 1 would run it; the started child's exit code."""
    from mullet.jobs.sandbox_backends import pid1

    parent = os.getpid()
    wakeup = os.pipe()
    try:
        child = pid1.start(command, env, False, wakeup)
    finally:
        if os.getpid() != parent:  # the child came back out of start(): main()'s finally makes that 125
            os._exit(125)
    for fd in wakeup:
        os.close(fd)
    return os.waitstatus_to_exitcode(os.waitpid(child, 0)[1])


@pytest.mark.parametrize("command, env, code", [
    (["true"], {"PATH": "/usr/bin:/bin"}, 0),
    (["true"], {"PATH": "/usr/bin:/bin", "BAD": "a\0b"}, 126),  # execvpe raises ValueError
    (["true"], {"PATH": "/usr/bin:/bin", "A=B": "x"}, 126),
    (["/nonexistent/command"], {}, 127),
    (["no-such-command-mullet"], {"PATH": "/usr/bin:/bin"}, 127),
])
def test_pid1_starts_the_command_or_exits_126_or_127(command, env, code):
    assert _start_and_wait(command, env) == code


def test_pid1_exits_126_for_a_file_it_cannot_execute(tmp_path):
    plain = tmp_path / "plain"
    plain.write_text("not a program")
    assert _start_and_wait([str(plain)], {}) == 126


def _read_env(monkeypatch, payload: bytes) -> dict:
    from mullet.jobs.sandbox_backends import pid1

    def die(message, code=125):
        raise SystemExit((code, message))

    monkeypatch.setattr(pid1, "die", die)
    read_end, write_end = os.pipe()
    os.write(write_end, payload)
    os.close(write_end)
    try:
        return pid1.read_environment(read_end)
    finally:
        os.close(read_end)


def test_pid1_reads_a_valid_environment(monkeypatch):
    assert _read_env(monkeypatch, b'{"A": "1", "_B2": "x=y", "C": ""}\nrest') == {"A": "1", "_B2": "x=y", "C": ""}


@pytest.mark.parametrize("payload, named", [
    (b'{"A=B": "x"}\n', "name"), (b'{"": "x"}\n', "name"), (b'{"A\\u0000B": "x"}\n', "name"),
    (b'{"A": "x\\u0000y"}\n', "NUL"), (b'{"A": 1}\n', "strings"), (b'{"A": null}\n', "strings"),
    (b'["A"]\n', "did not arrive"), (b"not json\n", "did not arrive"), (b'{"A": "x"}', "did not arrive")])
def test_pid1_refuses_an_environment_execve_would_not_take(monkeypatch, payload, named):
    with pytest.raises(SystemExit) as refused:
        _read_env(monkeypatch, payload)
    code, message = refused.value.code
    assert code == 125 and named in message
    assert "x\0y" not in message  # a value is never echoed


def test_pid1_never_blocks_writing_a_warning_when_its_wakeup_pipe_is_full(monkeypatch):
    import types

    from mullet.jobs.sandbox_backends import pid1

    class Exited(BaseException):
        pass

    def exit_(code):
        raise Exited(code)

    def stop_here(*args):
        raise RuntimeError("stop before forking")

    seen, pipes = {}, []

    def pipe():
        pipes.extend(os.pipe())
        return pipes[-2], pipes[-1]

    fake_signal = types.SimpleNamespace(**{n: getattr(signal, n) for n in dir(signal) if n.startswith("SIG")})
    fake_signal.signal = fake_signal.pthread_sigmask = lambda *args: None
    fake_signal.set_wakeup_fd = lambda fd, **kwargs: seen.setdefault("wakeup", (fd, kwargs))
    fake_os = types.SimpleNamespace(getpid=lambda: 1, pipe=pipe, set_blocking=os.set_blocking,
                                    kill=lambda *args: None, _exit=exit_, write=os.write)
    fake_ctypes = types.SimpleNamespace(CDLL=lambda *a, **k: types.SimpleNamespace(prctl=lambda *args: 0),
                                        get_errno=lambda: 0)
    monkeypatch.setattr(pid1, "os", fake_os)
    monkeypatch.setattr(pid1, "signal", fake_signal)
    monkeypatch.setattr(pid1, "ctypes", fake_ctypes)
    monkeypatch.setattr(pid1, "read_environment", lambda *args: {})
    monkeypatch.setattr(pid1, "start", stop_here)
    monkeypatch.setattr(pid1.sys, "argv", ["pid1", "split", "true"])
    try:
        with pytest.raises(Exited):
            pid1.main()
    finally:
        for fd in pipes:
            os.close(fd)
    assert seen["wakeup"] == (pipes[1], {"warn_on_full_buffer": False})


def test_podman_keeps_the_host_uid_when_rootless(monkeypatch, layout):
    monkeypatch.setattr(ct.shutil, "which", lambda name: f"/usr/bin/{name}")
    spec = SandboxSpec(cwd=layout["work"], writable=[layout["work"]])
    backend = PodmanBackend(SandboxSettings(backend="podman"))
    backend.rootless = True
    argv = backend.build_argv(["true"], spec)
    assert argv[0] == "/usr/bin/podman" and argv[argv.index("--user") + 2:argv.index("--user") + 4] == [
        "--userns", "keep-id"]
    backend.rootless = False
    assert "--userns" not in backend.build_argv(["true"], spec)
    assert backend.name == "podman" and isinstance(backend, SandboxBackend)


def test_symlinked_writable_paths_and_overlays_are_refused(fixed, layout, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(layout["work"])
    for spec in (SandboxSpec(cwd=layout["work"], writable=[link]),
                 SandboxSpec(cwd=layout["work"], writable=[layout["work"]], readonly_overlays=[link])):
        with pytest.raises(ValueError, match="symlink"):
            fixed.build_argv(["true"], spec)
    with pytest.raises(ValueError, match="not a directory"):
        fixed.build_argv(["true"], SandboxSpec(cwd=layout["work"], writable=[tmp_path / "missing"]))


def test_a_symlinked_component_inside_a_writable_path_is_refused(fixed, layout, tmp_path):
    # A command in the clone can swap a directory for a symlink between runs;
    # the root daemon would then follow it when it resolves the next mount.
    work = layout["work"]
    (tmp_path / "elsewhere" / "deps").mkdir(parents=True)
    (work / "pkg").symlink_to(tmp_path / "elsewhere")
    deps = work / "pkg" / "deps"
    for spec in (SandboxSpec(cwd=work, writable=[work], readonly_overlays=[deps]),
                 SandboxSpec(cwd=work, writable=[work, deps]),
                 SandboxSpec(cwd=work, writable=[work], expose_ro=[deps])):
        with pytest.raises(ValueError, match=f"{work / 'pkg'} is a symlink"):
            fixed.build_argv(["true"], spec)
    # A host symlink outside every writable path is the host's own business.
    (tmp_path / "tools-link").symlink_to(layout["tools"])
    fixed.build_argv(["true"], SandboxSpec(cwd=work, writable=[work], expose_ro=[tmp_path / "tools-link"]))
    with pytest.raises(ValueError, match="normalised"):
        fixed.build_argv(["true"], SandboxSpec(cwd=work, writable=[work], readonly_overlays=[work / ".." / "x"]))


def test_paths_are_checked_again_just_before_the_engine_starts(fixed, layout, tmp_path, monkeypatch):
    work = layout["work"]
    deps = work / "pkg" / "deps"
    deps.mkdir(parents=True)
    launch = fixed._launch(["true"], SandboxSpec(cwd=work, writable=[work], readonly_overlays=[deps]))
    # After the argv is built, the parent becomes a symlink to somewhere else.
    (tmp_path / "elsewhere" / "deps").mkdir(parents=True)
    (work / "pkg").rename(tmp_path / "old-pkg")
    (work / "pkg").symlink_to(tmp_path / "elsewhere")

    def refuse(*args, **kwargs):
        raise AssertionError("the engine was started")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    with pytest.raises(ValueError, match=f"{work / 'pkg'} is a symlink"):
        fixed._start(launch, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_the_exact_bind_sources_handed_to_docker_are_checked_again(fixed, layout, tmp_path, monkeypatch):
    # An exposed host symlink pointing into the clone: the argv carries its
    # resolved target. If the clone then swaps a directory on that path for a
    # symlink, resolving the exposed path again lands outside the clone and
    # looks fine, but the daemon follows the swapped-in symlink in the string it got.
    work = layout["work"]
    (work / "pkg" / "lib").mkdir(parents=True)
    exposed = tmp_path / "tool-link"
    exposed.symlink_to(work / "pkg" / "lib")
    launch = fixed._launch(["true"], SandboxSpec(cwd=work, writable=[work], expose_ro=[exposed]))
    source = str(work / "pkg" / "lib")
    assert f"type=bind,source={source},target={exposed},readonly" in launch.argv
    (tmp_path / "secrets" / "lib").mkdir(parents=True)
    (work / "pkg").rename(tmp_path / "old-pkg")
    (work / "pkg").symlink_to(tmp_path / "secrets")

    def refuse(*args, **kwargs):
        raise AssertionError("the engine was started")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    with pytest.raises(ValueError, match=f"{work / 'pkg'} is a symlink"):
        fixed._start(launch, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert source in launch.bind_sources


def test_a_bind_source_that_became_a_symlink_is_refused_before_the_engine_starts(fixed, layout, tmp_path,
                                                                                 monkeypatch):
    work = layout["work"]
    deps = work / "deps"
    deps.mkdir()
    launch = fixed._launch(["true"], SandboxSpec(cwd=work, writable=[work], readonly_overlays=[deps]))
    deps.rmdir()
    deps.symlink_to(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("the engine was started"))
    with pytest.raises(ValueError, match="symlink"):
        fixed._start(launch, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@pytest.mark.parametrize("name", ["a,b", 'a"b', "a\nb"])
def test_paths_the_mount_syntax_cannot_carry_are_refused(fixed, tmp_path, name):
    odd = tmp_path / name
    odd.mkdir()
    with pytest.raises(ValueError, match="can't mount"):
        fixed.build_argv(["true"], SandboxSpec(cwd=odd, writable=[odd]))


def test_an_exposed_symlink_whose_target_the_mount_syntax_cannot_carry_is_refused(fixed, layout, tmp_path):
    # The link's own name is harmless; the source the daemon gets is the resolved
    # target, and a comma there would end the --mount field early.
    odd = tmp_path / "tools,readonly=false"
    odd.mkdir()
    (tmp_path / "plain-link").symlink_to(odd)
    with pytest.raises(ValueError, match="can't mount"):
        fixed.build_argv(["true"], SandboxSpec(cwd=layout["work"], writable=[layout["work"]],
                                               expose_ro=[tmp_path / "plain-link"]))


def test_a_tmpfs_on_a_path_with_a_colon_is_refused(fixed, layout):
    # --tmpfs takes path:options, so a colon in the path would turn the rest into options.
    work = layout["work"]
    hidden = work / "secrets:rw,exec"
    hidden.mkdir()
    with pytest.raises(ValueError, match="colon"):
        fixed.build_argv(["true"], SandboxSpec(cwd=work, writable=[work], hide=[hidden]))


def test_a_path_both_hidden_and_writable_is_writable(fixed, layout):
    work = layout["work"]
    argv = fixed.build_argv(["true"], SandboxSpec(cwd=work, writable=[work], hide=[work]))
    assert f"type=bind,source={work},target={work}" in argv
    assert not any(a.startswith(f"{work}:") for a in argv)


# The environment never travels on the command line


def test_secrets_never_reach_the_argv_or_the_launcher(fixed, layout, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET_PARENT)
    seen = []

    class FakeProc:
        pid = 1
        returncode = 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(argv, **kwargs):
        seen.append((argv, kwargs, os.dup(kwargs["stdin"])))  # the client's stdin, read once the run ends
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    spec = SandboxSpec(cwd=layout["work"], writable=[layout["work"]], env={"MULLET_SPEC_TOKEN": SECRET_SPEC})
    fixed.run(["true"], spec, timeout=5, log_path=layout["work"] / "run.log")
    ((argv, kwargs, stdin),) = seen
    with os.fdopen(stdin, "rb") as pipe:
        payload = pipe.read()  # all of it: the lifeline is closed by now
    joined = "\0".join(argv)
    assert SECRET_PARENT not in joined and SECRET_SPEC not in joined and "MULLET_SPEC_TOKEN" not in joined
    assert not any(a in ("-e", "--env", "--env-file") for a in argv)
    assert SECRET_PARENT not in json.dumps(kwargs["env"]) and SECRET_SPEC not in json.dumps(kwargs["env"])
    # The spec's own value goes on the container's stdin, and the parent's never does.
    assert json.loads(payload)["MULLET_SPEC_TOKEN"] == SECRET_SPEC and payload.count(b"\n") == 1
    assert SECRET_PARENT.encode() not in payload and payload.endswith(b"}\n")


def test_the_payload_is_one_json_line_and_refuses_odd_names(fixed, layout):
    value = "it's $HOME `x`\nline2 é"
    spec = SandboxSpec(cwd=layout["work"], writable=[layout["work"]], env={"QUOTED": value})
    payload = ct.env_payload(spec)
    assert payload.count(b"\n") == 1 and json.loads(payload)["QUOTED"] == value
    for bad in ("A-B", "1A", "A B", ""):
        with pytest.raises(ValueError, match="variable name"):
            ct.env_payload(SandboxSpec(cwd=layout["work"], writable=[], env={bad: "v"}))


def test_an_oversized_environment_is_refused_before_anything_starts(fixed, layout, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a process was started")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    spec = SandboxSpec(cwd=layout["work"], writable=[layout["work"]], env={"BIG": "x" * 70000})
    log = layout["work"] / "logs" / "big.log"
    with pytest.raises(ValueError, match="environment"):
        fixed.run(["true"], spec, timeout=5, log_path=log)
    assert not log.parent.exists()


def _small_pipe():
    import fcntl

    read_end, write_end = os.pipe()
    fcntl.fcntl(write_end, fcntl.F_SETPIPE_SZ, 4096)  # what a user past pipe-user-pages-soft gets
    return read_end, write_end


def _near_cap_env() -> dict[str, str]:
    return {"BIG": "x" * (ct.MAX_ENV_BYTES - 2000)}  # the rest of build_env fits in what is left


@pytest.mark.timeout(60)
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc and Linux pipe and process features")
def test_a_client_that_never_reads_its_stdin_is_killed_not_waited_on_forever(fixed, layout, monkeypatch):
    # A pipe smaller than the payload, and a client that never reads: the old
    # write-then-start blocked forever before the client even existed.
    monkeypatch.setattr(ct, "_pipe", _small_pipe)
    monkeypatch.setattr(ct, "PAYLOAD_TIMEOUT", 1.0)
    fixed._cli = cli = _Cli({})
    real_launch = fixed._launch

    def deaf_client(command, spec, *, merged=False):
        launch = real_launch(command, spec, merged=merged)
        launch.argv = ["sleep", "60"]
        return launch

    monkeypatch.setattr(fixed, "_launch", deaf_client)
    spec = SandboxSpec(cwd=layout["work"], writable=[layout["work"]], env=_near_cap_env())
    started = time.monotonic()
    result = fixed.run(["true"], spec, timeout=30, log_path=layout["work"] / "deaf.log")
    assert time.monotonic() - started < 15 and result.exit_code != 0
    assert ["rm", "--force", "mullet-sbx-test"] in cli.calls


def test_the_launcher_environment_is_path_and_docker_location_only(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET_PARENT)
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/user/1000/docker.sock")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = ct.launcher_env()
    assert env["PATH"] == "/usr/bin:/bin" and env["DOCKER_HOST"].endswith("docker.sock")
    assert set(env) <= {"PATH", *ct.LAUNCHER_PASSTHROUGH}
    assert SECRET_PARENT not in json.dumps(env)


# check() and the nested sandbox, without Docker


class _Cli:
    """Stands in for the docker CLI: maps the first argument to (exit code, stdout, stderr)."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __call__(self, args, timeout=20):
        self.calls.append(args)
        code, out, err = self.answers.get(args[0], (0, "", ""))
        return subprocess.CompletedProcess(args, code, out, err)


@pytest.mark.parametrize("returncode, removed", [(0, False), (1, False), (143, False), (-9, True), (None, True)])
def test_a_client_that_did_not_see_its_container_end_is_followed_by_rm(returncode, removed):
    backend = DockerBackend(SandboxSettings())
    backend._cli = cli = _Cli({})
    read_end, write_end = os.pipe()
    backend._finish("mullet-sbx-x", write_end, returncode)
    assert cli.calls == ([["rm", "--force", "mullet-sbx-x"]] if removed else [])
    assert os.read(read_end, 1) == b""  # the lifeline is closed either way: the reader sees its end
    os.close(read_end)


def _dead_pid() -> int:
    done = subprocess.Popen(["true"])
    done.wait()
    return done.pid


def _labels(pid: int, start: int, boot: str | None = None, host: str | None = None) -> dict:
    return {"mullet.sandbox": "1", "mullet.owner": str(pid), "mullet.owner.uid": str(UID),
            "mullet.owner.start": str(start),
            "mullet.boot": boot or ct.boot_id(), "mullet.host": host or ct.host_id()}


def test_boot_and_host_ids_are_stable_and_non_empty():
    assert ct.boot_id() and ct.boot_id() == ct.boot_id() and ct.host_id() and ct.host_id() == ct.host_id()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc and Linux pipe and process features")
def test_reap_removes_only_containers_whose_owner_is_gone():
    containers = {
        "dead": _labels(_dead_pid(), 1),
        "recycled": _labels(os.getpid(), MY_TICKS - 100),  # the pid is alive, but not that process
        "rebooted": _labels(os.getpid(), MY_TICKS, boot="another-boot"),
        "alive": _labels(os.getpid(), MY_TICKS),
        "other-machine": _labels(_dead_pid(), 1, host="another-machine"),
        "unlabelled": {"mullet.sandbox": "1"},
    }
    inspect = "\n".join(f"{cid} /mullet-sbx-{cid} {json.dumps(labels)}" for cid, labels in containers.items())
    backend = DockerBackend(SandboxSettings())
    backend._cli = cli = _Cli({"ps": (0, "\n".join(containers), ""), "inspect": (0, inspect, "")})
    assert backend.reap_orphans() == 3
    assert cli.calls[0] == ["ps", "--all", "--quiet", "--no-trunc", "--filter", "label=mullet.sandbox=1"]
    assert cli.calls[1][:3] == ["inspect", "--format", "{{.Id}} {{.Name}} {{json .Config.Labels}}"]
    assert cli.calls[-1] == ["rm", "--force", "dead", "recycled", "rebooted"]


def test_a_clock_step_does_not_make_a_live_owner_look_gone(monkeypatch):
    # psutil's create_time adds the boot time the kernel derives from the wall
    # clock, so NTP or a resume moves it; the start in clock ticks doesn't move.
    labels = {"mullet.sandbox": "1", **ct.owner_labels()}
    real = psutil.Process.create_time
    monkeypatch.setattr(psutil.Process, "create_time", lambda self: real(self) + 3600.0)
    assert ct.owner_gone(labels) is False
    backend = DockerBackend(SandboxSettings())
    inspect = f"live /mullet-sbx-live {json.dumps(labels)}"
    backend._cli = cli = _Cli({"ps": (0, "live", ""), "inspect": (0, inspect, "")})
    assert backend.reap_orphans() == 0 and not any(call[0] == "rm" for call in cli.calls)


def test_the_owner_is_named_by_its_start_in_clock_ticks():
    labels = ct.owner_labels()
    assert labels["mullet.owner"] == str(os.getpid())
    assert labels["mullet.owner.start"] == str(ct.children.start_ticks(os.getpid()))
    assert "." not in labels["mullet.owner.start"]  # ticks, not a wall-clock time


def test_reap_skips_a_container_not_named_by_mullet():
    # `docker commit` of a sandbox container copies its labels into the image,
    # so a container started from that image carries a dead owner's labels.
    dead = _labels(_dead_pid(), 1)
    containers = {"sbx": ("/mullet-sbx-123-abcd", dead), "podman": ("mullet-sbx-9-ffff", dead),
                  "committed": ("/my-dev-box", dead), "lookalike": ("/x-mullet-sbx-1", dead),
                  "unnamed": ("", dead)}
    inspect = "\n".join(f"{cid} {name} {json.dumps(labels)}" for cid, (name, labels) in containers.items())
    backend = DockerBackend(SandboxSettings())
    backend._cli = cli = _Cli({"ps": (0, "\n".join(containers), ""), "inspect": (0, inspect, "")})
    assert backend.reap_orphans() == 2 and cli.calls[-1] == ["rm", "--force", "sbx", "podman"]
    assert cli.calls[1][:3] == ["inspect", "--format", "{{.Id}} {{.Name}} {{json .Config.Labels}}"]


def test_the_owner_labels_carry_the_uid():
    assert ct.owner_labels()["mullet.owner.uid"] == str(UID)


def test_reap_leaves_other_users_containers_alone():
    # With /proc mounted hidepid=2, another user's live Mullet is invisible, so
    # its pid looks gone; only the uid label keeps its container from being reaped.
    hidden = _dead_pid()
    containers = {
        "mine": {**_labels(hidden, 1), "mullet.owner.uid": str(UID)},
        "theirs": {**_labels(hidden, 1), "mullet.owner.uid": str(UID + 1)},
        "no-uid": {key: value for key, value in _labels(hidden, 1).items() if key != "mullet.owner.uid"},
        "odd-uid": {**_labels(hidden, 1), "mullet.owner.uid": f" {UID}"},
    }
    assert ct.owner_gone(containers["mine"])
    assert not any(ct.owner_gone(containers[key]) for key in ("theirs", "no-uid", "odd-uid"))
    inspect = "\n".join(f"{cid} /mullet-sbx-{cid} {json.dumps(labels)}" for cid, labels in containers.items())
    backend = DockerBackend(SandboxSettings())
    backend._cli = cli = _Cli({"ps": (0, "\n".join(containers), ""), "inspect": (0, inspect, "")})
    assert backend.reap_orphans() == 1 and cli.calls[-1] == ["rm", "--force", "mine"]


@pytest.fixture
def no_machine_id(monkeypatch):
    real = ct._read_id
    monkeypatch.setattr(ct, "_read_id", lambda *paths: "" if "/etc/machine-id" in paths else real(*paths))
    monkeypatch.setattr(ct.socket, "gethostname", lambda: "DESKTOP-SHARED")
    ct.host_id.cache_clear()
    yield
    ct.host_id.cache_clear()


def test_without_a_machine_id_the_host_is_private_to_this_install(no_machine_id, tmp_path, monkeypatch):
    # Two WSL2 distros on one Docker Desktop daemon: same hostname, same uid,
    # often the same home path, and no machine-id. They must not reap each other.
    ids = []
    for distro in ("ubuntu", "debian", "ubuntu"):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / distro / "state"))
        ct.host_id.cache_clear()
        ids.append(ct.host_id())
    assert ids[0] == ids[2] and ids[0] != ids[1]  # stable for one install, different for another
    prefix = f"DESKTOP-SHARED/{UID}/"
    assert all(i.startswith(prefix) and len(i) >= len(prefix) + 16 for i in ids)
    # The private part is random, not derived from the path: the same paths elsewhere get another one.
    (tmp_path / "ubuntu" / "state" / "mullet" / "sandbox-host-id").unlink()
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "ubuntu" / "state"))
    ct.host_id.cache_clear()
    assert ct.host_id() != ids[0]


def test_without_a_machine_id_or_a_writable_state_dir_the_host_still_names_this_install(
        no_machine_id, tmp_path, monkeypatch):
    for name in ("one", "two"):
        (tmp_path / name).write_text("")  # a file, so nothing can be made under it
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "one"))
    first = ct.host_id()
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "two"))
    ct.host_id.cache_clear()
    assert first.startswith(f"DESKTOP-SHARED/{UID}/") and ct.host_id() != first


def test_reap_does_nothing_when_the_engine_fails_or_nothing_is_left():
    backend = DockerBackend(SandboxSettings())
    backend._cli = cli = _Cli({"ps": (1, "", "Cannot connect to the Docker daemon")})
    assert backend.reap_orphans() == 0 and len(cli.calls) == 1
    backend._cli = cli = _Cli({"ps": (0, "", "")})
    assert backend.reap_orphans() == 0 and len(cli.calls) == 1


@pytest.mark.parametrize("configured, installed, expected", [
    ("bubblewrap", ("docker", "podman"), []), ("docker", ("docker", "podman"), ["docker"]),
    ("podman", (), []), ("auto", ("podman",), ["podman"]), ("auto", ("docker", "podman"), ["docker", "podman"])])
def test_reap_orphans_asks_the_engines_the_settings_name(monkeypatch, configured, installed, expected):
    monkeypatch.setattr(ct, "load_settings", lambda: SandboxSettings(backend=configured))
    monkeypatch.setattr(ct.shutil, "which", lambda name: f"/usr/bin/{name}" if name in installed else None)
    asked = []
    monkeypatch.setattr(ct.ContainerBackend, "reap_orphans", lambda self, deadline=None: asked.append(self.name) or 1)
    assert ct.reap_orphans() == len(expected) and asked == expected
    asked.clear()
    podman = ["podman"] if "podman" in installed else []
    assert ct.reap_orphans(["podman"]) == len(podman) and asked == podman


def test_reap_orphans_never_raises(monkeypatch):
    def broken():
        raise RuntimeError("config exploded")

    monkeypatch.setattr(ct, "load_settings", broken)
    assert ct.reap_orphans() == 0


def test_cleanup_paths_reap_orphaned_containers(monkeypatch, tmp_path):
    from mullet.cli import doctor_checks as dc
    from mullet.cli import job_cli
    from mullet.jobs import children

    calls = []
    monkeypatch.setattr(ct, "reap_orphans", lambda engines=None: calls.append(engines) or 0)
    children.kill_stale(tmp_path / "children.json")
    assert calls == [None]
    monkeypatch.setattr(job_cli, "stop_runner", lambda job_id: None)
    import argparse

    job_cli.cmd_stop(argparse.Namespace(job_id="j1"), None)
    assert calls == [None, None]
    monkeypatch.setattr(dc.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(ct, "reap_orphans", lambda engines=None: calls.append(engines) or 2)
    assert dc.orphan_container_rows() == [("Sandbox containers", "Removed 2 left behind by a Mullet process "
                                                                  "that died", "info")]
    assert calls[-1] == ["docker", "podman"]
    monkeypatch.setattr(ct, "reap_orphans", lambda engines=None: 0)
    assert dc.orphan_container_rows() == []


@pytest.fixture
def slow_engines(tmp_path, monkeypatch):
    """docker and podman that hang, as a wedged daemon makes them; each call is logged."""
    bin_dir, log = tmp_path / "bin", tmp_path / "calls.log"
    bin_dir.mkdir()
    for name in ct.ENGINES:
        script = bin_dir / name
        script.write_text(f"#!/bin/sh\necho {name} \"$1\" >> {log}\nexec sleep 30\n")
        script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setattr(ct, "load_settings", lambda: SandboxSettings(backend="auto"))
    return log


def test_reap_gives_each_engine_call_a_short_timeout_and_the_whole_reap_a_deadline():
    assert ct.REAP_CLI_TIMEOUT == 5 and ct.REAP_DEADLINE == 15
    backend = DockerBackend(SandboxSettings())
    timeouts = []

    def slow_cli(args, timeout=20):
        timeouts.append(timeout)
        return subprocess.CompletedProcess(args, 0, {"ps": "a b", "inspect": ""}.get(args[0], ""), "")

    backend._cli = slow_cli
    backend.reap_orphans()
    assert timeouts and all(0 < t <= ct.REAP_CLI_TIMEOUT for t in timeouts)
    timeouts.clear()
    assert backend.reap_orphans(deadline=time.monotonic() - 1) == 0 and timeouts == []


@pytest.mark.timeout(60)
def test_a_wedged_daemon_holds_the_reap_up_only_until_its_deadline(slow_engines, monkeypatch):
    monkeypatch.setattr(ct, "REAP_CLI_TIMEOUT", 1.0)
    monkeypatch.setattr(ct, "REAP_DEADLINE", 1.5)
    started = time.monotonic()
    assert ct.reap_orphans() == 0
    assert time.monotonic() - started < 4
    assert slow_engines.read_text().split("\n")[0] == "docker ps"


@pytest.mark.timeout(60)
def test_job_stop_asks_the_runner_to_stop_before_reaping(slow_engines, monkeypatch, capsys):
    import argparse

    from mullet.cli import job_cli

    monkeypatch.setattr(ct, "REAP_CLI_TIMEOUT", 1.0)
    monkeypatch.setattr(ct, "REAP_DEADLINE", 1.5)
    events = []
    monkeypatch.setattr(job_cli, "stop_runner", lambda job_id: events.append(("stop", slow_engines.exists())))
    started = time.monotonic()
    assert job_cli.cmd_stop(argparse.Namespace(job_id="j1"), None) == 0
    assert events == [("stop", False)]  # no engine had been asked anything yet
    assert slow_engines.exists() and time.monotonic() - started < 4
    assert "Asked the runner to stop" in capsys.readouterr().out


def test_job_stop_reaps_even_when_no_runner_is_working(monkeypatch):
    import argparse

    from mullet.cli import job_cli
    from mullet.errors import NotRunnable

    reaped = []

    def no_runner(job_id):
        raise NotRunnable("no runner is working on this job")

    monkeypatch.setattr(job_cli, "stop_runner", no_runner)
    monkeypatch.setattr(ct, "reap_orphans", lambda engines=None: reaped.append(engines) or 0)
    with pytest.raises(NotRunnable):
        job_cli.cmd_stop(argparse.Namespace(job_id="j1"), None)
    assert reaped == [None]


def test_check_refuses_off_linux(monkeypatch):
    monkeypatch.setattr(ct.sys, "platform", "darwin")
    status = DockerBackend(SandboxSettings()).check()
    assert not status.ok and status.backend == "docker" and "Linux" in status.reason


def test_check_reports_a_missing_binary(monkeypatch):
    monkeypatch.setattr(ct.sys, "platform", "linux")
    monkeypatch.setattr(ct.shutil, "which", lambda name: None)
    assert DockerBackend(SandboxSettings()).check() == SandboxCheck(False, "docker", "docker is not installed")
    assert PodmanBackend(SandboxSettings()).check() == SandboxCheck(False, "podman", "podman is not installed")


def test_check_reports_an_unreachable_daemon(monkeypatch):
    monkeypatch.setattr(ct.sys, "platform", "linux")
    monkeypatch.setattr(ct.shutil, "which", lambda name: f"/usr/bin/{name}")
    backend = DockerBackend(SandboxSettings())
    backend._cli = _Cli({"version": (1, "", "Cannot connect to the Docker daemon")})
    status = backend.check()
    assert not status.ok and "daemon" in status.reason and "Cannot connect" in status.reason


SECURE_INFO = (0, '["name=apparmor,profile=default","name=seccomp,profile=builtin","name=cgroupns"]\n', "")
GOOD_IMAGE = (0, json.dumps({"org.opencontainers.image.title": "mullet-sandbox",
                             "mullet.sandbox.version": __version__}), "")


def _healthy_cli(**answers):
    return _Cli({"version": (0, "29.9.2", ""), "info": SECURE_INFO, "image": GOOD_IMAGE, **answers})


def test_check_reports_a_missing_image_with_the_build_command(monkeypatch):
    monkeypatch.setattr(ct.sys, "platform", "linux")
    monkeypatch.setattr(ct.shutil, "which", lambda name: f"/usr/bin/{name}")
    backend = DockerBackend(SandboxSettings(image="mullet-sandbox:9.9.9"))
    backend._cli = _healthy_cli(image=(1, "", "No such image"))
    status = backend.check()
    assert not status.ok and "mullet-sandbox:9.9.9" in status.reason
    assert "mullet sandbox build-image" in status.reason


def test_check_fails_when_the_probe_container_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(ct.sys, "platform", "linux")
    monkeypatch.setattr(ct.shutil, "which", lambda name: f"/usr/bin/{name}")
    backend = DockerBackend(SandboxSettings())
    backend._cli = _healthy_cli()

    def failed_run(command, spec, *, timeout, log_path):
        log_path.write_text("docker: Error response from daemon: oci runtime error")
        return sandbox.RunResult(125, False, 0.1, log_path, log_path.read_text())

    monkeypatch.setattr(backend, "run", failed_run)
    status = backend.check()
    assert not status.ok and "oci runtime error" in status.reason


@pytest.mark.parametrize("labels", [
    "null", "{}", '{"org.opencontainers.image.title": "debian"}',
    '{"org.opencontainers.image.title": "mullet-sandbox"}',
    '{"org.opencontainers.image.title": "mullet-sandbox", "mullet.sandbox.version": ""}', "not json"])
def test_check_refuses_an_image_without_the_mullet_sandbox_labels(monkeypatch, labels):
    # Any image can be tagged mullet-sandbox:<version>; the labels say it was built from Mullet's Dockerfile.
    monkeypatch.setattr(ct.sys, "platform", "linux")
    monkeypatch.setattr(ct.shutil, "which", lambda name: f"/usr/bin/{name}")
    backend = DockerBackend(SandboxSettings())
    backend._cli = cli = _healthy_cli(image=(0, labels, ""))
    monkeypatch.setattr(backend, "_probe", lambda: pytest.fail("probed an unlabelled image"))
    status = backend.check()
    assert not status.ok and "not a Mullet sandbox image" in status.reason and "build-image" in status.reason
    assert ["image", "inspect", "--format", "{{json .Config.Labels}}", backend.settings.image] in cli.calls


def test_check_accepts_a_labelled_image_from_any_mullet_version(monkeypatch):
    # An image FROM mullet-sandbox inherits the labels, and PID 1 comes from Mullet, not the image.
    monkeypatch.setattr(ct.sys, "platform", "linux")
    monkeypatch.setattr(ct.shutil, "which", lambda name: f"/usr/bin/{name}")
    backend = DockerBackend(SandboxSettings(image="team/tools:3"))
    old = json.dumps({"org.opencontainers.image.title": "mullet-sandbox", "mullet.sandbox.version": "2.3.0"})
    backend._cli = _healthy_cli(image=(0, old, ""))
    monkeypatch.setattr(backend, "_probe", lambda: SandboxCheck(True, "docker"))
    assert backend.check().ok


@pytest.mark.parametrize("options", ['["name=apparmor,profile=default","name=cgroupns"]',
                                     '["name=seccomp,profile=unconfined"]', "[]", "null", "not json"])
def test_check_refuses_a_daemon_without_seccomp(monkeypatch, options):
    monkeypatch.setattr(ct.sys, "platform", "linux")
    monkeypatch.setattr(ct.shutil, "which", lambda name: f"/usr/bin/{name}")
    backend = DockerBackend(SandboxSettings())
    backend._cli = cli = _healthy_cli(info=(0, options, ""))
    monkeypatch.setattr(backend, "_probe", lambda: pytest.fail("probed a daemon without seccomp"))
    status = backend.check()
    assert not status.ok and "seccomp" in status.reason and "SecurityOptions" in status.reason
    assert "unshare" in status.reason  # the reason says what is at stake
    assert ["info", "--format", "{{json .SecurityOptions}}"] in cli.calls


@pytest.mark.parametrize("answer, rootless, refused", [
    ("true true", True, False), ("false true", False, False), ("true false", True, True), ("true", True, True)])
def test_podman_checks_seccomp_and_learns_whether_it_is_rootless(answer, rootless, refused):
    backend = PodmanBackend(SandboxSettings(backend="podman"))
    backend._cli = cli = _Cli({"info": (0, answer + "\n", "")})
    problem = backend._engine_checks()
    assert backend.rootless is rootless and bool(problem) is refused
    assert cli.calls == [["info", "--format", "{{.Host.Security.Rootless}} {{.Host.Security.SECCOMPEnabled}}"]]
    if refused:
        assert "SECCOMPEnabled" in problem and "unshare" in problem


def _probe_with(monkeypatch, backend, content: str):
    def fake_run(command, spec, *, timeout, log_path):
        (spec.cwd / "probe").write_text(content)
        log_path.write_text("")
        return sandbox.RunResult(0, False, 0.1, log_path, "")

    monkeypatch.setattr(backend, "run", fake_run)
    return backend._probe()


GOOD_STATUS = "Seccomp:\t2\nCapEff:\t0000000000000000\nNoNewPrivs:\t1\n"


def test_the_probe_passes_a_confined_container(monkeypatch):
    status = _probe_with(monkeypatch, DockerBackend(SandboxSettings()), f"{UID}\n{GOOD_STATUS}")
    assert status.ok, status.reason


@pytest.mark.parametrize("bad, named", [
    ("Seccomp:\t0", "seccomp"), ("Seccomp:\t1", "seccomp"), ("CapEff:\t00000000a80425fb", "capability"),
    ("CapEff:\t", "capability"), ("NoNewPrivs:\t0", "no-new-privileges")])
def test_the_probe_refuses_a_container_that_is_not_confined(monkeypatch, bad, named):
    lines = [line for line in GOOD_STATUS.splitlines() if not line.startswith(bad.split(":")[0])]
    status = _probe_with(monkeypatch, DockerBackend(SandboxSettings()), "\n".join([str(UID), *lines, bad]))
    assert not status.ok and named in status.reason and "all three" in status.reason, status.reason


def test_the_probe_refuses_when_the_kernel_reports_nothing(monkeypatch):
    status = _probe_with(monkeypatch, DockerBackend(SandboxSettings()), f"{UID}\n")
    assert not status.ok and status.reason.count("not shown") == 3, status.reason


def test_the_probe_refuses_a_container_that_runs_as_another_uid(monkeypatch):
    status = _probe_with(monkeypatch, DockerBackend(SandboxSettings()), f"{UID + 1}\n{GOOD_STATUS}")
    assert not status.ok and "own uid" in status.reason


def test_the_probe_refuses_files_owned_by_another_uid(monkeypatch):
    # The command claims our uid, but the file it made belongs to someone else.
    monkeypatch.setattr(ct.os, "getuid", lambda: UID + 1)
    status = _probe_with(monkeypatch, DockerBackend(SandboxSettings()), f"{UID + 1}\n{GOOD_STATUS}")
    assert not status.ok and "own uid" in status.reason


def test_nested_sandbox_is_refused_with_the_reason():
    backend = DockerBackend(SandboxSettings())
    assert backend.nested_check() is False
    with pytest.raises(SandboxUnavailable, match="seccomp"):
        backend.nested_argv([], [])


def test_opencode_names_the_backend_that_cannot_nest(monkeypatch, tmp_path):
    from mullet.jobs.executors import opencode as oc

    backend = DockerBackend(SandboxSettings())
    monkeypatch.setattr(sandbox, "selection", lambda: sandbox.Selection("docker", "test", backend))
    monkeypatch.setattr(sandbox, "check", lambda: SandboxCheck(True, "docker"))
    oc.nested_sandbox_ok.cache_clear()
    try:
        with pytest.raises(oc.ExecutorUnavailable, match="docker"):
            oc.OpencodeExecutor.check_runnable(oc.OpencodeExecutor.__new__(oc.OpencodeExecutor), tmp_path)
    finally:
        oc.nested_sandbox_ok.cache_clear()


# Selection


def test_auto_falls_back_to_docker_after_bubblewrap(monkeypatch):
    monkeypatch.setattr(BubblewrapBackend, "check", lambda self: SandboxCheck(False, "bubblewrap", "no userns"))
    monkeypatch.setattr(DockerBackend, "check", lambda self: SandboxCheck(True, "docker"))
    chosen = sandbox.select(SandboxSettings(backend="auto", from_config=True), platform="linux")
    assert chosen.name == "docker" and isinstance(chosen.backend, DockerBackend)
    assert sandbox.AUTO_ORDER["linux"] == ("bubblewrap", "docker", "podman")


def test_the_container_backends_get_the_users_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yaml").write_text("sandbox:\n  backend: podman\n  image: mine:1\n  pids_limit: 100\n")
    chosen = sandbox.selection()
    assert chosen.name == "podman" and isinstance(chosen.backend, PodmanBackend)
    assert chosen.backend.settings.image == "mine:1" and chosen.backend.settings.pids_limit == 100
    assert isinstance(get_backend("docker"), DockerBackend)


# Real containers


@pytest.fixture
def docker():
    return DockerBackend()


@pytest.fixture
def work(tmp_path):
    path = tmp_path / "work"
    path.mkdir()
    return path


def _running(name: str) -> bool:
    out = subprocess.run(["docker", "ps", "-q", "--filter", f"name=^{name}$"], capture_output=True, text=True,
                         env=ct.launcher_env(), timeout=30)
    return bool(out.stdout.strip())


def _wait_for(predicate, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return False


def _container_name(proc: subprocess.Popen) -> str:
    argv = psutil.Process(proc.pid).cmdline()
    return argv[argv.index("--name") + 1]


@needs_docker
def test_real_runs_as_the_host_uid_with_no_capabilities(docker, work, tmp_path):
    script = "id -u; id -g; grep -E '^(CapEff|CapBnd|CapPrm|NoNewPrivs)' /proc/self/status; touch made"
    result = docker.run_shell(script, SandboxSpec(cwd=work, writable=[work]), timeout=60,
                              log_path=tmp_path / "id.log")
    lines = result.log_path.read_text().splitlines()
    assert result.exit_code == 0, lines
    assert lines[:2] == [str(UID), str(GID)]
    for cap in ("CapEff", "CapBnd", "CapPrm"):
        assert f"{cap}:\t0000000000000000" in lines
    assert "NoNewPrivs:\t1" in lines
    assert (work / "made").stat().st_uid == UID


@needs_docker
def test_real_the_commands_stdin_is_dev_null_not_the_lifeline(docker, work, tmp_path):
    # On the lifeline, cat would block until the timeout, and a command could
    # hold the container open past Mullet's death by keeping it from EOF.
    result = docker.run_shell("readlink /proc/self/fd/0; cat; echo stdin-ended",
                              SandboxSpec(cwd=work, writable=[work]), timeout=20, log_path=tmp_path / "s.log")
    assert not result.timed_out and result.exit_code == 0
    assert result.log_path.read_text().split() == ["/dev/null", "stdin-ended"]


@needs_docker
def test_real_limits_come_from_the_settings(work, tmp_path):
    backend = DockerBackend(SandboxSettings(memory="256m", pids_limit=128))
    result = backend.run_shell("cat /sys/fs/cgroup/pids.max /sys/fs/cgroup/memory.max",
                               SandboxSpec(cwd=work, writable=[work]), timeout=60, log_path=tmp_path / "l.log")
    assert result.log_path.read_text().split() == ["128", str(256 * 1024 ** 2)]


@needs_docker
def test_real_writes_outside_the_clone_fail(docker, work, tmp_path):
    tool = tmp_path / "tool"
    tool.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    home_probe = Path.home() / "mullet-docker-probe"
    script = (f"echo in > inside; touch /probe; touch /usr/probe; echo x > {tool}/planted; "
              f"echo x > {outside}/escape; echo x > {home_probe}; echo x > /tmp/scratch && echo tmp-ok; "
              "echo done")
    spec = SandboxSpec(cwd=work, writable=[work], expose_ro=[tool])
    result = docker.run_shell(script, spec, timeout=60, log_path=tmp_path / "w.log")
    log = result.log_path.read_text()
    assert (work / "inside").read_text() == "in\n" and "done" in log and "tmp-ok" in log
    assert log.count("Read-only file system") == 3, log
    assert not (tool / "planted").exists() and not (outside / "escape").exists()
    assert not home_probe.exists() and not Path("/probe").exists()


@needs_docker
def test_real_no_network_at_all(docker, work, tmp_path):
    script = ("python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), 3)\" "
              "&& echo internet-reached || echo internet-blocked; "
              "python3 -c \"import socket; socket.getaddrinfo('example.com', 443)\" "
              "&& echo dns-reached || echo dns-blocked; "
              "ls /sys/class/net")
    result = docker.run_shell(script, SandboxSpec(cwd=work, writable=[work]), timeout=60,
                              log_path=tmp_path / "n.log")
    log = result.log_path.read_text()
    assert "internet-blocked" in log and "dns-blocked" in log
    assert log.split()[-1] == "lo" and log.split()[-2] == "dns-blocked", log  # loopback only


@needs_docker
def test_real_secrets_are_not_in_inspect_ps_or_pid1(docker, work, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET_PARENT)
    spec = SandboxSpec(cwd=work, writable=[work], env={"MULLET_SPEC_TOKEN": SECRET_SPEC})
    script = ("tr '\\0' '\\n' < /proc/1/environ; echo ---; tr '\\0' '\\n' < /proc/self/environ; "
              "echo ===; exec sleep 60")
    proc = docker.spawn(["/bin/sh", "-c", script], spec, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True)
    try:
        lines = []
        for line in proc.stdout:
            if line.strip() == "===":
                break
            lines.append(line)
        pid1, _, own = "".join(lines).partition("---")
        name = _container_name(proc)
        inspect = subprocess.run(["docker", "inspect", name], capture_output=True, text=True,
                                 env=ct.launcher_env(), timeout=30).stdout
        assert name in inspect
        host_side = [" ".join(p.info["cmdline"] or []) for p in psutil.process_iter(["cmdline"])]
        client_env = Path(f"/proc/{proc.pid}/environ").read_bytes().decode(errors="replace")
    finally:
        kill_group(proc)
    # Delivered, to the command only; PID 1 (which read it) is non-dumpable, so its environ can't be read.
    assert "Permission denied" in pid1 and f"MULLET_SPEC_TOKEN={SECRET_SPEC}" in own
    for secret in (SECRET_SPEC, "MULLET_SPEC_TOKEN"):
        assert secret not in pid1 and secret not in inspect and secret not in client_env
        assert not any(secret in cmd for cmd in host_side)
    for secret in (SECRET_PARENT, "OPENROUTER_API_KEY"):
        assert secret not in pid1 and secret not in own and secret not in inspect and secret not in client_env


@needs_docker
@pytest.mark.parametrize("how", ["term", "kill"])
def test_real_stopping_the_launcher_stops_the_container(docker, work, how):
    proc = docker.spawn(["sleep", "300"], SandboxSpec(cwd=work, writable=[work]), stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
    name = _container_name(proc)
    try:
        assert _wait_for(lambda: _running(name))
        if how == "term":
            kill_group(proc)
        else:  # no grace: the client dies at once, and stdin's end takes the container with it
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
        assert _wait_for(lambda: not _running(name)), f"{name} outlived its launcher"
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, env=ct.launcher_env(), timeout=30)


@needs_docker
def test_real_the_container_dies_with_mullet(work, tmp_path):
    # A Mullet process that dies without cleaning up (os._exit) must not leave its container running.
    child = textwrap.dedent(f"""
        import os, subprocess, sys, psutil
        from pathlib import Path
        from mullet.jobs.sandbox_backends.base import SandboxSpec
        from mullet.jobs.sandbox_backends.container import DockerBackend
        work = Path({str(work)!r})
        proc = DockerBackend().spawn(["sleep", "300"], SandboxSpec(cwd=work, writable=[work]),
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        argv = psutil.Process(proc.pid).cmdline()
        print(argv[argv.index("--name") + 1], flush=True)
        sys.stdin.readline()
        os._exit(0)
    """)
    runner = subprocess.Popen([sys.executable, "-c", child], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True, env={**os.environ})
    name = runner.stdout.readline().strip()
    try:
        assert name.startswith("mullet-sbx-") and _wait_for(lambda: _running(name))
        runner.stdin.write("go\n")
        runner.stdin.flush()
        runner.wait(timeout=20)
        assert _wait_for(lambda: not _running(name)), f"{name} outlived the Mullet process"
    finally:
        runner.kill()
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, env=ct.launcher_env(), timeout=30)


ATTACK = ("kill -9 $(ps -o ppid= -C cat) 2>/dev/null; kill -9 1 2>/dev/null; kill -STOP 1 2>/dev/null; "
          "kill -KILL -1 2>/dev/null; echo attacked; exec sleep 300")


@needs_docker
def test_real_the_command_cannot_disarm_the_lifeline_before_mullet_dies(work):
    # The command kills what watches stdin (the old watcher was its subshell's
    # child), tries PID 1 too, then Mullet dies without cleaning up (os._exit).
    child = textwrap.dedent(f"""
        import os, subprocess, sys, psutil
        from pathlib import Path
        from mullet.jobs.sandbox_backends.base import SandboxSpec
        from mullet.jobs.sandbox_backends.container import DockerBackend
        work = Path({str(work)!r})
        proc = DockerBackend().spawn(["/bin/sh", "-c", {ATTACK!r}], SandboxSpec(cwd=work, writable=[work]),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        argv = psutil.Process(proc.pid).cmdline()
        print(argv[argv.index("--name") + 1], flush=True)
        print(proc.stdout.readline().decode().strip(), flush=True)
        sys.stdin.readline()
        os._exit(0)
    """)
    runner = subprocess.Popen([sys.executable, "-c", child], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True, env={**os.environ})
    name = runner.stdout.readline().strip()
    try:
        assert name.startswith("mullet-sbx-")
        assert runner.stdout.readline().strip() == "attacked"
        time.sleep(1)
        assert _running(name), "the attack itself ended the container"
        runner.stdin.write("go\n")
        runner.stdin.flush()
        runner.wait(timeout=20)
        assert _wait_for(lambda: not _running(name), timeout=10), f"{name} outlived the Mullet process"
    finally:
        runner.kill()
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, env=ct.launcher_env(), timeout=30)


@needs_docker
@pytest.mark.timeout(120)
def test_real_a_payload_near_the_cap_arrives_through_a_small_pipe(docker, work, tmp_path, monkeypatch):
    monkeypatch.setattr(ct, "_pipe", _small_pipe)
    env = _near_cap_env()
    assert len(ct.env_payload(SandboxSpec(cwd=work, writable=[work], env=env))) > 50_000
    result = docker.run_shell('printf %s "$BIG" | wc -c', SandboxSpec(cwd=work, writable=[work], env=env),
                              timeout=60, log_path=tmp_path / "big.log")
    assert result.exit_code == 0 and result.log_path.read_text().split() == [str(len(env["BIG"]))]


@needs_docker
def test_real_reap_removes_a_container_whose_owner_is_dead_and_keeps_a_live_ones():
    image = DockerBackend().settings.image
    names = {"dead": f"mullet-sbx-reap-dead-{os.getpid()}", "alive": f"mullet-sbx-reap-alive-{os.getpid()}"}
    owners = {"dead": _labels(_dead_pid(), 1), "alive": _labels(os.getpid(), MY_TICKS)}
    try:
        for key, name in names.items():
            labels = [a for k, v in owners[key].items() for a in ("--label", f"{k}={v}")]
            subprocess.run(["docker", "run", "--detach", "--rm", "--name", name, *labels, "--pull", "never",
                            "--network", "none", "--cap-drop", "ALL", "--user", f"{UID}:{GID}",
                            "--entrypoint", "sleep", image, "300"],
                           check=True, capture_output=True, env=ct.launcher_env(), timeout=60)
        assert _wait_for(lambda: _running(names["dead"]) and _running(names["alive"]))
        assert DockerBackend().reap_orphans() >= 1
        assert _wait_for(lambda: not _running(names["dead"]), timeout=10)
        assert _running(names["alive"])
    finally:
        for name in names.values():
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, env=ct.launcher_env(), timeout=30)


@needs_docker
def test_real_pid1_refuses_an_environment_execve_would_not_take_and_says_why(docker, work, tmp_path):
    # A NUL can travel in the JSON payload, but no environment can hold it.
    spec = SandboxSpec(cwd=work, writable=[work], env={"MULLET_ODD": "a\0b"})
    result = docker.run(["/bin/sh", "-c", "touch ran"], spec, timeout=60, log_path=tmp_path / "nul.log")
    log = result.log_path.read_text()
    assert result.exit_code == 125 and "MULLET_ODD" in log and "NUL" in log, (result.exit_code, log)
    assert not (work / "ran").exists()
    missing = docker.run(["no-such-command-mullet"], SandboxSpec(cwd=work, writable=[work]), timeout=60,
                         log_path=tmp_path / "missing.log")
    assert missing.exit_code == 127 and "no-such-command-mullet" in missing.log_path.read_text()


@needs_docker
def test_real_pid1_can_be_neither_traced_nor_read_nor_signalled_from_inside(docker, work, tmp_path):
    # Same uid, no capabilities: PID 1 being non-dumpable is what refuses ptrace
    # and /proc/1/mem even where kernel.yama.ptrace_scope is 0.
    script = textwrap.dedent("""
        import ctypes, os, signal
        libc = ctypes.CDLL(None, use_errno=True)
        print("ptrace", libc.ptrace(16, 1, 0, 0), os.strerror(ctypes.get_errno()))  # PTRACE_ATTACH
        for path in ("/proc/1/mem", "/proc/1/environ", "/proc/1/fd/0"):
            try:
                open(path, "rb").read(1)
                print("read", path)
            except OSError as exc:
                print("refused", path, exc.strerror)
        for sig in (signal.SIGKILL, signal.SIGSTOP, signal.SIGSEGV):
            os.kill(1, sig)
        os.kill(1, signal.SIGCONT)
        print("state", open("/proc/1/stat").read().split()[2])
    """)
    result = docker.run(["python3", "-c", script], SandboxSpec(cwd=work, writable=[work]), timeout=60,
                        log_path=tmp_path / "p.log")
    log = result.log_path.read_text()
    assert result.exit_code == 0, log
    assert "ptrace -1" in log and "read /proc/" not in log and log.count("refused /proc/1/") == 3, log
    assert "state S" in log, log  # still alive and not stopped


@needs_docker
def test_real_a_timeout_removes_a_container_that_ignores_term(docker, work, tmp_path, monkeypatch):
    name = f"mullet-sbx-timeout-{os.getpid()}"
    monkeypatch.setattr(ct, "_new_name", lambda: name)
    try:
        result = docker.run_shell("trap '' TERM; sleep 120; echo late > late", SandboxSpec(cwd=work, writable=[work]),
                                  timeout=2, log_path=tmp_path / "t.log")
        assert result.timed_out and result.exit_code is None and result.duration_s < 30
        assert _wait_for(lambda: not _running(name)), f"{name} outlived its timeout"
        assert not (work / "late").exists()
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, env=ct.launcher_env(), timeout=30)


@needs_docker
def test_real_namespaces_cannot_be_nested_which_is_why_nested_check_is_false(docker, work, tmp_path):
    result = docker.run_shell("unshare -n true && echo nested-net; unshare -U -n true && echo nested-userns; "
                              "echo end", SandboxSpec(cwd=work, writable=[work]), timeout=60,
                              log_path=tmp_path / "u.log")
    log = result.log_path.read_text()
    assert "end" in log and "nested-net" not in log and "nested-userns" not in log
    assert docker.nested_check() is False


# `mullet sandbox build-image` and `mullet doctor`


def test_the_dockerfile_ships_with_the_package():
    from mullet.cli import sandbox_cli

    text = sandbox_cli.DOCKERFILE.read_text()
    assert sandbox_cli.DOCKERFILE.parent.name == "sandbox_backends"
    assert text.startswith("#") and "FROM " in text and "python3" in text and "git" in text
    instructions = [line.split()[0] for line in text.splitlines() if line and not line.startswith(("#", " "))]
    assert instructions[0] == "FROM" and not {"USER", "ENTRYPOINT", "CMD"} & set(instructions)


def test_the_base_image_is_pinned_by_digest_and_the_image_is_labelled():
    import re

    from mullet.cli import sandbox_cli

    text = sandbox_cli.DOCKERFILE.read_text()
    assert re.search(r"^FROM debian:trixie-slim@sha256:[0-9a-f]{64}$", text, re.MULTILINE), "pin the base image"
    assert 'org.opencontainers.image.title="mullet-sandbox"' in text
    assert re.search(r"^ARG MULLET_VERSION$", text, re.MULTILINE)  # no default: only a Mullet build sets it
    assert 'mullet.sandbox.version="${MULLET_VERSION}"' in text


def test_build_image_runs_the_engine_with_an_empty_context(monkeypatch, tmp_path):
    from mullet.cli import sandbox_cli

    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(sandbox_cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    seen = []

    def fake_run(command, **kwargs):
        context = Path(command[-1])
        seen.append((command, list(context.iterdir())))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sandbox_cli.subprocess, "run", fake_run)
    out = []
    assert sandbox_cli.cmd_sandbox(["build-image"], out=out.append, err=out.append) == 0
    ((command, context_files),) = seen
    assert command[:-1] == ["/usr/bin/docker", "build", "--tag", f"mullet-sandbox:{__version__}",
                            "--build-arg", f"MULLET_VERSION={__version__}", "--file", str(sandbox_cli.DOCKERFILE)]
    assert context_files == [] and not Path(command[-1]).exists()
    (tmp_path / "config.yaml").write_text("sandbox:\n  backend: podman\n  image: team/tools:3\n")
    assert sandbox_cli.cmd_sandbox(["build-image"], out=out.append, err=out.append) == 0
    assert seen[-1][0][0] == "/usr/bin/podman" and seen[-1][0][3] == "team/tools:3"
    assert sandbox_cli.cmd_sandbox(["build-image", "--engine", "docker", "--tag", "x:1"],
                                   out=out.append, err=out.append) == 0
    assert seen[-1][0][0] == "/usr/bin/docker" and seen[-1][0][3] == "x:1"


def test_build_image_reports_a_missing_engine_and_a_failed_build(monkeypatch, tmp_path):
    from mullet.cli import sandbox_cli

    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    errors = []
    monkeypatch.setattr(sandbox_cli.shutil, "which", lambda name: None)
    assert sandbox_cli.cmd_sandbox(["build-image"], out=errors.append, err=errors.append) == 1
    assert "docker is not installed" in errors[-1]
    monkeypatch.setattr(sandbox_cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(sandbox_cli.subprocess, "run", lambda c, **k: subprocess.CompletedProcess(c, 2))
    assert sandbox_cli.cmd_sandbox(["build-image"], out=errors.append, err=errors.append) == 1
    assert "build failed" in errors[-1]
    printed = []
    assert sandbox_cli.cmd_sandbox(["build-image", "--print"], out=printed.append, err=printed.append) == 0
    assert printed[0].startswith("/usr/bin/docker build --tag mullet-sandbox:")


def test_doctor_shows_the_container_engine_and_the_opencode_limit(monkeypatch, tmp_path):
    from mullet.cli import doctor_checks as dc

    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yaml").write_text("sandbox:\n  backend: docker\n")
    monkeypatch.setattr(dc.sys, "platform", "linux")
    monkeypatch.setattr(dc.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(dc, "_version", lambda argv: "Docker version 29.9.2")
    monkeypatch.setattr(ct, "reap_orphans", lambda engines=None: 1)
    monkeypatch.setattr(sandbox, "check", lambda: SandboxCheck(
        False, "docker", "the sandbox image mullet-sandbox:2.2.0 is not there; build it with: "
                         "mullet sandbox build-image"))
    rows = {row[0]: row for row in dc.sandbox_checks()}
    assert rows["Sandbox backend"][1].startswith("docker (set in the config file)")
    assert rows["Docker"] == ("Docker", "Docker version 29.9.2", "ok")
    assert rows["Job sandbox"][2] == "warn" and "mullet sandbox build-image" in rows["Job sandbox"][1]
    assert rows["opencode worker"][2] == "info" and "Bubblewrap" not in rows
    assert rows["Sandbox containers"][1].startswith("Removed 1 ")
    assert "--privileged" not in rows["Job sandbox"][1]
