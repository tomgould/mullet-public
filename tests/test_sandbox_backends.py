"""Tests for mullet.jobs.sandbox_backends and the backend selection in mullet.jobs.sandbox.

The contract tests (`test_contract_*`) are what every backend must pass. They
take the `backend` fixture, so a new backend joins them by adding its name to
CONTRACT_BACKENDS; one that can't run on this machine is skipped, unless
MULLET_REQUIRE_SANDBOX (comma-separated backend names, set in CI) names it.
The docker and podman backends need the image `mullet sandbox build-image` builds.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from mullet.cli import doctor_checks as dc
from mullet.cli.config_cli import cmd_config
from mullet.config.validate import validate_config
from mullet.jobs import sandbox
from mullet.jobs.sandbox_backends import (
    BACKENDS, BubblewrapBackend, DockerBackend, PodmanBackend, SandboxBackend, get_backend,
)
from mullet.jobs.sandbox_backends import settings as sb_settings
from mullet.jobs.sandbox_backends.base import PASSTHROUGH_ENV, SandboxCheck, SandboxSpec, SandboxUnavailable
from mullet.jobs.sandbox_backends.settings import SandboxSettings, load_settings, parse_settings

CONTRACT_BACKENDS = ["bubblewrap", "docker", "podman"]
# Variables a shell sets for itself, which no backend can keep out.
_SHELL_OWN = {"PWD", "OLDPWD", "SHLVL", "_"}


@pytest.fixture(autouse=True)
def fresh_selection():
    sandbox.clear_cache()
    yield
    sandbox.clear_cache()


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "cfg"))
    (tmp_path / "cfg").mkdir()

    def write(text: str) -> Path:
        path = tmp_path / "cfg" / "config.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    return write


@pytest.fixture
def no_processes(monkeypatch):
    """Fail the test if anything starts a process: nothing may run unsandboxed."""
    def refuse(*args, **kwargs):
        raise AssertionError(f"a process was started: {args[0] if args else kwargs}")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)


# The setting


def test_parse_defaults_when_absent():
    settings = parse_settings(None)
    assert settings.backend == "bubblewrap" and not settings.from_config and settings.problems == []


@pytest.mark.parametrize("value, expected", [("bubblewrap", "bubblewrap"), ("auto", "auto"),
                                             (" Auto ", "auto"), ("docker", "docker"),
                                             ("podman", "podman"), ("seatbelt", "seatbelt")])
def test_parse_accepts_known_names(value, expected):
    settings = parse_settings({"backend": value})
    assert settings.backend == expected and settings.from_config and settings.problems == []


@pytest.mark.parametrize("value", ["none", "off", "unsandboxed", "", 3, True, ["bubblewrap"]])
def test_parse_ignores_bad_values_and_keeps_the_default(value):
    settings = parse_settings({"backend": value})
    assert settings.backend == "bubblewrap" and not settings.from_config
    assert len(settings.problems) == 1 and "sandbox.backend" in settings.problems[0]


def test_parse_reports_a_non_mapping_and_unknown_keys():
    assert parse_settings("docker").problems == ["sandbox: expected a mapping; ignored"]
    settings = parse_settings({"backend": "auto", "extra": 1})
    assert settings.backend == "auto"
    assert settings.problems == ["sandbox.extra: unknown setting; ignored"]


def test_parse_the_container_settings():
    settings = parse_settings({"image": " registry.local/team/tools:1.2 ", "memory": "512M", "pids_limit": 64})
    assert (settings.image, settings.memory, settings.pids_limit) == ("registry.local/team/tools:1.2", "512m", 64)
    assert settings.problems == [] and not settings.from_config
    defaults = parse_settings({})
    assert (defaults.image, defaults.memory, defaults.pids_limit) == (sb_settings.DEFAULT_IMAGE, "8g", 4096)
    assert sb_settings.DEFAULT_IMAGE.startswith("mullet-sandbox:")


@pytest.mark.parametrize("key, value", [("image", ""), ("image", "a b"), ("image", "--privileged"),
                                        ("image", 3), ("memory", "8"), ("memory", "1m"), ("memory", "8t"),
                                        ("memory", 8), ("pids_limit", 10), ("pids_limit", True),
                                        ("pids_limit", "100"), ("pids_limit", 2_000_000)])
def test_parse_reports_bad_container_settings_and_keeps_defaults(key, value):
    settings = parse_settings({key: value})
    assert getattr(settings, key) == getattr(SandboxSettings(), key)
    assert len(settings.problems) == 1 and settings.problems[0].startswith(f"sandbox.{key}:")


def test_load_settings_reads_the_config_file(tmp_path):
    path = tmp_path / "config.yaml"
    assert load_settings(path) == SandboxSettings()
    path.write_text("sandbox:\n  backend: auto\n")
    assert load_settings(path).backend == "auto"
    path.write_text("sandbox: [unclosed\n")
    broken = load_settings(path)
    assert broken.backend == "bubblewrap" and "unreadable" in broken.problems[0]


def test_every_available_name_has_a_backend():
    assert set(BACKENDS) == set(sb_settings.AVAILABLE)
    assert not set(BACKENDS) & set(sb_settings.RESERVED)


def test_backends_satisfy_the_protocol():
    for name in BACKENDS:
        assert isinstance(get_backend(name), SandboxBackend)


# Selection


def test_default_is_bubblewrap(config):
    chosen = sandbox.selection()
    assert chosen.name == "bubblewrap" and chosen.reason == "the default"
    assert isinstance(chosen.backend, BubblewrapBackend)
    assert isinstance(sandbox.backend(), BubblewrapBackend)


def test_explicit_bubblewrap(config):
    config("sandbox:\n  backend: bubblewrap\n")
    chosen = sandbox.selection()
    assert chosen.name == "bubblewrap" and chosen.reason == "set in the config file"


def test_auto_picks_bubblewrap_when_it_works(monkeypatch):
    monkeypatch.setattr(BubblewrapBackend, "check", lambda self: SandboxCheck(True, "bwrap"))
    chosen = sandbox.select(SandboxSettings(backend="auto", from_config=True), platform="linux")
    assert chosen.name == "bubblewrap" and "auto" in chosen.reason
    assert chosen.probe == SandboxCheck(True, "bwrap")


def test_auto_refuses_when_no_backend_works(monkeypatch, config, no_processes):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(BubblewrapBackend, "check", lambda self: SandboxCheck(False, "bwrap", "no userns"))
    monkeypatch.setattr(DockerBackend, "check", lambda self: SandboxCheck(False, "docker", "no daemon"))
    monkeypatch.setattr(PodmanBackend, "check", lambda self: SandboxCheck(False, "podman", "no podman"))
    chosen = sandbox.select(SandboxSettings(backend="auto", from_config=True), platform="linux")
    assert chosen.backend is None and "no userns" in chosen.error
    assert "docker: no daemon" in chosen.error and "podman: no podman" in chosen.error
    config("sandbox:\n  backend: auto\n")
    assert not sandbox.check().ok and "no backend works here" in sandbox.check().reason
    with pytest.raises(SandboxUnavailable, match="no userns"):
        sandbox.run_shell("true", SandboxSpec(cwd=Path("/"), writable=[]), timeout=5,
                          log_path=Path(os.devnull))


def test_auto_on_a_platform_without_backends():
    chosen = sandbox.select(SandboxSettings(backend="auto", from_config=True), platform="darwin")
    assert chosen.backend is None and "no sandbox backend for darwin" in chosen.error


def test_unknown_backend_name_raises():
    with pytest.raises(SandboxUnavailable, match="unknown sandbox backend 'nope'"):
        get_backend("nope")


@pytest.mark.parametrize("name", ["seatbelt"])
def test_reserved_backends_are_unavailable(name):
    with pytest.raises(SandboxUnavailable, match=f"sandbox.backend is {name}.*does not have yet"):
        get_backend(name)


def test_unavailable_backend_refuses_every_entry_point(config, tmp_path, no_processes):
    config("sandbox:\n  backend: seatbelt\n")
    spec = SandboxSpec(cwd=tmp_path, writable=[tmp_path])
    status = sandbox.check()
    assert not status.ok and status.backend == "seatbelt" and "does not have yet" in status.reason
    calls = [
        lambda: sandbox.backend(),
        lambda: sandbox.run(["true"], spec, timeout=5, log_path=tmp_path / "a.log"),
        lambda: sandbox.run_shell("true", spec, timeout=5, log_path=tmp_path / "b.log"),
        lambda: sandbox.spawn(["true"], spec, stdout=None, stderr=None),
        lambda: sandbox.nested_argv([], []),
    ]
    for call in calls:
        with pytest.raises(SandboxUnavailable, match="seatbelt"):
            call()
    assert sandbox.nested_check() is False
    assert not (tmp_path / "a.log").exists() and not (tmp_path / "b.log").exists()


def test_a_failed_probe_refuses_and_starts_nothing(monkeypatch, tmp_path, no_processes):
    monkeypatch.setattr(sandbox, "check", lambda: SandboxCheck(False, "bwrap", "probe failed"))
    spec = SandboxSpec(cwd=tmp_path, writable=[tmp_path])
    for call in (lambda: sandbox.spawn(["true"], spec, stdout=None, stderr=None),
                 lambda: sandbox.run_shell("true", spec, timeout=5, log_path=tmp_path / "c.log")):
        with pytest.raises(SandboxUnavailable, match="probe failed"):
            call()
    assert sandbox.nested_check() is False


def test_a_bad_value_keeps_bubblewrap_and_reports_it(config):
    config("sandbox:\n  backend: none\n")
    chosen = sandbox.selection()
    assert chosen.name == "bubblewrap" and chosen.reason == "the default"
    assert any("sandbox.backend" in p for p in chosen.problems)


class _Recorder:
    """A backend that records what the facade hands it."""

    name = "recorder"

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def check(self):
        return SandboxCheck(True, "recorder")

    def build_argv(self, command, spec):
        self.calls.append(("build_argv", (command,)))
        return ["confined", *command]

    def spawn(self, command, spec, *, stdout, stderr, cwd=None, text=False):
        self.calls.append(("spawn", (command, cwd, text)))
        return "proc"

    def run(self, command, spec, *, timeout, log_path):
        self.calls.append(("run", (command, timeout)))
        return "ran"

    def run_shell(self, script, spec, *, timeout, log_path):
        self.calls.append(("run_shell", (script, timeout)))
        return "ran-shell"

    def nested_argv(self, readonly, hidden):
        self.calls.append(("nested_argv", (readonly, hidden)))
        return ["inner", "--"]

    def nested_check(self):
        return True


def test_the_facade_routes_every_call_to_the_selected_backend(monkeypatch, tmp_path):
    recorder = _Recorder()
    assert isinstance(recorder, SandboxBackend)
    monkeypatch.setattr(sandbox, "selection", lambda: sandbox.Selection("recorder", "test", recorder))
    spec = SandboxSpec(cwd=tmp_path, writable=[tmp_path])
    assert sandbox.spawn(["y"], spec, stdout=None, stderr=None, cwd=tmp_path, text=True) == "proc"
    assert sandbox.run(["z"], spec, timeout=3, log_path=tmp_path / "l") == "ran"
    assert sandbox.run_shell("echo", spec, timeout=4, log_path=tmp_path / "l") == "ran-shell"
    assert sandbox.nested_argv([tmp_path], []) == ["inner", "--"]
    assert sandbox.nested_check() is True
    assert [name for name, _ in recorder.calls] == ["spawn", "run", "run_shell", "nested_argv"]


def test_opencode_launches_and_nests_through_the_facade(monkeypatch, tmp_path):
    from mullet.jobs.executors import opencode as oc

    recorder = _Recorder()
    monkeypatch.setattr(sandbox, "selection", lambda: sandbox.Selection("recorder", "test", recorder))
    script = oc.shell_wrapper({}, [tmp_path], [])
    assert "inner --" in script and "bwrap" not in script


# Config, validation and doctor


def test_config_show_has_a_sandbox_section(config, capsys):
    import json

    config("sandbox:\n  backend: auto\n  extra: 1\n  pids_limit: 512\n")
    assert cmd_config(["show", "--json", "sandbox"]) == 0
    data = json.loads(capsys.readouterr().out)
    rows = {row["setting"]: row for row in data["sections"]["sandbox"]}
    assert list(rows) == ["sandbox.backend", "sandbox.image", "sandbox.memory", "sandbox.pids_limit"]
    row = rows["sandbox.backend"]
    assert row["value"] == "auto" and row["default"] == "bubblewrap"
    assert row["source"].endswith("config.yaml")
    assert rows["sandbox.pids_limit"]["value"] == 512 and rows["sandbox.pids_limit"]["default"] == 4096
    assert rows["sandbox.pids_limit"]["source"].endswith("config.yaml")
    assert rows["sandbox.image"]["value"] == sb_settings.DEFAULT_IMAGE and rows["sandbox.image"]["source"] == "default"
    assert rows["sandbox.memory"]["value"] == "8g"
    assert cmd_config(["show", "--json"]) == 0
    assert "sandbox.extra" in json.loads(capsys.readouterr().out)["unrecognised"]


def test_config_show_sandbox_default(config, capsys):
    assert cmd_config(["show", "sandbox"]) == 0
    out = capsys.readouterr().out
    assert "[sandbox]" in out and "sandbox.backend" in out and "default" in out


def test_validate_checks_the_sandbox_block():
    assert validate_config({"sandbox": {"backend": "auto"}}).valid
    bad = validate_config({"sandbox": {"backend": "off"}})
    assert not bad.valid and bad.issues[0].path == "sandbox.backend"
    warned = validate_config({"sandbox": {"backend": "bubblewrap", "x": 1}})
    assert warned.valid and warned.issues[0].severity == "warning"
    assert not any(i.path == "sandbox" for i in validate_config({"sandbox": {}}).issues)
    good = validate_config({"sandbox": {"backend": "docker", "image": "mine:1", "memory": "4g", "pids_limit": 256}})
    assert good.valid and not good.issues
    for key, value in (("image", "bad image"), ("memory", "lots"), ("pids_limit", 5)):
        bad = validate_config({"sandbox": {key: value}})
        assert not bad.valid and bad.issues[0].path == f"sandbox.{key}"


def test_doctor_shows_the_backend_and_why(config, monkeypatch):
    monkeypatch.setattr(sandbox, "check", lambda: SandboxCheck(True, "bwrap"))
    assert dc.sandbox_backend_row() == ("Sandbox backend", "bubblewrap (the default)", "ok")
    monkeypatch.setattr(dc.sys, "platform", "linux")
    assert dc.sandbox_checks()[0][0] == "Sandbox backend"


def test_doctor_warns_for_an_unavailable_backend(config):
    config("sandbox:\n  backend: seatbelt\n")
    name, detail, status = dc.sandbox_backend_row()
    assert status == "warn" and detail.startswith("None: sandbox.backend is seatbelt")


def test_doctor_warns_for_a_bad_setting(config, monkeypatch):
    monkeypatch.setattr(sandbox, "check", lambda: SandboxCheck(True, "bwrap"))
    config("sandbox:\n  backend: nope\n")
    name, detail, status = dc.sandbox_backend_row()
    assert status == "warn" and "bubblewrap (the default)" in detail and "Config:" in detail


# The contract every backend must satisfy


_PROBED: dict[str, SandboxCheck] = {}


@pytest.fixture(params=CONTRACT_BACKENDS)
def backend(request) -> SandboxBackend:
    # Defaults, not the throwaway config: the container backends need the image that
    # `mullet sandbox build-image` builds. A probe starts a container, so it runs once per backend.
    candidate = get_backend(request.param, SandboxSettings()) if request.param != "bubblewrap" \
        else get_backend(request.param)
    if request.param not in _PROBED:
        _PROBED[request.param] = candidate.check()
    status = _PROBED[request.param]
    if not status.ok:
        if request.param in os.environ.get("MULLET_REQUIRE_SANDBOX", "").split(","):
            pytest.fail(f"MULLET_REQUIRE_SANDBOX names {request.param}, which is unavailable: {status.reason}")
        pytest.skip(f"{request.param}: {status.reason}")
    return candidate


@pytest.fixture
def work(tmp_path):
    path = tmp_path / "work"
    path.mkdir()
    return path


def _shell(backend: SandboxBackend, script: str, spec: SandboxSpec, tmp_path: Path, timeout: float = 30):
    return backend.run_shell(script, spec, timeout=timeout, log_path=tmp_path / "logs" / "run.log")


def test_contract_writes_land_only_in_writable_paths(backend, work, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    home_probe = Path.home() / f"mullet-contract-{os.getpid()}"
    for spec in (SandboxSpec(cwd=work, writable=[work]),
                 SandboxSpec(cwd=work, writable=[work], expose_ro=[outside])):
        result = _shell(backend, f"echo ok > inside.txt; echo x > {outside}/escape.txt; echo x > {home_probe}",
                        spec, tmp_path)
        assert result.exit_code is not None and not result.timed_out
        assert (work / "inside.txt").read_text() == "ok\n"
        assert not (outside / "escape.txt").exists()
        assert not home_probe.exists()


def test_contract_root_and_exposed_paths_are_read_only(backend, work, tmp_path):
    tool = tmp_path / "tool"
    tool.mkdir()
    (tool / "bin.sh").write_text("echo tool-ran\n")
    spec = SandboxSpec(cwd=work, writable=[work], expose_ro=[tool])
    result = _shell(backend, f"sh {tool}/bin.sh; touch /mullet-contract-probe && echo root-writable; "
                             f"touch {tool}/planted && echo tool-writable; echo done", spec, tmp_path)
    assert "tool-ran" in result.tail and "done" in result.tail
    assert "root-writable" not in result.tail and "tool-writable" not in result.tail
    assert not (tool / "planted").exists()


def test_contract_no_network(backend, work, tmp_path):
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        script = (f"python3 -c \"import socket; socket.create_connection(('127.0.0.1', {port}), 2)\" "
                  "&& echo loopback-reached || echo loopback-blocked; "
                  "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), 2)\" "
                  "&& echo internet-reached || echo internet-blocked")
        result = _shell(backend, script, SandboxSpec(cwd=work, writable=[work]), tmp_path)
    finally:
        server.close()
    assert "loopback-blocked" in result.tail and "internet-blocked" in result.tail


def test_contract_environment_is_an_allowlist(backend, work, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-contract-must-not-leak")
    monkeypatch.setenv("MULLET_CONTRACT_SECRET", "contract-secret-value")
    spec = SandboxSpec(cwd=work, writable=[work], env={"MULLET_JUNIT": "/x/junit.xml"})
    result = _shell(backend, "env", spec, tmp_path)
    assert "sk-contract-must-not-leak" not in result.tail and "contract-secret-value" not in result.tail
    names = {line.split("=", 1)[0] for line in result.log_path.read_text().splitlines() if "=" in line}
    allowed = {*PASSTHROUGH_ENV, "HOME", "TMPDIR", "TERM", "MULLET_JUNIT", *_SHELL_OWN}
    assert names <= allowed, names - allowed
    assert "MULLET_JUNIT=/x/junit.xml" in result.tail and "TERM=dumb" in result.tail


def test_contract_timeout_kills_the_whole_tree(backend, work, tmp_path):
    started = time.monotonic()
    result = _shell(backend, "(sleep 3; touch late) & sleep 60", SandboxSpec(cwd=work, writable=[work]),
                    tmp_path, timeout=1)
    assert result.timed_out and result.exit_code is None
    assert time.monotonic() - started < 20
    time.sleep(4)
    assert not (work / "late").exists()


def test_contract_the_key_proxy_socket_is_reachable(backend, work, tmp_path):
    # A short path, as the opencode executor uses: Unix socket paths are limited to about 108 bytes.
    proxy_dir = Path(tempfile.mkdtemp(prefix="mullet-px-"))
    path = proxy_dir / "s"
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(path))
    server.listen(1)

    def answer():
        try:
            conn, _ = server.accept()
            conn.sendall(b"proxy-answered")
            conn.close()
        except OSError:
            pass

    thread = threading.Thread(target=answer, daemon=True)
    thread.start()
    try:
        spec = SandboxSpec(cwd=work, writable=[work], expose_ro=[proxy_dir])
        script = (f"python3 -c \"import socket; s = socket.socket(socket.AF_UNIX); s.connect('{path}'); "
                  "print(s.recv(64).decode())\"")
        result = _shell(backend, script, spec, tmp_path)
    finally:
        server.close()
        thread.join(timeout=5)
        path.unlink(missing_ok=True)
        proxy_dir.rmdir()
    assert "proxy-answered" in result.tail


def test_contract_spawn_confines_a_long_lived_worker(backend, work, tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_CONTRACT_SECRET", "contract-secret-value")
    outside = tmp_path / "outside"
    outside.mkdir()
    spec = SandboxSpec(cwd=work, writable=[work], expose_ro=[outside], env={"WORKER": "yes"})
    script = f"echo $WORKER; pwd; echo x > {outside}/escape || echo refused; env"
    proc = backend.spawn(["/bin/sh", "-c", script], spec, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    out, _ = proc.communicate(timeout=30)
    assert out.splitlines()[:2] == ["yes", str(work)] and "refused" in out
    assert not (outside / "escape").exists() and "contract-secret-value" not in out


# Per backend: the launcher's name, a flag that must always be there (a cleared
# environment), and the argv slice that means "no network".
_ARGV_MARKERS = {
    "bubblewrap": ("bwrap", "--clearenv", ["--unshare-net"]),
    "docker": ("docker", "--read-only", ["--network", "none"]),
    "podman": ("podman", "--read-only", ["--network", "none"]),
}


def _has(argv: list[str], part: list[str]) -> bool:
    return any(argv[i:i + len(part)] == part for i in range(len(argv)))


def test_contract_build_argv_is_never_the_bare_command(backend, work):
    launcher, always, no_net = _ARGV_MARKERS[backend.name]
    off = backend.build_argv(["true"], SandboxSpec(cwd=work, writable=[work]))
    assert off[-1] == "true" and len(off) > 1 and off[0] != "true"
    assert Path(off[0]).name == launcher and always in off and _has(off, no_net)
    on = backend.build_argv(["true"], SandboxSpec(cwd=work, writable=[work], network=True))
    assert Path(on[0]).name == launcher and always in on and not _has(on, no_net)


def test_contract_the_host_filesystem_is_read_only(backend, work, tmp_path):
    spec = SandboxSpec(cwd=work, writable=[work])
    result = _shell(backend, "touch /etc/mullet-contract-probe; touch /usr/mullet-contract-probe; "
                             "echo x > /etc/hostname", spec, tmp_path)
    assert result.tail.count("Read-only file system") == 3, result.tail
    assert not Path("/etc/mullet-contract-probe").exists() and not Path("/usr/mullet-contract-probe").exists()


def test_contract_process_environments_hold_no_parent_secret(backend, work, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-contract-launcher-leak")
    monkeypatch.setenv("MULLET_CONTRACT_SECRET", "contract-launcher-secret")
    spec = SandboxSpec(cwd=work, writable=[work])
    script = "tr '\\0' '\\n' < /proc/1/environ; echo ---; tr '\\0' '\\n' < /proc/self/environ"
    result = _shell(backend, script, spec, tmp_path)
    log = result.log_path.read_text()
    first, _, second = log.partition("---")
    # The container backends' PID 1 is non-dumpable, so its environ can't be read at all.
    if backend.name in ("docker", "podman"):
        assert "Permission denied" in first and "PATH=" in second, log
    else:
        assert "PATH=" in first and "PATH=" in second, log
    for secret in ("sk-contract-launcher-leak", "contract-launcher-secret", "OPENROUTER_API_KEY",
                   "MULLET_CONTRACT_SECRET"):
        assert secret not in log


def test_contract_a_refused_spec_leaves_no_log(backend, work, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(work)
    log = tmp_path / "logs" / "refused.log"
    with pytest.raises(ValueError, match="symlink"):
        backend.run(["true"], SandboxSpec(cwd=work, writable=[link]), timeout=5, log_path=log)
    assert not log.exists() and not log.parent.exists()


def test_contract_the_stop_check_runs_once_per_facade_call(backend, work, tmp_path, monkeypatch):
    from mullet.jobs import children

    calls = []
    monkeypatch.setattr(children, "check_not_stopping", lambda: calls.append(1))
    monkeypatch.setattr(sandbox, "selection", lambda: sandbox.Selection(backend.name, "test", backend))
    monkeypatch.setattr(sandbox, "check", lambda: SandboxCheck(True, backend.name))
    sandbox.run(["true"], SandboxSpec(cwd=work, writable=[work]), timeout=20, log_path=tmp_path / "a.log")
    sandbox.run_shell("true", SandboxSpec(cwd=work, writable=[work]), timeout=20, log_path=tmp_path / "b.log")
    assert len(calls) == 2


# The launcher's environment, the argv, and the cache


class _FakeProc:
    pid = 1
    returncode = 0

    def wait(self, timeout=None):
        return 0


@pytest.mark.parametrize("entry", ["spawn", "run"])
def test_launcher_starts_with_a_path_only_environment(entry, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-launcher-env-secret")
    monkeypatch.setenv("PATH", "/opt/fake/bin:/usr/bin")
    seen = []

    def fake_popen(argv, **kwargs):
        seen.append(kwargs)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    work = tmp_path / "work"
    work.mkdir()
    spec = SandboxSpec(cwd=work, writable=[work])
    bwrap = BubblewrapBackend()
    if entry == "spawn":
        bwrap.spawn(["true"], spec, stdout=None, stderr=None)
    else:
        bwrap.run(["true"], spec, timeout=5, log_path=tmp_path / "x.log")
    (kwargs,) = seen
    assert kwargs["env"] == {"PATH": "/opt/fake/bin:/usr/bin"}


def test_bubblewrap_argv_is_exactly_the_documented_line(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("LANG", "C.UTF-8")
    for name in ("LC_ALL", "LC_CTYPE", "TZ"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    work, deps, tools, hidden = (tmp_path / n for n in ("work", "deps", "tools", "hidden"))
    for path in (work, deps, tools, hidden, tmp_path / "home"):
        path.mkdir()
    head = ["/usr/bin/bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-ipc",
            "--unshare-uts", "--unshare-cgroup-try"]
    env = ["--clearenv", "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "LANG", "C.UTF-8",
           "--setenv", "K", "v", "--setenv", "HOME", str(tmp_path / "home"), "--setenv", "TMPDIR", "/tmp",
           "--setenv", "TERM", "dumb", "--chdir", str(work), "--", "echo", "hi"]
    spec = SandboxSpec(cwd=work, writable=[work], readonly_overlays=[deps], expose_ro=[tools],
                       hide=[hidden, tmp_path / "absent"], env={"K": "v"})
    mounts = ["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", str(hidden),
              "--ro-bind", str(tools.resolve()), str(tools), "--bind", str(work), str(work),
              "--ro-bind", str(deps), str(deps)]
    backend_ = BubblewrapBackend()
    assert backend_.build_argv(["echo", "hi"], spec) == [*head, "--unshare-net", *mounts, *env]
    spec.network = True
    resolver = Path("/run/systemd/resolve")
    dns = ["--ro-bind", str(resolver), str(resolver)] if resolver.is_dir() else []
    split = mounts.index("--ro-bind", mounts.index("--tmpfs"))
    assert backend_.build_argv(["echo", "hi"], spec) == [*head, *mounts[:split], *dns, *mounts[split:], *env]


def test_the_facade_no_longer_exposes_build_argv():
    assert not hasattr(sandbox, "build_argv") and "build_argv" not in sandbox.__all__


def test_the_executor_checks_the_sandbox_before_building_the_shell_wrapper(monkeypatch, tmp_path):
    from mullet.jobs.executors import opencode as oc

    source = Path(oc.__file__).read_text(encoding="utf-8")
    attempt = source[source.index("    def _attempt("):]
    assert attempt.index("self.check_runnable(work)") < attempt.index("shell_wrapper(")
    runnable = source[source.index("    def check_runnable("):source.index("    def _attempt(")]
    assert "sandbox.check()" in runnable and "raise ExecutorUnavailable" in runnable
    monkeypatch.setattr(sandbox, "selection", lambda: sandbox.Selection("none", "t", None, error="unavailable"))
    with pytest.raises(SandboxUnavailable):
        oc.shell_wrapper({}, [tmp_path], [])


def test_bubblewrap_reports_its_own_name(monkeypatch):
    backend_ = BubblewrapBackend()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert backend_.check() == SandboxCheck(False, "bubblewrap", "bubblewrap (bwrap) is not installed")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert backend_.check().backend == "bubblewrap"


def test_selection_and_probe_are_one_cached_value(monkeypatch, config):
    probes = []

    def probe(self):
        probes.append(1)
        return SandboxCheck(True, self.name)

    monkeypatch.setattr(BubblewrapBackend, "check", probe)
    chosen, status = sandbox.selection(), sandbox.check()
    assert sandbox.selection() is chosen and sandbox.check() is status and len(probes) == 1
    config("sandbox:\n  backend: seatbelt\n")
    sandbox.clear_cache()
    assert sandbox.selection() is not chosen and sandbox.selection().name == "seatbelt"
    assert sandbox.check().backend == "seatbelt" and not sandbox.check().ok


def test_concurrent_first_calls_agree(monkeypatch):
    probes = []

    def slow_probe(self):
        probes.append(1)
        time.sleep(0.2)
        return SandboxCheck(True, self.name)

    monkeypatch.setattr(BubblewrapBackend, "check", slow_probe)
    results: list[tuple] = []
    barrier = threading.Barrier(8)

    def first_call():
        barrier.wait()
        results.append((sandbox.selection(), sandbox.check()))

    threads = [threading.Thread(target=first_call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert len(results) == 8 and len(probes) == 1
    assert all(r[0] is results[0][0] and r[1] is results[0][1] for r in results)
