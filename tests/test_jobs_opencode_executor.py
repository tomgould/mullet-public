"""Tests for the opencode executor, using a fake `opencode` script inside the real sandbox.

The fake records how it was called from inside the sandbox and replays a
scripted `--format json` stream, so these tests check the command line, the
generated config and brief, the environment, the sandbox and the result
mapping without calling any model.
"""

import base64
import json
import re
import socket
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mullet.jobs import sandbox
from mullet.jobs.executors import opencode as oc
from mullet.jobs.executors.base import Budget, ExecutorUnavailable, NodeContext, Task
from mullet.jobs.executors.opencode import AGENT, OpencodeExecutor, bash_permission, build_config
from mullet.jobs.types import NodeSpec

from tests.jobs_helpers import FAKE_OPENCODE

needs_sandbox = pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)

FILES = {
    "app.py": "from helpers import double\n\ndef add(a, b):\n    return a + b\n",
    "helpers.py": "def double(x):\n    return x * 2\n",
    "tests/test_app.py": "from app import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    ".env": "SECRET=x\n",
    "uv.lock": "lock\n",
    "CLAUDE.md": "Use tabs.\n",
}


@pytest.fixture(autouse=True)
def fresh_model_cache():
    oc._models_cache.clear()
    yield
    oc._models_cache.clear()


KEY = "fake-provider-key-123"


class _Upstream(BaseHTTPRequestHandler):
    """The provider's API, on loopback: records what reached it."""

    seen: list[dict] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).seen.append({"path": self.path, "headers": dict(self.headers), "body": body.decode()})
        payload = b'data: {"choices": []}\n\ndata: [DONE]\n\n'
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST


@pytest.fixture
def upstream():
    _Upstream.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def fake(tmp_path, monkeypatch, upstream):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "opencode"
    script.write_text(FAKE_OPENCODE)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    config = tmp_path / "xdg-config"
    (config / "opencode").mkdir(parents=True)
    (config / "opencode" / "opencode.json").write_text(json.dumps({
        "provider": {
            "fakeai": {"npm": "@ai-sdk/openai-compatible", "api": f"{upstream}/v1", "env": ["FAKEAI_API_KEY"],
                       "models": {"m1": {"limit": {"context": 40000, "output": 4000},
                                         "headers": {"X-Secret": "model-level"}}}},
            "otherco": {"options": {"apiKey": "sk-other-literal-key-0000"}},
        },
        "mcp": {"mullet": {"type": "local", "command": ["mullet", "mcp"]}},
    }))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("FAKEAI_API_KEY", KEY)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-must-not-pass")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-must-not-pass")

    def scenario(**kwargs):
        (bin_dir / "scenario.json").write_text(json.dumps({"models": ["fakeai/m1", "fakeai/m2"], **kwargs}))

    scenario()
    return {"binary": str(script), "scenario": scenario, "bin": bin_dir, "tmp": tmp_path,
            "upstream": upstream, "config": config / "opencode" / "opencode.json"}


@pytest.fixture
def work(tmp_path):
    root = tmp_path / "jobroot"
    w = root / "work"
    w.mkdir(parents=True)
    (root / "git").mkdir()
    for rel, content in FILES.items():
        (w / rel).parent.mkdir(parents=True, exist_ok=True)
        (w / rel).write_text(content)
    return w


def make(fake, model="fakeai/m1", **kwargs) -> OpencodeExecutor:
    tmp = fake["tmp"]
    return OpencodeExecutor(tmp / "logs", model, home=tmp / "oc-home", binary=fake["binary"],
                            bash_commands=["python3 runtests.py"], **kwargs)


def calls(fake) -> list[dict]:
    log = fake["tmp"] / "oc-home" / "data" / "fake-opencode" / "calls.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def task(**kwargs) -> Task:
    ctx = NodeContext(parent_goal="make maths better", plan_notes="keep it small", files=sorted(FILES),
                      checks={"test_full": "python3 runtests.py"}, exclude=[".env", ".env.*", "uv.lock"])
    spec = NodeSpec(id="n1", goal="add a sub function next to add", scope=["app.py", "tests/**"],
                    acceptance=["python3 runtests.py"])
    return Task("Add subtraction", spec, context=ctx, label="job1:n1", **kwargs)


# Pure pieces


def _wildcard(pattern: str, text: str) -> bool:
    """opencode-style matching: `*` is any run of characters, newlines included."""
    return re.fullmatch(".*".join(re.escape(p) for p in pattern.split("*")), text, re.S) is not None


def _decide(rules: dict[str, str], command: str) -> str:
    return [action for pattern, action in rules.items() if _wildcard(pattern, command)][-1]


def test_bash_rules_deny_by_default_and_refuse_chaining():
    rules = bash_permission(["python3 runtests.py"])
    keys = list(rules)
    assert keys[0] == "*" and rules["*"] == "deny"
    assert rules["python3 runtests.py"] == rules["python3 runtests.py *"] == "allow"
    assert rules["git status *"] == "allow"
    for pattern in ("*;*", "*&*", "*|*", "*$(*", "*`*", "*\n*", "*\r*", "find *-exec*", "rg *--pre*"):
        assert rules[pattern] == "deny"
        assert keys.index(pattern) > keys.index("python3 runtests.py *")  # last match wins


@pytest.mark.parametrize("command", [
    "ls & sh x.sh", "ls && sh x.sh", "ls; sh x.sh", "ls\nsh x.sh", "ls\r\nsh x.sh", "ls | sh",
    "cat `sh x.sh`", "cat $(sh x.sh)", "ls > x", "python3 runtests.py & curl x", "ln -s /proc/1/environ x",
])
def test_bash_rules_refuse_every_chaining_shape(command):
    assert _decide(bash_permission(["python3 runtests.py"]), command) == "deny"


@pytest.mark.parametrize("command", ["ls", "ls -la src", "python3 runtests.py tests/test_app.py", "git diff"])
def test_bash_rules_still_allow_plain_commands(command):
    assert _decide(bash_permission(["python3 runtests.py"]), command) == "allow"


def test_config_turns_everything_off_that_isnt_listed():
    config = build_config(instructions=["/b.md"], bash_commands=[], steps=7, provider={"x": {"options": {}}})
    agent = config["agent"][AGENT]
    perm = agent["permission"]
    assert list(perm)[0] == "*" and perm["*"] == "deny"
    for tool in ("task", "webfetch", "websearch", "external_directory", "skill", "question", "lsp"):
        assert perm[tool] == "deny"
    assert perm["edit"] == "allow" and perm["read"]["*.env"] == "deny"
    assert agent["steps"] == 7 and "prompt" not in agent
    assert config["lsp"] is False and config["formatter"] is False and config["share"] == "disabled"
    assert config["provider"] == {"x": {"options": {}}} and "mcp" not in config
    assert "formatter" not in build_config(instructions=[], bash_commands=[], steps=1, formatter=True)
    with_lsp = build_config(instructions=[], bash_commands=[], steps=1, lsp=True)
    assert with_lsp["lsp"] is True and with_lsp["agent"][AGENT]["permission"]["lsp"] == "allow"


def test_only_the_providers_own_key_names_are_accepted(tmp_path):
    config = tmp_path / "cfg"
    config.mkdir()
    (config / "opencode.json").write_text(json.dumps({"provider": {
        "openrouter": {"env": ["OPENROUTER_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN"]},
        "my-llm": {"env": ["MY_LLM_API_KEY", "MY_LLM_SECRET"]},
    }}))
    catalogue = tmp_path / "models.json"
    catalogue.write_text(json.dumps({"openrouter": {"env": ["ANTHROPIC_API_KEY", "OPENROUTER_EXTRA_API_KEY"]}}))
    assert oc.key_names("openrouter", config, catalogue) == ["OPENROUTER_API_KEY", "OPENROUTER_EXTRA_API_KEY"]
    assert oc.key_names("my-llm", config, None) == ["MY_LLM_API_KEY"]
    assert oc.key_names("moonshotai", config, None) == ["MOONSHOT_API_KEY"]  # the fixed map


def test_a_worker_written_catalogue_is_never_read(fake, work):
    executor = make(fake)
    planted = executor.home / "cache" / "opencode" / "models.json"
    planted.parent.mkdir(parents=True)
    planted.write_text(json.dumps({"fakeai": {"env": ["OPENROUTER_API_KEY"],
                                              "models": {"m1": {"limit": {"context": 999}}}}}))
    assert "OPENROUTER_API_KEY" not in executor.keys()
    assert executor.context_length() == 40000
    executor._fresh_home()
    assert not planted.exists()  # the cache is rebuilt before every attempt


def test_fresh_home_keeps_sessions_but_drops_credentials_links_binaries_and_cache(fake, tmp_path):
    executor = make(fake)
    data = executor.home / "data" / "opencode"
    (data / "bin").mkdir(parents=True)
    (data / "bin" / "rg").write_text("#!/bin/sh\n")
    (data / "storage").mkdir()
    (data / "storage" / "session.json").write_text("{}")
    (data / "storage" / "helper").write_text("#!/bin/sh\n")
    (data / "storage" / "helper").chmod(0o755)
    (data / "opencode.db").write_text("sessions")
    (data / "auth.json").write_text("{}")
    (data / "mcp-auth.json").write_text("{}")
    (data / "link").symlink_to(tmp_path)
    (executor.home / "cache" / "opencode" / "packages").mkdir(parents=True)
    (executor.home / "state" / "opencode").mkdir(parents=True)
    (executor.home / "config" / "opencode").mkdir(parents=True)
    (executor.home / "config" / "opencode" / "opencode.json").write_text("{}")
    executor._fresh_home()
    assert (data / "opencode.db").read_text() == "sessions"
    assert (data / "storage" / "session.json").exists()
    assert not (data / "auth.json").exists() and not (data / "mcp-auth.json").exists()
    assert not (data / "link").is_symlink()
    assert not (data / "bin").exists() and not (data / "storage" / "helper").exists()
    assert not (executor.home / "cache" / "opencode" / "packages").exists()
    assert not (executor.home / "state" / "opencode").exists()
    assert not (executor.home / "config" / "opencode").exists()


def test_routes_for_the_providers_in_use(tmp_path):
    config = tmp_path / "cfg"
    config.mkdir()
    (config / "opencode.json").write_text(json.dumps({"provider": {
        "moonshot": {"api": "https://api.moonshot.ai/v1", "env": ["MOONSHOT_API_KEY"], "name": "Moonshot",
                     "models": {"kimi-k2.6": {"limit": {"context": 262144}}}},
    }}))
    catalogue = tmp_path / "models.json"
    catalogue.write_text(json.dumps({
        "openrouter": {"npm": "@openrouter/ai-sdk-provider", "api": "https://openrouter.ai/api/v1"},
        "moonshotai": {"npm": "@ai-sdk/openai-compatible", "api": "https://api.moonshot.ai/v1"},
        "anthropic": {"npm": "@ai-sdk/anthropic"},
    }))
    openrouter = oc.provider_route("openrouter", config, catalogue)
    assert (openrouter.upstream, openrouter.header) == ("https://openrouter.ai/api/v1", "bearer")
    entry = openrouter.sandbox_entry(with_key=True)
    assert entry["options"] == {"baseURL": f"http://127.0.0.1:{oc.PROXY_PORT}/api/v1", "apiKey": oc.DUMMY_KEY}
    kimi = oc.provider_route("moonshotai", config, catalogue)
    assert kimi.upstream == "https://api.moonshot.ai/v1"
    custom = oc.provider_route("moonshot", config, catalogue).sandbox_entry(with_key=True)
    assert custom["api"] == custom["options"]["baseURL"] == f"http://127.0.0.1:{oc.PROXY_PORT}/v1"
    assert "env" not in custom and custom["models"]["kimi-k2.6"]["limit"]["context"] == 262144
    assert oc.provider_route("anthropic", config, catalogue).header == "x-api-key"
    with pytest.raises(ExecutorUnavailable, match="can't tell where"):
        oc.provider_route("nowhere", config, catalogue)


def test_the_sandbox_config_never_carries_credentials(fake):
    executor = make(fake)
    entry = executor.route().sandbox_entry(with_key=True)
    text = json.dumps(entry)
    assert KEY not in text and "model-level" not in text and "FAKEAI_API_KEY" not in text
    assert entry["options"]["apiKey"] == oc.DUMMY_KEY


def test_a_key_from_the_users_own_config_is_used_on_the_host_only(fake, monkeypatch):
    monkeypatch.delenv("FAKEAI_API_KEY")
    monkeypatch.setattr("mullet.core.env.dotenv_value", lambda key: None)
    data = json.loads(fake["config"].read_text())
    data["provider"]["fakeai"]["options"] = {"apiKey": "sk-literal-in-config-123456"}
    fake["config"].write_text(json.dumps(data))
    executor = make(fake)
    assert executor.key() == "sk-literal-in-config-123456"
    assert "sk-literal" not in json.dumps(executor.route().sandbox_entry(with_key=True))
    data["provider"]["fakeai"]["options"] = {"apiKey": "{env:ANTHROPIC_API_KEY}"}  # not this provider's
    fake["config"].write_text(json.dumps(data))
    assert make(fake).key() is None


def test_secret_forms_cover_json_and_base64(tmp_path):
    forms = oc.secret_forms(KEY)
    assert KEY in forms
    assert '\\"' in "".join(oc.secret_forms('a"quoted-secret'))
    for offset in range(3):
        encoded = base64.b64encode(b"x" * offset + KEY.encode() + b"tail").decode()
        assert any(f in encoded for f in forms if f != KEY), offset
    path = tmp_path / "big.txt"
    path.write_bytes(b"a" * (6 * 1024 * 1024) + KEY.encode())  # past the first chunks
    assert oc._contains(path, [f.encode() for f in forms])


def test_opencode_models_runs_with_a_home_no_worker_has_seen(fake):
    executor = make(fake)
    planted = executor.home / "cache" / "opencode" / "packages" / "evil.js"
    planted.parent.mkdir(parents=True)
    planted.write_text("planted")
    executor.validate_model()
    env = json.loads((fake["bin"] / "models-calls.txt").read_text().splitlines()[-1])["env_values"]
    for var in ("XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
        assert not env[var].startswith(str(executor.home))
    assert planted.exists()  # and the job's home was left alone


def test_a_model_id_needs_a_provider():
    with pytest.raises(ExecutorUnavailable):
        OpencodeExecutor(Path("/tmp/x"), "kimi-k2.7-code")


def test_unknown_model_is_refused_with_the_nearest_ids(fake):
    fake["scenario"](models=["openrouter/moonshotai/kimi-k2.7-code", "openrouter/moonshotai/kimi-k2.6",
                             "opencode/big-pickle"])
    with pytest.raises(ExecutorUnavailable) as exc:
        make(fake, model="openrouter/moonshot/kimi-k2.7-code").validate_model()
    assert "openrouter/moonshotai/kimi-k2.7-code" in str(exc.value)


def test_a_provider_without_a_key_says_which_variable_to_set(fake, monkeypatch):
    monkeypatch.delenv("FAKEAI_API_KEY")
    monkeypatch.setattr("mullet.core.env.dotenv_value", lambda key: None)
    fake["scenario"](models=["opencode/big-pickle"])
    with pytest.raises(ExecutorUnavailable) as exc:
        make(fake).validate_model()
    assert "No API key was found for provider 'fakeai'" in str(exc.value) and "FAKEAI_API_KEY" in str(exc.value)


def test_the_model_list_is_fetched_once_per_process(fake):
    executor = make(fake)
    executor.validate_model()
    executor.validate_model()
    make(fake).validate_model()
    assert len((fake["bin"] / "models-calls.txt").read_text().splitlines()) == 1


def test_context_window_comes_from_opencodes_own_config(fake):
    assert make(fake).context_length() == 40000


def test_missing_cli_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(ExecutorUnavailable):
        OpencodeExecutor(tmp_path / "logs", "fakeai/m1").validate_model()


# Runs in the sandbox


@needs_sandbox
def test_command_line_config_and_brief(fake, work):
    result = make(fake).run(task(), work, Budget(max_turns=9))
    call = calls(fake)[-1]
    argv = call["argv"]
    assert argv[0] == "run" and argv[1].startswith("Do the step described in your Mullet task brief")
    assert argv[argv.index("--dir") + 1] == str(work)
    assert argv[argv.index("--agent") + 1] == AGENT
    assert argv[argv.index("--model") + 1] == "fakeai/m1"
    assert argv[argv.index("--format") + 1] == "json"
    assert argv[argv.index("--title") + 1] == "mullet job1:n1"
    assert "--pure" in argv and "--auto" not in argv and "--session" not in argv
    assert call["cwd"] == str(work)
    config = call["config"]
    assert config["agent"][AGENT]["steps"] == 9
    assert "mcp" not in config  # the user's own config, MCP servers and all, isn't loaded
    assert config["provider"]["fakeai"]["options"]["baseURL"] == f"http://127.0.0.1:{oc.PROXY_PORT}/v1"
    # The repo has CLAUDE.md and no AGENTS.md: CLAUDE.md is listed, since
    # opencode itself is told not to read Claude files.
    assert config["instructions"][1] == str(work / "CLAUDE.md")
    brief = call["brief"]
    for text in ("Add subtraction", "add a sub function next to add", "make maths better", "keep it small",
                 "python3 runtests.py", "## Outline of files in scope", "def add(a, b)", "## Repository map",
                 "one command per call"):
        assert text in brief
    assert result.error is None and not result.aborted


@needs_sandbox
def test_scope_tests_and_imports_are_attached_but_never_secrets(fake, work):
    make(fake).run(task(), work, Budget())
    attached = [str(Path(f).relative_to(work)) for f in calls(fake)[-1]["files"]]
    assert attached[:2] == ["app.py", "tests/test_app.py"]
    assert "helpers.py" in attached  # imported by app.py
    assert ".env" not in attached and "uv.lock" not in attached
    listing = (fake["tmp"] / "logs" / "opencode-001" / "files.txt").read_text()
    assert "helpers.py\t" in listing and "imported by app.py" in listing


@needs_sandbox
def test_environment_is_minimal_and_carries_no_key_at_all(fake, work):
    make(fake).run(task(), work, Budget())
    env = calls(fake)[-1]["env"]
    assert KEY not in json.dumps(env)
    assert not {"FAKEAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"} & set(env)
    for switch in ("OPENCODE_DISABLE_PROJECT_CONFIG", "OPENCODE_DISABLE_CLAUDE_CODE",
                   "OPENCODE_DISABLE_EXTERNAL_SKILLS", "OPENCODE_DISABLE_AUTOUPDATE", "OPENCODE_DISABLE_MODELS_FETCH"):
        assert env[switch] == "1"
    home = fake["tmp"] / "oc-home"
    assert env["XDG_DATA_HOME"] == str(home / "data")
    assert env["XDG_CACHE_HOME"] == str(home / "cache")
    assert env["XDG_CONFIG_HOME"] == str(home / "config")  # not the user's own config directory
    assert env["OPENCODE_CONFIG"].endswith("opencode-001/opencode.json")


@needs_sandbox
def test_the_provider_is_reached_only_through_the_proxy_with_the_real_key(fake, work):
    fake["scenario"](call_provider=True, direct_url=f"{fake['upstream']}/v1/models")
    result = make(fake).run(task(), work, Budget())
    call = calls(fake)[-1]
    assert call["provider_reply"].startswith("data: ")  # the round trip worked, stream and all
    assert call["direct"].startswith("refused")  # and nothing else is reachable
    assert [s["path"] for s in _Upstream.seen] == ["/v1/chat/completions"]
    assert _Upstream.seen[0]["headers"]["Authorization"] == f"Bearer {KEY}"  # added outside the sandbox
    assert call["config"]["provider"]["fakeai"]["options"]["apiKey"] == oc.DUMMY_KEY
    assert result.usage["provider_requests"] == 1 and result.usage["provider_host"] == "127.0.0.1"
    log = (fake["tmp"] / "logs" / "opencode-001" / "proxy.log").read_text()
    assert "POST /v1/chat/completions 200" in log and KEY not in log


@needs_sandbox
def test_nothing_in_the_sandbox_holds_the_key(fake, work):
    """Not the config, the brief, the environment, the attempt's files or the logs."""
    fake["scenario"](call_provider=True)
    make(fake).run(task(), work, Budget())
    attempt = fake["tmp"] / "logs" / "opencode-001"
    for path in attempt.iterdir():
        assert KEY.encode() not in path.read_bytes(), path.name
    assert KEY not in json.dumps(calls(fake)[-1])


@needs_sandbox
def test_the_sandbox_keeps_opencode_in_the_clone(fake, work, monkeypatch, tmp_path):
    home = tmp_path / "fake-home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text("PRIVATE")
    monkeypatch.setenv("HOME", str(home))
    git_dir = work.parent / "git"  # readable for git, never writable
    fake["scenario"](read=str(home / ".ssh" / "id_rsa"), escape=str(git_dir / "planted"),
                     write={"sub.py": "def sub(a, b):\n    return a - b\n"})
    make(fake).run(task(), work, Budget())
    call = calls(fake)[-1]
    assert call["read"].startswith("refused")
    assert call["escape"].startswith("refused") and not (git_dir / "planted").exists()
    assert (work / "sub.py").exists()


@needs_sandbox
def test_the_users_opencode_config_is_not_visible(fake, work):
    """It can hold a literal key for another provider."""
    fake["scenario"](read=str(fake["config"]))
    make(fake).run(task(), work, Budget())
    assert calls(fake)[-1]["read"].startswith("refused")


PROBE = '''\
import os, socket, sys
needle = bytes.fromhex(sys.argv[1])
hits = []
for pid in os.listdir("/proc"):
    if pid.isdigit():
        for name in ("environ", "cmdline"):
            try:
                with open(f"/proc/{pid}/{name}", "rb") as fh:
                    if needle in fh.read():
                        hits.append(f"{pid}/{name}")
            except OSError:
                pass
s = socket.socket()
s.settimeout(3)
net = "open" if s.connect_ex(("127.0.0.1", int(sys.argv[2]))) == 0 else "closed"
try:
    with open(os.path.join(sys.argv[3], "planted"), "w") as fh:
        fh.write("x")
    wrote = "yes"
except OSError:
    wrote = "no"
in_env = any(needle.decode() in v for v in os.environ.values())
proxy_seen = [p for p in os.listdir("/tmp") if p.startswith("mullet-px-") and os.listdir("/tmp/" + p)]
loop = socket.socket()
loop.settimeout(3)
proxy_port = "open" if loop.connect_ex(("127.0.0.1", int(sys.argv[4]))) == 0 else "closed"
print(f"proc-hits={len(hits)} net={net} key-in-env={in_env} home-write={wrote} "
      f"proxy-socket={'visible' if proxy_seen else 'hidden'} proxy-port={proxy_port} "
      f"FOO={os.environ.get('FOO')} LD_PRELOAD={os.environ.get('LD_PRELOAD')} "
      f"HTTPS_PROXY={os.environ.get('HTTPS_PROXY')} TMPDIR={os.environ.get('TMPDIR')}")
'''


@needs_sandbox
def test_the_workers_shell_has_no_network_no_key_and_no_way_to_the_proxy(fake, work):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    threading.Thread(target=lambda: listener.accept(), daemon=True).start()
    (work / "probe.py").write_text(PROBE)
    executor = make(fake, shell_env={"FOO": "bar", "LD_PRELOAD": "/tmp/evil.so", "HTTPS_PROXY": "http://evil:1",
                                     "TMPDIR": "/somewhere"})
    key_hex = KEY.encode().hex()
    fake["scenario"](shell=f"python3 probe.py {key_hex} {port} {executor.home / 'data'} {oc.PROXY_PORT}")
    executor.run(task(), work, Budget())
    listener.close()
    out = calls(fake)[-1]["shell"]
    assert "proc-hits=0" in out and "key-in-env=False" in out
    assert "net=closed" in out
    # Neither the proxy's socket nor opencode's loopback port to it is reachable from the shell.
    assert "proxy-socket=hidden" in out and "proxy-port=closed" in out
    assert "home-write=no" in out and not (executor.home / "data" / "planted").exists()
    # The profile's env reaches the shell, minus what changes loading or traffic, and Mullet's own win.
    assert "FOO=bar" in out and "LD_PRELOAD=None" in out and "HTTPS_PROXY=None" in out and "TMPDIR=/tmp" in out


@needs_sandbox
@pytest.mark.parametrize("command", ["ls & echo pwned", "ls\necho pwned", "ls; echo pwned", "echo $(id)"])
def test_the_shell_wrapper_refuses_chained_commands_itself(fake, work, command):
    fake["scenario"](shell=command)
    make(fake).run(task(), work, Budget())
    out = calls(fake)[-1]["shell"]
    assert "Mullet: one command per call" in out and "pwned" not in out.replace("echo pwned", "")


@needs_sandbox
def test_profile_env_never_reaches_opencode_itself(fake, work):
    make(fake, shell_env={"FOO": "bar", "OPENCODE_DISABLE_PROJECT_CONFIG": "0", "NODE_OPTIONS": "--require x"}).run(
        task(), work, Budget())
    env = calls(fake)[-1]["env"]
    assert "FOO" not in env and "NODE_OPTIONS" not in env
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"


def test_without_a_nested_sandbox_opencode_refuses_to_run(fake, work, monkeypatch):
    monkeypatch.setattr(oc, "nested_sandbox_ok", lambda: False)
    if not sandbox.check().ok:
        pytest.skip(sandbox.check().reason)
    with pytest.raises(ExecutorUnavailable, match="nested sandbox"):
        make(fake).run(task(), work, Budget())
    assert calls(fake) == []


def test_the_shell_wrapper_has_no_unsandboxed_fallback():
    script = oc.shell_wrapper({"FOO": "a b'c"}, [Path("/tmp/home")], [Path("/tmp/mullet-px-1")])
    for flag in ("--unshare-net", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--new-session",
                 "--proc /proc", "--tmpfs /tmp/mullet-px-1"):
        assert flag in script
    assert "--ro-bind /tmp/home /tmp/home" in script and "env -i 'FOO=a b'\"'\"'c'" in script
    assert script.index("FOO=") < script.index("PATH=")  # Mullet's own variables come last


def test_without_socat_opencode_refuses_to_run(fake, work, monkeypatch):
    if not sandbox.check().ok or not oc.nested_sandbox_ok():
        pytest.skip("needs the sandbox")
    real_which = oc.shutil.which
    monkeypatch.setattr(oc.shutil, "which", lambda name: None if name == "socat" else real_which(name))
    with pytest.raises(ExecutorUnavailable, match="socat"):
        make(fake).run(task(), work, Budget())


@needs_sandbox
@pytest.mark.parametrize("target", ["/proc/self/environ", "/proc", "/dev/null", "HOME/.ssh/id_rsa", "HOME",
                                    "/etc/hostname", "/etc", "/mnt", "/var/log", "/opt"])
def test_a_symlink_out_of_the_clone_refuses_the_run(fake, work, target, tmp_path):
    target = target.replace("HOME", str(Path.home()))
    (work / "innocent.txt").symlink_to(target)
    with pytest.raises(ExecutorUnavailable, match="symlinks that lead out of it"):
        make(fake).run(task(), work, Budget())
    assert calls(fake) == []


def test_links_inside_the_clone_or_to_system_and_toolchain_dirs_are_fine(work, tmp_path):
    (work / "alias.py").symlink_to("app.py")
    (work / "pybin").symlink_to("/usr/bin/env")
    toolchain = tmp_path / "toolchain"
    toolchain.mkdir()
    (work / "tool").symlink_to(toolchain)
    assert oc.escaping_symlinks(work, [toolchain]) == []
    assert oc.escaping_symlinks(work, []) == [("tool", str(toolchain))]


@needs_sandbox
def test_a_symlink_made_during_the_run_throws_the_attempt_away(fake, work):
    fake["scenario"](symlink={"leak": "/proc/self/environ"})
    result = make(fake).run(task(), work, Budget())
    assert result.aborted and "symlinks that lead out of the clone" in result.error
    assert result.session_id is None  # never resumed


@needs_sandbox
@pytest.mark.parametrize("scenario, where", [
    ({"leak_to": "leak.txt"}, "leak.txt"),
    ({"leak_to": "leak.txt", "leak_encoding": "base64"}, "leak.txt"),
    ({"leak_to": "leak.json", "leak_encoding": "json"}, "leak.json"),
    ({"leak_in_text": True}, "transcript.jsonl"),
])
def test_the_key_turning_up_anywhere_throws_the_attempt_away(fake, work, scenario, where):
    # The key can't get into the sandbox; the fake writes it as if it had.
    fake["scenario"](leak_value=KEY, **scenario)
    result = make(fake).run(task(), work, Budget())
    assert result.aborted and "API key turned up" in result.error and where in result.error
    assert KEY not in Path(result.transcript_path).read_text()
    assert KEY not in json.dumps(result.claim)


@needs_sandbox
def test_language_servers_are_off_unless_the_profile_turns_them_on(fake, work):
    make(fake).run(task(), work, Budget())
    call = calls(fake)[-1]
    assert call["config"]["lsp"] is False and "OPENCODE_EXPERIMENTAL_LSP_TOOL" not in call["env"]
    make(fake, lsp=True).run(task(), work, Budget())
    call = calls(fake)[-1]
    assert call["config"]["lsp"] is True and call["env"]["OPENCODE_EXPERIMENTAL_LSP_TOOL"] == "1"


@needs_sandbox
def test_success_maps_text_usage_cost_and_session(fake, work):
    fake["scenario"](text="DONE: added sub() and a test")
    result = make(fake).run(task(), work, Budget())
    assert result.exit_code == 0 and result.error is None and not result.aborted
    assert result.claim == {"status": "done", "summary": "DONE: added sub() and a test"}
    assert result.session_id == "ses_1"
    usage = result.usage
    assert usage["num_turns"] == 2 and usage["total_cost_usd"] == pytest.approx(0.03)
    assert usage["cost_reported"] is True
    assert usage["tokens"]["input"] == 150 and usage["tokens"]["cache_read"] == 5
    assert usage["tools"] == {"edit": 1} and usage["attached_files"] >= 2
    lines = Path(result.transcript_path).read_text().splitlines()
    assert json.loads(lines[-1])["type"] == "step_finish"


@needs_sandbox
def test_resume_continues_the_session_with_the_feedback(fake, work):
    executor = make(fake)
    first = executor.run(task(), work, Budget())
    feedback = "G5 failed: 1 tests regressed, e.g. test_app::test_add"
    executor.resume(task(feedback=[feedback]), work, Budget(), first.session_id)
    argv = calls(fake)[-1]["argv"]
    assert argv[argv.index("--session") + 1] == first.session_id
    assert "--title" not in argv and "--file" not in argv
    assert feedback in argv[1]
    assert feedback in calls(fake)[-1]["brief"]  # the brief is refreshed too


@needs_sandbox
@pytest.mark.parametrize("scenario, expected", [
    ({"error": "Model not found: fakeai/m1"}, "Model not found"),
    ({"exit": 3, "silent": True}, "opencode exited 3"),
    ({"silent": True}, "ended without a result"),
])
def test_provider_errors_and_crashes_abort_without_using_an_attempt(fake, work, scenario, expected):
    fake["scenario"](**scenario)
    result = make(fake).run(task(), work, Budget())
    assert result.aborted and expected in result.error


@needs_sandbox
def test_an_error_that_hangs_is_killed_after_a_grace_period(fake, work, monkeypatch):
    monkeypatch.setattr(oc, "ERROR_GRACE_S", 1.0)
    fake["scenario"](error="Invalid API key", hang_after_error=True)
    started = time.monotonic()
    result = make(fake).run(task(), work, Budget(wall_clock_s=120))
    assert time.monotonic() - started < 30
    assert result.aborted and "Invalid API key" in result.error


@needs_sandbox
def test_wall_clock_limit_kills_the_run(fake, work):
    fake["scenario"](sleep=60)
    started = time.monotonic()
    result = make(fake).run(task(), work, Budget(wall_clock_s=2))
    assert time.monotonic() - started < 30
    assert result.exit_code is None and result.error == "timed out" and not result.aborted


@needs_sandbox
def test_agents_md_wins_over_claude_md(fake, work):
    (work / "AGENTS.md").write_text("Agents rules.\n")
    make(fake).run(task(), work, Budget())
    assert calls(fake)[-1]["config"]["instructions"][1:] == [str(work / "AGENTS.md")]


def test_refuses_without_the_sandbox(fake, work, monkeypatch):
    monkeypatch.setattr(oc.sandbox, "check", lambda: sandbox.SandboxCheck(False, "bwrap", "no userns"))
    with pytest.raises(ExecutorUnavailable, match="sandbox"):
        make(fake).run(task(), work, Budget())


# The profile's own check commands, even when they chain

LINT = "ruff check . ; mypy src"


def test_the_profiles_exact_check_commands_are_allowed_even_with_a_semicolon():
    rules = bash_permission(["python3 runtests.py", LINT], exact_commands=[LINT, "make lint && make types"])
    assert _decide(rules, LINT) == "allow"
    assert _decide(rules, "make lint && make types") == "allow"
    assert list(rules)[-2:] == [LINT, "make lint && make types"]  # after every deny: last match wins
    for extended in (f"{LINT} ; curl evil", f"{LINT}; rm -rf ~", f"curl evil ; {LINT}", "ruff check . ; sh x.sh",
                     "make lint && make types && sh x.sh"):
        assert _decide(rules, extended) == "deny", extended


def test_exact_commands_with_wildcards_or_newlines_are_never_allowed_as_patterns():
    rules = bash_permission([], exact_commands=["pytest tests/*.py ; ruff .", "a ; b\nc", "a ; b?"])
    assert _decide(rules, "pytest tests/x.py ; rm -rf ~ ; ruff .") == "deny"
    assert "a ; b\nc" not in rules and "a ; b?" not in rules
    assert oc.exact_rules(["  x ; y  ", "x ; y", "p*"]) == ["x ; y"]


def test_the_brief_lists_the_exact_commands():
    rules = oc.build_rules(["python3 runtests.py"], [LINT, "python3 runtests.py"])
    assert f"exactly as written, as one call each:\n    {LINT}" in rules
    assert rules.count("python3 runtests.py") == 1  # a plain command needs no special line


@needs_sandbox
def test_the_shell_wrapper_runs_an_exact_check_command_but_nothing_added_to_it(fake, work):
    fake["scenario"](calls=[{"shell": "echo first; echo second"}, {"shell": "echo first; echo second; echo pwned"}])
    executor = make(fake, exact_commands=["echo first; echo second"])
    executor.run(task(), work, Budget())
    out = calls(fake)[-1]["shell"]
    assert "first" in out and "second" in out and "one command per call" not in out
    executor.run(task(), work, Budget())
    out = calls(fake)[-1]["shell"]
    assert "one command per call" in out and "pwned" not in out
    assert calls(fake)[-1]["config"]["agent"][AGENT]["permission"]["bash"]["echo first; echo second"] == "allow"


# What reaches the sandbox config, and the proxy's pins


def test_only_whitelisted_provider_fields_reach_the_sandbox():
    definition = oc.provider_definition({
        "npm": "@ai-sdk/openai-compatible", "name": "Moonshot", "api": "https://api.moonshot.ai/v1",
        "env": ["MOONSHOT_API_KEY"], "headers": {"X-Org": "secret-org"}, "mystery": {"anything": 1},
        "options": {"apiKey": "sk-x", "baseURL": "https://elsewhere", "timeout": 600000, "fetch": "x"},
        "models": {"kimi-k2.7-code": {"id": "kimi-k2.7-code-preview", "limit": {"context": 262144},
                                      "headers": {"X-Secret": "s"}, "provider": {"npm": "evil"},
                                      "options": {"reasoningEffort": "high", "apiKey": "sk-y", "baseURL": "x"}}},
    })
    assert definition == {
        "npm": "@ai-sdk/openai-compatible", "name": "Moonshot", "api": "https://api.moonshot.ai/v1",
        "options": {"timeout": 600000},
        "models": {"kimi-k2.7-code": {"id": "kimi-k2.7-code-preview", "limit": {"context": 262144},
                                      "options": {"reasoningEffort": "high"}}},
    }
    assert oc.api_model_ids("moonshotai/kimi-k2.7-code", definition) == ["kimi-k2.7-code", "kimi-k2.7-code-preview"]
    assert oc.api_model_ids("openrouter/moonshotai/kimi-k2.7-code", {}) == ["moonshotai/kimi-k2.7-code"]


def test_the_config_pins_the_small_model_to_the_attempts_model():
    config = build_config(instructions=[], bash_commands=[], steps=3, model="moonshotai/kimi-k2.7-code")
    assert config["model"] == config["small_model"] == "moonshotai/kimi-k2.7-code"


def test_moonshot_and_lm_studio_have_default_upstreams(tmp_path):
    assert oc.provider_route("moonshotai", tmp_path, None).upstream == "https://api.moonshot.ai/v1"
    assert oc.provider_route("lmstudio", tmp_path, None, "http://127.0.0.1:4321/v1").upstream == \
        "http://127.0.0.1:4321/v1"


@needs_sandbox
def test_an_error_event_is_a_provider_error(fake, work):
    fake["scenario"](error="429 Too Many Requests")
    result = make(fake).run(task(), work, Budget())
    assert result.aborted and result.provider_error


@needs_sandbox
def test_the_proxy_is_pinned_and_capped_per_attempt(fake, work, monkeypatch):
    seen = {}
    real = oc.KeyProxy

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(oc, "KeyProxy", spy)
    result = make(fake).run(task(), work, Budget(max_turns=10))
    assert seen["models"] == ["m1"] and seen["max_requests"] == 60 and seen["body_fields"] is None
    assert "provider_bytes" in result.usage and not result.provider_error

    executor = make(fake)
    executor.request_fields = {"provider": {"data_collection": "deny"}}
    executor.run(task(), work, Budget(max_turns=10))
    assert seen["body_fields"] == {"provider": {"data_collection": "deny"}}


@needs_sandbox
def test_a_step_priced_at_zero_counts_as_unreported(fake, work):
    fake["scenario"](costs=[0, 0])
    usage = make(fake).run(task(), work, Budget()).usage
    assert usage["cost_reported"] is False and usage["total_cost_usd"] == 0 and usage["tokens"]["input"] == 150


@needs_sandbox
def test_usage_is_passed_on_after_each_step_and_the_run_stops_past_its_spend(fake, work):
    fake["scenario"](pause_after_first_step=20)
    executor = make(fake)
    seen = []
    executor.on_progress = seen.append
    executor.spend_limit_usd = 0.005
    started = time.monotonic()
    result = executor.run(task(), work, Budget(wall_clock_s=120))
    assert time.monotonic() - started < 15  # stopped, not left to finish
    assert result.aborted and result.provider_error and "passed the job's remaining spend" in result.error
    assert seen and seen[0]["total_cost_usd"] == pytest.approx(0.01) and seen[0]["tokens"]["input"] == 100


@needs_sandbox
def test_without_a_reported_cost_the_running_cost_comes_from_the_tokens(fake, work):
    fake["scenario"](costs=[0, 0], pause_after_first_step=20)
    executor = make(fake)
    executor.price = {"input": 100.0, "output": 100.0}  # $/M: 125 tokens is $0.0125
    executor.spend_limit_usd = 0.01
    result = executor.run(task(), work, Budget(wall_clock_s=120))
    assert result.aborted and "passed the job's remaining spend" in result.error


@needs_sandbox
def test_the_proxy_only_serves_generation_paths(fake, work, monkeypatch):
    seen = {}
    real = oc.KeyProxy

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(oc, "KeyProxy", spy)
    make(fake).run(task(), work, Budget())
    assert set(seen["paths"]) == {"chat/completions", "completions", "messages", "responses", "models"}


def test_no_proxy_dir_is_left_when_an_attempt_fails_early(fake, work, monkeypatch, tmp_path):
    made = []
    real = oc.tempfile.mkdtemp

    def mkdtemp(suffix=None, prefix=None, dir=None):
        if prefix != "mullet-px-":
            return real(suffix, prefix, dir)
        made.append(Path(real(prefix=prefix, dir=str(tmp_path))))
        return str(made[-1])

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(oc.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(oc, "shell_wrapper", broken)
    monkeypatch.setattr(oc.OpencodeExecutor, "check_runnable", lambda self, work: None)
    with pytest.raises(OSError, match="disk full"):
        make(fake).run(task(), work, Budget())
    assert made and not made[0].exists()
