"""Shared fixture pieces for the mullet.jobs tests: a tiny repo and its test runner.

The runner is self-contained (no pytest needed inside the sandbox) and writes
junit XML to $MULLET_JUNIT, treating an exception named SkipTest as a skip.
"""

import subprocess
import textwrap
from pathlib import Path

RUNTESTS = textwrap.dedent('''\
    import importlib.util, os, sys
    from pathlib import Path
    from xml.sax.saxutils import quoteattr
    sys.path.insert(0, os.getcwd())
    files = sys.argv[1:] or sorted(str(p) for p in Path("tests").glob("test_*.py"))
    cases, failed = [], 0
    for f in files:
        name = Path(f).stem
        spec = importlib.util.spec_from_file_location(name, f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr in sorted(dir(mod)):
            if attr.startswith("test_") and callable(getattr(mod, attr)):
                try:
                    getattr(mod, attr)()
                    cases.append((name, attr, None))
                except Exception as exc:
                    if type(exc).__name__ == "SkipTest":
                        cases.append((name, attr, "skip"))
                    else:
                        failed += 1
                        cases.append((name, attr, repr(exc)))
    junit = os.environ.get("MULLET_JUNIT")
    if junit:
        with open(junit, "w") as fh:
            fh.write("<testsuite>")
            for cls, test, err in cases:
                fh.write(f"<testcase classname={quoteattr(cls)} name={quoteattr(test)}>")
                if err == "skip":
                    fh.write("<skipped/>")
                elif err:
                    fh.write(f"<failure message={quoteattr(err)}/>")
                fh.write("</testcase>")
            fh.write("</testsuite>")
    print(f"{len(cases)} tests, {failed} failed")
    sys.exit(1 if failed else 0)
''')

BASE_FILES = {
    ".gitignore": "__pycache__/\n",
    "app.py": "def add(a, b):\n    return a + b\n",
    "tests/test_app.py": (
        "from app import add\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n"
        "def test_add_zero():\n    assert add(0, 0) == 0\n"
    ),
    "runtests.py": RUNTESTS,
    "uv.lock": "lock\n",
    ".github/workflows/ci.yml": "on: push\n",
}

# A stand-in for the `claude` CLI: records how it was called (call.json next to
# it for the latest call, calls.jsonl for all of them) and replays the scenario
# in scenario.json as a stream-json transcript. A scenario with "calls" plays
# one entry per call, in order. The session id is `--resume`'s, or a new one.
FAKE_CLAUDE = textwrap.dedent('''\
    #!/usr/bin/env python3
    import json, os, sys, time
    from pathlib import Path
    here = Path(__file__).resolve().parent
    scenario = json.loads((here / "scenario.json").read_text())
    log = here / "calls.jsonl"
    index = len(log.read_text().splitlines()) if log.exists() else 0
    if "calls" in scenario:
        scenario = scenario["calls"][min(index, len(scenario["calls"]) - 1)]
    prompt = sys.stdin.read()
    argv = sys.argv[1:]
    session = argv[argv.index("--resume") + 1] if "--resume" in argv else f"sess-{index + 1}"
    call = {"argv": argv, "env": dict(os.environ), "cwd": os.getcwd(), "prompt": prompt}
    (here / "call.json").write_text(json.dumps(call))
    with open(log, "a") as fh:
        fh.write(json.dumps({"argv": argv, "prompt": prompt, "session": session}) + "\\n")
    def emit(event):
        print(json.dumps(event), flush=True)
    emit({"type": "system", "subtype": "init", "apiKeySource": "none", "model": "claude-test", "session_id": session})
    for rel, content in scenario.get("write", {}).items():
        path = Path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    if scenario.get("sleep"):
        time.sleep(scenario["sleep"])
    for info in scenario.get("rate_limits") or ([scenario["rate_limit"]] if scenario.get("rate_limit") else []):
        emit({"type": "rate_limit_event", "rate_limit_info": info})
    emit({"type": "assistant", "message": {"content": []}})
    result = {"type": "result", "subtype": scenario.get("subtype", "success"),
              "is_error": scenario.get("is_error", False), "num_turns": 3, "total_cost_usd": 0.12,
              "usage": {"output_tokens": 10}, "errors": scenario.get("errors", []), "session_id": session}
    if "structured" in scenario:
        result["structured_output"] = scenario["structured"]
    emit(result)
    sys.exit(scenario.get("exit", 0))
''')

# A stand-in for the `opencode` CLI. `opencode models` prints the scenario's
# "models". `opencode run` runs inside Mullet's sandbox, where only the clone
# and opencode's own home are writable, so it records each call (argv, the
# environment, the OPENCODE_CONFIG it was given, the attached files and the
# shell probe's output) in $XDG_DATA_HOME/fake-opencode/calls.jsonl (the data
# directory is the part of opencode's home that lasts between attempts), writes the
# scenario's files into the work tree, runs its "shell" command through $SHELL
# the way opencode's bash tool does, can call its provider at the base URL the
# config gives (the proxy's loopback port) and try a URL directly, and replays
# a `--format json` stream. A scenario with "calls" plays one entry per call,
# in order.
FAKE_OPENCODE = textwrap.dedent('''\
    #!/usr/bin/env python3
    import json, os, subprocess, sys, time
    from pathlib import Path
    here = Path(__file__).resolve().parent
    scenario = json.loads((here / "scenario.json").read_text())
    argv = sys.argv[1:]
    if argv[:1] == ["models"]:
        xdg = ("XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME")
        with open(here / "models-calls.txt", "a") as fh:
            fh.write(json.dumps({"env": sorted(os.environ), "cwd": os.getcwd(),
                                 "env_values": {k: os.environ.get(k, "") for k in xdg}}) + "\\n")
        print("\\n".join(scenario.get("models", ["fakeai/m1"])))
        sys.exit(scenario.get("models_exit", 0))
    state = Path(os.environ["XDG_DATA_HOME"]) / "fake-opencode"
    state.mkdir(parents=True, exist_ok=True)
    log = state / "calls.jsonl"
    index = len(log.read_text().splitlines()) if log.exists() else 0
    if "calls" in scenario:
        scenario = scenario["calls"][min(index, len(scenario["calls"]) - 1)]
    files = [argv[i + 1] for i, a in enumerate(argv) if a in ("--file", "-f")]
    session = argv[argv.index("--session") + 1] if "--session" in argv else f"ses_{index + 1}"
    config = json.loads(Path(os.environ["OPENCODE_CONFIG"]).read_text())
    brief = Path(config["instructions"][0]).read_text()
    shell_out = None
    if scenario.get("shell"):
        proc = subprocess.run([os.environ["SHELL"], "-c", scenario["shell"]], capture_output=True, text=True)
        shell_out = proc.stdout + proc.stderr
    home_probe = None
    if scenario.get("read"):
        try:
            home_probe = Path(scenario["read"]).read_text()
        except OSError as exc:
            home_probe = f"refused: {type(exc).__name__}"
    escape = None
    if scenario.get("escape"):
        try:
            Path(scenario["escape"]).write_text("escaped")
            escape = "written"
        except OSError as exc:
            escape = f"refused: {type(exc).__name__}"
    provider_reply = direct = None
    if scenario.get("call_provider"):
        import urllib.request
        entry = next(iter(config["provider"].values()))
        provider = entry["options"]
        # As opencode does, name the model: a pinned proxy refuses a request that doesn't.
        body = json.dumps({"model": next(iter(entry.get("models") or {"": 0})), "stream": True}).encode()
        req = urllib.request.Request(provider["baseURL"] + "/chat/completions", data=body,
                                     headers={"Authorization": "Bearer " + provider.get("apiKey", ""),
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                provider_reply = resp.read().decode()
        except Exception as exc:
            provider_reply = f"error: {exc}"
    if scenario.get("direct_url"):
        import urllib.request
        try:
            urllib.request.urlopen(scenario["direct_url"], timeout=5)
            direct = "reached"
        except Exception as exc:
            direct = f"refused: {type(exc).__name__}"
    with open(log, "a") as fh:
        fh.write(json.dumps({"argv": argv, "env": dict(os.environ), "cwd": os.getcwd(), "config": config,
                             "brief": brief, "files": files, "shell": shell_out, "read": home_probe,
                             "escape": escape, "work_files": sorted(os.listdir(".")),
                             "provider_reply": provider_reply, "direct": direct}) + "\\n")
    def emit(kind, **fields):
        print(json.dumps({"type": kind, "timestamp": 0, "sessionID": session, **fields}), flush=True)
    for rel, content in scenario.get("write", {}).items():
        path = Path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for rel, target in scenario.get("symlink", {}).items():
        os.symlink(target, rel)
    # The key a test says the executor holds, standing in for one that somehow got in.
    key = scenario.get("leak_value", "")
    if scenario.get("leak_encoding") == "base64":
        import base64
        key = base64.b64encode(("prefix-" + key).encode()).decode()
    elif scenario.get("leak_encoding") == "json":
        key = json.dumps({"k": key + "\\n"})
    if scenario.get("leak_to"):
        Path(scenario["leak_to"]).write_text("found it: " + key + "\\n")
    if scenario.get("leak_in_text"):
        scenario["text"] = "DONE: the key is " + key
    if scenario.get("sleep"):
        time.sleep(scenario["sleep"])
    if scenario.get("error"):
        emit("error", error={"name": "APIError", "data": {"message": scenario["error"]}})
        if scenario.get("hang_after_error"):
            time.sleep(600)
        sys.exit(scenario.get("exit", 1))
    if not scenario.get("silent"):
        emit("step_start", part={"type": "step-start"})
        emit("tool_use", part={"type": "tool", "tool": "edit", "state": {"status": "completed"}})
        costs = scenario.get("costs", [0.01, 0.02])
        emit("step_finish", part={"type": "step-finish", "tokens": {"input": 100, "output": 20,
             "cache": {"read": 5, "write": 0}}, "cost": costs[0]})
        if scenario.get("pause_after_first_step"):
            time.sleep(scenario["pause_after_first_step"])
        emit("step_start", part={"type": "step-start"})
        emit("text", part={"type": "text", "text": scenario.get("text", "DONE: did it")})
        emit("step_finish", part={"type": "step-finish", "tokens": {"input": 50, "output": 10}, "cost": costs[1]})
    sys.exit(scenario.get("exit", 0))
''')

FAKE_KEY = "sk-proj-" + "a1B2c3D4e5F6g7H8i9J0" * 2


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def write(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def make_source_repo(root: Path, files: dict[str, str] | None = None) -> Path:
    root.mkdir(parents=True)
    git(root, "init", "-q", "-b", "main")
    write(root, files or BASE_FILES)
    git(root, "add", ".")
    git(root, "commit", "-qm", "init")
    return root


def good_change(work: Path) -> None:
    (work / "app.py").write_text("def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n")
    with open(work / "tests" / "test_app.py", "a") as fh:
        fh.write("\ndef test_sub():\n    from app import sub\n    assert sub(5, 3) == 2\n")
