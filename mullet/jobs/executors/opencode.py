"""Attempt a node with opencode and any model it can reach, inside Mullet's sandbox.

opencode has no OS sandbox of its own, so Mullet runs it under bubblewrap
(`mullet.jobs.sandbox`) with no network at all, and the API key never goes
in. The provider's base URL points at a loopback port inside the sandbox,
where socat hands each connection to a Unix socket served by a proxy on the
host side (`mullet.jobs.executors.keyproxy`). The proxy forwards only to that
provider's API host, and swaps the dummy key opencode sends for the real one.
Whatever the worker reads, through any symlink or tool, there is no key in
the sandbox to find and no network to send anything to.

- writable: the clone, and a per-job home for opencode's data, state, cache
  and config. Never the user's own: those hold their credentials and every
  other session. Before each attempt the cache, state and config are
  rebuilt, and the data directory keeps its sessions but loses credential
  files, symlinks, `bin` directories and anything executable, so nothing a
  worker plants there (provider packages, a model catalogue, remote-config
  auth entries, binaries) outlives its run;
- read-only: the toolchains, the opencode install, the attempt's brief and
  config, the proxy's socket and the clone's git directory. Dependency
  directories stay read-only inside the clone. The user's own opencode
  config is not visible: Mullet generates the whole config, with only this
  provider's definition copied in and its credentials left out;
- environment: an allowlist. Nothing from the repo's profile.

The worker's shell commands go through a second, nested sandbox started
from a cleared environment: no network, its own session, pid, IPC and UTS
namespaces and /proc, opencode's home read-only, and the proxy's socket
hidden. It refuses chained commands itself as well, except the profile's
own check commands run exactly as written (a lint command with `;` in it,
say): those are allowed after every deny rule, and anything added to them
is still refused. Without a nested
sandbox, or without socat, the executor refuses to run. The profile's env,
minus anything that changes how programs load or where traffic goes,
applies only there.

Symlinks: opencode's file tools check paths as written, so a link in the
clone is followed wherever it goes. One leading anywhere but the clone,
/usr or a toolchain directory refuses the run. One made during the run, or
the key turning up in the transcript, the reply or a changed file (as
written, JSON-escaped or base64), throws the attempt away. With the key kept
out of the sandbox, those after-run checks are defence in depth.

Per attempt, Mullet writes the task brief (`mullet.jobs.context`) and an
`OPENCODE_CONFIG` that defines the `mullet-worker` agent: every tool denied
unless listed, edits allowed, the shell limited to read-only commands and
the profile's checks, no web, no subagents, nothing outside the clone,
`steps` set to the turn budget, and formatters off unless the profile asks.
Language servers stay off unless the user's own profile turns them on: they
run beside opencode, outside the nested shell, and execute repo code.
Project config in the clone (`opencode.json`, `.opencode/`) is switched off,
because a repo could otherwise redefine the agent and allow everything; so
are Claude Code's files and external skills. The brief rides in the agent's
instructions, and the files it lists are attached with `-f`, since one argv
string is capped at 128 KB on Linux.

The provider's definition reaches the sandbox through a whitelist of fields,
never its credentials or headers. The proxy is pinned to the attempt's model
(opencode's small model is set to the same one) and caps the attempt's
requests and bytes; a cap or a refused model counts as a provider error, so
a ladder job moves the step up.

The JSON event stream gives the transcript, usage, cost and the session id,
which `resume` continues with `-s` after a gate failure. What the model says
it did is kept as a claim; the gates decide.
"""

from __future__ import annotations

import base64
import copy
import difflib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mullet.jobs import children, context, sandbox
from mullet.jobs.executors.base import Budget, ExecResult, ExecutorUnavailable, Task, Watch, idle_text
from mullet.jobs.executors.claude_code import READ_ONLY_BASH
from mullet.jobs.executors import keyproxy
from mullet.jobs.executors.keyproxy import KeyProxy, ProxyError
from mullet.jobs.profile import safe_env

logger = logging.getLogger(__name__)

AGENT = "mullet-worker"
MODELS_TIMEOUT_S = 120.0
ERROR_GRACE_S = 20.0
PROXY_PORT = 4141  # inside the sandbox's own network namespace, so it can't clash
DUMMY_KEY = "mullet-proxy-adds-the-key"
# The API paths a worker's provider requests may use, under the provider's
# prefix: generation and the model list, never batch, file or upload endpoints.
API_PATHS: tuple[str, ...] = ("chat/completions", "completions", "messages", "responses", "models")
_SCAN_CHUNK = 1024 * 1024

# Where a provider's API lives when neither the user's config nor the model
# catalogue says: the defaults of the SDKs opencode uses for them.
_DEFAULT_UPSTREAMS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "groq": "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "xai": "https://api.x.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "moonshotai": "https://api.moonshot.ai/v1",  # the international host; "moonshot" in opencode is China's
    "lmstudio": "http://127.0.0.1:1234/v1",
}
# What a provider's or model's definition may carry into the sandbox config.
# Anything else (headers, env, keys, unknown fields) stays out.
_PROVIDER_FIELDS = frozenset({"npm", "name", "api", "models", "whitelist", "blacklist", "options"})
_MODEL_FIELDS = frozenset({
    "id", "name", "family", "limit", "cost", "tool_call", "reasoning", "attachment", "temperature",
    "modalities", "interleaved", "release_date", "status", "experimental", "options",
})
_MODEL_OPTION_FIELDS = frozenset({"reasoningEffort", "reasoning_effort", "thinking", "temperature", "topP",
                                  "include_reasoning", "reasoningSummary", "textVerbosity"})
# The header each SDK sends its key in; everything else uses a bearer token.
_KEY_HEADERS: dict[str, str] = {"@ai-sdk/anthropic": "x-api-key", "@ai-sdk/google": "x-goog-api-key"}

# Matched against the whole command, after the allowed prefixes, so a
# listed prefix can't carry a second command along with it. The shell
# wrapper refuses the same characters, whatever opencode's matcher does.
_CHAINING_BASH: tuple[str, ...] = (
    "*;*", "*&*", "*|*", "*$(*", "*`*", "*>*", "*<*", "*\n*", "*\r*",
)
_RISKY_BASH: tuple[str, ...] = ("find *-exec*", "find *-ok*", "find *-delete*", "find *-fprint*", "rg *--pre*")
_SECRET_READS: tuple[str, ...] = ("*.env", "*.env.*", ".env*", "*/.env*", "*.pem", "*.key")

# Outside the clone, a symlink may only point into these (and the toolchain roots).
_LINK_ROOTS: tuple[str, ...] = ("/usr",)

# Switches that keep a worker's opencode to Mullet's config and nothing else.
_OPENCODE_ENV: dict[str, str] = {
    "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
    "OPENCODE_DISABLE_CLAUDE_CODE": "1",
    "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
    "OPENCODE_DISABLE_AUTOUPDATE": "1",
    "OPENCODE_DISABLE_SHARE": "1",
}
# Inside the sandbox only: there is no network to fetch the catalogue with
# (Mullet copies the user's in).
_SANDBOX_ONLY_ENV: dict[str, str] = {"OPENCODE_DISABLE_MODELS_FETCH": "1"}
# What `opencode models` on the host keeps from Mullet's environment.
_HOST_ENV: tuple[str, ...] = (
    "PATH", "HOME", "LANG", "LC_ALL", "TZ", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "no_proxy", "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "SSL_CERT_DIR",
)

# Used when neither the user's opencode config nor its model catalogue names
# a provider's key variables.
_PROVIDER_KEYS: dict[str, tuple[str, ...]] = {
    "openrouter": ("OPENROUTER_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "moonshot": ("MOONSHOT_API_KEY",),
    "moonshotai": ("MOONSHOT_API_KEY",),
    "google": ("GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "opencode": ("OPENCODE_API_KEY",),
}
# Other names Mullet's own .env may use for the same key.
_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "MOONSHOT_API_KEY": ("KIMI_API_KEY",),
    "GOOGLE_GENERATIVE_AI_API_KEY": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "GEMINI_API_KEY": ("GOOGLE_API_KEY",),
}

_MODEL_LINE = re.compile(r"^[A-Za-z0-9][\w.~@-]*/\S+$")

_RULES = """\
- Your working directory is a private copy of the repository. Work only there.
- Change only files that match the scope above. Leave lockfiles, CI config,
  CLAUDE.md, AGENTS.md, opencode.json, .claude/, .mcp.json and .env files alone
  unless this step says otherwise.
- Any new test file must match the scope too. If the step needs one outside it,
  answer BLOCKED and say so rather than writing it elsewhere.
- Don't commit or run git commands that change anything. Mullet records your work.
- Make the smallest change that meets the goal. Write one test, then the code
  for it, one behaviour at a time.
- Don't invent new test helpers or seams just to make code testable. Leave no
  commented-out code and no TODO comments.
- Don't weaken, skip or delete tests to get them passing.
- Run the project's checks before you finish if you can. Shell commands run
  without network access.
- When you finish, reply with a short summary of what you changed. Start it with
  DONE, or with BLOCKED and the reason if you couldn't do this step."""


def build_rules(bash_commands: list[str], exact_commands: list[str] | None = None) -> str:
    """The rules section of an opencode worker's brief."""
    allowed = "\n".join(f"    {c}" for c in dict.fromkeys([*READ_ONLY_BASH, *bash_commands]))
    text = (f"{_RULES}\n- In the shell you may run commands that start with one of these, one command "
            "per call, with no pipes, redirects, `;`, `&&` or `$(...)`:\n" + allowed)
    exact = [c for c in exact_commands or [] if _CHAIN_CHARS.search(c)]
    if exact:
        text += ("\n- These project checks may be run exactly as written, as one call each:\n"
                 + "\n".join(f"    {c}" for c in exact))
    return text


# Characters that make a command more than one command: the shell wrapper and
# the chaining rules refuse them, except in a profile's own check commands.
_CHAIN_CHARS = re.compile(r"[;&|`<>\n\r]|\$\(")


def exact_rules(exact_commands: list[str] | None) -> list[str]:
    """The profile's check commands that may run exactly as written, though they chain.

    opencode matches patterns with `*` and `?` as wildcards and has no way
    to escape them, so a command with either would allow more than itself;
    those stay refused. Newlines are never allowed.
    """
    out = []
    for command in exact_commands or []:
        command = command.strip()
        if command and "\n" not in command and "\r" not in command and "*" not in command and "?" not in command:
            out.append(command)
    return list(dict.fromkeys(out))


def bash_permission(bash_commands: list[str], exact_commands: list[str] | None = None) -> dict[str, str]:
    """opencode bash rules: deny by default, allow the listed prefixes, deny chaining. Last match wins.

    The profile's own check commands (`exact_commands`) come last, each as the
    exact command only, so one that contains `;` or `&&` still runs while
    anything added to it is refused.
    """
    rules: dict[str, str] = {"*": "deny"}
    for command in dict.fromkeys([*READ_ONLY_BASH, *bash_commands]):
        command = command.strip()
        if command:
            rules[command] = "allow"
            rules[f"{command} *"] = "allow"
    for pattern in (*_RISKY_BASH, *_CHAINING_BASH):
        rules[pattern] = "deny"
    for command in exact_rules(exact_commands):
        rules.pop(command, None)  # re-added at the end, after the denies
        rules[command] = "allow"
    return rules


def build_config(*, instructions: list[str], bash_commands: list[str], steps: int,
                 formatter: bool = False, lsp: bool = False,
                 provider: dict[str, Any] | None = None, model: str | None = None,
                 exact_commands: list[str] | None = None) -> dict[str, Any]:
    """The per-attempt `OPENCODE_CONFIG`: the `mullet-worker` agent, its limits and its one provider.

    It is the whole of opencode's config in the sandbox: the user's own
    config directory isn't visible there. With `model`, opencode's small
    model (titles, summaries) is the same one, so every request uses the
    model the proxy is pinned to.
    """
    permission: dict[str, Any] = {
        "*": "deny",  # first, so any tool not listed here stays off
        "invalid": "allow",
        "read": {"*": "allow", **{p: "deny" for p in _SECRET_READS}},
        "edit": "allow",
        "glob": "allow",
        "grep": "allow",
        "list": "allow",
        "lsp": "allow" if lsp else "deny",
        "todowrite": "allow",
        "todoread": "allow",
        "bash": bash_permission(bash_commands, exact_commands),
        "task": "deny",
        "skill": "deny",
        "webfetch": "deny",
        "websearch": "deny",
        "external_directory": "deny",
        "doom_loop": "deny",
        "question": "deny",
    }
    config: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "instructions": instructions,
        "lsp": lsp,
        "snapshot": False,
        "share": "disabled",
        "autoupdate": False,
        "agent": {AGENT: {
            "description": "Mullet job worker: one gated step in a private clone",
            "mode": "primary",
            "steps": steps,
            "permission": permission,
        }},
    }
    if not formatter:
        config["formatter"] = False
    if provider:
        config["provider"] = provider
    if model:
        config["model"] = model
        config["small_model"] = model
    return config


def provider_of(model: str) -> str:
    """The provider id in an opencode model id (`provider/model`)."""
    return model.split("/", 1)[0]


@dataclass
class Route:
    """Where a provider's API is, how its key is sent, and what to tell opencode about it."""

    provider: str
    upstream: str  # the real base URL, reached only by the host-side proxy
    header: str  # "bearer", "x-api-key" or "x-goog-api-key"
    definition: dict[str, Any]  # the user's provider entry, credentials removed

    def sandbox_entry(self, with_key: bool) -> dict[str, Any]:
        """The provider's config inside the sandbox: its base URL is the proxy's loopback port."""
        entry = copy.deepcopy(self.definition)
        base = f"http://127.0.0.1:{PROXY_PORT}{urlsplit(self.upstream).path.rstrip('/')}"
        options = {k: v for k, v in (entry.get("options") or {}).items() if k == "timeout"}
        options["baseURL"] = base
        if with_key:
            options["apiKey"] = DUMMY_KEY  # opencode needs one to use the provider; the proxy swaps it
        entry["options"] = options
        if "api" in entry:
            entry["api"] = base
        return entry


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _plain(value: Any, depth: int = 0) -> Any:
    """Plain data only (no credential-looking keys), for model metadata like `limit` and `cost`."""
    if isinstance(value, dict) and depth < 4:
        return {k: _plain(v, depth + 1) for k, v in value.items()
                if isinstance(k, str) and k not in ("apiKey", "headers", "env")
                and not k.lower().endswith(("key", "token", "secret"))}
    if isinstance(value, list) and depth < 4:
        return [_plain(v, depth + 1) for v in value if _scalar(v) or isinstance(v, (dict, list))]
    return value if _scalar(value) else None


def provider_definition(configured: dict[str, Any]) -> dict[str, Any]:
    """The parts of the user's provider entry the sandbox may see: a whitelist, never credentials.

    The provider keeps its package, name, API URL, model filters and models;
    each model keeps its metadata and a few generation options. `options`
    at the provider level is rebuilt by `Route.sandbox_entry`.
    """
    out: dict[str, Any] = {}
    for key, value in configured.items():
        if key not in _PROVIDER_FIELDS:
            continue
        if key == "models" and isinstance(value, dict):
            models = {}
            for model_id, model in value.items():
                if not isinstance(model, dict):
                    continue
                entry = {k: _plain(v) for k, v in model.items() if k in _MODEL_FIELDS and k != "options"}
                options = model.get("options")
                if isinstance(options, dict):
                    kept = {k: _plain(v) for k, v in options.items() if k in _MODEL_OPTION_FIELDS}
                    if kept:
                        entry["options"] = kept
                models[str(model_id)] = entry
            out["models"] = models
        elif key in ("whitelist", "blacklist") and isinstance(value, list):
            out[key] = [v for v in value if isinstance(v, str)]
        elif key == "options" and isinstance(value, dict):
            # Rebuilt by Route.sandbox_entry: only the timeout survives.
            timeout = value.get("timeout")
            if timeout is False or (isinstance(timeout, int) and not isinstance(timeout, bool)):
                out["options"] = {"timeout": timeout}
        elif isinstance(value, str):
            out[key] = value
    return out


def api_model_ids(model: str, definition: dict[str, Any]) -> list[str]:
    """The model ids a request body may name for opencode id `model` (`provider/model`)."""
    model_id = model.split("/", 1)[1] if "/" in model else model
    ids = [model_id]
    entry = (definition.get("models") or {}).get(model_id)
    if isinstance(entry, dict) and isinstance(entry.get("id"), str):
        ids.append(entry["id"])
    return list(dict.fromkeys(ids))


def provider_route(provider: str, config_dir: Path, catalogue: Path | None,
                   upstream: str | None = None) -> Route:
    """How to reach `provider`: from the user's own config, then the catalogue, then the SDK defaults."""
    configured = (user_config(config_dir).get("provider") or {}).get(provider)
    configured = configured if isinstance(configured, dict) else {}
    listed = _read_json(catalogue).get(provider) if catalogue is not None else None
    listed = listed if isinstance(listed, dict) else {}
    options = configured.get("options") if isinstance(configured.get("options"), dict) else {}
    found = (upstream or options.get("baseURL") or configured.get("api") or listed.get("api")
             or _DEFAULT_UPSTREAMS.get(provider))
    if not isinstance(found, str) or not found:
        raise ExecutorUnavailable(
            f"can't tell where provider {provider!r}'s API is; set provider.{provider}.options.baseURL "
            "in your opencode config")
    npm = configured.get("npm") or listed.get("npm") or ""
    return Route(provider=provider, upstream=found.rstrip("/"), header=_KEY_HEADERS.get(str(npm), "bearer"),
                 definition=provider_definition(configured))


def configured_key(provider: str, config_dir: Path) -> str | None:
    """A key the user wrote into their own opencode config for `provider`, read on the host only."""
    configured = (user_config(config_dir).get("provider") or {}).get(provider)
    options = configured.get("options") if isinstance(configured, dict) else None
    value = options.get("apiKey") if isinstance(options, dict) else None
    if not isinstance(value, str) or not value:
        return None
    named = re.fullmatch(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}", value)
    if named:
        name = named.group(1)
        return resolve_keys([name]).get(name) if provider_key_name(provider, name) else None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        # opencode also reads JSONC; drop whole-line comments and trailing commas and try again.
        stripped = re.sub(r"^\s*//.*$", "", text, flags=re.M)
        stripped = re.sub(r",(\s*[}\]])", r"\1", stripped)
        try:
            data = json.loads(stripped)
        except ValueError:
            return {}
    return data if isinstance(data, dict) else {}


def user_config(config_dir: Path) -> dict[str, Any]:
    """The user's global opencode config, read for provider and MCP names only."""
    for name in ("opencode.json", "opencode.jsonc", "config.json"):
        data = _read_json(config_dir / name)
        if data:
            return data
    return {}


def user_catalogue() -> Path:
    """The user's own opencode model catalogue: public data, and out of every worker's reach."""
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "opencode" / "models.json"


def provider_key_name(provider: str, name: str) -> bool:
    """True if `name` is an API-key variable that belongs to `provider`.

    The fixed map, or `<PROVIDER>_..._API_KEY`. Anything else that a config
    or catalogue lists is ignored, so no record can make Mullet hand over an
    unrelated secret from its environment.
    """
    if name in _PROVIDER_KEYS.get(provider, ()):
        return True
    prefix = re.sub(r"[^A-Z0-9]+", "_", provider.upper()).strip("_")
    return bool(prefix) and re.fullmatch(rf"{re.escape(prefix)}(?:_[A-Z0-9]+)*_API_KEY", name) is not None


def key_names(provider: str, config_dir: Path, catalogue: Path | None) -> list[str]:
    """Variables that can hold `provider`'s API key, from opencode's own records, filtered.

    `config_dir` and `catalogue` must be the user's own, never anything a
    worker could have written.
    """
    names: list[str] = []
    configured = (user_config(config_dir).get("provider") or {}).get(provider) or {}
    if isinstance(configured, dict) and isinstance(configured.get("env"), list):
        names += [n for n in configured["env"] if isinstance(n, str)]
    if catalogue is not None:
        entry = _read_json(catalogue).get(provider) or {}
        if isinstance(entry, dict) and isinstance(entry.get("env"), list):
            names += [n for n in entry["env"] if isinstance(n, str)]
    names += list(_PROVIDER_KEYS.get(provider, ()))
    return [n for n in dict.fromkeys(names) if provider_key_name(provider, n)]


def resolve_keys(names: list[str]) -> dict[str, str]:
    """Values for `names` from the environment or Mullet's .env files, never from a project."""
    from mullet.core.env import dotenv_value

    out: dict[str, str] = {}
    for name in names:
        for candidate in (name, *_KEY_ALIASES.get(name, ())):
            value = os.environ.get(candidate) or dotenv_value(candidate)
            if value:
                out[name] = value
                break
    return out


def _model_context(providers: Any, provider: str, model_id: str) -> int | None:
    """`limit.context` for a model in a provider map shaped like opencode's config and catalogue."""
    entry = providers.get(provider) if isinstance(providers, dict) else None
    models = entry.get("models") if isinstance(entry, dict) else None
    model = models.get(model_id) if isinstance(models, dict) else None
    limit = model.get("limit") if isinstance(model, dict) else None
    value = limit.get("context") if isinstance(limit, dict) else None
    return value if isinstance(value, int) and value > 0 else None


def parse_models(text: str) -> list[str]:
    """Model ids from `opencode models` output."""
    return [line.strip() for line in text.splitlines() if _MODEL_LINE.match(line.strip())]


def nearest_models(model: str, known: list[str], n: int = 5) -> list[str]:
    """Known ids that look like `model`: close spellings, then ids sharing its last part."""
    close = difflib.get_close_matches(model, known, n=n, cutoff=0.5)
    tail = model.rsplit("/", 1)[-1].lower()
    sharing = [m for m in known if tail and tail in m.lower()]
    return list(dict.fromkeys([*close, *sharing]))[:n]


@lru_cache(maxsize=1)
def nested_sandbox_ok() -> bool:
    """Whether a sandbox can start another, network-less one inside it (for the worker's shell)."""
    return sandbox.nested_check()


def shell_wrapper(env: dict[str, str], readonly: list[Path], hidden: list[Path] | None = None,
                  exact_commands: list[str] | None = None) -> str:
    """The SHELL opencode runs the worker's commands with.

    It refuses a command that chains another (whatever opencode's own
    matcher made of it), unless it is exactly one of `exact_commands` (the
    profile's own checks), then runs it in a nested sandbox: no network, its
    own namespaces and /proc, `readonly` paths read-only, `hidden` paths
    covered, and an environment built from scratch (PATH-like basics plus
    `env`).
    """
    bash = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
    assignments = " ".join(shlex.quote(f"{k}={v}") for k, v in env.items())
    # The nested sandbox must start from a cleared environment (env -i below):
    # its own init process keeps whatever it was started with, and would show
    # it in /proc. The opencode home goes read-only, the proxy's socket is covered.
    nested = " ".join(shlex.quote(a) for a in sandbox.nested_argv(readonly, hidden or []))
    exact = exact_rules(exact_commands)
    # A quoted word in a case pattern is literal, so each one matches only itself.
    exact_check = ""
    if exact:
        exact_check = (
            'if [ "$#" -eq 2 ] && [ "$1" = "-c" ]; then\n'
            '  case "$2" in\n'
            f"    {'|'.join(shlex.quote(c) for c in exact)}) exact=1 ;;\n"
            "  esac\n"
            "fi\n"
        )
    return (
        "#!/bin/sh\n"
        "# Written by Mullet: the worker's shell commands run one at a time, without network,\n"
        "# API keys or a view of opencode's processes.\n"
        "nl='\n'\n"
        "cr=$(printf '\\r')\n"
        "exact=0\n"
        f"{exact_check}"
        'if [ "$exact" = 0 ]; then\n'
        '  case "$*" in\n'
        "    *\"$nl\"*|*\"$cr\"*|*'&'*|*';'*|*'|'*|*'`'*|*'$('*|*'>'*|*'<'*)\n"
        "      echo 'Mullet: one command per call, without ;, &, pipes, redirects or $(...)' >&2\n"
        "      exit 126 ;;\n"
        "  esac\n"
        "fi\n"
        # The profile's variables first, so Mullet's own win.
        f'exec /usr/bin/env -i {assignments} PATH="$PATH" HOME="$HOME" TMPDIR=/tmp TERM=dumb '
        f'LANG="${{LANG:-C.UTF-8}}" {nested} {bash} "$@"\n'
    )


def install_dirs(binary: str) -> list[Path]:
    """Directories the opencode install needs inside the sandbox: its bin dir and its package."""
    link = Path(binary)
    real = link.resolve()
    dirs = [link.parent, real.parent]
    for parent in list(real.parents)[:3]:
        if (parent / "package.json").is_file():
            dirs.append(parent)
            break
    try:
        with open(real, "rb") as fh:
            script = fh.read(2) == b"#!"
    except OSError:
        script = False
    node = shutil.which("node")
    if script and node:  # an npm wrapper script needs node beside it
        dirs.append(Path(node).parent)
    return list(dict.fromkeys(dirs))


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def escaping_symlinks(work: Path, allowed: list[Path]) -> list[tuple[str, str]]:
    """Symlinks in `work` whose targets opencode's file tools must never reach: (path, target).

    opencode checks paths as written, so a link inside the clone is followed
    wherever it goes. A target is fine only inside the clone, under /usr (a
    virtualenv's python) or in one of the `allowed` read-only roots (the
    toolchains). Anything else, /proc, /etc, /mnt, another user's home or
    the job's own files, is refused.
    """
    root = os.path.realpath(work)
    roots = [root, *(os.path.realpath(p) for p in _LINK_ROOTS), *(os.path.realpath(p) for p in allowed)]
    found: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in [*dirnames, *filenames]:
            full = os.path.join(dirpath, name)
            if not os.path.islink(full):
                continue
            target = os.path.realpath(full)
            if not any(_under(target, r) for r in roots):
                found.append((os.path.relpath(full, root), target))
    return sorted(found)


def files_changed_since(work: Path, since: float, skip: list[str]) -> list[Path]:
    """Regular files under `work` whose status changed after `since` (ctime can't be set back)."""
    root = Path(work)
    skipped = {root / s.strip("/") for s in skip if s.strip("/")}
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        dirnames[:] = [d for d in dirnames if here / d not in skipped]
        for name in filenames:
            path = here / name
            try:
                st = os.lstat(path)
            except OSError:
                continue
            if st.st_ctime >= since and path.is_file() and not path.is_symlink():
                out.append(path)
    return out


def secret_forms(secret: str) -> list[str]:
    """`secret` as written, JSON-escaped, and the parts of its base64 that don't depend on alignment."""
    forms = [secret, json.dumps(secret)[1:-1]]
    raw = secret.encode()
    for offset in range(3):
        encoded = base64.b64encode(b"\0" * offset + raw).decode()
        core = encoded[4 if offset else 0:-4]  # drop the groups the padding bytes touch
        if len(core) >= 8:
            forms.append(core)
    return list(dict.fromkeys(f for f in forms if f))


def _contains(path: Path, needles: list[bytes]) -> bool:
    """True if any needle is in the file, read in chunks that overlap by the longest needle."""
    overlap = max((len(n) for n in needles), default=1)
    tail = b""
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(_SCAN_CHUNK)
                if not chunk:
                    return False
                window = tail + chunk
                if any(n in window for n in needles):
                    return True
                tail = window[-overlap:]
    except OSError:
        return False


def _redact_file(path: Path, secrets: list[str]) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for secret in secrets:
        text = text.replace(secret, "[redacted]")
    path.write_text(text, encoding="utf-8")


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


@dataclass
class _Stream:
    lines: int = 0
    session_id: str | None = None
    steps: int = 0
    cost: float = 0.0
    cost_reported: bool = False  # opencode prices steps from the catalogue; without a price, no cost
    tokens: dict[str, int] = field(default_factory=dict)
    tools: dict[str, int] = field(default_factory=dict)
    texts: list[str] = field(default_factory=list)  # text since the last step started
    final_text: str = ""
    error: str | None = None
    error_at: float | None = None
    spend_stopped: bool = False
    step_tools: int = 0  # tool calls since the last step started
    watch: Watch | None = None


# A step that ends for one of these reasons, or with text and no tool calls when
# opencode gives no reason, is the worker's final reply.
_FINAL_REASONS = frozenset({"stop", "end_turn"})


def _add_tokens(total: dict[str, int], tokens: dict[str, Any], prefix: str = "") -> None:
    for key, value in tokens.items():
        if isinstance(value, dict):
            _add_tokens(total, value, f"{prefix}{key}_")
        elif isinstance(value, (int, float)):
            total[prefix + key] = total.get(prefix + key, 0) + int(value)


def consume(stream, transcript_path: Path, state: _Stream) -> None:
    """Read `opencode run --format json` events into `state`, copying every line to the transcript.

    Events carry `type` and `sessionID`; `part` holds the detail: `text`
    parts, `tool_use` with `part.tool`, and `step_finish` with tokens and
    cost for the step. An `error` event is opencode's own diagnosis of a
    failed request (a bad key, an unknown model, a provider outage).
    """
    with open(transcript_path, "w", encoding="utf-8") as transcript:
        for line in stream:
            transcript.write(line)
            transcript.flush()
            state.lines += 1
            if state.watch is not None:
                state.watch.output()
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            state.session_id = state.session_id or event.get("sessionID")
            kind = event.get("type")
            part = event.get("part") or {}
            if kind == "step_start":
                state.texts = []
                state.step_tools = 0
                if state.watch is not None:
                    state.watch.resumed()
            elif kind == "text":
                if part.get("text"):
                    state.texts.append(str(part["text"]))
                    state.final_text = "".join(state.texts)
            elif kind == "tool_use":
                name = str(part.get("tool") or "?")
                state.tools[name] = state.tools.get(name, 0) + 1
                state.step_tools += 1
            elif kind == "step_finish":
                state.steps += 1
                reason = part.get("reason")
                final = reason in _FINAL_REASONS if reason else bool(state.texts) and not state.step_tools
                if final and state.watch is not None:
                    state.watch.finished()
                _add_tokens(state.tokens, part.get("tokens") or {})
                if part.get("cost") is not None:
                    try:
                        cost = float(part["cost"])
                    except (TypeError, ValueError):
                        cost = 0.0
                    if cost > 0:
                        # opencode prices a step from its catalogue; $0 means it has
                        # no price there, so it counts as not reported.
                        state.cost += cost
                        state.cost_reported = True
            elif kind == "error":
                err = event.get("error") or {}
                data = err.get("data") if isinstance(err, dict) else None
                message = (data or {}).get("message") if isinstance(data, dict) else None
                state.error = str(message or (err.get("message") if isinstance(err, dict) else err) or err)
                state.error_at = time.monotonic()


_models_lock = threading.Lock()
_models_cache: dict[tuple[str, str], list[str]] = {}


class OpencodeExecutor:
    """Runs `opencode run` in the clone, sandboxed, as the `mullet-worker` agent."""

    name = "opencode"

    def __init__(self, logs_dir: Path, model: str, *, home: Path | None = None,
                 bash_commands: list[str] | None = None, read_paths: list[Path] | None = None,
                 deps_paths: list[str] | None = None, shell_env: dict[str, str] | None = None,
                 formatter: bool = False, lsp: bool = False, binary: str | None = None,
                 config_dir: Path | None = None, context_share: float = context.BUDGET_SHARE,
                 preload_files: bool = True, upstream: str | None = None,
                 exact_commands: list[str] | None = None, max_requests: int | None = None,
                 max_bytes: int = keyproxy.MAX_BYTES, pin_model: bool = True,
                 request_fields: dict[str, Any] | None = None):
        if not model or "/" not in model:
            raise ExecutorUnavailable(f"opencode needs a model id like provider/model, not {model!r}")
        self.logs_dir = logs_dir
        self.model = model
        self.home = home or logs_dir / "opencode-home"
        self.bash_commands = list(bash_commands or [])
        # The profile's check commands, which may run exactly as written even when they chain.
        self.exact_commands = list(exact_commands or [])
        # Per-attempt caps in the key proxy: requests (default: four per turn,
        # plus some), bytes, and requests only for this model.
        self.max_requests = max_requests
        self.max_bytes = max_bytes
        self.pin_model = pin_model
        # Fields the key proxy sets on every request body (OpenRouter host routing).
        self.request_fields = dict(request_fields or {})
        # Set by the runner before each attempt: the job's remaining spend (None:
        # no limit), $/M token prices for when no cost is reported, and a
        # callback that gets the usage so far after every step.
        self.spend_limit_usd: float | None = None
        self.price: dict[str, float] = {}
        # True when the user set a price for this model (with cache rates, if it has them).
        # Only then are tokens turned into dollars from the provider's own usage reports.
        self.price_configured = False
        self._proxy: KeyProxy | None = None
        self.on_progress: Callable[[dict[str, Any]], None] | None = None
        self.read_paths = list(read_paths or [])
        self.deps_paths = list(deps_paths or [])
        # The profile's env: only ever for the worker's nested shell, never opencode itself.
        self.shell_env = safe_env(dict(shell_env or {}))
        self.formatter = formatter
        self.lsp = lsp
        self.binary = binary or shutil.which("opencode")
        # The user's own opencode config: read on the host for this provider's
        # definition and key, never shown to the sandbox.
        xdg_config = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        self.config_dir = config_dir or Path(xdg_config) / "opencode"
        self.upstream = upstream  # the provider's real base URL, if not the configured one
        self.context_share = context_share
        self.preload_files = preload_files
        self._seq = 0
        self._context_length: int | None = None

    # Setup and checks

    def _dirs(self, home: Path | None = None) -> dict[str, Path]:
        home = home or self.home
        return {"XDG_DATA_HOME": home / "data", "XDG_STATE_HOME": home / "state",
                "XDG_CACHE_HOME": home / "cache", "XDG_CONFIG_HOME": home / "config"}

    def catalogue(self) -> Path | None:
        """The model catalogue Mullet trusts: the user's own, else the one `list_models` fetched on the host."""
        for path in (user_catalogue(), self.logs_dir / "models.json"):
            if path.is_file():
                return path
        return None

    def _fresh_home(self, home: Path | None = None) -> None:
        """Make opencode's home safe to start from: nothing a worker left can steer the next run.

        The cache (provider packages, the model catalogue, language servers),
        the state and the config directory are rebuilt; the catalogue comes
        from the user's own cache. The data directory keeps its sessions, so
        a session can be resumed, but loses credential files (an auth entry
        can pull in remote config), symlinks, `bin` directories and anything
        executable.
        """
        dirs = self._dirs(home)
        for name in ("XDG_CACHE_HOME", "XDG_STATE_HOME", "XDG_CONFIG_HOME"):
            _remove(dirs[name])
        data = dirs["XDG_DATA_HOME"]
        if data.is_symlink() or (data.exists() and not data.is_dir()):
            _remove(data)
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        for dirpath, dirnames, filenames in os.walk(data, topdown=True, followlinks=False):
            for name in list(dirnames):
                full = Path(dirpath) / name
                if full.is_symlink() or name == "bin":
                    _remove(full)
                    dirnames.remove(name)
            for name in filenames:
                full = Path(dirpath) / name
                mode = os.lstat(full).st_mode
                if (stat.S_ISLNK(mode) or not stat.S_ISREG(mode) or mode & 0o111
                        or (name.endswith(".json") and "auth" in name)):
                    full.unlink()
        catalogue = self.catalogue()
        if catalogue is not None:
            mine = dirs["XDG_CACHE_HOME"] / "opencode" / "models.json"
            mine.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(catalogue, mine)
            except OSError:
                pass

    def key_names(self) -> list[str]:
        """Variables that can hold the key for this model's provider."""
        return key_names(provider_of(self.model), self.config_dir, self.catalogue())

    def keys(self) -> dict[str, str]:
        """The provider's key under each variable name it has, for host-side opencode only."""
        return resolve_keys(self.key_names())

    def key(self) -> str | None:
        """The provider's real key, for the host-side proxy: Mullet's environment first, then the user's config."""
        found = self.keys()
        return next(iter(found.values()), None) or configured_key(provider_of(self.model), self.config_dir)

    def route(self) -> Route:
        """Where this model's provider is, and how its key is sent."""
        return provider_route(provider_of(self.model), self.config_dir, self.catalogue(), self.upstream)

    def _base_env(self, home: Path | None = None) -> dict[str, str]:
        env = {k: str(v) for k, v in self._dirs(home).items()}
        env.update(_OPENCODE_ENV)
        if self.lsp:
            env["OPENCODE_EXPERIMENTAL_LSP_TOOL"] = "1"
        return env

    def list_models(self) -> list[str]:
        """Model ids opencode offers with the keys this executor passes. Cached per process.

        `opencode models` runs on the host, so it gets a fresh home that no
        worker ever saw, never the job's, and the user's own config.
        """
        if not self.binary:
            raise ExecutorUnavailable("opencode is not on PATH (install it with: npm i -g opencode-ai)")
        cache_key = (self.binary, provider_of(self.model))
        with _models_lock:
            if cache_key in _models_cache:
                return _models_cache[cache_key]
        with tempfile.TemporaryDirectory(prefix="mullet-opencode-models-") as tmp:
            home = Path(tmp)
            self._fresh_home(home)
            env = {k: os.environ[k] for k in _HOST_ENV if k in os.environ}
            env.update(self._base_env(home))
            env["XDG_CONFIG_HOME"] = str(self.config_dir.parent)
            env.update(self.keys())
            try:
                proc = subprocess.run([self.binary, "models"], capture_output=True, text=True, env=env,
                                      cwd=home, timeout=MODELS_TIMEOUT_S, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ExecutorUnavailable(f"`opencode models` failed: {exc}") from exc
            fetched = home / "cache" / "opencode" / "models.json"
            if self.catalogue() is None and fetched.is_file():
                # No catalogue of the user's own: keep the one opencode just fetched, out of the sandbox.
                self.logs_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(fetched, self.logs_dir / "models.json")
        if proc.returncode != 0:
            raise ExecutorUnavailable(f"`opencode models` exited {proc.returncode}: {proc.stderr.strip()[-300:]}")
        found = parse_models(proc.stdout)
        with _models_lock:
            _models_cache[cache_key] = found
        return found

    def validate_model(self) -> None:
        """Raise ExecutorUnavailable, with the nearest valid ids, unless opencode offers this model."""
        known = self.list_models()
        if self.model in known:
            return
        provider = provider_of(self.model)
        message = f"opencode has no model {self.model!r}."
        near = nearest_models(self.model, known)
        if near:
            message += f" Nearest: {', '.join(near)}."
        names = self.key_names()
        if names and not self.keys() and not any(m.startswith(provider + "/") for m in known):
            # Without its key a provider's models aren't listed at all.
            message += (f" No API key was found for provider {provider!r} either: set {' or '.join(names)} "
                        "in Mullet's .env (~/.config/mullet/.env) or the environment.")
        raise ExecutorUnavailable(message + " See `opencode models`.")

    def context_length(self) -> int:
        """The model's context window: opencode's own figure if it has one, else Mullet's registry."""
        if self._context_length is None:
            provider, model_id = self.model.split("/", 1)
            catalogue = self.catalogue()
            sources = [user_config(self.config_dir).get("provider"), _read_json(catalogue) if catalogue else {}]
            limit = None
            for providers in sources:
                limit = _model_context(providers, provider, model_id)
                if limit:
                    break
            if not limit:
                from mullet.core.providers.model_info import get_context_length
                limit = get_context_length(self.model)
            self._context_length = int(limit)
        return self._context_length

    # Attempts

    def run(self, task: Task, work: Path, budget: Budget) -> ExecResult:
        """Attempt `task` in `work` in a new session, with the brief and context files attached."""
        return self._attempt(task, work, budget, session_id=None)

    def resume(self, task: Task, work: Path, budget: Budget, session_id: str) -> ExecResult:
        """Continue `session_id` on the tree it left, with the newest gate feedback."""
        return self._attempt(task, work, budget, session_id=session_id)

    def command(self, work: Path, message: str, *, title: str, files: list[Path],
                session_id: str | None) -> list[str]:
        """The opencode command line. The message comes first: `-f` takes every word after it."""
        if not self.binary:
            raise ExecutorUnavailable("opencode is not on PATH (install it with: npm i -g opencode-ai)")
        argv = [self.binary, "run", message, "--pure", "--dir", str(work), "--agent", AGENT,
                "--model", self.model, "--format", "json"]
        if session_id:
            argv += ["--session", session_id]
        else:
            argv += ["--title", title]
        for path in files:
            argv += ["--file", str(path)]
        return argv

    def spec(self, work: Path, attempt_dir: Path, env: dict[str, str], proxy_dir: Path) -> sandbox.SandboxSpec:
        """The sandbox opencode runs in: no network, and the only way out is the proxy's socket."""
        git_dir = work.parent / "git"  # a JobClone keeps it beside the work tree, out of the worker's reach
        expose = [*self.read_paths, *install_dirs(self.binary or "opencode"), attempt_dir, proxy_dir]
        if git_dir.is_dir():
            expose.append(git_dir)
        return sandbox.SandboxSpec(
            cwd=work,
            writable=[work, *self._dirs().values()],
            readonly_overlays=[work / p.strip("/") for p in self.deps_paths if p.strip("/")],
            expose_ro=[p for p in expose if p.exists()],
            network=False,
            env=env,
        )

    @staticmethod
    def launcher(socat: str, socket_path: Path, argv: list[str]) -> list[str]:
        """Inside the sandbox: socat on the loopback port, passing to the proxy's socket, then opencode."""
        listen = f"TCP-LISTEN:{PROXY_PORT},bind=127.0.0.1,fork,reuseaddr"
        script = (
            f"{shlex.quote(socat)} {listen} UNIX-CONNECT:{shlex.quote(str(socket_path))} &\n"
            "i=0\n"
            f"until {shlex.quote(socat)} -u OPEN:/dev/null TCP:127.0.0.1:{PROXY_PORT} 2>/dev/null; do\n"
            "  i=$((i+1))\n"
            '  if [ "$i" -gt 200 ]; then echo "mullet: the provider port never opened" >&2; exit 97; fi\n'
            "  sleep 0.05\n"
            "done\n"
            'exec "$@"\n'
        )
        return ["/bin/sh", "-c", script, "mullet-launch", *argv]

    def _next_dir(self) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        while True:  # a resumed job reuses logs_dir, so skip numbers already used
            self._seq += 1
            path = self.logs_dir / f"opencode-{self._seq:03d}"
            try:
                path.mkdir()
                return path
            except FileExistsError:
                continue

    def allowed_link_roots(self) -> list[Path]:
        """Read-only roots outside the clone that a symlink in it may point into: the toolchains."""
        return [*self.read_paths, *install_dirs(self.binary or "opencode")]

    def check_runnable(self, work: Path) -> None:
        """Refuse to start opencode where its sandbox, or the clone, can't keep it in."""
        status = sandbox.check()
        if not status.ok:
            raise ExecutorUnavailable(f"opencode only runs inside Mullet's sandbox, which is unavailable: {status.reason}")
        if not nested_sandbox_ok():
            raise ExecutorUnavailable(
                "opencode needs a nested sandbox for the worker's shell (no network, no API key), "
                f"and the {sandbox.selection().name} sandbox backend can't start one inside another here")
        if not shutil.which("socat"):
            raise ExecutorUnavailable("opencode reaches its provider through socat, which isn't installed "
                                      "(for example: sudo apt install socat)")
        links = escaping_symlinks(work, self.allowed_link_roots())
        if links:
            shown = "; ".join(f"{path} -> {target}" for path, target in links[:5])
            raise ExecutorUnavailable(
                f"refusing to run opencode on this node: the clone has symlinks that lead out of it, which "
                f"opencode's file tools would follow ({shown}). Remove them, or use the Claude executor.")

    def _attempt(self, task: Task, work: Path, budget: Budget, *, session_id: str | None) -> ExecResult:
        self.check_runnable(work)
        self.validate_model()
        route = self.route()
        key = self.key()
        self._fresh_home()
        attempt_dir = self._next_dir()
        picked: list[context.ContextFile] = []
        if self.preload_files and session_id is None:
            tokens = context.context_budget(self.model, self.context_share, context_length=self.context_length())
            picked = context.select_context_files(task, work, tokens)
        brief = attempt_dir / "brief.md"
        brief.write_text(context.build_brief(
            task, work, attached=picked, rules=build_rules(self.bash_commands, self.exact_commands),
            attached_note="They are attached to your first message, so you don't need to read them again."))
        instructions = [str(brief), *project_rules(work)]
        config = build_config(instructions=instructions, bash_commands=self.bash_commands,
                              steps=budget.max_turns, formatter=self.formatter, lsp=self.lsp,
                              provider={route.provider: route.sandbox_entry(with_key=bool(key))},
                              model=self.model, exact_commands=self.exact_commands)
        config_path = attempt_dir / "opencode.json"
        config_path.write_text(json.dumps(config, indent=2))
        (attempt_dir / "files.txt").write_text("".join(f"{f.path}\t{f.tokens}\t{f.reason}\n" for f in picked))
        # A short path: Unix socket paths are limited to about 108 bytes. Removed
        # in the finally below whatever happens after this line.
        proxy_dir = Path(tempfile.mkdtemp(prefix="mullet-px-", dir="/tmp" if os.path.isdir("/tmp") else None))
        try:
            return self._run_with_proxy(task, work, budget, session_id, route, key, attempt_dir, config_path,
                                        picked, proxy_dir)
        finally:
            shutil.rmtree(proxy_dir, ignore_errors=True)

    def _run_with_proxy(self, task: Task, work: Path, budget: Budget, session_id: str | None, route: Route,
                        key: str | None, attempt_dir: Path, config_path: Path,
                        picked: list[context.ContextFile], proxy_dir: Path) -> ExecResult:
        shell = attempt_dir / "shell"
        shell.write_text(shell_wrapper(self.shell_env, list(self._dirs().values()), [proxy_dir],
                                       self.exact_commands))
        shell.chmod(0o755)

        if session_id:
            message = resume_message(task)
        else:
            message = ("Do the step described in your Mullet task brief, which is in your instructions. "
                       + (f"The {len(picked)} attached files are from the repository, loaded for you. " if picked else "")
                       + "When you have finished, reply with a short summary that starts with DONE, "
                       "or with BLOCKED and the reason if you couldn't do the step.")
        # Mullet's own settings only: nothing from the repo's profile, and no key.
        env = {**self._base_env(), **_SANDBOX_ONLY_ENV, "OPENCODE_CONFIG": str(config_path), "SHELL": str(shell)}
        title = f"mullet {task.label or task.node.id}"
        argv = self.command(work, message, title=title, files=[work / f.path for f in picked],
                            session_id=session_id)
        started = time.time()
        max_requests = self.max_requests or max(50, budget.max_turns * 4 + 20)
        try:
            proxy = KeyProxy(proxy_dir / "s", route.upstream, key, header=route.header,
                             log_path=attempt_dir / "proxy.log", max_requests=max_requests,
                             max_bytes=self.max_bytes,
                             models=api_model_ids(self.model, route.definition) if self.pin_model else None,
                             paths=API_PATHS if self.pin_model else None,
                             body_fields=self.request_fields or None,
                             allowance_usd=self.spend_limit_usd,
                             pricer=self._price_tokens if self.price_configured else None)
        except ProxyError as exc:
            raise ExecutorUnavailable(f"can't reach provider {route.provider!r} safely: {exc}") from exc
        self._proxy = proxy
        try:
            with proxy:
                launch = self.launcher(shutil.which("socat") or "socat", proxy.socket_path, argv)
                result = self._launch(launch, work, attempt_dir, env, budget, proxy_dir)
        finally:
            self._proxy = None
        result.usage.update(attached_files=len(picked), attached_tokens=sum(f.tokens for f in picked),
                            resumed=bool(session_id), lsp=self.lsp, provider_host=urlsplit(route.upstream).hostname,
                            provider_requests=proxy.requests, provider_bytes=proxy.bytes)
        self._take_provider_cost(result.usage, proxy)
        if proxy.capped or proxy.refused_models:
            # Mullet's own caps stopped the run: like a provider's, not the worker's failure.
            result.usage["proxy_refused"] = proxy.capped or "a request named another model"
            result.provider_error = True
            if result.aborted or result.error:
                result.error = f"{result.error or 'stopped'} ({result.usage['proxy_refused']})"
        return self._after_run(result, work, attempt_dir, [key] if key else [], started)

    def _after_run(self, result: ExecResult, work: Path, attempt_dir: Path, keys: list[str],
                   started: float) -> ExecResult:
        """Throw the attempt away if it made a link out of the clone or the key turned up anywhere.

        Both are not the worker's honest work, and the runner resets the tree
        of an aborted attempt, so neither can reach a commit or a resume. The
        key never enters the sandbox, so finding it means something is badly
        wrong; the check stays as defence in depth.
        """
        problems = []
        links = escaping_symlinks(work, self.allowed_link_roots())
        if links:
            problems.append("it made symlinks that lead out of the clone: "
                            + "; ".join(f"{p} -> {t}" for p, t in links[:5]))
        secrets = [form for k in keys if len(k) >= 8 for form in secret_forms(k)]
        if secrets:
            needles = [s.encode() for s in secrets]
            logs = [attempt_dir / "transcript.jsonl", attempt_dir / "stderr.log"]
            leaked = [p for p in logs if _contains(p, needles)]
            for path in leaked:
                _redact_file(path, secrets)
            changed = [p for p in files_changed_since(work, started - 2, self.deps_paths) if _contains(p, needles)]
            if leaked or changed:
                where = [p.name for p in leaked] + [str(p.relative_to(work)) for p in changed]
                problems.append(f"the provider's API key turned up in {', '.join(where[:5])}")
            if result.claim and any(s in str(result.claim) for s in secrets):
                summary = str(result.claim.get("summary", ""))
                for secret in secrets:
                    summary = summary.replace(secret, "[redacted]")
                result.claim = {**result.claim, "summary": summary}
                problems.append("the provider's API key turned up in the worker's reply")
        if problems:
            result.aborted = True
            result.session_id = None  # never resume a session that saw any of this
            result.error = "attempt thrown away: " + "; ".join(problems)
        return result

    def _launch(self, argv: list[str], work: Path, attempt_dir: Path, env: dict[str, str],
                budget: Budget, proxy_dir: Path) -> ExecResult:
        transcript = attempt_dir / "transcript.jsonl"
        stderr_path = attempt_dir / "stderr.log"
        state = _Stream(watch=Watch(budget.idle_s, budget.grace_s))
        started = time.monotonic()
        timed_out = False
        stop: str | None = None
        seen_steps = 0
        children.check_not_stopping()
        with open(stderr_path, "wb") as stderr:
            proc = sandbox.spawn(argv, self.spec(work, attempt_dir, env, proxy_dir),
                                 stdout=subprocess.PIPE, stderr=stderr, cwd=work, text=True)
        reader = threading.Thread(target=consume, args=(proc.stdout, transcript, state), daemon=True)
        reader.start()
        with children.track(proc):
            try:
                while proc.poll() is None:
                    now = time.monotonic()
                    if now - started > budget.wall_clock_s:
                        timed_out = True
                        _kill_group(proc)
                        break
                    state.watch.poll_file(stderr_path, now)
                    stop = state.watch.verdict(now)
                    if stop is not None:
                        _kill_group(proc)  # silent past the idle limit, or done but not exiting
                        break
                    if state.error_at is not None and time.monotonic() - state.error_at > ERROR_GRACE_S:
                        _kill_group(proc)  # opencode reported a fatal error but hasn't exited
                        break
                    if state.steps != seen_steps:
                        seen_steps = state.steps
                        if self._progress(state, time.monotonic() - started):
                            _kill_group(proc)  # past the job's remaining spend
                            break
                    try:
                        proc.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        pass
            except BaseException:  # Ctrl-C or a stop: take opencode and its children with us
                _kill_group(proc)
                raise
        reader.join(timeout=30)
        return self._result(proc.returncode, timed_out, state, transcript, attempt_dir,
                            time.monotonic() - started, stop=stop, idle_s=budget.idle_s)

    def _usage(self, state: _Stream, duration: float) -> dict[str, Any]:
        # The proxy has seen every response, so its running cost goes into each progress record:
        # an attempt whose runner is killed keeps what the provider has been charged so far.
        proxied = self._proxy.spent_usd if self._proxy is not None else 0.0
        cost = max(state.cost, proxied)
        return {
            "duration_s": round(duration, 1),
            "num_turns": state.steps,
            "total_cost_usd": round(cost, 6),
            "cost_reported": state.cost_reported or proxied > state.cost,
            "tokens": dict(state.tokens),
            "tools": dict(state.tools),
            "model": self.model,
            "session_id": state.session_id,
        }

    def _price_tokens(self, tokens: dict[str, int]) -> float:
        from mullet.jobs.ladder import token_cost

        return token_cost(tokens, self.price)

    @staticmethod
    def _take_provider_cost(usage: dict[str, Any], proxy: KeyProxy) -> None:
        """Record what the provider's own responses said, and never leave the cost below it.

        opencode prices steps from its own catalogue, which can lack a model's cache
        rates; the provider's stated cost (or the user's configured price) is the better figure.
        """
        summary = proxy.usage_summary()
        usage.update(summary)
        if summary["proxy_cost_usd"] > float(usage.get("total_cost_usd") or 0.0):
            usage["total_cost_usd"] = summary["proxy_cost_usd"]
            usage["cost_reported"] = True
            usage["cost_source"] = "provider" if summary["proxy_provider_cost_usd"] > 0 else "configured price"

    def running_cost(self, state: _Stream) -> float:
        """The attempt's cost so far: the most of what opencode reported, what the provider said, or tokens at `self.price`."""
        from mullet.jobs.ladder import token_cost

        proxied = self._proxy.spent_usd if self._proxy is not None else 0.0
        return max(state.cost, proxied) if max(state.cost, proxied) > 0 else token_cost(state.tokens, self.price)

    def _progress(self, state: _Stream, duration: float) -> bool:
        """After a step: pass the usage so far on, and say whether the attempt is past its spend."""
        if self.on_progress is not None:
            try:
                self.on_progress(self._usage(state, duration))
            except Exception as exc:  # recording progress must never break the run
                logger.warning("opencode: progress callback failed: %s", exc)
        if self.spend_limit_usd is not None and self.running_cost(state) > self.spend_limit_usd:
            state.spend_stopped = True
            return True
        return False

    def _result(self, code: int | None, timed_out: bool, state: _Stream, transcript: Path,
                attempt_dir: Path, duration: float, *, stop: str | None = None,
                idle_s: float | None = None) -> ExecResult:
        usage = self._usage(state, duration)
        text = state.final_text.strip()
        claim = None
        if text:
            first = text.split(None, 1)[0].strip(":.").upper() if text.split() else ""
            status = {"DONE": "done", "BLOCKED": "blocked"}.get(first, "unknown")
            claim = {"status": status, "summary": text[-2000:]}
        if stop == "grace":
            # It gave its final reply and then didn't exit: killed, and judged like any finished run.
            code = None
            usage["exited_after_grace"] = True
        error, aborted, provider_error, idle_stopped = None, False, False, False
        if state.spend_stopped:
            # Not the worker's failure: the job ran out of money for this rung.
            error, aborted, provider_error = (f"stopped: this attempt's cost (${self.running_cost(state):.2f}) passed "
                                              f"the job's remaining spend (${self.spend_limit_usd:.2f})"), True, True
        elif timed_out:
            error = "timed out"
        elif state.error:
            # The provider or opencode refused (a bad key, an unknown model, an outage):
            # not the worker's doing, so it doesn't use up an attempt.
            error, aborted, provider_error = f"opencode reported an error: {state.error[:500]}", True, True
        elif stop == "idle":
            # Like a provider stop: opencode hung, so it doesn't use up an attempt.
            error, aborted, provider_error, idle_stopped = f"stopped: {idle_text(idle_s or 0)}", True, True, True
            usage["idle_stopped"] = True
        elif code not in (0, None):
            error, aborted = f"opencode exited {code}: {_tail(attempt_dir / 'stderr.log')}", True
        elif state.steps == 0 and not text:
            error, aborted = f"opencode ended without a result (exit {code})", True
        return ExecResult(exit_code=None if timed_out or idle_stopped else code, transcript_path=str(transcript),
                          usage=usage, claim=claim, aborted=aborted, error=error, session_id=state.session_id,
                          provider_error=provider_error, idle_stopped=idle_stopped)


def project_rules(work: Path) -> list[str]:
    """The repo's own agent rules: AGENTS.md, or CLAUDE.md when there is none.

    With project config switched off, opencode no longer finds AGENTS.md by
    itself, so Mullet lists it in the instructions.
    """
    for name in ("AGENTS.md", "CLAUDE.md"):
        if (work / name).is_file() and not (work / name).is_symlink():
            return [str(work / name)]
    return []


def resume_message(task: Task) -> str:
    """What a resumed session is told: the newest gate feedback."""
    latest = task.feedback[-1] if task.feedback else "Mullet's checks failed, with no details recorded."
    return ("Mullet ran its checks on your work and they failed. The files are as you left them. "
            "Fix the problems below, keeping to the rules in your brief, then reply with a summary "
            f"that starts with DONE or BLOCKED.\n\n{latest}")


def _tail(path: Path, chars: int = 400) -> str:
    text = sandbox.read_tail(path, chars).strip()
    return text or "(no output on stderr)"


def _kill_group(proc: subprocess.Popen, grace: float = 10.0) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue
