"""Per-repo project profiles: how to set up a job's clone and which commands are its gates.

A profile starts from what can be detected in the repo, is overridden by the
user's stored profile, and then by an optional in-repo `.mullet/project.yaml`.
Whoever controls the repo controls that last file, so for the safety fields it
can only tighten what it inherits. Commands are shell strings that the gate
runner executes later inside the sandbox; nothing here runs them.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import posixpath
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator

from mullet.core.agent.project_detect import detect_project
from mullet.jobs.paths import profiles_dir

logger = logging.getLogger(__name__)

REPO_PROFILE = ".mullet/project.yaml"

# Repo-relative POSIX globs: `**` spans directories, and a pattern with no `/`
# matches the basename at any depth. Gate G2 does the matching.
# .gitignore is left out on purpose: editing it is routine, and files it hides
# are removed before gates run. .gitattributes is in, because attributes can
# change how git reads and diffs every other file.
DEFAULT_PROTECTED: tuple[str, ...] = (
    ".git",
    ".git/**",
    ".gitmodules",
    ".gitattributes",
    ".lfsconfig",
    ".github/workflows/**",
    ".gitlab-ci.yml",
    ".husky/**",
    ".github/actions/**",
    ".pre-commit-config.yaml",
    # Files the user's own tools run or read when they open the repo or merge the branch:
    # editors, dev containers, shell hooks, and package-manager settings.
    ".envrc",
    ".vscode/**",
    ".devcontainer/**",
    ".npmrc",
    ".yarnrc",
    ".yarnrc.yml",
    ".pypirc",
    "AGENTS.md",
    "CLAUDE.md",
    ".claude/**",
    ".mcp.json",
    "opencode.json",
    ".mullet/**",
    ".env",
    ".env.*",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
    "composer.lock",
    "Cargo.lock",
    "go.sum",
    # Test runner configuration: changing it can switch tests off without
    # touching a test file.
    "pytest.ini",
    "tox.ini",
    "jest.config.*",
    "vitest.config.*",
    "playwright.config.*",
    "karma.conf.*",
    ".mocharc*",
    "phpunit.xml",
    "phpunit.xml.dist",
)

# Toolchains exposed read-only when they exist. For tools that keep
# credentials or other projects beside their binaries (~/.cargo, ~/go), only
# the bin directory is exposed. Setup gets per-job caches from the runner,
# never the user's own.
_TOOLCHAIN_DIRS = (
    "~/.local/bin",
    "~/.local/share/uv",
    "~/.cargo/bin",
    "~/.rustup",
    "~/.nvm",
    "~/.pyenv",
    "~/go/bin",
    "~/.bun",
    "~/.deno",
    "~/.volta",
    "~/.local/share/pnpm",
)

_PHP_FRAMEWORKS = (
    ("laravel/framework", "laravel"),
    ("drupal/core-recommended", "drupal"),
    ("drupal/core", "drupal"),
    ("symfony/framework-bundle", "symfony"),
)

_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_COMMIT = re.compile(r"[0-9a-fA-F]{7,64}")
_TOML_TABLE = re.compile(r"^\s*\[\[?([^\[\]]+)\]\]?\s*(?:#.*)?$")
_TOML_DEV_KEY = re.compile(r"""^\s*["']?dev["']?\s*=""")
# A `test` target, alone or among others, but not `test-x:`, `test :=` or `.PHONY: test`.
_MAKE_TEST = re.compile(r"^(?:[^\s:#=]+[ \t]+)*test(?:[ \t]+[^\s:#=]+)*[ \t]*(?::(?![:=])|::(?!=))", re.M)


class ProfileError(ValueError):
    """A stored or in-repo profile could not be read, or is invalid."""


def _repo_relative(value: str) -> str:
    if not value or "\0" in value:
        raise ValueError("paths must be non-empty")
    if value.startswith(("/", "~")):
        raise ValueError(f"{value!r} must be relative to the repo root")
    parts = value.split("/")
    if ".." in parts:
        raise ValueError(f"{value!r} must not contain '..'")
    if "" in parts or "." in parts:
        normal = posixpath.normpath(value)
        hint = f"; use {normal!r}" if normal != "." else ""
        raise ValueError(f"{value!r} is not normalised{hint}")
    return value


def _home_or_absolute(value: str) -> str:
    if "\0" in value or not value.startswith(("/", "~/")):
        raise ValueError(f"{value!r} must be absolute or start with ~/")
    if ".." in value.split("/"):
        raise ValueError(f"{value!r} must not contain '..'")
    return value


class Checks(BaseModel):
    """Gate commands; `None` means the check is not configured.

    Test commands may write junit XML to `$MULLET_JUNIT`, and `{tests}` in
    `test_targeted` is replaced by space-separated, shell-quoted test paths.
    """

    model_config = ConfigDict(extra="forbid")

    lint: str | None = None
    typecheck: str | None = None
    build: str | None = None
    test_full: str | None = None
    test_targeted: str | None = None
    junit: bool = False

    @field_validator("lint", "typecheck", "build", "test_full", "test_targeted")
    @classmethod
    def _command(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None or not value.strip():
            return None
        if info.field_name == "test_targeted" and "{tests}" not in value:
            raise ValueError("test_targeted must contain {tests}")
        return value


class Timeouts(BaseModel):
    """Wall-clock limits, in seconds."""

    model_config = ConfigDict(extra="forbid")

    setup: int = Field(default=1800, ge=1)
    check: int = Field(default=900, ge=1)
    test_full: int = Field(default=3600, ge=1)


class Profile(BaseModel):
    """How to set up one repo's clone, and which commands are its gates."""

    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    repo_id: str
    language: str = "unknown"
    framework: str = ""
    setup: list[str] = Field(default_factory=list)
    checks: Checks = Field(default_factory=Checks)
    timeouts: Timeouts = Field(default_factory=Timeouts)
    protected_paths: list[str] = Field(default_factory=lambda: list(DEFAULT_PROTECTED))
    deps_paths: list[str] = Field(default_factory=list)
    data_class: Literal["private", "public"] = "private"
    copy_in: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    expose_ro: list[str] = Field(default_factory=list)
    setup_writable: list[str] = Field(default_factory=list)
    # Executors whose model provider may see this repo's code. Claude only by
    # default; only the user's stored profile can add to it.
    allowed_executors: list[str] = Field(default_factory=lambda: ["claude-code"])
    formatter: bool = False  # let an external executor run code formatters on what it edits
    # Language servers for the opencode worker. Off by default: they run beside
    # opencode, outside the worker's no-network shell, and execute repo code.
    # Only the user's stored profile can turn them on.
    opencode_lsp: bool = False
    # The delegation ladder (mullet.jobs.ladder): `auto` lets mechanical steps
    # go to cheaper workers the profile allows; `claude-only` keeps every step
    # on Claude. A repo's own file can only choose claude-only.
    ladder: Literal["auto", "claude-only"] = "auto"
    # A repo-relative Markdown file of coding standards, shown to the G7
    # reviewer only (never the worker), read from the job's base commit.
    review_standards: str | None = None

    @field_validator("review_standards")
    @classmethod
    def _standards_path(cls, value: str | None) -> str | None:
        return None if value is None else _repo_relative(value)

    @field_validator("allowed_executors")
    @classmethod
    def _executor_names(cls, values: list[str]) -> list[str]:
        for value in values:
            if not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", value):
                raise ValueError(f"{value!r} is not an executor name")
        return list(dict.fromkeys(values))

    @field_validator("repo_id")
    @classmethod
    def _repo_id(cls, value: str) -> str:
        if not _REPO_ID.fullmatch(value):
            raise ValueError(f"{value!r} is not a safe repo id")
        return value

    @field_validator("protected_paths", "deps_paths", "copy_in")
    @classmethod
    def _relative_paths(cls, values: list[str]) -> list[str]:
        return [_repo_relative(v) for v in values]

    @field_validator("expose_ro", "setup_writable")
    @classmethod
    def _host_paths(cls, values: list[str]) -> list[str]:
        return [_home_or_absolute(v) for v in values]

    @field_validator("env")
    @classmethod
    def _env(cls, env: dict[str, str]) -> dict[str, str]:
        for key, value in env.items():
            if not _ENV_KEY.fullmatch(key):
                raise ValueError(f"env key {key!r} is not a valid shell identifier")
            upper = key.upper()
            if upper in ("PATH", "HOME") or upper.startswith("MULLET_"):
                raise ValueError(f"env must not set {key}")
            if "\0" in value:
                raise ValueError(f"env value for {key} contains a NUL byte")
        return env


# Never passed on from a profile's env, which a repo can set: these change
# how programs load code, where their traffic goes, or which agent config and
# credentials apply.
_UNSAFE_ENV_PREFIXES: tuple[str, ...] = ("LD_", "DYLD_", "OPENCODE_", "ANTHROPIC_", "CLAUDE_")
_UNSAFE_ENV_NAMES = frozenset({"NODE_OPTIONS", "BUN_OPTIONS"})


def unsafe_env_name(name: str) -> bool:
    """True for variables a profile may not pass to any command Mullet runs."""
    upper = name.upper()
    return upper in _UNSAFE_ENV_NAMES or upper.endswith("_PROXY") or upper.startswith(_UNSAFE_ENV_PREFIXES)


def safe_env(env: dict[str, str]) -> dict[str, str]:
    """A profile's env without the variables in `unsafe_env_name`, for check commands and worker shells."""
    dropped = sorted(k for k in env if unsafe_env_name(k))
    if dropped:
        logger.warning("profile env: ignoring %s, which a profile may not set", ", ".join(dropped))
    return {k: v for k, v in env.items() if not unsafe_env_name(k)}


def expand_home(p: str) -> Path:
    """Expand a leading `~/` to the user's home directory; other paths are returned unchanged."""
    if p == "~":
        return Path.home()
    if p.startswith("~/"):
        return Path.home() / p[2:]
    return Path(p)


def repo_id(repo: Path) -> str:
    """Filename-safe id for a repo: its sanitised directory name and a hash of its resolved path."""
    resolved = Path(repo).expanduser().resolve()
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", resolved.name).strip("._-")[:48].rstrip("._-")
    digest = hashlib.sha1(os.fsencode(str(resolved)), usedforsecurity=False).hexdigest()[:8]
    return f"{name or 'repo'}-{digest}"


def profile_path(rid: str) -> Path:
    """Where the user's stored profile for repo id `rid` lives."""
    if not _REPO_ID.fullmatch(rid):
        raise ValueError(f"{rid!r} is not a safe repo id")
    return profiles_dir() / f"{rid}.yaml"


def detect_profile(repo: Path) -> Profile:
    """Build a profile from what the repo's files show.

    Detection reads the working tree, so run it on a tree no worker has touched.
    """
    root = Path(repo).expanduser().resolve()
    info = detect_project(str(root))
    found: dict[str, Any] = {"language": info.language, "framework": info.framework}
    if info.language == "python":
        found.update(_python_defaults(root))
    elif (root / "composer.json").is_file():
        found.update(_php_defaults(root))
    elif info.language in ("javascript", "typescript"):
        found.update(_node_defaults(root, info.package_manager))
    elif info.language == "go":
        found.update(setup=["go mod download"], checks={"test_full": "go test ./...", "lint": "go vet ./..."})
    elif info.language == "rust":
        rust_checks = {"test_full": "cargo test", "build": "cargo build"}
        if shutil.which("cargo-clippy"):
            rust_checks["lint"] = "cargo clippy --all-targets -- -D warnings"
        found.update(setup=["cargo fetch"], checks=rust_checks)
    checks = found.setdefault("checks", {})
    if checks.get("test_full") is None and _make_has_test(root):
        checks["test_full"] = "make test"
    return Profile(
        repo_id=repo_id(root),
        expose_ro=[p for p in _TOOLCHAIN_DIRS if expand_home(p).is_dir()],
        setup_writable=[],
        **found,
    )


def load_profile(repo: Path, base_sha: str | None = None) -> Profile:
    """Detected defaults, overridden by the stored profile, then by the in-repo `.mullet/project.yaml`.

    With `base_sha` the in-repo file is read from that commit, so only committed
    content counts and working-tree edits change nothing; without it the
    working-tree file is read. The in-repo file may change commands, setup,
    timeouts, deps_paths, env and review_standards, but it can only add protected paths, move
    data_class to private, remove allowed executors and set `ladder: claude-only`, and its expose_ro,
    setup_writable, copy_in and opencode_lsp are ignored. Raises `ProfileError` when a file
    is unreadable or invalid.
    """
    root = Path(repo).expanduser().resolve()
    detected = detect_profile(root)
    merged = detected.model_dump(mode="json")

    stored_path = profile_path(detected.repo_id)
    stored = _read_optional(stored_path)
    if stored is not None:
        merged = _overlay(merged, _parse_layer(stored, str(stored_path)))
        _validate(merged, str(stored_path))

    if base_sha is None:
        source = str(root / REPO_PROFILE)
        raw = _read_working_repo_file(root)
    else:
        source = f"{base_sha}:{REPO_PROFILE}"
        raw = _read_committed_repo_file(root, base_sha)
    if raw is not None:
        merged = _apply_repo_layer(merged, _parse_layer(raw, source), source)
    return _validate(merged, source)


def user_profile(repo: Path) -> Profile:
    """The profile without the repo's own `.mullet/project.yaml`: detected defaults and the stored profile.

    What only the user (or Mullet's own detection) wrote, for decisions a repo
    must not be able to make, such as which commands a worker may run as written.
    """
    root = Path(repo).expanduser().resolve()
    detected = detect_profile(root)
    merged = detected.model_dump(mode="json")
    stored_path = profile_path(detected.repo_id)
    stored = _read_optional(stored_path)
    if stored is not None:
        merged = _overlay(merged, _parse_layer(stored, str(stored_path)))
    return _validate(merged, str(stored_path))


def whole_check_commands(profile: Profile) -> list[str]:
    """The profile's whole check commands: lint, typecheck, build and the full test run."""
    checks = profile.checks
    commands = (checks.lint, checks.typecheck, checks.build, checks.test_full)
    return list(dict.fromkeys(c.strip() for c in commands if c and c.strip()))


def save_profile(profile: Profile) -> Path:
    """Write `profile` to the user's profile store atomically and return its path."""
    target = profile_path(profile.repo_id)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(profile_to_yaml(profile))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return target


def profile_to_yaml(profile: Profile) -> str:
    """Render `profile` as YAML in field order, for storing it or showing it in a plan."""
    return yaml.safe_dump(
        profile.model_dump(mode="json"), sort_keys=False, default_flow_style=False, allow_unicode=True
    )


def _python_defaults(root: Path) -> dict[str, Any]:
    tables = _toml_tables(_read_text(root / "pyproject.toml"))
    if (root / "uv.lock").is_file():
        setup = ["uv sync --frozen"]
    else:
        has_dev_extra = any(_TOML_DEV_KEY.match(line) for line in tables.get("project.optional-dependencies", []))
        dev_reqs = " -r requirements-dev.txt" if (root / "requirements-dev.txt").is_file() else ""
        setup = ["python3 -m venv .venv"]
        if has_dev_extra:
            setup.append(".venv/bin/pip install -e '.[dev]'")
        elif (root / "pyproject.toml").is_file() or (root / "setup.py").is_file():
            setup.append(f".venv/bin/pip install -e .{dev_reqs}")
        else:
            setup.append(f".venv/bin/pip install -r requirements.txt{dev_reqs}")
    pytest = '.venv/bin/python -m pytest -q --junitxml="$MULLET_JUNIT"'
    checks: dict[str, Any] = {"test_full": pytest, "test_targeted": f"{pytest} {{tests}}", "junit": True}
    has_ruff = any(name == "tool.ruff" or name.startswith("tool.ruff.") for name in tables)
    if has_ruff or (root / "ruff.toml").is_file() or (root / ".ruff.toml").is_file():
        checks["lint"] = ".venv/bin/ruff check ."
    return {"setup": setup, "deps_paths": [".venv"], "checks": checks}


def _node_defaults(root: Path, pm: str) -> dict[str, Any]:
    package = _read_json(root / "package.json")
    scripts = _mapping(package.get("scripts"))
    if pm != "npm":
        install = f"{pm} install --frozen-lockfile"
    elif (root / "package-lock.json").is_file() or (root / "npm-shrinkwrap.json").is_file():
        install = "npm ci"
    else:
        # Without --no-package-lock, npm would add a protected lockfile to the tree.
        install = "npm install --no-package-lock"
    checks: dict[str, Any] = {}
    test = scripts.get("test")
    if isinstance(test, str) and test.strip() and "no test specified" not in test:
        checks["test_full"] = f"{pm} test"
    if "lint" in scripts:
        checks["lint"] = f"{pm} run lint"
    if "build" in scripts:
        checks["build"] = f"{pm} run build"
    deps = {**_mapping(package.get("dependencies")), **_mapping(package.get("devDependencies"))}
    if (root / "tsconfig.json").is_file() and "typescript" in deps:
        checks["typecheck"] = "npx tsc --noEmit"
    return {"setup": [install], "deps_paths": ["node_modules"], "checks": checks}


def _php_defaults(root: Path) -> dict[str, Any]:
    composer = _read_json(root / "composer.json")
    required = {**_mapping(composer.get("require")), **_mapping(composer.get("require-dev"))}
    framework = next((name for package, name in _PHP_FRAMEWORKS if package in required), "")
    checks: dict[str, Any] = {}
    if (root / "phpunit.xml").is_file() or (root / "phpunit.xml.dist").is_file():
        phpunit = 'vendor/bin/phpunit --log-junit "$MULLET_JUNIT"'
        checks = {"test_full": phpunit, "test_targeted": f"{phpunit} {{tests}}", "junit": True}
    return {
        "language": "php",
        "framework": framework,
        "setup": ["composer install --no-interaction --no-progress"],
        "deps_paths": ["vendor"],
        "checks": checks,
    }


def _make_has_test(root: Path) -> bool:
    return any(_MAKE_TEST.search(_read_text(root / name)) for name in ("GNUmakefile", "makefile", "Makefile"))


def _toml_tables(text: str) -> dict[str, list[str]]:
    # Table name to body lines: enough to spot a table or a key, not a TOML parser.
    tables: dict[str, list[str]] = {"": []}
    current = ""
    for line in text.splitlines():
        match = _TOML_TABLE.match(line)
        if match:
            current = re.sub(r"[\s\"']", "", match.group(1))
            tables.setdefault(current, [])
        else:
            tables[current].append(line)
    return tables


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(_read_text(path) or "{}")
    except ValueError:
        logger.warning("profile: %s is not valid JSON; ignoring it for detection", path)
        return {}
    return _mapping(data)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _read_optional(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:
        raise ProfileError(f"{path}: {exc}") from exc


def _read_working_repo_file(root: Path) -> bytes | None:
    path = root / REPO_PROFILE
    if path.is_symlink() or path.parent.is_symlink():
        raise ProfileError(f"{path}: refusing to follow a symlink")
    return _read_optional(path)


def _read_committed_repo_file(root: Path, base_sha: str) -> bytes | None:
    if not _COMMIT.fullmatch(base_sha):
        raise ProfileError(f"base_sha {base_sha!r} is not a commit hash")
    listing = _git(root, "ls-tree", "-z", f"{base_sha}^{{commit}}", "--", REPO_PROFILE)
    if not listing.stdout:
        return None
    mode, kind, oid = listing.stdout.partition(b"\t")[0].decode().split()
    if kind != "blob" or mode == "120000":
        raise ProfileError(f"{base_sha}:{REPO_PROFILE} is not a regular file")
    return _git(root, "cat-file", "blob", oid).stdout


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", *args],
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProfileError(f"git {args[0]} failed in {root}: {exc}") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise ProfileError(f"git {args[0]} failed in {root}: {stderr}")
    return result


def _parse_layer(raw: bytes, source: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ProfileError(f"{source}: invalid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ProfileError(f"{source}: expected a mapping of profile fields")
    # The id always comes from the repo's path, never from a file.
    data.pop("repo_id", None)
    return data


def _overlay(base: dict[str, Any], layer: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in layer.items():
        if key in ("checks", "timeouts") and isinstance(value, dict):
            merged[key] = {**base.get(key, {}), **value}
        else:
            merged[key] = value
    return merged


def _apply_repo_layer(base: dict[str, Any], layer: dict[str, Any], source: str) -> dict[str, Any]:
    layer = dict(layer)
    # copy_in is here too: a repo's own file must not be able to ask for the
    # user's local secrets (say .env) to be placed where its tests can read them.
    # opencode_lsp runs repo code beside the worker, with network and its key.
    for key in ("expose_ro", "setup_writable", "copy_in", "opencode_lsp"):
        if key in layer:
            layer.pop(key)
            logger.warning("%s: ignoring %s, which only your stored profile can set", source, key)
    extra_protected = layer.pop("protected_paths", None)
    data_class = layer.pop("data_class", None)
    executors = layer.pop("allowed_executors", None)
    ladder = layer.pop("ladder", None)
    merged = _overlay(base, layer)
    if ladder is not None:
        # The repo may keep its code on Claude, never send it further.
        if ladder == "claude-only":
            merged["ladder"] = ladder
        elif ladder != base.get("ladder", "auto"):
            logger.warning("%s: ignoring ladder: %s; only your stored profile can loosen it", source, ladder)
    if extra_protected is not None:
        if not isinstance(extra_protected, list) or not all(isinstance(p, str) for p in extra_protected):
            raise ProfileError(f"{source}: protected_paths must be a list of strings")
        merged["protected_paths"] = list(dict.fromkeys([*base["protected_paths"], *extra_protected]))
    if executors is not None:
        # The repo may narrow the list, never widen it.
        if not isinstance(executors, list) or not all(isinstance(e, str) for e in executors):
            raise ProfileError(f"{source}: allowed_executors must be a list of strings")
        dropped = [e for e in executors if e not in base["allowed_executors"]]
        if dropped:
            logger.warning("%s: ignoring allowed_executors %s; only your stored profile can allow them",
                           source, ", ".join(dropped))
        merged["allowed_executors"] = [e for e in base["allowed_executors"] if e in executors]
    if data_class is not None:
        if data_class == "public" and base["data_class"] != "public":
            logger.warning("%s: ignoring data_class: public; only your stored profile can set it", source)
        else:
            merged["data_class"] = data_class
    return merged


def _validate(data: dict[str, Any], source: str) -> Profile:
    try:
        return Profile.model_validate(data)
    except ValidationError as exc:
        raise ProfileError(f"{source}: {exc}") from exc
