"""The one .env loader (mullet.core.env) and the local LLM URL settings.

Precedence: the process environment, then config_dir()/.env, then the
source checkout's .env (only for a git checkout of Mullet itself). The
current directory's .env is never read.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from mullet.core import env as mullet_env
from mullet.core.paths import config_dir


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def checkout(tmp_path, monkeypatch) -> Path:
    """A fake checkout whose .env the loader treats as Mullet's own."""
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(mullet_env, "source_checkout", lambda: root)
    return root


@pytest.fixture
def no_checkout(monkeypatch) -> None:
    monkeypatch.setattr(mullet_env, "source_checkout", lambda: None)


def _fake_package(root: Path, monkeypatch) -> None:
    """Make mullet.core.env believe it lives at root/mullet/core/env.py."""
    (root / "mullet" / "core").mkdir(parents=True)
    monkeypatch.setattr(mullet_env, "__file__", str(root / "mullet" / "core" / "env.py"))


# ---------------------------------------------------------------------------
# Detecting a source checkout
# ---------------------------------------------------------------------------

class TestSourceCheckout:
    def test_a_git_checkout_of_mullet(self, tmp_path, monkeypatch):
        _fake_package(tmp_path, monkeypatch)
        (tmp_path / ".git").mkdir()
        _write(tmp_path / "pyproject.toml", '[project]\nname = "mullet"\nversion = "1"\n')
        assert mullet_env.source_checkout() == tmp_path

    def test_a_git_worktree_has_a_dot_git_file(self, tmp_path, monkeypatch):
        _fake_package(tmp_path, monkeypatch)
        _write(tmp_path / ".git", "gitdir: /elsewhere\n")
        _write(tmp_path / "pyproject.toml", "[project]\nname = 'mullet'\n")
        assert mullet_env.source_checkout() == tmp_path

    def test_an_installed_copy_has_no_git(self, tmp_path, monkeypatch):
        _fake_package(tmp_path, monkeypatch)
        _write(tmp_path / "pyproject.toml", '[project]\nname = "mullet"\n')
        assert mullet_env.source_checkout() is None

    def test_another_project_is_not_mullet(self, tmp_path, monkeypatch):
        _fake_package(tmp_path, monkeypatch)
        (tmp_path / ".git").mkdir()
        _write(tmp_path / "pyproject.toml", '[project]\nname = "mullet-plugin"\n')
        assert mullet_env.source_checkout() is None

    def test_no_pyproject_is_not_a_checkout(self, tmp_path, monkeypatch):
        _fake_package(tmp_path, monkeypatch)
        (tmp_path / ".git").mkdir()
        assert mullet_env.source_checkout() is None


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------

class TestLoadDotenv:
    def test_files_in_precedence_order(self, checkout):
        user = _write(config_dir() / ".env", "A=1\n")
        repo = _write(checkout / ".env", "A=2\n")
        assert mullet_env.dotenv_files() == [user, repo]

    def test_missing_files_are_skipped(self, checkout):
        assert mullet_env.dotenv_files() == []

    def test_environment_beats_config_beats_checkout(self, checkout, monkeypatch):
        for name in ("MT_ENV_ONLY", "MT_BOTH", "MT_ALL", "MT_REPO_ONLY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("MT_ALL", "from-process")
        _write(config_dir() / ".env", "MT_BOTH=from-config\nMT_ALL=from-config\n")
        _write(checkout / ".env", "MT_BOTH=from-checkout\nMT_ALL=from-checkout\nMT_REPO_ONLY=from-checkout\n")

        loaded = mullet_env.load_dotenv()

        assert loaded == [config_dir() / ".env", checkout / ".env"]
        assert os.environ["MT_ALL"] == "from-process"
        assert os.environ["MT_BOTH"] == "from-config"
        assert os.environ["MT_REPO_ONLY"] == "from-checkout"
        for name in ("MT_BOTH", "MT_ALL", "MT_REPO_ONLY"):
            monkeypatch.delenv(name, raising=False)

    def test_an_installed_copy_reads_only_the_config_file(self, no_checkout, monkeypatch):
        monkeypatch.delenv("MT_INSTALLED", raising=False)
        _write(config_dir() / ".env", "MT_INSTALLED=yes\n")
        assert mullet_env.load_dotenv() == [config_dir() / ".env"]
        assert os.environ["MT_INSTALLED"] == "yes"
        monkeypatch.delenv("MT_INSTALLED", raising=False)

    def test_follows_mullet_config_dir(self, no_checkout, monkeypatch, tmp_path):
        monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "cfg"))
        monkeypatch.delenv("MT_CUSTOM", raising=False)
        _write(tmp_path / "cfg" / ".env", "MT_CUSTOM=1\n")
        mullet_env.load_dotenv()
        assert os.environ["MT_CUSTOM"] == "1"
        monkeypatch.delenv("MT_CUSTOM", raising=False)

    def test_never_reads_the_current_directory(self, no_checkout, monkeypatch, tmp_path):
        project = tmp_path / "some-project"
        _write(project / ".env", "MT_FROM_CWD=leaked\nOPENROUTER_API_KEY=leaked\n")
        monkeypatch.chdir(project)
        monkeypatch.delenv("MT_FROM_CWD", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        assert mullet_env.load_dotenv() == []
        assert "MT_FROM_CWD" not in os.environ
        assert "OPENROUTER_API_KEY" not in os.environ
        assert mullet_env.dotenv_value("MT_FROM_CWD") is None

    def test_never_reads_the_legacy_dotenv(self, no_checkout, monkeypatch):
        _write(Path.home() / ".mullet" / ".env", "MT_LEGACY=1\n")
        monkeypatch.delenv("MT_LEGACY", raising=False)
        assert mullet_env.load_dotenv() == []
        assert "MT_LEGACY" not in os.environ

    def test_values_are_never_printed(self, checkout, monkeypatch, capsys):
        monkeypatch.delenv("MT_SECRETISH", raising=False)
        _write(config_dir() / ".env", "MT_SECRETISH=hunter2\nnot a line\n")
        mullet_env.load_dotenv()
        captured = capsys.readouterr()
        assert "hunter2" not in captured.out + captured.err
        monkeypatch.delenv("MT_SECRETISH", raising=False)


class TestDotenvValue:
    def test_config_file_wins_over_checkout(self, checkout):
        _write(config_dir() / ".env", "MULLET_DELEGATION=off\n")
        _write(checkout / ".env", "MULLET_DELEGATION=on\n")
        assert mullet_env.dotenv_value("MULLET_DELEGATION") == "off"

    def test_falls_through_to_checkout(self, checkout):
        _write(config_dir() / ".env", "OTHER=1\n")
        _write(checkout / ".env", "MULLET_DELEGATION=off\n")
        assert mullet_env.dotenv_value("MULLET_DELEGATION") == "off"

    def test_ignores_the_environment(self, checkout, monkeypatch):
        monkeypatch.setenv("MULLET_DELEGATION", "on")
        assert mullet_env.dotenv_value("MULLET_DELEGATION") is None

    def test_delegation_switch_reads_the_config_file(self, no_checkout, monkeypatch):
        """A hook runs where nothing loaded .env; the switch still comes through."""
        monkeypatch.delenv("MULLET_DELEGATION", raising=False)
        _write(config_dir() / ".env", "MULLET_DELEGATION=off\n")
        assert mullet_env.delegation_enabled() is False


class TestParser:
    CONTENT = (
        "# comment\n"
        "\n"
        "PLAIN=value\n"
        'DOUBLE="double quoted"\n'
        "SINGLE='single quoted'\n"
        "export EXPORTED=yes\n"
        "  SPACED = padded \n"
        "NO_EQUALS_LINE\n"
    )
    EXPECTED = {
        "PLAIN": "value",
        "DOUBLE": "double quoted",
        "SINGLE": "single quoted",
        "EXPORTED": "yes",
        "SPACED": "padded",
    }

    def test_with_python_dotenv(self, tmp_path):
        values = mullet_env.read_dotenv(_write(tmp_path / ".env", self.CONTENT))
        for key, value in self.EXPECTED.items():
            assert values[key] == value
        assert "NO_EQUALS_LINE" not in values

    def test_without_python_dotenv(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "dotenv", None)
        assert mullet_env.read_dotenv(_write(tmp_path / ".env", self.CONTENT)) == self.EXPECTED

    def test_unreadable_file_is_empty(self, tmp_path):
        assert mullet_env.read_dotenv(tmp_path / "missing.env") == {}


class TestEntryPointsShareTheLoader:
    def test_cli_and_mcp_server_use_the_core_loader(self):
        import mullet.cli.main as cli_main
        import mullet.core.providers.budget_probe as budget_probe
        import mullet.server.mcp as mcp_server

        assert cli_main.load_dotenv is mullet_env.load_dotenv
        assert mcp_server.load_dotenv is mullet_env.load_dotenv
        assert budget_probe.load_dotenv is mullet_env.load_dotenv
        assert not hasattr(cli_main, "_load_dotenv")
        assert not hasattr(mcp_server, "_load_dotenv")


# ---------------------------------------------------------------------------
# Local LLM endpoints
# ---------------------------------------------------------------------------

class TestLocalLlmUrls:
    def test_defaults(self):
        assert mullet_env.local_llm_url() == "http://localhost:1234/v1"
        assert mullet_env.local_llm_host() == "http://localhost:1234"
        assert mullet_env.ollama_url() == "http://localhost:11434"

    def test_overrides(self, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_URL", "http://gpu-box:8080/v1/")
        monkeypatch.setenv("OLLAMA_URL", "http://gpu-box:11434/")
        assert mullet_env.local_llm_url() == "http://gpu-box:8080/v1"
        assert mullet_env.local_llm_host() == "http://gpu-box:8080"
        assert mullet_env.ollama_url() == "http://gpu-box:11434"

    def test_url_without_v1_is_its_own_host(self, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_URL", "http://gpu-box:8080")
        assert mullet_env.local_llm_host() == "http://gpu-box:8080"

    def test_consumers_honour_the_setting(self, monkeypatch):
        from mullet.core.providers.lmstudio_manager import LMStudioManager
        from mullet.core.providers.local_detection import _fallback_urls
        from mullet.core.providers.ollama import OllamaConfig
        from mullet.knowledge.engine.llm import LMStudioNativeProvider

        monkeypatch.setenv("LOCAL_LLM_URL", "http://gpu-box:8080/v1")
        monkeypatch.setenv("OLLAMA_URL", "http://ollama-box:9999")
        manager = LMStudioManager()
        assert manager.base_url == "http://gpu-box:8080"
        assert manager.port == 8080
        assert LMStudioNativeProvider().host == "http://gpu-box:8080"
        assert OllamaConfig().base_url == "http://ollama-box:9999"
        assert _fallback_urls()[0] == "http://ollama-box:9999/v1"

    def test_an_explicit_port_still_wins(self, monkeypatch):
        from mullet.core.providers.lmstudio_manager import LMStudioManager
        monkeypatch.setenv("LOCAL_LLM_URL", "http://gpu-box:8080/v1")
        assert LMStudioManager(port=5678).base_url == "http://localhost:5678"
