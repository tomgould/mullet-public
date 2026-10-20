"""Tests for pure data model classes across the codebase.

Covers:
- mullet.knowledge.engine.logging_config (AuditConfig, load_audit_config)
- mullet.knowledge.sources.github_models (GitHubSourceConfig, RepoInfo)
- mullet.core.orchestration.swarm_types (SwarmTask, SwarmContext, TaskResult, SwarmResult)
"""
from __future__ import annotations

import pytest

from mullet.knowledge.engine.logging_config import (
    AuditConfig,
    get_default_audit_config,
    load_audit_config,
)
from mullet.knowledge.sources.github_models import GitHubSourceConfig, RepoInfo
from mullet.core.orchestration.swarm_types import (
    SwarmContext,
    SwarmResult,
    SwarmTask,
)


# ---------------------------------------------------------------------------
# AuditConfig
# ---------------------------------------------------------------------------

class TestAuditConfig:
    def test_default_should_log_is_true(self) -> None:
        c = AuditConfig()
        assert c.should_log() is True

    def test_disabled_config_should_not_log(self) -> None:
        c = AuditConfig(enabled=False)
        assert c.should_log() is False

    def test_log_level_none_should_not_log(self) -> None:
        c = AuditConfig(enabled=True, log_level="none")
        assert c.should_log() is False

    def test_is_minimal(self) -> None:
        c = AuditConfig(log_level="minimal")
        assert c.is_minimal() is True
        assert c.is_full() is False

    def test_is_full(self) -> None:
        c = AuditConfig(log_level="full")
        assert c.is_full() is True
        assert c.is_minimal() is False

    def test_default_log_level_is_full(self) -> None:
        c = AuditConfig()
        assert c.is_full() is True


class TestLoadAuditConfig:
    def test_load_with_all_fields(self) -> None:
        cfg = load_audit_config({
            "audit": {
                "enabled": False,
                "log_level": "minimal",
                "include_chunks": False,
                "include_scores": False,
                "include_timing": False,
                "retention_days": 30,
            }
        })
        assert cfg.enabled is False
        assert cfg.log_level == "minimal"
        assert cfg.include_chunks is False
        assert cfg.retention_days == 30

    def test_load_missing_audit_section_uses_defaults(self) -> None:
        cfg = load_audit_config({})
        assert cfg.enabled is True
        assert cfg.log_level == "full"
        assert cfg.retention_days is None

    def test_load_partial_audit_section(self) -> None:
        cfg = load_audit_config({"audit": {"enabled": False}})
        assert cfg.enabled is False
        assert cfg.include_chunks is True  # default

    def test_get_default_audit_config(self) -> None:
        cfg = get_default_audit_config()
        assert cfg.enabled is True
        assert cfg.should_log() is True


# ---------------------------------------------------------------------------
# GitHubSourceConfig
# ---------------------------------------------------------------------------

class TestGitHubSourceConfig:
    def test_owner_repo_shorthand_parsed(self) -> None:
        cfg = GitHubSourceConfig(repo="anthropics/claude-code")
        assert cfg.owner == "anthropics"
        assert cfg.repo == "claude-code"

    def test_owner_not_overridden_when_set(self) -> None:
        cfg = GitHubSourceConfig(owner="my-org", repo="my-org/my-repo")
        # owner is already set — shorthand parsing skips
        assert cfg.owner == "my-org"
        assert cfg.repo == "my-org/my-repo"

    def test_from_dict(self) -> None:
        cfg = GitHubSourceConfig.from_dict({
            "owner": "acme",
            "repo": "api",
            "token_env": "GITHUB_TOKEN",
            "include_repos": ["repo1"],
            "exclude_repos": ["private-repo"],
            "include_forks": True,
            "include_private": False,
            "sync_on_ingest": False,
        })
        assert cfg.owner == "acme"
        assert cfg.repo == "api"
        assert cfg.token_env == "GITHUB_TOKEN"
        assert cfg.include_repos == ["repo1"]
        assert cfg.include_forks is True
        assert cfg.include_private is False
        assert cfg.sync_on_ingest is False

    def test_from_dict_with_defaults(self) -> None:
        cfg = GitHubSourceConfig.from_dict({})
        assert cfg.owner == ""
        assert cfg.include_forks is False
        assert cfg.include_private is True
        assert cfg.sync_on_ingest is True

    def test_token_reads_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("MY_GH_TOKEN", "ghp_secret123")
        cfg = GitHubSourceConfig(token_env="MY_GH_TOKEN")
        assert cfg.token == "ghp_secret123"

    def test_token_returns_none_when_no_env_var(self, monkeypatch) -> None:
        monkeypatch.delenv("UNSET_TOKEN_VAR", raising=False)
        cfg = GitHubSourceConfig(token_env="UNSET_TOKEN_VAR")
        assert cfg.token is None

    def test_token_returns_none_when_token_env_empty(self) -> None:
        cfg = GitHubSourceConfig()
        assert cfg.token is None


class TestRepoInfo:
    def test_from_api_response(self) -> None:
        data = {
            "name": "my-repo",
            "owner": {"login": "acme"},
            "html_url": "https://github.com/acme/my-repo",
            "clone_url": "https://github.com/acme/my-repo.git",
            "private": True,
            "description": "My repo description",
            "default_branch": "main",
            "size": 1234,
            "fork": False,
        }
        r = RepoInfo.from_api_response(data)
        assert r.name == "my-repo"
        assert r.owner == "acme"
        assert r.private is True
        assert r.size_kb == 1234
        assert r.is_fork is False
        assert r.default_branch == "main"

    def test_from_api_response_null_description(self) -> None:
        data = {
            "name": "x",
            "owner": {"login": "org"},
            "description": None,
        }
        r = RepoInfo.from_api_response(data)
        assert r.description == ""

    def test_from_api_response_defaults(self) -> None:
        data = {"name": "minimal", "owner": {"login": "u"}}
        r = RepoInfo.from_api_response(data)
        assert r.default_branch == "main"
        assert r.is_fork is False
        assert r.private is False


# ---------------------------------------------------------------------------
# SwarmTypes
# ---------------------------------------------------------------------------

class TestSwarmTask:
    def test_defaults(self) -> None:
        t = SwarmTask(id="t1", instruction="Do something")
        assert t.context_refs == []
        assert t.expected_output_shape == "markdown report"
        assert t.tier == "STANDARD"
        assert t.apply is False

    def test_explicit_values(self) -> None:
        t = SwarmTask(id="t2", instruction="Build it", tier="LOW", apply=True)
        assert t.tier == "LOW"
        assert t.apply is True


class TestSwarmContext:
    def test_valid_construction(self) -> None:
        ctx = SwarmContext(files=["a.py", "b.py"], items=["item1"])
        assert ctx.files == ["a.py", "b.py"]
        assert ctx.items == ["item1"]

    def test_invalid_files_type_raises(self) -> None:
        with pytest.raises(TypeError, match="files must be list"):
            SwarmContext(files="not-a-list")

    def test_invalid_items_type_raises(self) -> None:
        with pytest.raises(TypeError, match="items must be list"):
            SwarmContext(items="not-a-list")

    def test_non_string_in_files_raises(self) -> None:
        with pytest.raises(TypeError, match="files\\[0\\]"):
            SwarmContext(files=[123])

    def test_non_string_in_items_raises(self) -> None:
        with pytest.raises(TypeError, match="items\\[0\\]"):
            SwarmContext(items=[None])

    def test_defaults(self) -> None:
        ctx = SwarmContext()
        assert ctx.files == []
        assert ctx.output_shape == "markdown report"
        assert ctx.concurrency == 20
        assert ctx.apply is False


class TestSwarmResult:
    def test_minimal_construction(self) -> None:
        r = SwarmResult(goal="test goal")
        assert r.tasks == []
        assert r.results == []
        assert r.synthesis == ""
        assert r.fell_back is False
        assert r.partial_failure is False
        assert r.wall_clock_seconds == 0.0
