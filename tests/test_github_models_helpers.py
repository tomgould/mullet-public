"""Tests for GitHubSourceConfig and RepoInfo pure data models."""

import os

from mullet.knowledge.sources.github_models import GitHubSourceConfig, RepoInfo


# ---------------------------------------------------------------------------
# GitHubSourceConfig — construction and defaults
# ---------------------------------------------------------------------------

class TestGitHubSourceConfigDefaults:
    def test_owner_defaults_empty(self):
        c = GitHubSourceConfig()
        assert c.owner == ""

    def test_repo_defaults_empty(self):
        c = GitHubSourceConfig()
        assert c.repo == ""

    def test_token_env_defaults_empty(self):
        c = GitHubSourceConfig()
        assert c.token_env == ""

    def test_include_repos_defaults_empty(self):
        c = GitHubSourceConfig()
        assert c.include_repos == []

    def test_exclude_repos_defaults_empty(self):
        c = GitHubSourceConfig()
        assert c.exclude_repos == []

    def test_include_forks_defaults_false(self):
        c = GitHubSourceConfig()
        assert c.include_forks is False

    def test_include_private_defaults_true(self):
        c = GitHubSourceConfig()
        assert c.include_private is True

    def test_sync_on_ingest_defaults_true(self):
        c = GitHubSourceConfig()
        assert c.sync_on_ingest is True


# ---------------------------------------------------------------------------
# GitHubSourceConfig — shorthand repo parsing
# ---------------------------------------------------------------------------

class TestGitHubSourceConfigShorthand:
    def test_shorthand_parses_owner(self):
        c = GitHubSourceConfig(repo="myowner/myrepo")
        assert c.owner == "myowner"

    def test_shorthand_parses_repo(self):
        c = GitHubSourceConfig(repo="myowner/myrepo")
        assert c.repo == "myrepo"

    def test_no_slash_no_owner_set(self):
        c = GitHubSourceConfig(repo="noslash")
        assert c.owner == ""
        assert c.repo == "noslash"

    def test_existing_owner_not_overwritten(self):
        c = GitHubSourceConfig(owner="existing", repo="other/repo")
        assert c.owner == "existing"
        assert c.repo == "other/repo"

    def test_triple_slash_splits_once(self):
        c = GitHubSourceConfig(repo="owner/repo/extra")
        assert c.owner == "owner"
        assert c.repo == "repo/extra"

    def test_empty_repo_no_shorthand(self):
        c = GitHubSourceConfig(repo="")
        assert c.owner == ""


# ---------------------------------------------------------------------------
# GitHubSourceConfig — token property
# ---------------------------------------------------------------------------

class TestGitHubSourceConfigToken:
    def test_no_token_env_returns_none(self):
        c = GitHubSourceConfig()
        assert c.token is None

    def test_token_env_set_returns_env_value(self):
        os.environ["TEST_GH_TOKEN_MULLET"] = "secret123"
        c = GitHubSourceConfig(token_env="TEST_GH_TOKEN_MULLET")
        assert c.token == "secret123"
        del os.environ["TEST_GH_TOKEN_MULLET"]

    def test_token_env_missing_from_env_returns_none(self):
        os.environ.pop("MULLET_NONEXISTENT_TOKEN_XYZ", None)
        c = GitHubSourceConfig(token_env="MULLET_NONEXISTENT_TOKEN_XYZ")
        assert c.token is None

    def test_token_returns_string_or_none(self):
        c = GitHubSourceConfig(token_env="")
        result = c.token
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# GitHubSourceConfig — from_dict
# ---------------------------------------------------------------------------

class TestGitHubSourceConfigFromDict:
    def test_owner_set(self):
        c = GitHubSourceConfig.from_dict({"owner": "tg"})
        assert c.owner == "tg"

    def test_repo_set(self):
        c = GitHubSourceConfig.from_dict({"repo": "mullet"})
        assert c.repo == "mullet"

    def test_token_env_set(self):
        c = GitHubSourceConfig.from_dict({"token_env": "MY_TOKEN"})
        assert c.token_env == "MY_TOKEN"

    def test_include_repos_set(self):
        c = GitHubSourceConfig.from_dict({"include_repos": ["repo1", "repo2"]})
        assert c.include_repos == ["repo1", "repo2"]

    def test_exclude_repos_set(self):
        c = GitHubSourceConfig.from_dict({"exclude_repos": ["forked"]})
        assert c.exclude_repos == ["forked"]

    def test_include_forks_set(self):
        c = GitHubSourceConfig.from_dict({"include_forks": True})
        assert c.include_forks is True

    def test_include_private_set_false(self):
        c = GitHubSourceConfig.from_dict({"include_private": False})
        assert c.include_private is False

    def test_sync_on_ingest_set_false(self):
        c = GitHubSourceConfig.from_dict({"sync_on_ingest": False})
        assert c.sync_on_ingest is False

    def test_empty_dict_uses_defaults(self):
        c = GitHubSourceConfig.from_dict({})
        assert c.owner == ""
        assert c.include_forks is False
        assert c.include_private is True

    def test_returns_instance(self):
        c = GitHubSourceConfig.from_dict({})
        assert isinstance(c, GitHubSourceConfig)


# ---------------------------------------------------------------------------
# RepoInfo — from_api_response
# ---------------------------------------------------------------------------

def _api_data(**overrides):
    base = {
        "name": "myrepo",
        "owner": {"login": "tg"},
        "html_url": "https://github.com/tg/myrepo",
        "clone_url": "https://github.com/tg/myrepo.git",
        "private": True,
        "description": "A test repo",
        "default_branch": "main",
        "size": 1234,
        "fork": False,
    }
    base.update(overrides)
    return base


class TestRepoInfoFromApiResponse:
    def test_name_stored(self):
        ri = RepoInfo.from_api_response(_api_data())
        assert ri.name == "myrepo"

    def test_owner_from_login(self):
        ri = RepoInfo.from_api_response(_api_data())
        assert ri.owner == "tg"

    def test_url_stored(self):
        ri = RepoInfo.from_api_response(_api_data())
        assert ri.url == "https://github.com/tg/myrepo"

    def test_clone_url_stored(self):
        ri = RepoInfo.from_api_response(_api_data())
        assert ri.clone_url == "https://github.com/tg/myrepo.git"

    def test_private_stored(self):
        ri = RepoInfo.from_api_response(_api_data())
        assert ri.private is True

    def test_description_stored(self):
        ri = RepoInfo.from_api_response(_api_data())
        assert ri.description == "A test repo"

    def test_null_description_becomes_empty_string(self):
        ri = RepoInfo.from_api_response(_api_data(description=None))
        assert ri.description == ""

    def test_default_branch_stored(self):
        ri = RepoInfo.from_api_response(_api_data())
        assert ri.default_branch == "main"

    def test_custom_branch(self):
        ri = RepoInfo.from_api_response(_api_data(default_branch="develop"))
        assert ri.default_branch == "develop"

    def test_size_kb_stored(self):
        ri = RepoInfo.from_api_response(_api_data())
        assert ri.size_kb == 1234

    def test_is_fork_false(self):
        ri = RepoInfo.from_api_response(_api_data())
        assert ri.is_fork is False

    def test_is_fork_true(self):
        ri = RepoInfo.from_api_response(_api_data(fork=True))
        assert ri.is_fork is True

    def test_missing_html_url_defaults_empty(self):
        data = _api_data()
        del data["html_url"]
        ri = RepoInfo.from_api_response(data)
        assert ri.url == ""

    def test_missing_clone_url_defaults_empty(self):
        data = _api_data()
        del data["clone_url"]
        ri = RepoInfo.from_api_response(data)
        assert ri.clone_url == ""

    def test_missing_private_defaults_false(self):
        data = _api_data()
        del data["private"]
        ri = RepoInfo.from_api_response(data)
        assert ri.private is False

    def test_missing_size_defaults_zero(self):
        data = _api_data()
        del data["size"]
        ri = RepoInfo.from_api_response(data)
        assert ri.size_kb == 0

    def test_returns_repo_info_instance(self):
        ri = RepoInfo.from_api_response(_api_data())
        assert isinstance(ri, RepoInfo)
