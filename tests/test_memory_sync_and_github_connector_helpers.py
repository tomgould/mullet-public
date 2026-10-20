"""Tests for memory_sync_cmd pure helpers and github_connector filter helpers."""

import json

from mullet.cli.memory_sync_cmd import (
    _dry_run_observations,
    _dry_run_decisions,
)
from mullet.knowledge.sources.github_connector import (
    GitHubSourceConnector,
    GitHubSourceConfig,
    RepoInfo,
)


# ---------------------------------------------------------------------------
# _dry_run_observations
# ---------------------------------------------------------------------------

class TestDryRunObservations:
    def test_no_summaries_dir_returns_zero(self, tmp_path):
        assert _dry_run_observations(tmp_path) == 0

    def test_empty_summaries_dir_returns_zero(self, tmp_path):
        summaries = tmp_path / ".mullet" / "observations" / "summaries"
        summaries.mkdir(parents=True)
        assert _dry_run_observations(tmp_path) == 0

    def test_counts_json_files(self, tmp_path):
        summaries = tmp_path / ".mullet" / "observations" / "summaries"
        summaries.mkdir(parents=True)
        for i in range(3):
            (summaries / f"obs_{i}.json").write_text(json.dumps({"summary": f"s{i}"}))
        assert _dry_run_observations(tmp_path) == 3

    def test_non_json_not_counted(self, tmp_path):
        summaries = tmp_path / ".mullet" / "observations" / "summaries"
        summaries.mkdir(parents=True)
        (summaries / "obs_0.json").write_text("{}")
        (summaries / "obs_0.txt").write_text("text")
        assert _dry_run_observations(tmp_path) == 1

    def test_returns_int(self, tmp_path):
        assert isinstance(_dry_run_observations(tmp_path), int)


# ---------------------------------------------------------------------------
# _dry_run_decisions
# ---------------------------------------------------------------------------

class TestDryRunDecisions:
    def test_no_decisions_file_returns_zero(self, tmp_path):
        assert _dry_run_decisions(tmp_path) == 0

    def test_empty_decisions_file_returns_zero(self, tmp_path):
        decisions = tmp_path / ".mullet" / "decisions.jsonl"
        decisions.parent.mkdir(parents=True)
        decisions.write_text("")
        assert _dry_run_decisions(tmp_path) == 0

    def test_counts_non_empty_lines(self, tmp_path):
        decisions = tmp_path / ".mullet" / "decisions.jsonl"
        decisions.parent.mkdir(parents=True)
        lines = [
            json.dumps({"kind": "arch", "excerpt": "Use REST", "decision": "REST API"}),
            json.dumps({"kind": "tech", "excerpt": "Use Python", "decision": "Python 3.10"}),
        ]
        decisions.write_text("\n".join(lines))
        assert _dry_run_decisions(tmp_path) == 2

    def test_blank_lines_not_counted(self, tmp_path):
        decisions = tmp_path / ".mullet" / "decisions.jsonl"
        decisions.parent.mkdir(parents=True)
        decisions.write_text(
            json.dumps({"kind": "a", "excerpt": "x"}) + "\n\n"
        )
        assert _dry_run_decisions(tmp_path) == 1

    def test_returns_int(self, tmp_path):
        assert isinstance(_dry_run_decisions(tmp_path), int)


# ---------------------------------------------------------------------------
# GitHubSourceConnector._filter_repos
# ---------------------------------------------------------------------------

def _make_connector(include=None, exclude=None):
    cfg = GitHubSourceConfig(
        owner="testorg",
        include_repos=include or [],
        exclude_repos=exclude or [],
    )
    return GitHubSourceConnector(config=cfg)


def _make_repos(names):
    return [
        RepoInfo(
            name=n,
            owner="testorg",
            url=f"https://github.com/testorg/{n}",
            clone_url=f"https://github.com/testorg/{n}.git",
            private=False,
            description="",
            default_branch="main",
            size_kb=100,
            is_fork=False,
        )
        for n in names
    ]


class TestFilterRepos:
    def test_no_filters_returns_all(self):
        connector = _make_connector()
        repos = _make_repos(["repo1", "repo2", "repo3"])
        filtered = connector._filter_repos(repos)
        assert len(filtered) == 3

    def test_include_filter_matches(self):
        connector = _make_connector(include=["repo1", "repo2"])
        repos = _make_repos(["repo1", "repo2", "repo3"])
        filtered = connector._filter_repos(repos)
        names = {r.name for r in filtered}
        assert "repo1" in names
        assert "repo2" in names
        assert "repo3" not in names

    def test_exclude_filter_removes_repos(self):
        connector = _make_connector(exclude=["repo3"])
        repos = _make_repos(["repo1", "repo2", "repo3"])
        filtered = connector._filter_repos(repos)
        names = {r.name for r in filtered}
        assert "repo3" not in names
        assert "repo1" in names

    def test_include_case_insensitive(self):
        connector = _make_connector(include=["REPO1"])
        repos = _make_repos(["repo1"])
        filtered = connector._filter_repos(repos)
        assert len(filtered) == 1

    def test_exclude_case_insensitive(self):
        connector = _make_connector(exclude=["REPO3"])
        repos = _make_repos(["repo1", "repo2", "repo3"])
        filtered = connector._filter_repos(repos)
        names = {r.name for r in filtered}
        assert "repo3" not in names

    def test_empty_repos_returns_empty(self):
        connector = _make_connector()
        assert connector._filter_repos([]) == []

    def test_returns_list(self):
        connector = _make_connector()
        assert isinstance(connector._filter_repos([]), list)
