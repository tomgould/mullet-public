"""Tests for json_extractor pure helpers and github_connector filter logic."""


from mullet.knowledge.code.extractors.json_extractor import (
    _strip_jsonc,
    JSONExtractor,
)
from mullet.knowledge.sources.github_connector import GitHubSourceConnector
from mullet.knowledge.sources.github_models import GitHubSourceConfig, RepoInfo


# ---------------------------------------------------------------------------
# _strip_jsonc
# ---------------------------------------------------------------------------

class TestStripJsonc:
    def test_plain_json_unchanged(self):
        src = '{"key": "value"}'
        result = _strip_jsonc(src)
        assert result == src

    def test_single_line_comment_stripped(self):
        src = '{"key": "value"} // this is a comment'
        result = _strip_jsonc(src)
        assert "//" not in result
        assert '"key"' in result

    def test_block_comment_stripped(self):
        src = '/* header */ {"key": "value"}'
        result = _strip_jsonc(src)
        assert "/*" not in result
        assert '"key"' in result

    def test_trailing_comma_removed(self):
        src = '{"a": 1, "b": 2,}'
        result = _strip_jsonc(src)
        # Should be parseable as JSON after stripping
        import json
        data = json.loads(result)
        assert data == {"a": 1, "b": 2}

    def test_trailing_comma_in_array(self):
        src = '[1, 2, 3,]'
        result = _strip_jsonc(src)
        import json
        data = json.loads(result)
        assert data == [1, 2, 3]

    def test_comment_inside_string_preserved(self):
        src = '{"url": "http://example.com"}'
        result = _strip_jsonc(src)
        assert "http://example.com" in result

    def test_empty_string_returns_empty(self):
        assert _strip_jsonc("") == ""

    def test_returns_string(self):
        assert isinstance(_strip_jsonc('{"a":1}'), str)

    def test_multiline_block_comment(self):
        src = '{\n/* this is\na block comment */\n"key": "val"\n}'
        result = _strip_jsonc(src)
        assert "block comment" not in result
        assert '"key"' in result

    def test_double_slash_in_string_preserved(self):
        src = '{"path": "//double/slash"}'
        result = _strip_jsonc(src)
        assert "//double/slash" in result


# ---------------------------------------------------------------------------
# JSONExtractor._detect_json_type
# ---------------------------------------------------------------------------

class TestDetectJsonType:
    def setup_method(self):
        self.extractor = JSONExtractor()

    def test_composer_json_by_filename(self):
        result = self.extractor._detect_json_type({}, "composer.json")
        assert result == "composer"

    def test_composer_lock_by_filename(self):
        result = self.extractor._detect_json_type({}, "composer.lock")
        assert result == "composer-lock"

    def test_package_json_by_filename(self):
        result = self.extractor._detect_json_type({}, "package.json")
        assert result == "package"

    def test_package_lock_by_filename(self):
        result = self.extractor._detect_json_type({}, "package-lock.json")
        assert result == "package-lock"

    def test_tsconfig_by_filename(self):
        result = self.extractor._detect_json_type({}, "tsconfig.json")
        assert result == "tsconfig"

    def test_jsconfig_by_filename(self):
        result = self.extractor._detect_json_type({}, "jsconfig.json")
        assert result == "tsconfig"

    def test_generic_unknown_filename(self):
        result = self.extractor._detect_json_type({"foo": "bar"}, "myfile.json")
        assert result == "generic"

    def test_tsconfig_by_content(self):
        data = {"compilerOptions": {"target": "es6"}}
        result = self.extractor._detect_json_type(data, "config.json")
        assert result == "tsconfig"

    def test_composer_by_content(self):
        data = {"require": {"php": "^8.0"}, "autoload": {"psr-4": {}}}
        result = self.extractor._detect_json_type(data, "unknown.json")
        assert result == "composer"

    def test_returns_string(self):
        result = self.extractor._detect_json_type({}, "any.json")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# JSONExtractor.extract
# ---------------------------------------------------------------------------

class TestJSONExtractorExtract:
    def setup_method(self):
        self.extractor = JSONExtractor()

    def test_language_is_json(self):
        assert self.extractor.language == "json"

    def test_empty_source_returns_chunk(self):
        result = self.extractor.extract("{}", "test.json")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_invalid_json_returns_raw_chunk(self):
        result = self.extractor.extract("not json", "test.json")
        assert len(result) == 1
        assert result[0].symbol_type == "document"

    def test_package_json_extracted(self):
        src = '{"name": "my-app", "version": "1.0.0", "dependencies": {}, "scripts": {}}'
        result = self.extractor.extract(src, "package.json")
        assert len(result) >= 1
        types = {c.symbol_type for c in result}
        assert len(types) > 0

    def test_generic_small_json_returns_document(self):
        src = '{"key": "value"}'
        result = self.extractor.extract(src, "settings.json")
        assert result[0].symbol_type == "document"

    def test_chunk_has_source_file(self):
        result = self.extractor.extract('{"a":1}', "my.json")
        assert all(c.source_file == "my.json" for c in result)

    def test_tsconfig_json_extracted(self):
        src = '{"compilerOptions": {"target": "es6"}, "include": ["src"]}'
        result = self.extractor.extract(src, "tsconfig.json")
        assert len(result) >= 1

    def test_jsonc_with_comments_parsed(self):
        src = '{\n  // a comment\n  "key": "value"\n}'
        result = self.extractor.extract(src, "config.json")
        assert len(result) >= 1

    def test_array_json_returns_document(self):
        src = '[1, 2, 3]'
        result = self.extractor.extract(src, "data.json")
        assert result[0].symbol_type == "document"

    def test_returns_list(self):
        assert isinstance(self.extractor.extract("{}", "f.json"), list)


# ---------------------------------------------------------------------------
# GitHubSourceConnector._filter_repos
# ---------------------------------------------------------------------------

def _make_repo(name: str, owner: str = "org") -> RepoInfo:
    return RepoInfo(
        name=name,
        owner=owner,
        url=f"https://github.com/{owner}/{name}",
        clone_url=f"https://github.com/{owner}/{name}.git",
        private=False,
        description="",
        default_branch="main",
        size_kb=0,
        is_fork=False,
    )


def _make_connector(include=None, exclude=None) -> GitHubSourceConnector:
    cfg = GitHubSourceConfig(
        owner="org",
        include_repos=include or [],
        exclude_repos=exclude or [],
    )
    return GitHubSourceConnector(cfg)


class TestFilterRepos:
    def test_no_filter_returns_all(self):
        connector = _make_connector()
        repos = [_make_repo("alpha"), _make_repo("beta")]
        result = connector._filter_repos(repos)
        assert len(result) == 2

    def test_include_filter_keeps_only_matching(self):
        connector = _make_connector(include=["alpha"])
        repos = [_make_repo("alpha"), _make_repo("beta")]
        result = connector._filter_repos(repos)
        assert len(result) == 1
        assert result[0].name == "alpha"

    def test_exclude_filter_removes_matching(self):
        connector = _make_connector(exclude=["beta"])
        repos = [_make_repo("alpha"), _make_repo("beta")]
        result = connector._filter_repos(repos)
        assert len(result) == 1
        assert result[0].name == "alpha"

    def test_include_case_insensitive(self):
        connector = _make_connector(include=["Alpha"])
        repos = [_make_repo("alpha"), _make_repo("gamma")]
        result = connector._filter_repos(repos)
        assert len(result) == 1

    def test_exclude_case_insensitive(self):
        connector = _make_connector(exclude=["BETA"])
        repos = [_make_repo("alpha"), _make_repo("beta")]
        result = connector._filter_repos(repos)
        assert len(result) == 1
        assert result[0].name == "alpha"

    def test_empty_repo_list_returns_empty(self):
        connector = _make_connector(include=["anything"])
        result = connector._filter_repos([])
        assert result == []

    def test_both_filters_applied(self):
        connector = _make_connector(include=["alpha", "beta"], exclude=["beta"])
        repos = [_make_repo("alpha"), _make_repo("beta"), _make_repo("gamma")]
        result = connector._filter_repos(repos)
        assert len(result) == 1
        assert result[0].name == "alpha"
