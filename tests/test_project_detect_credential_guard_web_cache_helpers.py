"""Tests for project_detect, credential_guard, and web_cache pure helpers."""

import os

from mullet.core.agent.project_detect import (
    ProjectInfo,
    detect_project,
    format_project_context,
    _file_contains,
)
from mullet.core.tools.credential_guard import (
    CredentialGuard,
    CREDENTIAL_FILENAMES,
    CREDENTIAL_EXTENSIONS,
    CREDENTIAL_PATTERNS,
)
from mullet.knowledge.sources.web_cache import (
    CrawlCache,
    _get_header,
    CACHE_FILENAME,
    CACHE_VERSION,
)


# ---------------------------------------------------------------------------
# ProjectInfo defaults
# ---------------------------------------------------------------------------

class TestProjectInfoDefaults:
    def test_language_unknown(self):
        assert ProjectInfo().language == "unknown"

    def test_framework_empty(self):
        assert ProjectInfo().framework == ""

    def test_package_manager_empty(self):
        assert ProjectInfo().package_manager == ""

    def test_test_command_empty(self):
        assert ProjectInfo().test_command == ""

    def test_has_git_false(self):
        assert ProjectInfo().has_git is False

    def test_has_docker_false(self):
        assert ProjectInfo().has_docker is False

    def test_has_ci_false(self):
        assert ProjectInfo().has_ci is False

    def test_entry_points_empty(self):
        assert ProjectInfo().entry_points == []


# ---------------------------------------------------------------------------
# detect_project — language detection
# ---------------------------------------------------------------------------

class TestDetectProjectLanguage:
    def test_empty_dir_unknown(self, tmp_path):
        info = detect_project(str(tmp_path))
        assert info.language == "unknown"

    def test_requirements_txt_is_python(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests\n")
        info = detect_project(str(tmp_path))
        assert info.language == "python"

    def test_setup_py_is_python(self, tmp_path):
        (tmp_path / "setup.py").write_text("from setuptools import setup\n")
        info = detect_project(str(tmp_path))
        assert info.language == "python"

    def test_pyproject_toml_is_python(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[build-system]\n")
        info = detect_project(str(tmp_path))
        assert info.language == "python"

    def test_python_uses_pip(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests\n")
        info = detect_project(str(tmp_path))
        assert info.package_manager == "pip"

    def test_python_test_command(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests\n")
        info = detect_project(str(tmp_path))
        assert "pytest" in info.test_command

    def test_package_json_is_javascript(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "test"}\n')
        info = detect_project(str(tmp_path))
        assert info.language == "javascript"

    def test_tsconfig_makes_typescript(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "test"}\n')
        (tmp_path / "tsconfig.json").write_text("{}\n")
        info = detect_project(str(tmp_path))
        assert info.language == "typescript"

    def test_cargo_toml_is_rust(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname = \"app\"\n")
        info = detect_project(str(tmp_path))
        assert info.language == "rust"

    def test_rust_test_command(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\n")
        info = detect_project(str(tmp_path))
        assert info.test_command == "cargo test"

    def test_go_mod_is_go(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/app\n")
        info = detect_project(str(tmp_path))
        assert info.language == "go"

    def test_pom_xml_is_java_maven(self, tmp_path):
        (tmp_path / "pom.xml").write_text("<project></project>\n")
        info = detect_project(str(tmp_path))
        assert info.language == "java"
        assert info.package_manager == "maven"

    def test_build_gradle_is_java_gradle(self, tmp_path):
        (tmp_path / "build.gradle").write_text("plugins {}\n")
        info = detect_project(str(tmp_path))
        assert info.language == "java"
        assert info.package_manager == "gradle"


class TestDetectProjectFramework:
    def test_manage_py_detects_django(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("django\n")
        (tmp_path / "manage.py").write_text("# django manage\n")
        info = detect_project(str(tmp_path))
        assert info.framework == "django"

    def test_fastapi_in_requirements(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("fastapi\nuvicorn\n")
        info = detect_project(str(tmp_path))
        assert info.framework == "fastapi"


class TestDetectProjectFeatureFlags:
    def test_git_dir_sets_has_git(self, tmp_path):
        (tmp_path / ".git").mkdir()
        info = detect_project(str(tmp_path))
        assert info.has_git is True

    def test_no_git_dir_has_git_false(self, tmp_path):
        info = detect_project(str(tmp_path))
        assert info.has_git is False

    def test_dockerfile_sets_has_docker(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM ubuntu\n")
        info = detect_project(str(tmp_path))
        assert info.has_docker is True

    def test_github_workflows_sets_has_ci(self, tmp_path):
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        info = detect_project(str(tmp_path))
        assert info.has_ci is True


class TestDetectProjectEntryPoints:
    def test_main_py_detected(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("x\n")
        (tmp_path / "main.py").write_text("# main\n")
        info = detect_project(str(tmp_path))
        assert "main.py" in info.entry_points

    def test_app_py_detected(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("x\n")
        (tmp_path / "app.py").write_text("# app\n")
        info = detect_project(str(tmp_path))
        assert "app.py" in info.entry_points

    def test_no_entry_points_when_absent(self, tmp_path):
        info = detect_project(str(tmp_path))
        assert info.entry_points == []


# ---------------------------------------------------------------------------
# format_project_context
# ---------------------------------------------------------------------------

class TestFormatProjectContext:
    def test_unknown_language_returns_empty(self):
        assert format_project_context(ProjectInfo()) == ""

    def test_python_has_language_header(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("x\n")
        info = detect_project(str(tmp_path))
        result = format_project_context(info)
        assert "python" in result

    def test_test_command_in_output(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("x\n")
        info = detect_project(str(tmp_path))
        result = format_project_context(info)
        assert "pytest" in result

    def test_framework_in_output_when_set(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("fastapi\n")
        info = detect_project(str(tmp_path))
        info.framework = "fastapi"
        result = format_project_context(info)
        assert "fastapi" in result

    def test_docker_mentioned_when_present(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("x\n")
        (tmp_path / "Dockerfile").write_text("FROM ubuntu\n")
        info = detect_project(str(tmp_path))
        result = format_project_context(info)
        assert "Docker" in result or "docker" in result.lower()

    def test_returns_string(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("x\n")
        info = detect_project(str(tmp_path))
        assert isinstance(format_project_context(info), str)


# ---------------------------------------------------------------------------
# _file_contains
# ---------------------------------------------------------------------------

class TestFileContains:
    def test_returns_true_when_found(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello fastapi world\n")
        assert _file_contains(f, "fastapi") is True

    def test_returns_false_when_not_found(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world\n")
        assert _file_contains(f, "django") is False

    def test_case_insensitive(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("FASTAPI\n")
        assert _file_contains(f, "fastapi") is True

    def test_nonexistent_file_returns_false(self, tmp_path):
        assert _file_contains(tmp_path / "nope.txt", "anything") is False


# ---------------------------------------------------------------------------
# CredentialGuard constants
# ---------------------------------------------------------------------------

class TestCredentialGuardConstants:
    def test_credential_filenames_nonempty(self):
        assert len(CREDENTIAL_FILENAMES) > 0

    def test_credential_extensions_nonempty(self):
        assert len(CREDENTIAL_EXTENSIONS) > 0

    def test_credential_patterns_nonempty(self):
        assert len(CREDENTIAL_PATTERNS) > 0

    def test_dot_env_in_filenames(self):
        assert ".env" in CREDENTIAL_FILENAMES

    def test_pem_in_extensions(self):
        assert ".pem" in CREDENTIAL_EXTENSIONS


# ---------------------------------------------------------------------------
# CredentialGuard.is_credential_file
# ---------------------------------------------------------------------------

class TestIsCredentialFile:
    def setup_method(self):
        self.guard = CredentialGuard()

    def test_dot_env_is_credential(self):
        assert self.guard.is_credential_file("/path/.env") is True

    def test_id_rsa_is_credential(self):
        assert self.guard.is_credential_file("/home/user/.ssh/id_rsa") is True

    def test_pem_file_is_credential(self):
        assert self.guard.is_credential_file("/certs/server.pem") is True

    def test_credentials_json_is_credential(self):
        assert self.guard.is_credential_file("/app/credentials.json") is True

    def test_app_py_not_credential(self):
        assert self.guard.is_credential_file("/src/app.py") is False

    def test_config_json_not_credential(self):
        assert self.guard.is_credential_file("/app/config.json") is False

    def test_readme_md_not_credential(self):
        assert self.guard.is_credential_file("/README.md") is False


# ---------------------------------------------------------------------------
# CredentialGuard.scan_content
# ---------------------------------------------------------------------------

class TestScanContent:
    def setup_method(self):
        self.guard = CredentialGuard()

    def test_safe_text_returns_empty(self):
        assert self.guard.scan_content("Hello world, nothing here.") == []

    def test_returns_list(self):
        assert isinstance(self.guard.scan_content("hello"), list)

    def test_api_key_detected(self):
        result = self.guard.scan_content("api_key = abc123def456ghi789jkl")
        assert len(result) >= 1

    def test_github_pat_detected(self):
        result = self.guard.scan_content("ghp_" + "A" * 36)
        assert len(result) >= 1

    def test_jwt_detected(self):
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        result = self.guard.scan_content(jwt)
        assert len(result) >= 1

    def test_openai_key_detected(self):
        result = self.guard.scan_content("sk-" + "a" * 40)
        assert len(result) >= 1

    def test_empty_string_returns_empty(self):
        assert self.guard.scan_content("") == []

    def test_returns_pattern_strings(self):
        result = self.guard.scan_content("api_key = abc123def456ghi789jkl")
        assert all(isinstance(p, str) for p in result)


# ---------------------------------------------------------------------------
# CredentialGuard.warn_if_credential
# ---------------------------------------------------------------------------

class TestWarnIfCredential:
    def setup_method(self):
        self.guard = CredentialGuard()

    def test_env_file_returns_warning(self):
        result = self.guard.warn_if_credential("/app/.env")
        assert result is not None
        assert isinstance(result, str)

    def test_warning_contains_filename(self):
        result = self.guard.warn_if_credential("/app/.env")
        assert ".env" in result

    def test_safe_file_returns_none(self):
        assert self.guard.warn_if_credential("/src/app.py") is None

    def test_pem_returns_warning(self):
        assert self.guard.warn_if_credential("/certs/server.pem") is not None


# ---------------------------------------------------------------------------
# _get_header (web_cache)
# ---------------------------------------------------------------------------

class TestGetHeader:
    def test_exact_match(self):
        assert _get_header({"etag": '"abc"'}, "etag") == '"abc"'

    def test_case_insensitive_lookup(self):
        assert _get_header({"Content-Type": "text/html"}, "content-type") == "text/html"

    def test_case_insensitive_key(self):
        assert _get_header({"etag": '"v1"'}, "ETag") == '"v1"'

    def test_missing_key_returns_empty(self):
        assert _get_header({"content-type": "text/html"}, "etag") == ""

    def test_empty_dict_returns_empty(self):
        assert _get_header({}, "etag") == ""


# ---------------------------------------------------------------------------
# CrawlCache constants
# ---------------------------------------------------------------------------

class TestCrawlCacheConstants:
    def test_cache_filename_defined(self):
        assert isinstance(CACHE_FILENAME, str)
        assert len(CACHE_FILENAME) > 0

    def test_cache_version_integer(self):
        assert isinstance(CACHE_VERSION, int)


# ---------------------------------------------------------------------------
# CrawlCache — init and page_count
# ---------------------------------------------------------------------------

class TestCrawlCacheInit:
    def test_initial_page_count_zero(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        assert cache.page_count == 0

    def test_cache_path_under_dir(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        assert str(tmp_path) in cache.cache_path


# ---------------------------------------------------------------------------
# CrawlCache — update_page
# ---------------------------------------------------------------------------

class TestCrawlCacheUpdatePage:
    def test_update_increments_page_count(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {}, "content")
        assert cache.page_count == 1

    def test_update_multiple_pages(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/a", {}, "content a")
        cache.update_page("/b", {}, "content b")
        assert cache.page_count == 2

    def test_update_same_url_no_duplicate(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {}, "v1")
        cache.update_page("/page", {}, "v2")
        assert cache.page_count == 1

    def test_etag_stored(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {"etag": '"v1"'}, "content")
        headers = cache.get_page_headers("/page")
        assert headers.get("If-None-Match") == '"v1"'

    def test_last_modified_stored(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {"last-modified": "Mon, 01 Jan 2024 00:00:00 GMT"}, "content")
        headers = cache.get_page_headers("/page")
        assert "If-Modified-Since" in headers

    def test_links_stored(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {}, "content", ["/link1", "/link2"])
        assert cache.get_page_links("/page") == ["/link1", "/link2"]


# ---------------------------------------------------------------------------
# CrawlCache — get_page_headers
# ---------------------------------------------------------------------------

class TestCrawlCacheGetPageHeaders:
    def test_unknown_url_returns_empty(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        assert cache.get_page_headers("/missing") == {}

    def test_no_etag_no_if_none_match(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {}, "content")
        headers = cache.get_page_headers("/page")
        assert "If-None-Match" not in headers

    def test_returns_dict(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {}, "content")
        assert isinstance(cache.get_page_headers("/page"), dict)


# ---------------------------------------------------------------------------
# CrawlCache — is_page_unchanged
# ---------------------------------------------------------------------------

class TestCrawlCacheIsPageUnchanged:
    def test_same_content_is_unchanged(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {}, "hello content")
        assert cache.is_page_unchanged("/page", "hello content") is True

    def test_different_content_not_unchanged(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {}, "original")
        assert cache.is_page_unchanged("/page", "modified") is False

    def test_missing_page_not_unchanged(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        assert cache.is_page_unchanged("/missing", "anything") is False


# ---------------------------------------------------------------------------
# CrawlCache — get_page_links
# ---------------------------------------------------------------------------

class TestCrawlCacheGetPageLinks:
    def test_missing_page_returns_empty(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        assert cache.get_page_links("/missing") == []

    def test_page_without_links_returns_empty(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {}, "content")
        assert cache.get_page_links("/page") == []


# ---------------------------------------------------------------------------
# CrawlCache — save and load
# ---------------------------------------------------------------------------

class TestCrawlCacheSaveLoad:
    def test_save_creates_file(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {}, "content")
        cache.save()
        assert os.path.exists(cache.cache_path)

    def test_load_restores_page_count(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {}, "content")
        cache.save()
        cache2 = CrawlCache(str(tmp_path))
        cache2.load()
        assert cache2.page_count == 1

    def test_load_restores_unchanged_check(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {}, "hello content")
        cache.save()
        cache2 = CrawlCache(str(tmp_path))
        cache2.load()
        assert cache2.is_page_unchanged("/page", "hello content") is True

    def test_load_missing_file_no_error(self, tmp_path):
        cache = CrawlCache(str(tmp_path / "subdir"))
        cache.load()  # should not raise
        assert cache.page_count == 0
