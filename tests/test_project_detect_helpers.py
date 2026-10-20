"""Tests for mullet.core.agent.project_detect — pure project detection."""


from unittest.mock import patch

from mullet.core.agent.project_detect import (
    ProjectInfo,
    detect_project,
    format_project_context,
    _file_contains,
)


# ---------------------------------------------------------------------------
# ProjectInfo dataclass
# ---------------------------------------------------------------------------

class TestProjectInfo:
    def test_defaults(self):
        info = ProjectInfo()
        assert info.language == "unknown"
        assert info.framework == ""
        assert info.package_manager == ""
        assert info.test_command == ""
        assert info.has_git is False
        assert info.has_docker is False
        assert info.has_ci is False
        assert info.entry_points == []


# ---------------------------------------------------------------------------
# _file_contains
# ---------------------------------------------------------------------------

class TestFileContains:
    def test_text_present(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("django==4.2")
        assert _file_contains(f, "django")

    def test_case_insensitive(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("FastAPI==0.100")
        assert _file_contains(f, "fastapi")

    def test_text_absent(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("requests==2.28")
        assert not _file_contains(f, "django")

    def test_nonexistent_file_returns_false(self, tmp_path):
        assert not _file_contains(tmp_path / "nonexistent.txt", "anything")

    def test_empty_file_returns_false(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert not _file_contains(f, "django")

    def test_read_error_returns_false_and_logs_warning(self, tmp_path, caplog):
        f = tmp_path / "test.txt"
        f.write_text("content")
        with patch("pathlib.Path.read_text", side_effect=OSError("read error")):
            with caplog.at_level("WARNING", logger="mullet.core.agent.project_detect"):
                assert not _file_contains(f, "content")
        assert "failed to read file" in caplog.text


# ---------------------------------------------------------------------------
# detect_project — Python
# ---------------------------------------------------------------------------

class TestDetectProjectPython:
    def test_pyproject_toml_detects_python(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.pytest]")
        info = detect_project(str(tmp_path))
        assert info.language == "python"

    def test_requirements_txt_detects_python(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests\n")
        info = detect_project(str(tmp_path))
        assert info.language == "python"

    def test_setup_py_detects_python(self, tmp_path):
        (tmp_path / "setup.py").write_text("from setuptools import setup")
        info = detect_project(str(tmp_path))
        assert info.language == "python"

    def test_python_package_manager_pip(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[build-system]")
        info = detect_project(str(tmp_path))
        assert info.package_manager == "pip"

    def test_python_test_command(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[build-system]")
        info = detect_project(str(tmp_path))
        assert "pytest" in info.test_command

    def test_django_detected_via_manage_py(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[build-system]")
        (tmp_path / "manage.py").write_text("#!/usr/bin/env python")
        info = detect_project(str(tmp_path))
        assert info.framework == "django"

    def test_fastapi_detected_via_requirements(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("fastapi==0.100")
        info = detect_project(str(tmp_path))
        assert info.framework == "fastapi"

    def test_flask_detected_via_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('dependencies = ["flask"]')
        info = detect_project(str(tmp_path))
        assert info.framework == "flask"

    def test_aiohttp_detected_via_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('dependencies = ["aiohttp"]')
        info = detect_project(str(tmp_path))
        assert info.framework == "aiohttp"


# ---------------------------------------------------------------------------
# detect_project — JavaScript / TypeScript
# ---------------------------------------------------------------------------

class TestDetectProjectJS:
    def test_package_json_detects_javascript(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "app"}')
        info = detect_project(str(tmp_path))
        assert info.language == "javascript"

    def test_tsconfig_detects_typescript(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "app"}')
        (tmp_path / "tsconfig.json").write_text("{}")
        info = detect_project(str(tmp_path))
        assert info.language == "typescript"

    def test_yarn_lock_detects_yarn(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "app"}')
        (tmp_path / "yarn.lock").write_text("")
        info = detect_project(str(tmp_path))
        assert info.package_manager == "yarn"

    def test_pnpm_lock_detects_pnpm(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "app"}')
        (tmp_path / "pnpm-lock.yaml").write_text("")
        info = detect_project(str(tmp_path))
        assert info.package_manager == "pnpm"

    def test_npm_is_default_package_manager(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "app"}')
        info = detect_project(str(tmp_path))
        assert info.package_manager == "npm"

    def test_nextjs_framework_detected(self, tmp_path):
        (tmp_path / "package.json").write_text('{"dependencies": {"next": "13.0"}}')
        info = detect_project(str(tmp_path))
        assert info.framework == "nextjs"

    def test_react_framework_detected(self, tmp_path):
        (tmp_path / "package.json").write_text('{"dependencies": {"react": "18.0"}}')
        info = detect_project(str(tmp_path))
        assert info.framework == "react"

    def test_vue_framework_detected(self, tmp_path):
        (tmp_path / "package.json").write_text('{"dependencies": {"vue": "3.0"}}')
        info = detect_project(str(tmp_path))
        assert info.framework == "vue"

    def test_express_framework_detected(self, tmp_path):
        (tmp_path / "package.json").write_text('{"dependencies": {"express": "4.0"}}')
        info = detect_project(str(tmp_path))
        assert info.framework == "express"

    def test_svelte_framework_detected(self, tmp_path):
        (tmp_path / "package.json").write_text('{"dependencies": {"svelte": "4.0"}}')
        info = detect_project(str(tmp_path))
        assert info.framework == "svelte"


# ---------------------------------------------------------------------------
# detect_project — other languages
# ---------------------------------------------------------------------------

class TestDetectProjectOtherLanguages:
    def test_cargo_toml_detects_rust(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "myapp"')
        info = detect_project(str(tmp_path))
        assert info.language == "rust"
        assert info.package_manager == "cargo"
        assert "cargo test" in info.test_command

    def test_go_mod_detects_go(self, tmp_path):
        (tmp_path / "go.mod").write_text("module myapp\n")
        info = detect_project(str(tmp_path))
        assert info.language == "go"
        assert "go test" in info.test_command

    def test_pom_xml_detects_java_maven(self, tmp_path):
        (tmp_path / "pom.xml").write_text("<project/>")
        info = detect_project(str(tmp_path))
        assert info.language == "java"
        assert info.package_manager == "maven"

    def test_build_gradle_detects_java_gradle(self, tmp_path):
        (tmp_path / "build.gradle").write_text("apply plugin: 'java'")
        info = detect_project(str(tmp_path))
        assert info.language == "java"
        assert info.package_manager == "gradle"

    def test_empty_dir_unknown_language(self, tmp_path):
        info = detect_project(str(tmp_path))
        assert info.language == "unknown"


# ---------------------------------------------------------------------------
# detect_project — infrastructure detection
# ---------------------------------------------------------------------------

class TestDetectProjectInfra:
    def test_git_dir_detected(self, tmp_path):
        (tmp_path / ".git").mkdir()
        info = detect_project(str(tmp_path))
        assert info.has_git is True

    def test_no_git_dir(self, tmp_path):
        info = detect_project(str(tmp_path))
        assert info.has_git is False

    def test_dockerfile_detected(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM python:3.10")
        info = detect_project(str(tmp_path))
        assert info.has_docker is True

    def test_docker_compose_detected(self, tmp_path):
        (tmp_path / "docker-compose.yml").write_text("version: '3'")
        info = detect_project(str(tmp_path))
        assert info.has_docker is True

    def test_github_workflows_detects_ci(self, tmp_path):
        wf = tmp_path / ".github" / "workflows"
        wf.mkdir(parents=True)
        info = detect_project(str(tmp_path))
        assert info.has_ci is True

    def test_entry_points_detected(self, tmp_path):
        (tmp_path / "main.py").write_text("# main")
        (tmp_path / "pyproject.toml").write_text("[build-system]")
        info = detect_project(str(tmp_path))
        assert "main.py" in info.entry_points


# ---------------------------------------------------------------------------
# format_project_context
# ---------------------------------------------------------------------------

class TestFormatProjectContext:
    def test_unknown_language_returns_empty(self):
        info = ProjectInfo()
        assert format_project_context(info) == ""

    def test_includes_language(self):
        info = ProjectInfo(language="python")
        out = format_project_context(info)
        assert "python" in out

    def test_framework_included_when_set(self):
        info = ProjectInfo(language="python", framework="django")
        out = format_project_context(info)
        assert "django" in out

    def test_framework_omitted_when_empty(self):
        info = ProjectInfo(language="python")
        out = format_project_context(info)
        assert "Framework:" not in out

    def test_test_command_included(self):
        info = ProjectInfo(language="python", test_command="python -m pytest")
        out = format_project_context(info)
        assert "pytest" in out

    def test_docker_flag_included(self):
        info = ProjectInfo(language="python", has_docker=True)
        out = format_project_context(info)
        assert "Docker" in out

    def test_ci_flag_included(self):
        info = ProjectInfo(language="python", has_ci=True)
        out = format_project_context(info)
        assert "CI" in out

    def test_docker_omitted_when_false(self):
        info = ProjectInfo(language="python", has_docker=False)
        out = format_project_context(info)
        assert "Docker" not in out

    def test_package_manager_included(self):
        info = ProjectInfo(language="javascript", package_manager="npm")
        out = format_project_context(info)
        assert "Package manager: npm" in out
