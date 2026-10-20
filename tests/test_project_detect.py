"""Tests for project type auto-detection."""

from mullet.core.agent.project_detect import detect_project, format_project_context, ProjectInfo


def test_detect_python(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "myapp"\n')
    info = detect_project(str(tmp_path))
    assert info.language == "python"
    assert info.package_manager == "pip"
    assert "pytest" in info.test_command


def test_detect_python_django(tmp_path):
    (tmp_path / "pyproject.toml").write_text('dependencies = ["django"]\n')
    (tmp_path / "manage.py").write_text("#!/usr/bin/env python\n")
    info = detect_project(str(tmp_path))
    assert info.language == "python"
    assert info.framework == "django"


def test_detect_python_fastapi(tmp_path):
    (tmp_path / "requirements.txt").write_text("fastapi\nuvicorn\n")
    info = detect_project(str(tmp_path))
    assert info.language == "python"
    assert info.framework == "fastapi"


def test_detect_javascript(tmp_path):
    (tmp_path / "package.json").write_text('{"name": "myapp"}\n')
    info = detect_project(str(tmp_path))
    assert info.language == "javascript"
    assert info.package_manager == "npm"


def test_detect_typescript(tmp_path):
    (tmp_path / "package.json").write_text('{"name": "myapp"}\n')
    (tmp_path / "tsconfig.json").write_text("{}\n")
    info = detect_project(str(tmp_path))
    assert info.language == "typescript"


def test_detect_nextjs(tmp_path):
    (tmp_path / "package.json").write_text('{"dependencies": {"next": "^14"}}\n')
    (tmp_path / "tsconfig.json").write_text("{}\n")
    info = detect_project(str(tmp_path))
    assert info.framework == "nextjs"


def test_detect_react(tmp_path):
    (tmp_path / "package.json").write_text('{"dependencies": {"react": "^18"}}\n')
    info = detect_project(str(tmp_path))
    assert info.framework == "react"


def test_detect_yarn(tmp_path):
    (tmp_path / "package.json").write_text('{"name": "myapp"}\n')
    (tmp_path / "yarn.lock").write_text("")
    info = detect_project(str(tmp_path))
    assert info.package_manager == "yarn"


def test_detect_rust(tmp_path):
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "myapp"\n')
    info = detect_project(str(tmp_path))
    assert info.language == "rust"
    assert info.package_manager == "cargo"
    assert "cargo test" in info.test_command


def test_detect_go(tmp_path):
    (tmp_path / "go.mod").write_text("module example.com/myapp\n")
    info = detect_project(str(tmp_path))
    assert info.language == "go"
    assert "go test" in info.test_command


def test_detect_java_maven(tmp_path):
    (tmp_path / "pom.xml").write_text("<project></project>\n")
    info = detect_project(str(tmp_path))
    assert info.language == "java"
    assert info.package_manager == "maven"


def test_detect_java_gradle(tmp_path):
    (tmp_path / "build.gradle").write_text("plugins { id 'java' }\n")
    info = detect_project(str(tmp_path))
    assert info.language == "java"
    assert info.package_manager == "gradle"


def test_detect_git(tmp_path):
    (tmp_path / ".git").mkdir()
    info = detect_project(str(tmp_path))
    assert info.has_git is True


def test_detect_docker(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM python:3.12\n")
    info = detect_project(str(tmp_path))
    assert info.has_docker is True


def test_detect_ci(tmp_path):
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    info = detect_project(str(tmp_path))
    assert info.has_ci is True


def test_detect_unknown(tmp_path):
    info = detect_project(str(tmp_path))
    assert info.language == "unknown"


def test_detect_entry_points(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
    (tmp_path / "main.py").write_text("print('hello')\n")
    info = detect_project(str(tmp_path))
    assert "main.py" in info.entry_points


def test_format_context_python():
    info = ProjectInfo(language="python", framework="fastapi", test_command="pytest", package_manager="pip")
    ctx = format_project_context(info)
    assert "python" in ctx
    assert "fastapi" in ctx
    assert "pytest" in ctx


def test_format_context_unknown():
    info = ProjectInfo(language="unknown")
    ctx = format_project_context(info)
    assert ctx == ""


def test_detect_mullet_project():
    """Detect the actual Mullet project."""
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    info = detect_project(project_root)
    assert info.language == "python"
    assert info.has_git is True
    assert "pytest" in info.test_command
