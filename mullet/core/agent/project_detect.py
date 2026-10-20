"""Auto-detect project type and configuration from directory contents.

Used to provide context-specific tool recommendations and prompts.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProjectInfo:
    """Detected project information."""
    language: str = "unknown"  # python, javascript, typescript, rust, go, java, etc.
    framework: str = ""  # django, fastapi, react, nextjs, etc.
    package_manager: str = ""  # pip, npm, yarn, cargo, go mod, etc.
    test_command: str = ""  # pytest, npm test, cargo test, etc.
    has_git: bool = False
    has_docker: bool = False
    has_ci: bool = False
    entry_points: list[str] = field(default_factory=list)


def detect_project(directory: str = ".") -> ProjectInfo:
    """Detect project type from directory contents."""
    d = Path(directory)
    info = ProjectInfo()

    # Git
    info.has_git = (d / ".git").exists()

    # Docker
    info.has_docker = (d / "Dockerfile").exists() or (d / "docker-compose.yml").exists()

    # CI
    info.has_ci = (d / ".github" / "workflows").exists() or (d / ".gitlab-ci.yml").exists()

    # Python
    if (d / "pyproject.toml").exists() or (d / "setup.py").exists() or (d / "requirements.txt").exists():
        info.language = "python"
        info.package_manager = "pip"
        info.test_command = "python -m pytest"

        # Detect framework
        if _file_contains(d / "pyproject.toml", "django") or (d / "manage.py").exists():
            info.framework = "django"
        elif _file_contains(d / "pyproject.toml", "fastapi") or _file_contains(d / "requirements.txt", "fastapi"):
            info.framework = "fastapi"
        elif _file_contains(d / "pyproject.toml", "flask"):
            info.framework = "flask"
        elif _file_contains(d / "pyproject.toml", "aiohttp"):
            info.framework = "aiohttp"

    # JavaScript/TypeScript
    elif (d / "package.json").exists():
        info.package_manager = "npm"
        if (d / "yarn.lock").exists():
            info.package_manager = "yarn"
        elif (d / "pnpm-lock.yaml").exists():
            info.package_manager = "pnpm"

        if (d / "tsconfig.json").exists():
            info.language = "typescript"
        else:
            info.language = "javascript"

        info.test_command = f"{info.package_manager} test"

        # Detect framework
        if _file_contains(d / "package.json", "next"):
            info.framework = "nextjs"
        elif _file_contains(d / "package.json", "react"):
            info.framework = "react"
        elif _file_contains(d / "package.json", "vue"):
            info.framework = "vue"
        elif _file_contains(d / "package.json", "express"):
            info.framework = "express"
        elif _file_contains(d / "package.json", "svelte"):
            info.framework = "svelte"

    # Rust
    elif (d / "Cargo.toml").exists():
        info.language = "rust"
        info.package_manager = "cargo"
        info.test_command = "cargo test"

    # Go
    elif (d / "go.mod").exists():
        info.language = "go"
        info.package_manager = "go"
        info.test_command = "go test ./..."

    # Java
    elif (d / "pom.xml").exists():
        info.language = "java"
        info.package_manager = "maven"
        info.test_command = "mvn test"
    elif (d / "build.gradle").exists() or (d / "build.gradle.kts").exists():
        info.language = "java"
        info.package_manager = "gradle"
        info.test_command = "gradle test"

    # Entry points
    for name in ["main.py", "app.py", "index.js", "index.ts", "main.rs", "main.go", "src/main.py", "src/index.ts"]:
        if (d / name).exists():
            info.entry_points.append(name)

    return info


def format_project_context(info: ProjectInfo) -> str:
    """Format project info for system prompt injection."""
    if info.language == "unknown":
        return ""

    parts = [f"# Project: {info.language}"]
    if info.framework:
        parts.append(f"Framework: {info.framework}")
    if info.test_command:
        parts.append(f"Test command: `{info.test_command}`")
    if info.package_manager:
        parts.append(f"Package manager: {info.package_manager}")
    if info.has_docker:
        parts.append("Docker: available")
    if info.has_ci:
        parts.append("CI: configured")

    return "\n".join(parts)


def _file_contains(path: Path, text: str) -> bool:
    """Check if a file exists and contains text (case-insensitive)."""
    try:
        if path.exists():
            return text.lower() in path.read_text(errors="replace").lower()
    except Exception as _exc:
        logger.warning("project_detect: failed to read file %s for content check: %s", path, _exc)
    return False
