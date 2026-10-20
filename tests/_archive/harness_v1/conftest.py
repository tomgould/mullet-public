"""Pytest fixtures for the Mullet harness test suite."""

import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "harness: Mullet harness layer tests")
    config.addinivalue_line("markers", "tui: Mullet harness TUI tests (requires textual)")


@pytest.fixture
def tmp_session_store(tmp_path: Path) -> Path:
    """Yield a SessionStore-compatible directory with pre-created sessions subdir."""
    store = tmp_path / "abc123def456-cwd-name" / "sessions"
    store.mkdir(parents=True)
    return store


@pytest.fixture
def mock_provider():
    """Yield a mock async provider that records calls and returns fixed responses."""
    class _MockProvider:
        def __init__(self) -> None:
            self.calls: list[tuple] = []
            self.responses: deque[dict] = deque()

        async def __call__(self, *args, **kwargs) -> dict:
            self.calls.append((args, kwargs))
            if self.responses:
                return self.responses.popleft()
            return {"role": "assistant", "content": "OK", "stop_reason": "end_turn", "tool_uses": []}

    return _MockProvider()


@pytest.fixture
def fake_router_decision() -> SimpleNamespace:
    """Yield a fake router decision with Kimi STANDARD tier defaults."""
    return SimpleNamespace(provider="kimi", model="kimi-k2.5", tier="STANDARD", input_tokens=100, output_tokens=200)


@pytest.fixture
def isolated_profiles_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Yield a temporary profiles directory with an empty profiles.yaml."""
    config_dir = tmp_path / "mullet_config"
    config_dir.mkdir()
    (config_dir / "profiles.yaml").write_text("---\n")
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(config_dir))
    return config_dir


@pytest.fixture
def sample_skill_md(tmp_path: Path) -> tuple[Path, dict]:
    """Yield a temporary SKILL.md path and its parsed frontmatter dict."""
    content = (
        "---\n"
        "name: test-skill\n"
        "description: A test skill\n"
        "tools: [Bash, Read]\n"
        "---\n"
        "# Test Skill\n"
        "Body content here.\n"
    )
    path = tmp_path / "SKILL.md"
    path.write_text(content)
    expected_dict = {"name": "test-skill", "description": "A test skill", "tools": ["Bash", "Read"]}
    return path, expected_dict


@pytest.fixture
def sample_plugin_dir(tmp_path: Path) -> Path:
    """Yield a temporary plugin directory with a valid plugin.json manifest."""
    plugin_dir = tmp_path / "test-plugin" / ".claude-plugin"
    plugin_dir.mkdir(parents=True)
    manifest = {"name": "test-plugin", "version": "1.0.0"}
    (plugin_dir / "plugin.json").write_text(json.dumps(manifest))
    return plugin_dir.parent
