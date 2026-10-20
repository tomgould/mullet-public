"""Basic validity tests for GitHub Actions workflow files."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.harness


class TestCiWorkflows:
    def test_test_yml_parses_as_yaml(self) -> None:
        path = Path(".github/workflows/test.yml")
        assert path.exists()
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict)
        assert data.get("name") == "Test"

    def test_lint_yml_parses_as_yaml(self) -> None:
        path = Path(".github/workflows/lint.yml")
        assert path.exists()
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict)
        assert data.get("name") == "Lint"

    def test_test_yml_includes_python_versions(self) -> None:
        path = Path(".github/workflows/test.yml")
        data = yaml.safe_load(path.read_text())
        matrix = data["jobs"]["test"]["strategy"]["matrix"]
        versions = matrix["python-version"]
        assert "3.10" in versions
        assert "3.11" in versions
        assert "3.12" in versions
