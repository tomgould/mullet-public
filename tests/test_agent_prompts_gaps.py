"""Tests for uncovered lines in mullet.core.agent.prompts."""

from unittest.mock import patch
from mullet.core.agent.prompts import build_base_prompt


def test_build_base_prompt_handles_detect_project_exception(tmp_path, caplog):
    with patch("mullet.core.agent.project_detect.detect_project", side_effect=RuntimeError("detection failed")):
        with caplog.at_level("WARNING", logger="mullet.core.agent.prompts"):
            result = build_base_prompt(cwd=str(tmp_path))
    assert isinstance(result, str)
    assert "project detection failed" in caplog.text
