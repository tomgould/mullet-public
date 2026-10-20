"""Tests for first-run onboarding helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from mullet.harness.tui.onboarding import (
    detect_setup_issues,
    first_run_banner,
    is_first_run,
    mark_onboarded,
)

pytestmark = [pytest.mark.harness, pytest.mark.tui]


class TestIsFirstRun:
    def test_is_first_run_with_no_marker(self, tmp_path: Path) -> None:
        marker = tmp_path / ".onboarded"
        assert is_first_run(marker_path=marker) is True

    def test_is_first_run_after_marked(self, tmp_path: Path) -> None:
        marker = tmp_path / ".onboarded"
        mark_onboarded(marker_path=marker)
        assert is_first_run(marker_path=marker) is False


class TestMarkOnboarded:
    def test_mark_onboarded_creates_marker(self, tmp_path: Path) -> None:
        marker = tmp_path / "deep" / ".onboarded"
        assert not marker.exists()
        mark_onboarded(marker_path=marker)
        assert marker.exists()


class TestFirstRunBanner:
    def test_first_run_banner_text(self) -> None:
        banner = first_run_banner()
        assert "Welcome to Mullet" in banner
        assert "/help" in banner
        assert "harness-init" in banner
        assert "provider list" in banner
        assert "check your config" in banner


class TestDetectSetupIssues:
    def test_detect_setup_issues_no_kimi_warns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        issues = detect_setup_issues()
        assert any("KIMI_API_KEY" in issue for issue in issues)

    def test_detect_setup_issues_with_kimi_passes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KIMI_API_KEY", "fake-key")
        # Create .mullet/ so that passes too
        (tmp_path / ".mullet").mkdir()
        monkeypatch.chdir(tmp_path)
        issues = detect_setup_issues()
        assert issues == []

    def test_detect_setup_issues_no_mullet_warns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KIMI_API_KEY", "fake-key")
        monkeypatch.chdir(tmp_path)
        issues = detect_setup_issues()
        assert any(".mullet/" in issue for issue in issues)
