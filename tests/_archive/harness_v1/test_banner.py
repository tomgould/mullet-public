"""Tests for the Mullet ASCII banner."""

from __future__ import annotations

from mullet.cli.banner import render_banner


def test_render_banner_includes_version() -> None:
    banner = render_banner(version="9.9.9", no_color=True)
    assert "9.9.9" in banner
    assert "Mullet" in banner or "__" in banner


def test_render_banner_no_color_returns_plain() -> None:
    banner = render_banner(no_color=True)
    assert "\033[" not in banner
    assert "Cloud in the front" in banner


def test_render_banner_default_version() -> None:
    import mullet

    banner = render_banner(no_color=True)
    assert mullet.__version__ in banner
