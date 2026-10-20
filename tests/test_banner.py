"""Tests for mullet.cli.banner — logo rendering."""
from unittest.mock import patch
from mullet.cli.banner import render_banner, _LOGO_LINES


class TestRenderBanner:
    def test_no_color_contains_logo(self):
        banner = render_banner(no_color=True)
        assert "Mullet" in banner or "__" in banner

    def test_no_color_contains_version(self):
        banner = render_banner(version="1.2.3", no_color=True)
        assert "1.2.3" in banner

    def test_no_color_contains_tagline(self):
        banner = render_banner(no_color=True)
        assert "Local GPU" in banner

    def test_custom_version(self):
        banner = render_banner(version="9.9.9", no_color=True)
        assert "9.9.9" in banner
        assert "8.8.8" not in banner

    def test_logo_lines_present_in_banner(self):
        banner = render_banner(no_color=True)
        for line in _LOGO_LINES:
            stripped = line.strip()
            if stripped:
                assert stripped in banner

    def test_with_color_doesnt_crash(self):
        banner = render_banner(no_color=False)
        assert banner  # non-empty

    def test_version_defaults_to_package_version(self):
        import mullet
        banner = render_banner(no_color=True)
        assert mullet.__version__ in banner

    def test_rich_exception_falls_back_to_plain(self):
        """Lines 41-42: Rich Console raises → exception caught, plain banner returned."""
        with patch("rich.console.Console", side_effect=Exception("rich broken")):
            banner = render_banner(no_color=False)
        assert "v" in banner  # plain fallback still contains version string
