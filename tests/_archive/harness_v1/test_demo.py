"""Tests for `mullet demo` interactive walkthrough."""

from unittest.mock import MagicMock, patch


from mullet.cli.demo_cli import cmd_demo


class TestDemoCLI:
    """Test the `mullet demo` walkthrough."""

    def test_demo_runs_in_fake_mode_without_provider(self, capsys):
        """With no provider found, demo should run in fake mode and return 0."""
        with patch("mullet.cli.demo_cli._detect_provider", return_value=(None, "✗ No provider available. Demo running in fake-mode.")):
            rc = cmd_demo(["--no-color"])
        assert rc == 0
        captured = capsys.readouterr()
        out = (captured.out + captured.err).lower()
        assert "fake-mode" in out or "no provider available" in out

    def test_demo_with_kimi_calls_provider_once(self, capsys, monkeypatch):
        """When KIMI_API_KEY is set, the demo should attempt a real provider call."""
        monkeypatch.setenv("KIMI_API_KEY", "sk-test123")
        mock_provider = MagicMock()
        mock_provider.health_check.return_value = True
        mock_provider.name = "kimi"
        mock_provider.model = "kimi-k2.6"

        def _fake_stream(*args, **kwargs):
            class _Chunk:
                content = "A harness routes tasks across LLMs."
                finish_reason = "stop"
            yield _Chunk()

        mock_provider.create_stream.side_effect = _fake_stream

        with patch("mullet.cli.demo_cli._resolve_provider_instance", return_value=mock_provider):
            rc = cmd_demo(["--no-color"])

        assert rc == 0
        assert mock_provider.create_stream.called
        captured = capsys.readouterr()
        assert "harness" in captured.out.lower() or "harness" in captured.err.lower()

    def test_demo_skip_install_check_works(self, capsys):
        """--skip-install-check should bypass the pip dependency check."""
        rc = cmd_demo(["--no-color", "--skip-install-check"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "skipping install check" in captured.out.lower() or "skip install check" in captured.out.lower()

    def test_demo_outputs_skill_count(self, capsys):
        """Demo should print the number of skills discovered."""
        with patch("mullet.cli.demo_cli._load_skills_for_demo", return_value=[{"name": "test-skill", "scope": "builtin"}]):
            rc = cmd_demo(["--no-color"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "1 skill" in captured.out.lower() or "1 skills" in captured.out.lower()

    def test_demo_shows_provider_status(self, capsys):
        """Demo should show provider detection status."""
        with patch("mullet.cli.demo_cli._detect_provider", return_value=(None, "✗ No provider available. Demo running in fake-mode.")):
            rc = cmd_demo(["--no-color"])
        assert rc == 0
        captured = capsys.readouterr()
        out = (captured.out + captured.err).lower()
        assert "no provider available" in out or "fake-mode" in out
