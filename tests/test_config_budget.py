"""Tests for mullet.cli.config_cli budget subcommand."""

from __future__ import annotations

from pathlib import Path


from mullet.cli.config_cli import cmd_config



def _setup_isolated(tmp_path: Path, monkeypatch):
    """Patch all filesystem helpers to operate under tmp_path.

    Patches both the re-exported name in config_cli and the canonical name in
    spend_caps so that load_budget_config() also looks in tmp_path.
    """
    mullet_dir = tmp_path / ".mullet"

    # config_cli._find_mullet_dir must return our tmp dir (it normally walks up)
    monkeypatch.setattr(
        "mullet.cli.config_cli._find_mullet_dir",
        lambda: mullet_dir,
    )

    # config_cli._find_budget_file (re-exported from spend_caps) controls where
    # existing files are looked up during set/monthly/reset.
    def _mock_find_budget_file_config():
        yaml_path = mullet_dir / "budget.yaml"
        return yaml_path if yaml_path.exists() else None

    monkeypatch.setattr(
        "mullet.cli.config_cli._find_budget_file",
        _mock_find_budget_file_config,
    )

    # spend_caps.load_budget_config → _find_budget_file in spend_caps module
    def _mock_find_budget_file_spend(cwd: Path | None = None) -> Path | None:
        yaml_path = mullet_dir / "budget.yaml"
        return yaml_path if yaml_path.exists() else None

    monkeypatch.setattr(
        "mullet.core.providers.spend_caps._find_budget_file",
        _mock_find_budget_file_spend,
    )

    return mullet_dir


class TestConfigBudget:
    def test_budget_show_empty(self, tmp_path, monkeypatch, capsys):
        """No budget file → prints 'No spend caps configured', returns 0."""
        _setup_isolated(tmp_path, monkeypatch)

        rc = cmd_config(["budget", "show"])
        captured = capsys.readouterr()

        assert rc == 0
        assert "No spend caps configured" in captured.out

    def test_budget_set_provider(self, tmp_path, monkeypatch, capsys):
        """Set a provider cap → file written with correct YAML structure."""
        mullet_dir = _setup_isolated(tmp_path, monkeypatch)

        rc = cmd_config(["budget", "set", "openrouter", "5.0"])
        captured = capsys.readouterr()

        assert rc == 0
        assert "Set openrouter cap to $5.00" in captured.out

        budget_file = mullet_dir / "budget.yaml"
        assert budget_file.exists()
        text = budget_file.read_text(encoding="utf-8")
        assert "openrouter:" in text
        assert "cap_usd: 5.0" in text

    def test_budget_set_invalid_provider(self, tmp_path, monkeypatch, capsys):
        """Unknown provider name → returns 1 with error on stderr."""
        _setup_isolated(tmp_path, monkeypatch)

        rc = cmd_config(["budget", "set", "fakeprovider", "5.0"])
        captured = capsys.readouterr()

        assert rc == 1
        assert "Unknown provider" in captured.err

    def test_budget_monthly(self, tmp_path, monkeypatch, capsys):
        """Set monthly total cap → yaml has monthly_total_cap_usd."""
        mullet_dir = _setup_isolated(tmp_path, monkeypatch)

        rc = cmd_config(["budget", "monthly", "50.0"])
        captured = capsys.readouterr()

        assert rc == 0
        assert "Set monthly total cap to $50.00" in captured.out

        budget_file = mullet_dir / "budget.yaml"
        assert budget_file.exists()
        text = budget_file.read_text(encoding="utf-8")
        assert "monthly_total_cap_usd: 50.0" in text

    def test_budget_reset_no_file(self, tmp_path, monkeypatch, capsys):
        """Reset with no existing file → returns 0, prints 'No budget configuration'."""
        _setup_isolated(tmp_path, monkeypatch)

        rc = cmd_config(["budget", "reset"])
        captured = capsys.readouterr()

        assert rc == 0
        assert "No budget configuration" in captured.out

    def test_budget_reset_existing(self, tmp_path, monkeypatch, capsys):
        """Reset when file exists → file deleted, returns 0."""
        mullet_dir = _setup_isolated(tmp_path, monkeypatch)
        mullet_dir.mkdir(parents=True, exist_ok=True)
        budget_file = mullet_dir / "budget.yaml"
        budget_file.write_text("kimi:\n  cap_usd: 20.0\n", encoding="utf-8")

        rc = cmd_config(["budget", "reset"])
        captured = capsys.readouterr()

        assert rc == 0
        assert "Budget configuration reset" in captured.out
        assert not budget_file.exists()

    def test_budget_set_merges_existing(self, tmp_path, monkeypatch, capsys):
        """Setting a second provider merges with the existing YAML."""
        mullet_dir = _setup_isolated(tmp_path, monkeypatch)

        cmd_config(["budget", "set", "openrouter", "10.0"])
        cmd_config(["budget", "set", "kimi", "20.0"])

        budget_file = mullet_dir / "budget.yaml"
        text = budget_file.read_text(encoding="utf-8")
        assert "openrouter:" in text
        assert "kimi:" in text

    def test_budget_set_missing_args(self, tmp_path, monkeypatch, capsys):
        """set without enough args → returns 1."""
        _setup_isolated(tmp_path, monkeypatch)

        rc = cmd_config(["budget", "set", "kimi"])
        captured = capsys.readouterr()

        assert rc == 1
        assert "Usage" in captured.err

    def test_budget_monthly_missing_args(self, tmp_path, monkeypatch, capsys):
        """monthly without amount → returns 1."""
        _setup_isolated(tmp_path, monkeypatch)

        rc = cmd_config(["budget", "monthly"])
        captured = capsys.readouterr()

        assert rc == 1
        assert "Usage" in captured.err

    def test_budget_show_with_caps(self, tmp_path, monkeypatch, capsys):
        """Show after setting caps → displays provider and cap amount."""
        mullet_dir = _setup_isolated(tmp_path, monkeypatch)
        mullet_dir.mkdir(parents=True, exist_ok=True)
        (mullet_dir / "budget.yaml").write_text(
            "openrouter:\n  cap_usd: 10.0\nmonthly_total_cap_usd: 50.0\n",
            encoding="utf-8",
        )

        rc = cmd_config(["budget", "show"])
        captured = capsys.readouterr()

        assert rc == 0
        combined = captured.out + captured.err
        assert "openrouter" in combined
        assert "10" in combined
