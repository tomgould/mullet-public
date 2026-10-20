"""Tests for the onboarding checklist (mullet-onboard, scripts/onboard.py)."""

from unittest.mock import patch

from mullet.cli import onboard


def test_help_exits_cleanly(capsys):
    assert onboard.main(["--help"]) == 0
    assert "Usage: mullet-onboard" in capsys.readouterr().out


def test_checklist_runs_and_gives_accurate_anthropic_advice(capsys):
    with patch.object(onboard, "_detect_lm_studio", return_value=(False, None)), \
            patch.object(onboard, "_detect_ram_gb", return_value=16.0), \
            patch("builtins.input", return_value=""):
        assert onboard.main([]) == 0
    out = capsys.readouterr().out
    assert "personalised setup checklist" in out
    assert "ANTHROPIC_API_KEY is optional" in out
    assert "Do NOT add ANTHROPIC_API_KEY" not in out


def test_source_checkout_script_uses_the_package_module():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "onboard.py"
    spec = importlib.util.spec_from_file_location("onboard_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main is onboard.main
