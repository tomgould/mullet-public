"""Tests for scripts/hooks/preset_prompt.py."""

import json
import os
import subprocess
import sys
from pathlib import Path

from mullet.core.presets import PresetManager

HOOK = Path(__file__).resolve().parent.parent / "scripts" / "hooks" / "preset_prompt.py"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_hook(tmp_path: Path, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT),
        "MULLET_CONFIG_DIR": str(tmp_path),
    }
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input="{}",
        capture_output=True,
        text=True,
        env=env,
    )


def test_preset_hook_outputs_valid_json_with_active_preset(tmp_path):
    """Hook emits valid JSON with the expected shape and mentions the active preset."""
    # Seed the config directory by constructing PresetManager first
    mgr = PresetManager(config_dir=tmp_path)
    active = mgr.active()

    result = _run_hook(tmp_path)
    assert result.returncode == 0
    stdout = result.stdout.strip()
    assert stdout, "Expected non-empty stdout from successful hook run"
    data = json.loads(stdout)
    assert "hookSpecificOutput" in data
    hso = data["hookSpecificOutput"]
    assert hso.get("hookEventName") == "SessionStart"
    assert "additionalContext" in hso
    assert active.name in hso["additionalContext"]
    assert active.label in hso["additionalContext"]


def test_preset_hook_fail_open_on_bad_config_dir():
    """Hook exits 0 and emits empty or valid stdout even when config dir is unusable."""
    bad_path = Path("/dev/null/should_not_exist")
    result = _run_hook(tmp_path=bad_path)
    assert result.returncode == 0
    stdout = result.stdout.strip()
    # Either empty or parseable JSON — must not crash
    if stdout:
        json.loads(stdout)
