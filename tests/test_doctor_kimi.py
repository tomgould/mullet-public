"""Tests for the Kimi check in `mullet doctor` (D8).

Each test invokes the doctor as a subprocess with a controlled environment so
that the system's real KIMI_API_KEY (if any) cannot influence the result, and
with a fake `claude` on PATH, since the doctor runs `claude --version`.
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.usefixtures("fake_claude")


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_doctor(env_overrides: dict, extra_args: list[str] | None = None,
                timeout: int = 20) -> subprocess.CompletedProcess:
    """Run `python -m mullet.cli.main doctor` with a sanitised environment.

    Strips both KIMI_API_KEY and MOONSHOT_API_KEY from the inherited env first,
    then applies overrides — so individual tests start from a known baseline.
    """
    env = os.environ.copy()
    # Set to empty string rather than pop: dotenv uses override=False so it
    # won't overwrite existing vars; popping lets it re-inject from .env.
    env["KIMI_API_KEY"] = ""
    env["MOONSHOT_API_KEY"] = ""
    env["PYTHONPATH"] = _REPO_ROOT
    env.update(env_overrides)
    args = [sys.executable, "-m", "mullet.cli.main", "doctor"]
    if extra_args:
        args.extend(extra_args)
    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout,
        cwd="/tmp", env=env,
    )


def test_doctor_reports_kimi_key_set():
    """KIMI_API_KEY present → green Kimi line in output.

    Failure mode if regressed: doctor would not mention 'Kimi' at all,
    leaving users blind to whether their key is loaded.
    """
    result = _run_doctor({"KIMI_API_KEY": "sk-test-doctor"})
    assert result.returncode == 0, result.stderr
    assert "Kimi API" in result.stdout
    # Green check uses the literal ✓ from icons map
    # (find a line that contains both ✓ and 'Kimi')
    assert any("Kimi API" in line and "✓" in line for line in result.stdout.splitlines()), \
        f"Expected a Kimi API ok line in:\n{result.stdout}"
    assert "KIMI_API_KEY set" in result.stdout


def test_doctor_reports_kimi_key_missing():
    """No Kimi env var → info line, not an error / not counted in denominator.

    Failure mode if regressed: doctor either crashes, marks Kimi as a failure,
    or silently omits Kimi entirely.
    """
    result = _run_doctor({})
    assert result.returncode == 0, result.stderr
    assert "Kimi API" in result.stdout
    assert "KIMI_API_KEY not set" in result.stdout
    # Should be an info-style line (not ✓, not !)
    kimi_lines = [line for line in result.stdout.splitlines() if "Kimi API" in line]
    assert kimi_lines, "Kimi line should appear even when key missing"
    assert any("·" in line for line in kimi_lines), \
        f"Kimi-missing line should use info icon (·): {kimi_lines}"


def test_doctor_reports_moonshot_fallback_key():
    """MOONSHOT_API_KEY set without KIMI_API_KEY → still treated as present.

    Failure mode if regressed: users with only MOONSHOT_API_KEY would get a
    misleading 'not set' message even though Kimi will work fine.
    """
    result = _run_doctor({"MOONSHOT_API_KEY": "sk-test-moonshot"})
    assert result.returncode == 0, result.stderr
    assert "Kimi API" in result.stdout
    assert "MOONSHOT_API_KEY set" in result.stdout
    assert "not set" not in next(line for line in result.stdout.splitlines() if "Kimi API" in line)


def test_doctor_kimi_check_does_not_crash_on_probe_failure():
    """--full mode with bogus base URL → warn line, doctor still exits 0.

    Failure mode if regressed: doctor would crash (non-zero exit) when the
    Kimi endpoint is unreachable, breaking sysadmin workflows on offline boxes.
    """
    result = _run_doctor(
        {
            "KIMI_API_KEY": "sk-test-bogus",
            # Force the probe to fail fast — RFC 6761 reserved TLD that won't resolve
            "KIMI_BASE_URL": "https://invalid.test.invalid/v1",
        },
        extra_args=["--full"],
    )
    assert result.returncode == 0, result.stderr
    assert "Kimi API probe" in result.stdout
    # Probe failure should appear as a warn line (!), not crash
    probe_lines = [line for line in result.stdout.splitlines() if "Kimi API probe" in line]
    assert probe_lines, "--full should produce a Kimi API probe line"
    assert any("!" in line or "failed" in line for line in probe_lines), \
        f"Probe failure should warn, not silently pass: {probe_lines}"


def test_doctor_counter_reflects_kimi_presence():
    """Denominator includes Kimi iff key is present.

    Failure mode if regressed: users without a Kimi key would see their
    X/Y score drop for no reason, or users with a key wouldn't get credit.
    """
    # Without Kimi key
    no_key = _run_doctor({})
    assert no_key.returncode == 0
    no_key_line = [line for line in no_key.stdout.splitlines()
                   if "checks passed" in line]
    assert no_key_line, f"Expected 'checks passed' line in:\n{no_key.stdout}"
    no_key_x, no_key_y = _parse_counter(no_key_line[0])

    # With Kimi key
    with_key = _run_doctor({"KIMI_API_KEY": "sk-test-counter"})
    assert with_key.returncode == 0
    with_key_line = [line for line in with_key.stdout.splitlines()
                     if "checks passed" in line]
    assert with_key_line, f"Expected 'checks passed' line in:\n{with_key.stdout}"
    with_key_x, with_key_y = _parse_counter(with_key_line[0])

    # Adding a Kimi key should raise the numerator by at least 1 (the Kimi-specific
    # provider check). Additional checks may also be Kimi-dependent,
    # so the delta may be ≥1 but is always ≥1.
    assert with_key_y >= no_key_y, (
        f"Denominator should not shrink when Kimi key present: "
        f"no-key={no_key_y}, with-key={with_key_y}"
    )
    assert with_key_x >= no_key_x + 1, (
        f"Numerator should grow by ≥1 when Kimi key present: "
        f"no-key={no_key_x}, with-key={with_key_x}"
    )


def _parse_counter(line: str) -> tuple[int, int]:
    """Extract (passed, total) from a 'X/Y checks passed' line."""
    import re
    m = re.search(r"(\d+)/(\d+)\s+checks passed", line)
    assert m, f"Could not parse counter from: {line!r}"
    return int(m.group(1)), int(m.group(2))
