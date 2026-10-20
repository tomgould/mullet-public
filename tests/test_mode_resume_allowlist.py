"""G25 — session-resume restores allowed_providers from saved mode state.

When a user passes --providers kimi,haiku and the orchestration mode (ralph
or autopilot) crashes / is paused mid-run, --resume must rehydrate the
allowlist from the saved state. Otherwise the next delegation could pick a
provider outside the user's envelope, breaking trust.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mullet.core.orchestration.autopilot import AutopilotMode
from mullet.core.orchestration.ralph import RalphMode


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    """Each test runs in its own tmp dir so .mullet/state files do not leak."""
    monkeypatch.chdir(tmp_path)


# --- Mode save() persistence tests ------------------------------------------


def test_ralph_save_persists_allowed_providers():
    """Fails because RalphMode.execute save() call does not include
    'allowed_providers' in the state dict — only prd/iteration/task today."""
    ralph = RalphMode()
    ralph._allowed_providers = frozenset({"kimi", "haiku"})

    captured: dict = {}

    def _capture(state):
        captured.update(state)

    with patch.object(ralph._state, "save", side_effect=_capture):
        # Trigger persistence path directly with the helper the production
        # code is expected to call.
        ralph._persist_state({"prd": {}, "iteration": 1, "task": "t"})

    assert captured.get("allowed_providers") == ["haiku", "kimi"], (
        "save() must include allowed_providers as a stable-sorted list; "
        f"got {captured.get('allowed_providers')!r}"
    )


def test_ralph_save_with_no_allowlist_uses_none():
    """Fails because helper does not exist or does not preserve None vs missing key."""
    ralph = RalphMode()
    ralph._allowed_providers = None

    captured: dict = {}

    def _capture(state):
        captured.update(state)

    with patch.object(ralph._state, "save", side_effect=_capture):
        ralph._persist_state({"prd": {}, "iteration": 1, "task": "t"})

    assert "allowed_providers" in captured, (
        "save() must always set the key, even when value is None, "
        "so resume can distinguish 'unrestricted' from 'legacy state'."
    )
    assert captured["allowed_providers"] is None


def test_autopilot_save_persists_allowed_providers():
    """Fails because AutopilotMode does not persist allowed_providers in any save() call."""
    auto = AutopilotMode()
    auto._allowed_providers = frozenset({"kimi"})

    captured: dict = {}

    def _capture(state):
        captured.update(state)

    with patch.object(auto._state, "save", side_effect=_capture):
        auto._persist_state({"phase": "plan", "task": "t", "plan": "p"})

    assert captured.get("allowed_providers") == ["kimi"]


# --- run_mode --resume restoration tests ------------------------------------


def _resume_args(providers: str | None = None) -> argparse.Namespace:
    """Minimal args namespace exercising the --resume path of run_mode.

    We exit before build_agent runs by stubbing the mode constructor so the
    test does not depend on a working LLM; we only need to confirm
    args.providers gets set correctly during the resume preamble.
    """
    return SimpleNamespace(
        resume=True,
        providers=providers,
        provider=None,
        speed=None,
        model=None,
        verbose=False,
        toolsets=None,
        max_iterations=None,
        knowledge=None,
        tier=None,
    )


def test_run_mode_resume_restores_providers_on_args():
    """Fails because run_mode --resume does not read 'allowed_providers'
    from saved state and write it back onto args.providers before build_agent."""
    from mullet.cli import main as cli_main

    args = _resume_args(providers=None)
    saved = {
        "phase": "plan",
        "task": "t",
        "plan": "p",
        "allowed_providers": ["kimi", "haiku"],
        "_timestamp": 9_999_999_999,  # force not-stale
    }

    with patch("mullet.core.orchestration.state.ModeState.load", return_value=saved), \
         patch.object(cli_main, "build_agent", side_effect=RuntimeError("stop here")):
        with pytest.raises(RuntimeError, match="stop here"):
            cli_main.run_mode("ralph", "", args)

    assert args.providers == "haiku,kimi", (
        "After --resume, args.providers must be the comma-joined sorted "
        f"saved allowlist. Got {args.providers!r}."
    )


def test_run_mode_resume_preserves_explicit_providers_arg():
    """Fails because run_mode would overwrite the user's explicit --providers
    with the saved allowlist — user intent on this invocation must win."""
    from mullet.cli import main as cli_main

    args = _resume_args(providers="gemini")
    saved = {
        "phase": "plan",
        "task": "t",
        "plan": "p",
        "allowed_providers": ["kimi"],
        "_timestamp": 9_999_999_999,
    }

    with patch("mullet.core.orchestration.state.ModeState.load", return_value=saved), \
         patch.object(cli_main, "build_agent", side_effect=RuntimeError("stop here")):
        with pytest.raises(RuntimeError, match="stop here"):
            cli_main.run_mode("ralph", "", args)

    assert args.providers == "gemini", (
        "Explicit --providers on this run wins over saved state; "
        f"got {args.providers!r}"
    )
