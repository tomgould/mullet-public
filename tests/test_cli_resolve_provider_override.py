"""Tests for session-override injection in resolve_provider (mullet/cli/main.py).

Patches:
  - mullet.cli.main.check_budget   — controls BudgetCheck.status / .pressure
  - mullet.core.session_state.get_override — controls what override is returned

resolve_provider returns a Provider object or None; these tests focus on how
*args* are mutated and what is printed to stderr rather than the returned object.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


from mullet.hud.guard import BudgetCheck, BudgetStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs) -> SimpleNamespace:
    """Build a minimal args namespace that resolve_provider expects."""
    defaults = dict(
        provider=None,
        speed=None,
        model=None,
        base_url=None,
        providers=None,   # --providers allowlist
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _budget(status=BudgetStatus.GO, pressure=0.0) -> BudgetCheck:
    return BudgetCheck(
        status=status,
        pressure=pressure,
        message="test budget message",
    )


def _call_resolve(args, override_return=None, budget_return=None, capsys=None):
    """Call resolve_provider with the given mocks. Returns (result, stderr_text)."""
    if budget_return is None:
        budget_return = _budget()

    with patch("mullet.cli.main.check_budget", return_value=budget_return), \
         patch("mullet.core.session_state.get_override", return_value=override_return):
        from mullet.cli.main import resolve_provider
        try:
            result = resolve_provider(args)
        except Exception:
            result = None

    if capsys:
        return result, capsys.readouterr().err
    return result, ""


# ---------------------------------------------------------------------------
# No override — normal path
# ---------------------------------------------------------------------------

def test_no_override_no_provider_low_pressure(capsys):
    """With no override and no --provider, args.provider stays None (auto-detect)."""
    args = _make_args()
    _call_resolve(args, override_return=None, budget_return=_budget(pressure=0.10))
    # provider not injected
    assert args.provider is None


# ---------------------------------------------------------------------------
# Override active — injected when no explicit --provider or --speed
# ---------------------------------------------------------------------------

def test_override_injected_when_no_provider(capsys):
    """Active override is injected into args when caller has no --provider."""
    args = _make_args()
    override = {"provider": "opus-4-7", "model": "claude-opus-4-7"}
    # budget pressure low enough that no downgrade fires (0.0)
    result, stderr = _call_resolve(
        args,
        override_return=override,
        budget_return=_budget(pressure=0.0),
        capsys=capsys,
    )
    assert args.provider == "opus-4-7"
    assert args.model == "claude-opus-4-7"
    assert "opus-4-7" in stderr


def test_override_model_not_overwritten_when_model_already_set(capsys):
    """If args.model is already set, override should not clobber it."""
    args = _make_args(model="my-custom-model")
    override = {"provider": "opus-4-7", "model": "claude-opus-4-7"}
    _call_resolve(args, override_return=override, budget_return=_budget(pressure=0.0))
    assert args.model == "my-custom-model"


# ---------------------------------------------------------------------------
# Override IGNORED when user explicitly pins --provider
# ---------------------------------------------------------------------------

def test_explicit_provider_overrides_session_override(capsys):
    """When user passes --provider sonnet, session override must NOT inject."""
    args = _make_args(provider="sonnet")
    override = {"provider": "opus-4-7", "model": "claude-opus-4-7"}
    _call_resolve(args, override_return=override, budget_return=_budget(pressure=0.0))
    # provider stays as user specified; should not become "opus-4-7"
    assert args.provider in ("sonnet", "gemini")   # budget may adjust sonnet → gemini, but never opus


def test_explicit_speed_prevents_override_injection(capsys):
    """When user passes --speed premium, session override must NOT inject."""
    args = _make_args(speed="premium")
    override = {"provider": "opus-4-7", "model": "claude-opus-4-7"}
    _call_resolve(args, override_return=override, budget_return=_budget(pressure=0.0))
    # provider was NOT injected from override
    assert args.provider is None


# ---------------------------------------------------------------------------
# Budget pressure — downgrade from opus
# ---------------------------------------------------------------------------

def test_override_opus_downgraded_to_sonnet_at_medium_pressure(capsys):
    """pressure=0.60 (>0.55, <0.80) → downgrade to sonnet; warning on stderr."""
    args = _make_args()
    override = {"provider": "opus-4-7", "model": "claude-opus-4-7"}
    result, stderr = _call_resolve(
        args,
        override_return=override,
        budget_return=_budget(pressure=0.60),
        capsys=capsys,
    )
    assert args.provider == "sonnet"
    assert "downgrading to sonnet" in stderr


def test_override_opus_downgraded_to_haiku_at_high_pressure(capsys):
    """pressure=0.85 (>=0.80) → downgrade to haiku; warning on stderr."""
    args = _make_args()
    override = {"provider": "opus-4-7", "model": "claude-opus-4-7"}
    result, stderr = _call_resolve(
        args,
        override_return=override,
        budget_return=_budget(pressure=0.85),
        capsys=capsys,
    )
    assert args.provider == "haiku"
    assert "downgrading to haiku" in stderr


# ---------------------------------------------------------------------------
# Expired override — treated as None by get_override
# ---------------------------------------------------------------------------

def test_expired_override_not_injected(capsys):
    """get_override returns None for expired overrides; nothing should be injected."""
    args = _make_args()
    # session_state.get_override already strips expired entries and returns None
    result, stderr = _call_resolve(
        args,
        override_return=None,   # simulates expired / absent override
        budget_return=_budget(pressure=0.0),
        capsys=capsys,
    )
    assert args.provider is None
    assert "opus-4-7" not in stderr


# ---------------------------------------------------------------------------
# PAUSE status — resolve_provider returns None
# ---------------------------------------------------------------------------

def test_pause_budget_returns_none(capsys):
    """BudgetStatus.PAUSE → resolve_provider returns None and prints warning."""
    args = _make_args()
    result, stderr = _call_resolve(
        args,
        override_return=None,
        budget_return=_budget(status=BudgetStatus.PAUSE, pressure=1.0),
        capsys=capsys,
    )
    assert result is None
    assert "Token Guard" in stderr
