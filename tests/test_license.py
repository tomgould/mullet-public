"""Tests for license checker."""

import base64
import json
import time
import pytest
from mullet.config.license import LicenseChecker, LicenseError


def test_solo_mode_skips_check(monkeypatch):
    monkeypatch.setenv("MULLET_MODE", "solo")
    checker = LicenseChecker()
    checker.check_on_startup()  # Should not raise


def test_enterprise_no_key_trial(monkeypatch, tmp_path):
    monkeypatch.setenv("MULLET_MODE", "enterprise")
    monkeypatch.delenv("MULLET_LICENSE_KEY", raising=False)
    # Point trial file to temp
    monkeypatch.setattr("mullet.config.license.Path.home", lambda: tmp_path)

    checker = LicenseChecker()
    # First call: trial starts (14 days remaining)
    checker.check_on_startup()  # Should not raise (trial active)


def test_valid_license_key(monkeypatch, tmp_path):
    monkeypatch.setenv("MULLET_MODE", "enterprise")
    claims = {
        "customer": "Test Corp",
        "max_users": 50,
        "expires_at": time.time() + 86400 * 365,
    }
    key = base64.b64encode(json.dumps(claims).encode()).decode()
    monkeypatch.setenv("MULLET_LICENSE_KEY", key)

    checker = LicenseChecker()
    monkeypatch.setattr(checker, "_count_users", lambda: 5)
    checker.check_on_startup()  # Should not raise


def test_expired_license(monkeypatch):
    monkeypatch.setenv("MULLET_MODE", "enterprise")
    claims = {
        "customer": "Expired Corp",
        "expires_at": time.time() - 86400,
    }
    key = base64.b64encode(json.dumps(claims).encode()).decode()
    monkeypatch.setenv("MULLET_LICENSE_KEY", key)

    checker = LicenseChecker()
    with pytest.raises(LicenseError, match="expired"):
        checker.check_on_startup()


def test_invalid_key_format(monkeypatch):
    monkeypatch.setenv("MULLET_MODE", "enterprise")
    monkeypatch.setenv("MULLET_LICENSE_KEY", "not-valid-base64-key")

    checker = LicenseChecker()
    with pytest.raises(LicenseError, match="Invalid"):
        checker.check_on_startup()


def test_trial_expired_raises(monkeypatch, tmp_path):
    """Line 42: trial_remaining <= 0 → LicenseError."""
    monkeypatch.setenv("MULLET_MODE", "enterprise")
    monkeypatch.delenv("MULLET_LICENSE_KEY", raising=False)
    monkeypatch.setattr("mullet.config.license.Path.home", lambda: tmp_path)

    checker = LicenseChecker()
    monkeypatch.setattr(checker, "_trial_days_remaining", lambda: 0)
    with pytest.raises(LicenseError, match="Enterprise license required"):
        checker.check_on_startup()


def test_user_limit_exceeded_raises(monkeypatch, tmp_path):
    """Line 64: current > max_users → LicenseError."""
    monkeypatch.setenv("MULLET_MODE", "enterprise")
    claims = {"customer": "Tiny Corp", "max_users": 2}
    key = base64.b64encode(json.dumps(claims).encode()).decode()
    monkeypatch.setenv("MULLET_LICENSE_KEY", key)

    checker = LicenseChecker()
    monkeypatch.setattr(checker, "_count_users", lambda: 5)
    with pytest.raises(LicenseError, match="License allows 2 users"):
        checker.check_on_startup()


def test_read_license_file_from_disk(monkeypatch, tmp_path):
    """Line 96: license key read from .mullet/license.key file."""
    monkeypatch.setenv("MULLET_MODE", "enterprise")
    monkeypatch.delenv("MULLET_LICENSE_KEY", raising=False)

    claims = {"customer": "File Corp", "max_users": 10}
    key = base64.b64encode(json.dumps(claims).encode()).decode()

    license_path = tmp_path / ".mullet" / "license.key"
    license_path.parent.mkdir(parents=True)
    license_path.write_text(key)

    import mullet.config.license as lic_mod
    monkeypatch.setattr(lic_mod, "LICENSE_FILE_LOCATIONS", [str(license_path)])

    checker = LicenseChecker()
    monkeypatch.setattr(checker, "_count_users", lambda: 1)
    checker.check_on_startup()  # Should not raise


def test_trial_days_remaining_elapsed(monkeypatch, tmp_path):
    """Lines 107-112: trial file exists with old timestamp → computes remaining days."""
    from mullet.core.paths import state_dir
    trial_file = state_dir() / ".trial_start"
    trial_file.parent.mkdir(parents=True, exist_ok=True)
    # Set start 7 days ago → 7 days remaining
    trial_file.write_text(str(time.time() - 86400 * 7))

    checker = LicenseChecker()
    remaining = checker._trial_days_remaining()
    assert 6 <= remaining <= 7


def test_trial_days_remaining_corrupt_file(monkeypatch, tmp_path):
    """Line 116: trial file has corrupt content → returns 0."""
    from mullet.core.paths import state_dir
    trial_file = state_dir() / ".trial_start"
    trial_file.parent.mkdir(parents=True, exist_ok=True)
    trial_file.write_text("not-a-float")

    checker = LicenseChecker()
    assert checker._trial_days_remaining() == 0


def test_count_users_db_exception(monkeypatch):
    """Lines 116-122: _count_users swallows exceptions → returns 0."""
    checker = LicenseChecker()
    import sys
    monkeypatch.setitem(sys.modules, "mullet.server.db", None)
    assert checker._count_users() == 0


def test_count_users_db_success(monkeypatch):
    """Lines 118-120: _count_users reads from real DB mock → returns count."""
    from unittest.mock import MagicMock
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchone.return_value = (7,)

    mock_module = MagicMock()
    mock_module.init_db.return_value = mock_db

    import sys
    monkeypatch.setitem(sys.modules, "mullet.server.db", mock_module)

    checker = LicenseChecker()
    assert checker._count_users() == 7
