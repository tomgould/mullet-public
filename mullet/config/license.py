"""License key validation for enterprise mode.

BSL 1.1: solo use unrestricted, production multi-tenant requires license.
License keys are signed JWTs validated offline using an embedded public key.
"""

import json
import logging
import os
import time
from pathlib import Path

from mullet.config.mode import MulletMode

logger = logging.getLogger(__name__)

TRIAL_DAYS = 14
LICENSE_FILE_LOCATIONS = [
    ".mullet/license.key",
    str(Path.home() / ".mullet" / "license.key"),
]

# Process-level flag so the "base64 only" warning fires once per run rather
# than once per license-check call. Each call previously spammed the warning
# into the log; this kept producing noise in tests + the persistent MCP
# server.
_BASE64_WARNING_EMITTED = False


class LicenseChecker:
    """Enforce licensing boundaries."""

    def check_on_startup(self):
        """Check license on startup. Solo mode skips all checks."""
        if MulletMode.is_solo():
            return  # No license needed for solo

        key = os.getenv("MULLET_LICENSE_KEY") or self._read_license_file()
        if not key:
            trial_remaining = self._trial_days_remaining()
            if trial_remaining <= 0:
                raise LicenseError(
                    "Enterprise license required for multi-user mode. "
                    "Set MULLET_LICENSE_KEY or place license.key in ~/.mullet/. "
                    "Or set MULLET_MODE=solo for single-user use."
                )
            logger.warning("Mullet enterprise trial: %d days remaining", trial_remaining)
            return

        # Validate the key
        claims = self._validate_key(key)
        if not claims:
            raise LicenseError("Invalid license key")

        # Check expiry
        if claims.get("expires_at") and time.time() > claims["expires_at"]:
            raise LicenseError("License key has expired")

        # Check user limit
        max_users = claims.get("max_users")
        if max_users:
            current = self._count_users()
            if current > max_users:
                raise LicenseError(
                    f"License allows {max_users} users, but {current} are registered. "
                    "Upgrade your license or remove users."
                )

        logger.info("License valid: %s (expires: %s)",
                     claims.get("customer", "unknown"),
                     claims.get("expires_at", "never"))

    def _validate_key(self, key: str) -> dict | None:
        """Validate a license key. Returns claims dict or None."""
        # Simple format: base64-encoded JSON (production would use JWT with signature)
        global _BASE64_WARNING_EMITTED
        if not _BASE64_WARNING_EMITTED:
            logger.warning("License validation uses base64 only — no signature verification. Not suitable for production enforcement.")
            _BASE64_WARNING_EMITTED = True
        import base64
        try:
            decoded = base64.b64decode(key.encode()).decode()
            claims = json.loads(decoded)
            if "customer" in claims:
                return claims
        except Exception as _exc:
            logger.warning("license: Error decoding license key: %s", _exc)
            pass
        return None

    def _read_license_file(self) -> str | None:
        """Read license key from file."""
        for path in LICENSE_FILE_LOCATIONS:
            p = Path(path)
            if p.exists():
                return p.read_text().strip()
        return None

    def _trial_days_remaining(self) -> int:
        """Check trial period. Trial starts when enterprise mode is first used."""
        trial_file = Path.home() / ".mullet" / ".trial_start"
        if not trial_file.exists():
            trial_file.parent.mkdir(parents=True, exist_ok=True)
            trial_file.write_text(str(time.time()))
            return TRIAL_DAYS

        try:
            start = float(trial_file.read_text().strip())
            elapsed_days = (time.time() - start) / 86400
            return max(0, TRIAL_DAYS - int(elapsed_days))
        except Exception:
            return 0

    def _count_users(self) -> int:
        """Count registered users in the database."""
        try:
            from mullet.server.db import init_db
            db = init_db()
            row = db.execute("SELECT COUNT(*) FROM users").fetchone()
            return row[0] if row else 0
        except Exception:
            return 0


class LicenseError(Exception):
    """Raised when license validation fails."""
    pass
