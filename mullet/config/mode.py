"""MULLET_MODE toggle — solo vs enterprise."""

import os


class MulletMode:
    """Identity helpers for the `MULLET_MODE` env var (solo vs enterprise)."""

    SOLO = "solo"
    ENTERPRISE = "enterprise"

    @staticmethod
    def current() -> str:
        """Return the active mode string from `$MULLET_MODE` (defaults to "solo")."""
        return os.getenv("MULLET_MODE", "solo").lower()

    @staticmethod
    def is_solo() -> bool:
        """True when running in single-user "solo" mode."""
        return MulletMode.current() == MulletMode.SOLO

    @staticmethod
    def is_enterprise() -> bool:
        """True when running in multi-user "enterprise" mode."""
        return MulletMode.current() == MulletMode.ENTERPRISE
