"""Environment-driven settings — single source of truth."""
import os
from pathlib import Path

# .env at the root of a source checkout. The CLI and the MCP server load it
# into os.environ when they start; settings that hooks also read (see
# delegation_enabled) fall back to it, because a hook runs in Claude Code's
# environment, where nothing has loaded it.
_REPO_DOTENV = Path(__file__).resolve().parent.parent.parent / ".env"

_ON_VALUES = frozenset({"1", "on", "true", "yes", "enabled"})
_OFF_VALUES = frozenset({"0", "off", "false", "no", "disabled"})

DELEGATION_OFF_MESSAGE = (
    "Delegation is turned off (MULLET_DELEGATION=off). "
    "Set it to on in .env to use it."
)


def get_pools_dir() -> Path:
    """Return the canonical Trawl pools directory.

    Precedence:
    1. ``TRAWL_POOLS_DIR`` env var (if set and non-empty, used verbatim)
    2. Default: ``~/.mullet/pools``
    """
    override = os.environ.get("TRAWL_POOLS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".mullet" / "pools"


def _dotenv_value(key: str, path: Path | None = None) -> str | None:
    """Return ``key``'s value from the checkout's .env file, or None.

    A small line parser (KEY=value, optional quotes, # comments), so it works
    without python-dotenv. Reads only the one key asked for.
    """
    env_path = path or _REPO_DOTENV
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    return None


def _flag(value: str | None, default: bool) -> bool:
    """Read an on/off setting; anything unrecognised keeps the default."""
    if value is None:
        return default
    value = value.strip().lower()
    if value in _ON_VALUES:
        return True
    if value in _OFF_VALUES:
        return False
    return default


def delegation_enabled() -> bool:
    """True unless ``MULLET_DELEGATION`` is ``off``.

    Read from the environment first, then from the checkout's .env, so the
    Claude Code hooks see the same switch as the MCP server. Default: on.
    """
    value = os.environ.get("MULLET_DELEGATION")
    if value is None:
        value = _dotenv_value("MULLET_DELEGATION")
    return _flag(value, default=True)


def openrouter_referer() -> str | None:
    """The HTTP-Referer sent to OpenRouter for app attribution, if any.

    ``MULLET_OPENROUTER_REFERER`` (a URL). OpenRouter treats the header as
    optional, so by default it isn't sent.
    """
    value = (os.environ.get("MULLET_OPENROUTER_REFERER") or "").strip()
    return value or None


def no_ai_attribution() -> bool:
    """True when ``MULLET_NO_AI_ATTRIBUTION`` is on.

    Then `mullet init`, `mullet setup-claude` and `mullet autopilot init`
    write a "no Co-Authored-By or other AI attribution in commits" rule into
    the projects they set up. Default: off, so projects keep their own
    conventions.
    """
    return _flag(os.environ.get("MULLET_NO_AI_ATTRIBUTION"), default=False)
