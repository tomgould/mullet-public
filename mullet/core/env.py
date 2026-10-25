"""Environment-driven settings — single source of truth.

Also the one place Mullet loads ``.env`` files. Precedence, highest first:

1. the process environment;
2. ``config_dir()/.env`` (``~/.config/mullet/.env`` by default);
3. the ``.env`` at the root of the source checkout, only when Mullet runs from
   a git checkout of Mullet itself.

The current working directory's ``.env`` is never read: it belongs to
whatever project Mullet happens to be working on.
"""
import os
import re
from pathlib import Path

from mullet.core import paths

_ON_VALUES = frozenset({"1", "on", "true", "yes", "enabled"})
_OFF_VALUES = frozenset({"0", "off", "false", "no", "disabled"})

DELEGATION_OFF_MESSAGE = (
    "Delegation is turned off (MULLET_DELEGATION=off). "
    "Set it to on in .env to use it."
)

_PROJECT_NAME = re.compile(r"""^\s*name\s*=\s*["']mullet["']\s*$""", re.MULTILINE)


def get_pools_dir() -> Path:
    """Return the canonical Trawl pools directory.

    Precedence:
    1. ``TRAWL_POOLS_DIR`` env var (if set and non-empty, used verbatim)
    2. Default: ``pools`` under the data directory (see ``mullet.core.paths``)
    """
    return paths.pools_dir()


def source_checkout() -> Path | None:
    """Root of the git checkout Mullet runs from, or None for an installed copy.

    It counts only as a checkout of Mullet itself: the package's parent
    directory holds a ``pyproject.toml`` naming the project ``mullet`` and a
    ``.git`` entry (a directory, or a file in a git worktree).
    """
    root = Path(__file__).resolve().parent.parent.parent
    if not (root / ".git").exists():
        return None
    try:
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    return root if _PROJECT_NAME.search(text) else None


def dotenv_files() -> list[Path]:
    """The ``.env`` files Mullet reads, highest precedence first."""
    candidates = [paths.config_dir() / ".env"]
    root = source_checkout()
    if root is not None:
        candidates.append(root / ".env")
    return [path for path in candidates if path.is_file()]


def _parse_dotenv(text: str) -> dict[str, str]:
    """A small line parser (KEY=value, optional quotes, # comments, export)."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name:
            values[name] = value.strip().strip('"').strip("'")
    return values


def read_dotenv(path: Path) -> dict[str, str]:
    """The settings in one ``.env`` file, or an empty dict if it can't be read.

    Uses python-dotenv when it is installed (quoting, escapes, inline
    comments), and the small parser above otherwise.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        dotenv_values = None
    try:
        if dotenv_values is not None:
            return {k: v for k, v in dotenv_values(path).items() if v is not None}
        return _parse_dotenv(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return {}


def load_dotenv() -> list[Path]:
    """Load Mullet's ``.env`` files into ``os.environ``; return the files read.

    A variable already in the environment is never overwritten, and a file
    earlier in ``dotenv_files()`` wins over a later one. Values are never
    printed or logged.
    """
    loaded = []
    for path in dotenv_files():
        for name, value in read_dotenv(path).items():
            os.environ.setdefault(name, value)
        loaded.append(path)
    return loaded


def dotenv_value(key: str) -> str | None:
    """``key``'s value from Mullet's ``.env`` files (not the environment), or None.

    For code that runs where nothing has loaded them, such as a Claude Code
    hook.
    """
    for path in dotenv_files():
        values = read_dotenv(path)
        if key in values:
            return values[key]
    return None


def local_llm_url() -> str:
    """Base URL of the local OpenAI-compatible server (LM Studio by default).

    ``LOCAL_LLM_URL``, default ``http://localhost:1234/v1``. It ends in
    ``/v1``; use ``local_llm_host()`` for the server root.
    """
    value = (os.environ.get("LOCAL_LLM_URL") or "").strip().rstrip("/")
    return value or "http://localhost:1234/v1"


def local_llm_host() -> str:
    """``local_llm_url()`` without its trailing ``/v1``, for LM Studio's own API."""
    url = local_llm_url()
    return url[: -len("/v1")] if url.endswith("/v1") else url


def local_preferred_models() -> list[str]:
    """Local model IDs from ``MULLET_LOCAL_PREFERRED_MODELS``, comma-separated.

    Highest preference first; empty when unset. See
    ``mullet.core.providers.lmstudio_manager.preferred_models()`` for the
    config-file and built-in fallbacks.
    """
    value = os.environ.get("MULLET_LOCAL_PREFERRED_MODELS") or ""
    return [name.strip() for name in value.split(",") if name.strip()]


def ollama_url() -> str:
    """Root URL of the Ollama server: ``OLLAMA_URL``, default ``http://localhost:11434``."""
    value = (os.environ.get("OLLAMA_URL") or "").strip().rstrip("/")
    return value or "http://localhost:11434"


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


def delegation_switch_on() -> bool:
    """True unless ``MULLET_DELEGATION`` alone is off: the environment first, then Mullet's .env files."""
    value = os.environ.get("MULLET_DELEGATION")
    if value is None:
        value = dotenv_value("MULLET_DELEGATION")
    return _flag(value, default=True)


def _strategy_off() -> bool:
    """True when a fresh session strategy says ``delegation: off``; any error counts as no."""
    try:
        from mullet.core.strategy import delegation_off

        return delegation_off()
    except Exception:
        return False


def delegation_enabled() -> bool:
    """True unless ``MULLET_DELEGATION`` is ``off`` or the session strategy says off.

    The switch is read from the environment first, then from Mullet's .env
    files, so the Claude Code hooks see the same switch as the MCP server.
    A fresh ``mullet strategy`` with ``delegation: off`` counts the same; a
    strategy that can't be read changes nothing. Default: on.
    """
    return delegation_switch_on() and not _strategy_off()


def delegation_off_message() -> str:
    """Why delegation is off: names ``MULLET_DELEGATION`` or the session strategy."""
    if not delegation_switch_on():
        return DELEGATION_OFF_MESSAGE
    from mullet.core.strategy import STRATEGY_OFF_MESSAGE

    return STRATEGY_OFF_MESSAGE


def openrouter_referer() -> str | None:
    """The HTTP-Referer sent to OpenRouter for app attribution, if any.

    ``MULLET_OPENROUTER_REFERER`` (a URL). OpenRouter treats the header as
    optional, so by default it isn't sent.
    """
    value = (os.environ.get("MULLET_OPENROUTER_REFERER") or "").strip()
    return value or None


def no_ai_attribution() -> bool:
    """True when ``MULLET_NO_AI_ATTRIBUTION`` is on.

    Then `mullet init` and `mullet setup-claude` write a "no Co-Authored-By
    or other AI attribution in commits" rule into the projects they set up.
    Default: off, so projects keep their own conventions.
    """
    return _flag(os.environ.get("MULLET_NO_AI_ATTRIBUTION"), default=False)
