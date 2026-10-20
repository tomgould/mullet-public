"""Remember models a provider has said it can no longer serve.

OpenRouter withdraws models, and especially `:free` slots, without notice. A
request for one then fails with a 404 such as "This model is unavailable for
free. The paid version is available now - use this slug instead: ...". The
provider records the model here and moves to its next candidate, and the
routers leave recorded models out of their candidate lists, so later tasks
don't hit the same dead model.

Same persisted-JSON-with-TTL pattern as opencode_tool's known-bad models and
the learned constraints: entries expire after a day, so a model that comes
back is tried again without anyone clearing a cache.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Tests point this at a temporary file; None means ~/.mullet/unavailable_models.json.
STORE_PATH: Path | None = None

TTL_SECONDS = 24 * 3600

_lock = threading.Lock()

# Messages OpenRouter (and OpenAI-compatible servers generally) use when the
# model itself, rather than the request, is the problem.
_UNAVAILABLE_PATTERNS = re.compile(
    r"unavailable for free"
    r"|no endpoints found"
    r"|is not a valid model id"
    r"|model[^.]{0,80}?\b(?:not found|does not exist|is unavailable|is no longer available|has been deprecated)",
    re.IGNORECASE,
)
_SUGGESTED_SLUG = re.compile(r"use this slug instead:\s*([A-Za-z0-9._:/-]+)", re.IGNORECASE)


def _store_path() -> Path:
    return STORE_PATH or Path.home() / ".mullet" / "unavailable_models.json"


def is_model_unavailable_error(exc: BaseException) -> bool:
    """True when ``exc`` says the requested model can't be served at all.

    Matches OpenRouter's wording for a withdrawn or unknown model (the 404
    "unavailable for free" and "No endpoints found for ..." replies, the 400
    "... is not a valid model ID") and any other 404 that talks about the
    model. Rate limits, server errors and bad parameters are not matched:
    those are retried or reshaped elsewhere.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return False
    message = str(exc)
    if _UNAVAILABLE_PATTERNS.search(message):
        return True
    return status == 404 and "model" in message.lower()


def suggested_replacement(exc: BaseException) -> str | None:
    """The slug OpenRouter suggests in its error, if it names one."""
    match = _SUGGESTED_SLUG.search(str(exc))
    return match.group(1).rstrip(".,;") if match else None


def _load() -> dict:
    """{provider: {model: {"message", "ts"}}}, still-fresh entries only."""
    path = _store_path()
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
    except Exception:
        return {}
    cutoff = time.time() - TTL_SECONDS
    fresh: dict = {}
    for provider, models in (data or {}).items():
        if not isinstance(models, dict):
            continue
        kept = {m: v for m, v in models.items() if isinstance(v, dict) and v.get("ts", 0) >= cutoff}
        if kept:
            fresh[provider] = kept
    return fresh


def mark_unavailable(provider: str, model: str, message: str = "") -> None:
    """Record that ``provider`` can't serve ``model``. Never raises."""
    try:
        with _lock:
            data = _load()
            data.setdefault(provider, {})[model] = {"message": message[:300], "ts": time.time()}
            path = _store_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=1)
                os.replace(tmp, path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
    except Exception as exc:
        logger.debug("model_availability: could not record %s/%s: %s", provider, model, exc)


def unavailable_models(provider: str) -> set[str]:
    """Models ``provider`` has recently said it can't serve."""
    return set(_load().get(provider, {}))


def is_unavailable(provider: str, model: str) -> bool:
    return model in unavailable_models(provider)


def without_unavailable(provider: str, candidates: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """``candidates`` ((model, context) pairs) minus recorded unavailable models."""
    bad = unavailable_models(provider)
    if not bad:
        return list(candidates)
    return [(m, ctx) for m, ctx in candidates if m not in bad]
