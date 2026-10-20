"""Kimi K2 provider — Moonshot AI's OpenAI-compatible endpoint.

Service address: https://api.moonshot.ai/v1 (international, default).
China mainland users set KIMI_BASE_URL=https://api.moonshot.cn/v1.
Accepts KIMI_API_KEY or MOONSHOT_API_KEY (the two refer to the same key).

Default model is ``kimi-k2.7-code`` — Moonshot's strongest coding model:
256K context, native multimodal, ~30% fewer reasoning tokens vs K2.6, ~10%
better on agentic benchmarks. Thinking is ALWAYS ON for K2.7-Code (disabling
it via the API returns an error). For budget-sensitive or non-coding tasks use ``kimi-k2.5``.

Also available: ``kimi-k2.7-code-highspeed`` (~5-6x faster output, ~2x price).

Ref: https://platform.kimi.ai/docs/models
"""

import logging
import os

from mullet.core.providers.openai_compat import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# platform.kimi.ai (international) — default
# platform.moonshot.cn users should set KIMI_BASE_URL=https://api.moonshot.cn/v1
_KIMI_BASE_URL_DEFAULT = "https://api.moonshot.ai/v1"
_KIMI_DEFAULT_MODEL = "kimi-k2.7-code"

_DEPRECATED_KIMI_MODELS: dict[str, tuple[str, str]] = {
    "kimi-k2-0711-preview": ("kimi-k2.5", "2026-05-25"),
    "kimi-k2-0905-preview": ("kimi-k2.5", "2026-05-25"),
}


class KimiProvider(OpenAICompatibleProvider):
    """Kimi K2 (Moonshot AI) via OpenAI-compatible API.

    Requires KIMI_API_KEY or MOONSHOT_API_KEY environment variable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _KIMI_DEFAULT_MODEL,
        base_url: str | None = None,
    ):
        resolved_key = (
            api_key
            or os.environ.get("KIMI_API_KEY")
            or os.environ.get("MOONSHOT_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "Kimi API key not found. Set KIMI_API_KEY or MOONSHOT_API_KEY."
            )
        resolved_url = (
            base_url
            or os.environ.get("KIMI_BASE_URL")
            or _KIMI_BASE_URL_DEFAULT
        )
        super().__init__(
            base_url=resolved_url,
            api_key=resolved_key,
            model=model,
        )
        self.name = "kimi"
        if self.model in _DEPRECATED_KIMI_MODELS:
            replacement, eol_date = _DEPRECATED_KIMI_MODELS[self.model]
            logger.warning(
                "Kimi model %r is deprecated (EOL %s) — switching to %r. "
                "Update your config to suppress this warning.",
                self.model, eol_date, replacement,
            )
            self.model = replacement
        logger.debug("KimiProvider initialised (model=%s, url=%s)", self.model, resolved_url)
