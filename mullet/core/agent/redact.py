"""Secret redaction for logs and tool output.

Applies pattern matching to mask API keys, tokens, and credentials
before they reach log files, verbose output, or streamed responses.

Short tokens (< 18 chars) are fully masked. Longer tokens preserve
the first 6 and last 4 characters for debuggability.

Ported from Hermes agent engine.
"""

import logging
import os
import re

# Snapshot at import time — runtime env mutations cannot disable redaction mid-session
_REDACT_ENABLED = os.getenv("MULLET_REDACT_SECRETS", "true").lower() not in ("0", "false", "no", "off")

# Known API key prefixes — match prefix + contiguous token chars
_PREFIX_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{10,}",           # OpenAI / OpenRouter / Anthropic
    r"ghp_[A-Za-z0-9]{10,}",            # GitHub PAT (classic)
    r"github_pat_[A-Za-z0-9_]{10,}",    # GitHub PAT (fine-grained)
    r"gho_[A-Za-z0-9]{10,}",            # GitHub OAuth
    r"ghu_[A-Za-z0-9]{10,}",            # GitHub user-to-server
    r"ghs_[A-Za-z0-9]{10,}",            # GitHub server-to-server
    r"ghr_[A-Za-z0-9]{10,}",            # GitHub refresh
    r"xox[baprs]-[A-Za-z0-9-]{10,}",    # Slack tokens
    r"AIza[A-Za-z0-9_-]{30,}",          # Google API keys
    r"AKIA[A-Z0-9]{16}",                # AWS Access Key ID
    r"sk_live_[A-Za-z0-9]{10,}",        # Stripe live
    r"sk_test_[A-Za-z0-9]{10,}",        # Stripe test
    r"SG\.[A-Za-z0-9_-]{10,}",          # SendGrid
    r"hf_[A-Za-z0-9]{10,}",             # HuggingFace
    r"r8_[A-Za-z0-9]{10,}",             # Replicate
    r"npm_[A-Za-z0-9]{10,}",            # npm access token
    r"pypi-[A-Za-z0-9_-]{10,}",         # PyPI token
    r"pplx-[A-Za-z0-9]{10,}",           # Perplexity
    r"tvly-[A-Za-z0-9]{10,}",           # Tavily
    r"exa_[A-Za-z0-9]{10,}",            # Exa search
]

# ENV assignment: KEY=value where KEY looks like a secret
_SECRET_ENV_NAMES = r"(?:API_?KEY|AUTH_?KEY|CLIENT_?SECRET|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)"
_ENV_ASSIGN_RE = re.compile(
    rf"([A-Z_]*{_SECRET_ENV_NAMES}[A-Z_]*)\s*=\s*(['\"]?)(\S+)\2",
    re.IGNORECASE,
)

# JSON fields: "apiKey": "value", "token": "value", etc.
_JSON_KEY_NAMES = r"(?:api_?[Kk]ey|token|secret|password|access_token|refresh_token|auth_token|bearer|client_secret|private_key)"
_JSON_FIELD_RE = re.compile(
    rf'("{_JSON_KEY_NAMES}")\s*:\s*"([^"]+)"',
    re.IGNORECASE,
)

# Authorization headers
_AUTH_HEADER_RE = re.compile(
    r"(Authorization:\s*Bearer\s+)(\S+)",
    re.IGNORECASE,
)

# Private key blocks
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----"
)

# Database connection string passwords
_DB_CONNSTR_RE = re.compile(
    r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:]+:)([^@]+)(@)",
    re.IGNORECASE,
)

# Compile known prefix patterns into one alternation
_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(" + "|".join(_PREFIX_PATTERNS) + r")(?![A-Za-z0-9_-])"
)


def _mask_token(token: str) -> str:
    """Mask a token, preserving prefix for long tokens."""
    if len(token) < 18:
        return "***"
    return f"{token[:6]}...{token[-4:]}"


def redact_sensitive_text(text: str | None) -> str | None:
    """Apply all redaction patterns to a block of text.

    Safe to call on any string — non-matching text passes through unchanged.
    Disabled when MULLET_REDACT_SECRETS=false.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    if not _REDACT_ENABLED:
        return text

    # Known prefixes (sk-, ghp_, etc.)
    text = _PREFIX_RE.sub(lambda m: _mask_token(m.group(1)), text)

    # ENV assignments: OPENAI_API_KEY=sk-abc...
    def _redact_env(m):
        name, quote, value = m.group(1), m.group(2), m.group(3)
        return f"{name}={quote}{_mask_token(value)}{quote}"
    text = _ENV_ASSIGN_RE.sub(_redact_env, text)

    # JSON fields: "apiKey": "value"
    def _redact_json(m):
        key, value = m.group(1), m.group(2)
        return f'{key}: "{_mask_token(value)}"'
    text = _JSON_FIELD_RE.sub(_redact_json, text)

    # Authorization headers
    text = _AUTH_HEADER_RE.sub(
        lambda m: m.group(1) + _mask_token(m.group(2)),
        text,
    )

    # Private key blocks
    text = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)

    # Database connection string passwords
    text = _DB_CONNSTR_RE.sub(lambda m: f"{m.group(1)}***{m.group(3)}", text)

    return text


class RedactingFormatter(logging.Formatter):
    """Log formatter that redacts secrets from all log messages."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with secret redaction applied."""
        original = super().format(record)
        return redact_sensitive_text(original)
