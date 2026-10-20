"""Credential guard — detect sensitive files and credential patterns.

Warns before reading credential files (.env, .pem, id_rsa, etc.)
and scans content for API keys, tokens, and passwords.
"""

import os
import re


# Filenames that commonly contain credentials
CREDENTIAL_FILENAMES = frozenset({
    ".env", ".env.local", ".env.development", ".env.production", ".env.staging",
    ".pem", ".key", ".p12", ".pfx", ".cer", ".crt",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
    "credentials.json", "client_secret.json", "service_account.json",
    "secrets.yaml", "secrets.yml", ".htpasswd",
    "token", "api_key", "auth_token", "access_token", "refresh_token",
    "private_key.txt", "secret.txt", "password.txt",
})

# File extensions that are credential-like
CREDENTIAL_EXTENSIONS = frozenset({
    ".pem", ".key", ".p12", ".pfx", ".cer", ".crt", ".jks", ".keystore",
})

# Regex patterns for credential-like content
CREDENTIAL_PATTERNS = [
    re.compile(r"(?i)api[_-]?key\s*[=:]\s*\S{16,}"),
    re.compile(r"(?i)token\s*[=:]\s*\S{16,}"),
    re.compile(r"(?i)password\s*[=:]\s*\S{8,}"),
    re.compile(r"(?i)secret\s*[=:]\s*\S{16,}"),
    re.compile(r"(?i)authorization:\s*bearer\s+\S{16,}"),
    re.compile(r"(?i)x-api-key:\s*\S{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key
    re.compile(r"(?i)aws_secret_access_key\s*[=:]\s*\S{30,}"),
    re.compile(r"(?i)client_secret\s*[=:]\s*\S{16,}"),
    re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----"),
    re.compile(r"ssh-(?:rsa|ed25519|ecdsa)\s+[A-Za-z0-9+/]{30,}"),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),  # GitHub PAT
    re.compile(r"gho_[A-Za-z0-9]{36}"),  # GitHub OAuth
    re.compile(r"sk-[A-Za-z0-9_-]{32,}"),  # OpenAI key
    re.compile(r"sk_live_[A-Za-z0-9]{24,}"),  # Stripe live key
    re.compile(r"xoxb-[0-9]{10,13}-[0-9]{10,13}-[A-Za-z0-9]{24}"),  # Slack bot
    re.compile(r"(?i)database_url\s*[=:]\s*\w+://\S+@\S+"),
    re.compile(r"AIza[A-Za-z0-9_-]{35}"),  # Google API key
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # JWT
    re.compile(r"(?i)smtp_password\s*[=:]\s*\S{8,}"),
]


class CredentialGuard:
    """Detect credential files and patterns in content."""

    def is_credential_file(self, path: str) -> bool:
        """Check if a file path looks like a credential file."""
        basename = os.path.basename(path).lower()
        ext = os.path.splitext(path)[1].lower()

        # Check filename
        if basename in CREDENTIAL_FILENAMES:
            return True

        # Check extension
        if ext in CREDENTIAL_EXTENSIONS:
            return True

        # Check if filename contains credential-like words
        for pattern in CREDENTIAL_FILENAMES:
            if pattern in basename:
                return True

        return False

    def scan_content(self, text: str) -> list[str]:
        """Scan text for credential-like patterns. Returns list of matched pattern descriptions."""
        matches: list[str] = []
        for pattern in CREDENTIAL_PATTERNS:
            if pattern.search(text):
                matches.append(pattern.pattern)
        return matches

    def warn_if_credential(self, path: str) -> str | None:
        """Return a warning message if the path is a credential file, else None."""
        if self.is_credential_file(path):
            return (
                f"WARNING: '{os.path.basename(path)}' appears to be a credential file. "
                "Reading it may expose sensitive data to the LLM context."
            )
        return None
