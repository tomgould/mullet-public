"""Tests for secret redaction engine."""

import mullet.core.agent.redact as _redact_mod
from unittest.mock import patch

from mullet.core.agent.redact import redact_sensitive_text, _mask_token


def test_mask_short_token():
    assert _mask_token("short") == "***"


def test_mask_long_token():
    result = _mask_token("sk-abcdefghijklmnopqrstuvwxyz")
    assert result.startswith("sk-abc")
    assert result.endswith("wxyz")
    assert "..." in result


def test_none_passthrough():
    assert redact_sensitive_text(None) is None


def test_empty_passthrough():
    assert redact_sensitive_text("") == ""


def test_normal_text_unchanged():
    text = "This is a normal sentence with no secrets."
    assert redact_sensitive_text(text) == text


def test_openai_key():
    text = "My key is sk-abc123def456ghi789jkl012mno345"
    result = redact_sensitive_text(text)
    assert "sk-abc" not in result or "..." in result
    assert "abc123def456" not in result


def test_github_pat():
    text = "Token: ghp_1234567890abcdef1234567890abcdef12345678"
    result = redact_sensitive_text(text)
    assert "1234567890abcdef" not in result


def test_github_fine_grained():
    text = "github_pat_abcdefghij1234567890_longtoken"
    result = redact_sensitive_text(text)
    assert "abcdefghij" not in result


def test_slack_token():
    text = "xoxb-123456789012-abcdefghij"
    result = redact_sensitive_text(text)
    assert "123456789012" not in result


def test_aws_key():
    text = "AWS key: AKIAIOSFODNN7EXAMPLE"
    result = redact_sensitive_text(text)
    assert "IOSFODNN7EXAMPLE" not in result


def test_stripe_key():
    text = "sk_live_abc123def456ghi789jkl012mno"
    result = redact_sensitive_text(text)
    assert "abc123def456" not in result


def test_sendgrid_key():
    text = "SG.abcdefghijklmnop.qrstuvwxyz"
    result = redact_sensitive_text(text)
    assert "abcdefghijklmnop" not in result


def test_huggingface_token():
    text = "hf_abcdefghijklmnopqrst"
    result = redact_sensitive_text(text)
    assert "abcdefghijklmnopqrst" not in result


def test_env_assignment():
    text = "export OPENAI_API_KEY=sk-verylongsecretapikey1234567890"
    result = redact_sensitive_text(text)
    assert "OPENAI_API_KEY=" in result  # key name preserved
    assert "verylongsecret" not in result


def test_json_field():
    text = '{"apiKey": "super_secret_value_12345678"}'
    result = redact_sensitive_text(text)
    assert "super_secret" not in result
    assert '"apiKey"' in result  # field name preserved


def test_auth_header():
    text = "Authorization: Bearer sk-verylongtokenabcdef1234567890"
    result = redact_sensitive_text(text)
    assert "Authorization: Bearer" in result
    assert "verylongtoken" not in result


def test_private_key():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIE...long...key\n-----END RSA PRIVATE KEY-----"
    result = redact_sensitive_text(text)
    assert "[REDACTED PRIVATE KEY]" in result
    assert "MIIE" not in result


def test_postgres_connection():
    text = "postgres://admin:super_secret_pw@db.example.com:5432/mydb"
    result = redact_sensitive_text(text)
    assert "super_secret_pw" not in result
    assert "postgres://admin:" in result
    assert "@db.example.com" in result


def test_mysql_connection():
    text = "mysql://root:p@ssw0rd@localhost/app"
    result = redact_sensitive_text(text)
    assert "p@ssw0rd" not in result


def test_mongodb_connection():
    text = "mongodb+srv://user:secretpass@cluster0.example.net/db"
    result = redact_sensitive_text(text)
    assert "secretpass" not in result


def test_multiple_secrets_in_one_text():
    text = (
        "export ANTHROPIC_API_KEY=sk-ant-abc123def456ghi789jkl\n"
        "ghp_1234567890abcdefghijklmno\n"
        "postgres://user:dbpass123@localhost/app\n"
    )
    result = redact_sensitive_text(text)
    assert "abc123def456" not in result
    assert "1234567890abcdef" not in result
    assert "dbpass123" not in result


def test_non_string_coerced():
    assert redact_sensitive_text(42) == "42"
    assert redact_sensitive_text(True) == "True"


def test_redact_disabled_returns_text_unchanged():
    """Line 100: _REDACT_ENABLED=False → early return without scanning."""
    text = "sk-abcdefghijklmnopqrstuvwxyz"
    with patch.object(_redact_mod, "_REDACT_ENABLED", False):
        result = redact_sensitive_text(text)
    assert result == text


def test_redacting_formatter_redacts_secrets():
    """Lines 137-138: RedactingFormatter.format() calls redact on log output."""
    import logging
    from mullet.core.agent.redact import RedactingFormatter
    formatter = RedactingFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="secret key: sk-abcdefghijklmnopqrstuvwxyz", args=(), exc_info=None,
    )
    result = formatter.format(record)
    assert "abcdefghijklm" not in result
