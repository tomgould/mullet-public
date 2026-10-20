"""Tests for the credential guard module."""

import pytest
from mullet.core.tools.credential_guard import CredentialGuard


@pytest.fixture
def guard():
    return CredentialGuard()


# ── is_credential_file ───────────────────────────────────────────────────────

def test_env_file(guard):
    assert guard.is_credential_file("/path/to/.env")

def test_env_local(guard):
    assert guard.is_credential_file("/path/.env.local")

def test_pem_file(guard):
    assert guard.is_credential_file("/path/key.pem")

def test_id_rsa(guard):
    assert guard.is_credential_file("/home/user/.ssh/id_rsa")

def test_credentials_json(guard):
    assert guard.is_credential_file("/app/credentials.json")

def test_secrets_yaml(guard):
    assert guard.is_credential_file("/config/secrets.yaml")

def test_token_file(guard):
    assert guard.is_credential_file("/tmp/token")

def test_not_credential(guard):
    assert not guard.is_credential_file("/path/main.py")
    assert not guard.is_credential_file("/path/README.md")
    assert not guard.is_credential_file("/path/test.txt")


# ── scan_content ─────────────────────────────────────────────────────────────

def test_scan_api_key(guard):
    content = "api_key = sk-abcdef1234567890abcdef1234567890"
    assert len(guard.scan_content(content)) > 0

def test_scan_bearer_token(guard):
    content = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.test.payload"
    assert len(guard.scan_content(content)) > 0

def test_scan_aws_key(guard):
    content = "AKIAIOSFODNN7EXAMPLE"
    assert len(guard.scan_content(content)) > 0

def test_scan_private_key(guard):
    content = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAK"
    assert len(guard.scan_content(content)) > 0

def test_scan_github_pat(guard):
    content = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
    assert len(guard.scan_content(content)) > 0

def test_scan_openai_key(guard):
    content = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
    assert len(guard.scan_content(content)) > 0

def test_scan_database_url(guard):
    content = "DATABASE_URL = postgresql://user:pass@host:5432/db"
    assert len(guard.scan_content(content)) > 0

def test_scan_no_credentials(guard):
    content = "This is a safe document with no sensitive information."
    assert guard.scan_content(content) == []

def test_scan_jwt(guard):
    content = "token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    assert len(guard.scan_content(content)) > 0


# ── warn_if_credential ───────────────────────────────────────────────────────

def test_warn_credential_file(guard):
    warning = guard.warn_if_credential("/path/.env")
    assert warning is not None
    assert "WARNING" in warning

def test_no_warn_normal_file(guard):
    assert guard.warn_if_credential("/path/main.py") is None
