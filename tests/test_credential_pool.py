"""Tests for credential pool."""

import time
from mullet.core.providers.credential_pool import CredentialPool, Credential


def test_single_credential():
    pool = CredentialPool()
    pool.add("sk-key-1", "key1")
    cred = pool.get_next()
    assert cred is not None
    assert cred.api_key == "sk-key-1"


def test_round_robin():
    pool = CredentialPool(strategy="round_robin")
    pool.add("sk-1", "a")
    pool.add("sk-2", "b")
    pool.add("sk-3", "c")

    keys = [pool.get_next().api_key for _ in range(6)]
    assert keys == ["sk-1", "sk-2", "sk-3", "sk-1", "sk-2", "sk-3"]


def test_least_used():
    pool = CredentialPool(strategy="least_used")
    pool.add("sk-1", "a")
    pool.add("sk-2", "b")

    # Use sk-1 twice
    c1 = pool.get_next()  # Gets least used (either)
    pool.get_next()  # Gets the other

    # Now sk-2 should be least used
    c = pool.get_next()
    # One of them should have 1 request, the other 2


def test_exhaustion_cooldown():
    pool = CredentialPool()
    pool.add("sk-1", "a")
    pool.add("sk-2", "b")

    pool.mark_exhausted("sk-1", error_code=429)
    cred = pool.get_next()
    assert cred.api_key == "sk-2"  # sk-1 is cooling down


def test_all_exhausted():
    pool = CredentialPool()
    pool.add("sk-1", "a")

    pool.mark_exhausted("sk-1")
    cred = pool.get_next()
    assert cred is None  # No available credentials


def test_success_resets_failures():
    pool = CredentialPool()
    pool.add("sk-1", "a")

    pool.mark_exhausted("sk-1")
    # Manually reset cooldown for test
    pool._credentials[0].exhausted_until = 0

    pool.mark_success("sk-1", tokens=100)
    assert pool._credentials[0].consecutive_failures == 0
    assert pool._credentials[0].total_tokens == 100


def test_available_count():
    pool = CredentialPool()
    pool.add("sk-1")
    pool.add("sk-2")

    assert pool.available_count == 2
    pool.mark_exhausted("sk-1")
    assert pool.available_count == 1


def test_status():
    pool = CredentialPool()
    pool.add("sk-1", "key1")
    pool.get_next()

    status = pool.status()
    assert len(status) == 1
    assert status[0]["name"] == "key1"
    assert status[0]["requests"] == 1


def test_credential_is_available():
    c = Credential(api_key="sk-1")
    assert c.is_available is True

    c.exhausted_until = time.time() + 100
    assert c.is_available is False

    c.exhausted_until = time.time() - 1
    assert c.is_available is True


def test_cooldown_remaining_zero_when_not_exhausted():
    c = Credential(api_key="sk-1")
    assert c.cooldown_remaining == 0


def test_cooldown_remaining_positive_when_exhausted():
    c = Credential(api_key="sk-1")
    c.exhausted_until = time.time() + 100
    assert c.cooldown_remaining > 0


def test_empty_pool_returns_none():
    pool = CredentialPool()
    assert pool.get_next() is None


def test_total_count():
    pool = CredentialPool()
    assert pool.total_count == 0
    pool.add("sk-1")
    pool.add("sk-2")
    assert pool.total_count == 2


def test_mark_exhausted_402_quota():
    pool = CredentialPool()
    pool.add("sk-1", "key1")
    pool.mark_exhausted("sk-1", error_code=402)
    c = pool._credentials[0]
    # 24h cooldown
    assert c.exhausted_until > time.time() + 80000


def test_mark_exhausted_unknown_key_noop():
    pool = CredentialPool()
    pool.add("sk-1")
    # Should not raise; unknown key is silently ignored
    pool.mark_exhausted("sk-unknown")
    assert pool.available_count == 1


def test_mark_success_updates_tokens():
    pool = CredentialPool()
    pool.add("sk-1", "key1")
    pool.mark_success("sk-1", tokens=500)
    assert pool._credentials[0].total_tokens == 500


def test_status_empty_pool():
    pool = CredentialPool()
    assert pool.status() == []


def test_round_robin_skips_exhausted():
    pool = CredentialPool(strategy="round_robin")
    pool.add("sk-1", "a")
    pool.add("sk-2", "b")
    pool.mark_exhausted("sk-1")
    cred = pool.get_next()
    assert cred.api_key == "sk-2"
