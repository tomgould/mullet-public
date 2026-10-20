"""Tests for mullet.core.providers.credential_pool — Credential and CredentialPool."""

import time
from mullet.core.providers.credential_pool import Credential, CredentialPool


# ---------------------------------------------------------------------------
# Credential
# ---------------------------------------------------------------------------

class TestCredential:
    def test_defaults(self):
        c = Credential(api_key="abc123")
        assert c.name == ""
        assert c.total_requests == 0
        assert c.total_tokens == 0
        assert c.exhausted_until == 0
        assert c.consecutive_failures == 0

    def test_is_available_by_default(self):
        c = Credential(api_key="key")
        assert c.is_available is True

    def test_not_available_during_cooldown(self):
        c = Credential(api_key="key", exhausted_until=time.time() + 3600)
        assert c.is_available is False

    def test_available_after_cooldown_expires(self):
        c = Credential(api_key="key", exhausted_until=time.time() - 1)
        assert c.is_available is True

    def test_cooldown_remaining_zero_when_not_exhausted(self):
        c = Credential(api_key="key")
        assert c.cooldown_remaining == 0

    def test_cooldown_remaining_positive_during_cooldown(self):
        c = Credential(api_key="key", exhausted_until=time.time() + 100)
        assert c.cooldown_remaining > 0
        assert c.cooldown_remaining <= 100

    def test_cooldown_remaining_zero_after_expiry(self):
        c = Credential(api_key="key", exhausted_until=time.time() - 10)
        assert c.cooldown_remaining == 0


# ---------------------------------------------------------------------------
# CredentialPool — basic operations
# ---------------------------------------------------------------------------

class TestCredentialPoolAdd:
    def test_add_increases_count(self):
        pool = CredentialPool()
        pool.add("key1")
        assert pool.total_count == 1

    def test_add_sets_name(self):
        pool = CredentialPool()
        pool.add("key1", name="primary")
        assert pool.status()[0]["name"] == "primary"

    def test_add_autogenerates_name(self):
        pool = CredentialPool()
        pool.add("key1")
        assert pool.status()[0]["name"].startswith("key_")

    def test_empty_pool_count_zero(self):
        pool = CredentialPool()
        assert pool.total_count == 0
        assert pool.available_count == 0


# ---------------------------------------------------------------------------
# CredentialPool.get_next — strategies
# ---------------------------------------------------------------------------

class TestCredentialPoolGetNext:
    def test_empty_pool_returns_none(self):
        pool = CredentialPool()
        assert pool.get_next() is None

    def test_returns_credential(self):
        pool = CredentialPool()
        pool.add("key1")
        cred = pool.get_next()
        assert cred is not None
        assert cred.api_key == "key1"

    def test_increments_total_requests(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.get_next()
        pool.get_next()
        assert pool.status()[0]["requests"] == 2

    def test_round_robin_cycles(self):
        pool = CredentialPool(strategy="round_robin")
        pool.add("key1", name="k1")
        pool.add("key2", name="k2")
        first = pool.get_next()
        second = pool.get_next()
        assert first.api_key != second.api_key

    def test_least_used_picks_lower_count(self):
        pool = CredentialPool(strategy="least_used")
        pool.add("key1")
        pool.add("key2")
        # Exhaust key1 manually
        pool._credentials[0].total_requests = 100
        cred = pool.get_next()
        assert cred.api_key == "key2"

    def test_random_strategy_returns_credential(self):
        pool = CredentialPool(strategy="random")
        pool.add("key1")
        pool.add("key2")
        cred = pool.get_next()
        assert cred is not None

    def test_all_exhausted_returns_none(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.mark_exhausted("key1", error_code=429)
        assert pool.get_next() is None


# ---------------------------------------------------------------------------
# CredentialPool.mark_exhausted
# ---------------------------------------------------------------------------

class TestMarkExhausted:
    def test_429_sets_1h_cooldown(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.mark_exhausted("key1", error_code=429)
        cred = pool._credentials[0]
        assert cred.cooldown_remaining > 3500

    def test_402_sets_24h_cooldown(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.mark_exhausted("key1", error_code=402)
        cred = pool._credentials[0]
        assert cred.cooldown_remaining > 86000

    def test_increments_consecutive_failures(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.mark_exhausted("key1", error_code=429)
        assert pool._credentials[0].consecutive_failures == 1

    def test_unknown_key_no_crash(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.mark_exhausted("nonexistent_key", error_code=429)  # Should not raise


# ---------------------------------------------------------------------------
# CredentialPool.mark_success
# ---------------------------------------------------------------------------

class TestMarkSuccess:
    def test_resets_consecutive_failures(self):
        pool = CredentialPool()
        pool.add("key1")
        pool._credentials[0].consecutive_failures = 5
        pool.mark_success("key1", tokens=0)
        assert pool._credentials[0].consecutive_failures == 0

    def test_increments_total_tokens(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.mark_success("key1", tokens=500)
        pool.mark_success("key1", tokens=300)
        assert pool._credentials[0].total_tokens == 800


# ---------------------------------------------------------------------------
# CredentialPool.status
# ---------------------------------------------------------------------------

class TestCredentialPoolStatus:
    def test_returns_list(self):
        pool = CredentialPool()
        pool.add("key1")
        assert isinstance(pool.status(), list)

    def test_status_has_required_keys(self):
        pool = CredentialPool()
        pool.add("key1", name="primary")
        s = pool.status()[0]
        for key in ("name", "available", "requests", "tokens", "cooldown", "failures"):
            assert key in s

    def test_available_true_for_fresh_key(self):
        pool = CredentialPool()
        pool.add("key1")
        assert pool.status()[0]["available"] is True

    def test_available_false_for_exhausted_key(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.mark_exhausted("key1", error_code=429)
        assert pool.status()[0]["available"] is False

    def test_available_count_correct(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.add("key2")
        pool.mark_exhausted("key1", error_code=429)
        assert pool.available_count == 1
