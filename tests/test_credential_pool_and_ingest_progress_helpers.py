"""Tests for CredentialPool/Credential and IngestionProgress/IngestionCheckpoint pure helpers."""

import time

from mullet.core.providers.credential_pool import Credential, CredentialPool
from mullet.knowledge.engine.ingest_progress import IngestionProgress, IngestionCheckpoint


# ---------------------------------------------------------------------------
# Credential
# ---------------------------------------------------------------------------

class TestCredentialDefaults:
    def test_api_key_stored(self):
        c = Credential(api_key="test-key")
        assert c.api_key == "test-key"

    def test_name_defaults_empty(self):
        c = Credential(api_key="key")
        assert c.name == ""

    def test_total_requests_defaults_zero(self):
        c = Credential(api_key="key")
        assert c.total_requests == 0

    def test_total_tokens_defaults_zero(self):
        c = Credential(api_key="key")
        assert c.total_tokens == 0

    def test_exhausted_until_defaults_zero(self):
        c = Credential(api_key="key")
        assert c.exhausted_until == 0

    def test_consecutive_failures_defaults_zero(self):
        c = Credential(api_key="key")
        assert c.consecutive_failures == 0

    def test_is_available_initially_true(self):
        c = Credential(api_key="key")
        assert c.is_available is True

    def test_cooldown_remaining_zero_initially(self):
        c = Credential(api_key="key")
        assert c.cooldown_remaining == 0


class TestCredentialExhausted:
    def test_active_cooldown_not_available(self):
        c = Credential(api_key="key", exhausted_until=time.time() + 3600)
        assert c.is_available is False

    def test_expired_cooldown_is_available(self):
        c = Credential(api_key="key", exhausted_until=time.time() - 1)
        assert c.is_available is True

    def test_cooldown_remaining_positive_when_exhausted(self):
        c = Credential(api_key="key", exhausted_until=time.time() + 3600)
        assert c.cooldown_remaining > 0

    def test_cooldown_remaining_zero_when_available(self):
        c = Credential(api_key="key")
        assert c.cooldown_remaining == 0


# ---------------------------------------------------------------------------
# CredentialPool — construction
# ---------------------------------------------------------------------------

class TestCredentialPoolInit:
    def test_empty_pool_has_zero_total(self):
        pool = CredentialPool()
        assert pool.total_count == 0

    def test_empty_pool_has_zero_available(self):
        pool = CredentialPool()
        assert pool.available_count == 0

    def test_get_next_on_empty_returns_none(self):
        pool = CredentialPool()
        assert pool.get_next() is None

    def test_add_credential(self):
        pool = CredentialPool()
        pool.add("key1")
        assert pool.total_count == 1

    def test_add_multiple_credentials(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.add("key2")
        assert pool.total_count == 2

    def test_available_count_matches_available(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.add("key2")
        assert pool.available_count == 2


# ---------------------------------------------------------------------------
# CredentialPool — get_next (round robin)
# ---------------------------------------------------------------------------

class TestCredentialPoolRoundRobin:
    def test_single_key_always_returned(self):
        pool = CredentialPool(strategy="round_robin")
        pool.add("key1")
        assert pool.get_next().api_key == "key1"

    def test_two_keys_alternate(self):
        pool = CredentialPool(strategy="round_robin")
        pool.add("key1")
        pool.add("key2")
        first = pool.get_next().api_key
        second = pool.get_next().api_key
        assert first != second

    def test_wraps_around_after_all_keys(self):
        pool = CredentialPool(strategy="round_robin")
        pool.add("key1")
        pool.add("key2")
        k1 = pool.get_next()
        k2 = pool.get_next()
        k3 = pool.get_next()
        assert k3.api_key == k1.api_key

    def test_all_exhausted_returns_none(self):
        pool = CredentialPool(strategy="round_robin")
        pool.add("key1")
        pool.mark_exhausted("key1", 429)
        assert pool.get_next() is None


# ---------------------------------------------------------------------------
# CredentialPool — get_next (least_used)
# ---------------------------------------------------------------------------

class TestCredentialPoolLeastUsed:
    def test_picks_key_with_fewer_requests(self):
        pool = CredentialPool(strategy="least_used")
        pool.add("busy")
        pool.add("free")
        pool._credentials[0].total_requests = 10
        pool._credentials[1].total_requests = 0
        result = pool.get_next()
        assert result.api_key == "free"


# ---------------------------------------------------------------------------
# CredentialPool — mark_exhausted and mark_success
# ---------------------------------------------------------------------------

class TestCredentialPoolMarkExhausted:
    def test_mark_429_makes_unavailable(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.mark_exhausted("key1", 429)
        cred = pool._credentials[0]
        assert cred.is_available is False

    def test_mark_402_longer_cooldown(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.mark_exhausted("key1", 402)
        cred = pool._credentials[0]
        assert cred.cooldown_remaining > 3600  # > 1 hour

    def test_mark_429_shorter_cooldown_than_402(self):
        pool = CredentialPool()
        pool.add("key429")
        pool.add("key402")
        pool.mark_exhausted("key429", 429)
        pool.mark_exhausted("key402", 402)
        c429 = pool._credentials[0]
        c402 = pool._credentials[1]
        assert c402.cooldown_remaining > c429.cooldown_remaining

    def test_mark_nonexistent_key_no_error(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.mark_exhausted("nonexistent", 429)  # should not raise


class TestCredentialPoolMarkSuccess:
    def test_resets_consecutive_failures(self):
        pool = CredentialPool()
        pool.add("key1")
        pool._credentials[0].consecutive_failures = 5
        pool.mark_success("key1", tokens=0)
        assert pool._credentials[0].consecutive_failures == 0

    def test_accumulates_tokens(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.mark_success("key1", tokens=1000)
        assert pool._credentials[0].total_tokens == 1000

    def test_mark_nonexistent_key_no_error(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.mark_success("nonexistent", tokens=0)  # should not raise


# ---------------------------------------------------------------------------
# CredentialPool — status
# ---------------------------------------------------------------------------

class TestCredentialPoolStatus:
    def test_returns_list(self):
        pool = CredentialPool()
        pool.add("key1", "main")
        assert isinstance(pool.status(), list)

    def test_status_has_one_entry_per_credential(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.add("key2")
        assert len(pool.status()) == 2

    def test_status_entry_has_required_keys(self):
        pool = CredentialPool()
        pool.add("key1", "main")
        entry = pool.status()[0]
        for key in ("name", "available", "requests", "tokens", "cooldown", "failures"):
            assert key in entry

    def test_status_shows_available_true(self):
        pool = CredentialPool()
        pool.add("key1")
        assert pool.status()[0]["available"] is True

    def test_empty_pool_status_is_empty_list(self):
        pool = CredentialPool()
        assert pool.status() == []


# ---------------------------------------------------------------------------
# IngestionProgress
# ---------------------------------------------------------------------------

class TestIngestionProgressPercent:
    def test_zero_of_ten_is_zero_percent(self):
        ip = IngestionProgress(total_files=10)
        assert ip.percent == 0.0

    def test_five_of_ten_is_fifty_percent(self):
        ip = IngestionProgress(total_files=10, processed_files=5)
        assert ip.percent == 50.0

    def test_ten_of_ten_is_100_percent(self):
        ip = IngestionProgress(total_files=10, processed_files=10)
        assert ip.percent == 100.0

    def test_zero_total_returns_100_percent(self):
        ip = IngestionProgress(total_files=0)
        assert ip.percent == 100.0


class TestIngestionProgressStatus:
    def test_initial_status_in_progress(self):
        ip = IngestionProgress(total_files=10)
        assert ip.status == "in_progress"

    def test_completed_when_all_processed(self):
        ip = IngestionProgress(total_files=5, processed_files=5)
        assert ip.status == "completed"

    def test_completed_with_errors_when_failures(self):
        ip = IngestionProgress(total_files=5, processed_files=5)
        ip.failed_files = ["bad.py"]
        assert ip.status == "completed_with_errors"

    def test_in_progress_when_partial(self):
        ip = IngestionProgress(total_files=10, processed_files=3)
        assert ip.status == "in_progress"


class TestIngestionProgressEta:
    def test_eta_none_when_no_processed(self):
        ip = IngestionProgress(total_files=10)
        assert ip.eta_seconds is None

    def test_eta_positive_when_processing(self):
        ip = IngestionProgress(total_files=100, processed_files=1, start_time=time.time() - 10)
        eta = ip.eta_seconds
        assert eta is not None
        assert eta > 0

    def test_format_eta_calculating_when_none(self):
        ip = IngestionProgress(total_files=10)
        assert "calculating" in ip.format_eta()

    def test_format_eta_returns_string(self):
        ip = IngestionProgress(total_files=10, processed_files=1, start_time=time.time() - 1)
        assert isinstance(ip.format_eta(), str)

    def test_format_eta_shows_seconds(self):
        ip = IngestionProgress(total_files=10, processed_files=9, start_time=time.time() - 1)
        eta = ip.format_eta()
        assert "s" in eta  # seconds or minutes format


# ---------------------------------------------------------------------------
# IngestionCheckpoint
# ---------------------------------------------------------------------------

class TestIngestionCheckpointSaveLoad:
    def test_save_creates_file(self, tmp_path):
        cp = IngestionCheckpoint(
            client_name="test", processed_files=["a.py"], total_files=["a.py", "b.py"]
        )
        path = tmp_path / "checkpoint.json"
        cp.save(path)
        assert path.exists()

    def test_load_returns_checkpoint(self, tmp_path):
        cp = IngestionCheckpoint(
            client_name="test", processed_files=["a.py"], total_files=["a.py", "b.py"]
        )
        path = tmp_path / "cp.json"
        cp.save(path)
        loaded = IngestionCheckpoint.load(path)
        assert loaded is not None

    def test_load_client_name_preserved(self, tmp_path):
        cp = IngestionCheckpoint(client_name="my-client", processed_files=[], total_files=[])
        path = tmp_path / "cp.json"
        cp.save(path)
        loaded = IngestionCheckpoint.load(path)
        assert loaded.client_name == "my-client"

    def test_load_processed_files_preserved(self, tmp_path):
        cp = IngestionCheckpoint(
            client_name="c", processed_files=["a.py", "b.py"], total_files=["a.py", "b.py"]
        )
        path = tmp_path / "cp.json"
        cp.save(path)
        loaded = IngestionCheckpoint.load(path)
        assert loaded.processed_files == ["a.py", "b.py"]

    def test_load_chunks_indexed_preserved(self, tmp_path):
        cp = IngestionCheckpoint(
            client_name="c", processed_files=[], total_files=[], chunks_indexed=99
        )
        path = tmp_path / "cp.json"
        cp.save(path)
        loaded = IngestionCheckpoint.load(path)
        assert loaded.chunks_indexed == 99

    def test_load_nonexistent_returns_none(self, tmp_path):
        result = IngestionCheckpoint.load(tmp_path / "nonexistent.json")
        assert result is None

    def test_checkpoint_path_returns_correct_filename(self, tmp_path):
        path = IngestionCheckpoint.checkpoint_path(tmp_path)
        assert path.name == ".ingest_checkpoint.json"

    def test_checkpoint_path_under_given_dir(self, tmp_path):
        path = IngestionCheckpoint.checkpoint_path(tmp_path)
        assert str(tmp_path) in str(path)
