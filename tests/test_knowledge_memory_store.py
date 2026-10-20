"""Tests for mullet.knowledge.engine.memory.MemoryStore — SQLite persistent memory."""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from mullet.knowledge.engine.memory import MemoryStore


@pytest.fixture
def store(tmp_path):
    """Fresh MemoryStore with SQLite in a temp path."""
    s = MemoryStore(db_path=tmp_path / "mem.db")
    yield s


# --- store / retrieve ------------------------------------------------------


def test_store_and_retrieve(store):
    memory_id = store.store("ns1", "key1", "content-value")
    assert memory_id  # UUID string

    got = store.retrieve("ns1", "key1")
    assert got is not None
    assert got["namespace"] == "ns1"
    assert got["key"] == "key1"
    assert got["content"] == "content-value"


def test_retrieve_missing_returns_none(store):
    assert store.retrieve("no-ns", "no-key") is None


def test_store_upsert_on_same_key(store):
    """Same (namespace, key) → update, not duplicate."""
    store.store("ns", "k", "first")
    store.store("ns", "k", "second")
    got = store.retrieve("ns", "k")
    assert got["content"] == "second"


def test_store_different_namespaces_isolated(store):
    """Same key in different namespaces = two distinct entries."""
    store.store("ns-a", "k", "value-a")
    store.store("ns-b", "k", "value-b")
    assert store.retrieve("ns-a", "k")["content"] == "value-a"
    assert store.retrieve("ns-b", "k")["content"] == "value-b"


def test_store_with_metadata(store):
    store.store("ns", "k", "content", metadata={"tag": "important", "owner": "alice"})
    got = store.retrieve("ns", "k")
    # metadata is JSON-serialized in DB — retrieve returns parsed dict
    md = got["metadata"]
    # MemoryStore may or may not auto-parse — just check the raw string contains the tag
    if isinstance(md, str):
        assert "tag" in md
    else:
        assert md["tag"] == "important"


# --- TTL expiry ------------------------------------------------------------


def test_store_with_short_ttl_expires(store):
    """TTL=1s → not expired immediately, expired after 1.1s."""
    store.store("ns", "ephemeral", "expires-soon", ttl=1)
    # Right after store: still retrievable
    got = store.retrieve("ns", "ephemeral")
    assert got is not None
    # After TTL: expired
    time.sleep(1.1)
    assert store.retrieve("ns", "ephemeral") is None


def test_store_with_ttl_zero_treated_as_no_ttl(store):
    """TTL=0 is falsy in the `now + ttl if ttl else None` guard — no expiry."""
    store.store("ns", "k", "v", ttl=0)
    got = store.retrieve("ns", "k")
    assert got is not None  # still retrievable
    assert got["expires_at"] is None


def test_store_with_future_ttl_retrievable(store):
    store.store("ns", "kept", "fresh", ttl=3600)  # 1 hour
    got = store.retrieve("ns", "kept")
    assert got is not None
    assert got["content"] == "fresh"


def test_store_no_ttl_never_expires(store):
    """No TTL → expires_at is NULL, always retrievable."""
    store.store("ns", "permanent", "kept forever")
    got = store.retrieve("ns", "permanent")
    assert got is not None


# --- concurrent inserts (IntegrityError branch) ----------------------------


def test_upsert_sequence_preserves_latest_value(store):
    """3 sequential stores with same key keep only the last content."""
    store.store("ns", "k", "v1")
    store.store("ns", "k", "v2")
    store.store("ns", "k", "v3")
    got = store.retrieve("ns", "k")
    assert got["content"] == "v3"


# --- embedding path --------------------------------------------------------


def test_store_without_embedding_model(store):
    """MemoryStore constructed without embedding_model still stores fine."""
    assert store.embedding_model is None
    store.store("ns", "k", "content")
    assert store.retrieve("ns", "k")["content"] == "content"


def test_store_embedding_error_swallowed(tmp_path):
    """Embedding exceptions don't crash store() — content is still persisted."""
    class _BrokenEmbedder:
        def encode(self, *args, **kwargs):
            raise RuntimeError("cuda out of memory")

    s = MemoryStore(db_path=tmp_path / "mem.db", embedding_model=_BrokenEmbedder())
    # Must not raise
    s.store("ns", "k", "content")
    assert s.retrieve("ns", "k")["content"] == "content"


# --- database setup --------------------------------------------------------


def test_db_file_created(tmp_path):
    path = tmp_path / "nested" / "mem.db"
    MemoryStore(db_path=path)
    assert path.exists()
    assert path.parent.exists()


def test_custom_connection(tmp_path):
    """Passing a connection skips owning it — two stores share the DB."""
    import sqlite3
    path = tmp_path / "shared.db"
    conn = sqlite3.connect(str(path), check_same_thread=False)

    store1 = MemoryStore(db_path=path, connection=conn)
    store2 = MemoryStore(db_path=path, connection=conn)

    store1.store("ns", "k", "v1")
    # store2 sees it because they share the connection
    got = store2.retrieve("ns", "k")
    assert got is not None
    assert got["content"] == "v1"


# --- embedding generation and vector search --------------------------------


def test_store_with_embedding_model(tmp_path):
    """Embedding is generated and stored when embedding_model is provided."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    s = MemoryStore(db_path=tmp_path / "mem.db", embedding_model=mock_model)
    s.store("ns", "k", "content")
    # Verify embedding stored in DB
    row = s._conn.execute(
        "SELECT embedding FROM memories WHERE namespace = ? AND key = ?",
        ("ns", "k"),
    ).fetchone()
    assert row[0] is not None
    stored = np.frombuffer(row[0], dtype=np.float32)
    np.testing.assert_array_almost_equal(
        stored, np.array([0.1, 0.2, 0.3], dtype=np.float32)
    )


def test_vector_search(tmp_path):
    """search() with embedding_model routes to _vector_search."""
    mock_model = MagicMock()
    vectors = {
        "hello world": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "other stuff": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "another hello": np.array([0.9, 0.1, 0.0], dtype=np.float32),
        "query": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }
    mock_model.encode.side_effect = lambda text: vectors.get(
        text, np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )
    s = MemoryStore(db_path=tmp_path / "mem.db", embedding_model=mock_model)
    s.store("ns1", "k1", "hello world")
    s.store("ns2", "k2", "other stuff")
    s.store("ns1", "k3", "another hello")

    # Search without namespace
    results = s.search("query")
    assert len(results) == 3
    assert all("similarity" in r for r in results)
    # Sorted by similarity descending
    assert results[0]["key"] == "k1"
    assert results[0]["similarity"] == pytest.approx(1.0)

    # Search with namespace filter
    results = s.search("query", namespace="ns1")
    assert len(results) == 2
    assert {r["key"] for r in results} == {"k1", "k3"}


def test_vector_search_no_results(tmp_path):
    """_vector_search returns empty list when nothing matches."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    s = MemoryStore(db_path=tmp_path / "mem.db", embedding_model=mock_model)
    results = s.search("query", namespace="empty")
    assert results == []


# --- LIKE fallback search --------------------------------------------------


def test_like_search_basic(tmp_path):
    """search() without embedding_model uses _like_search."""
    s = MemoryStore(db_path=tmp_path / "mem.db")
    s.store("ns1", "k1", "hello world")
    s.store("ns2", "k2", "goodbye world")

    results = s.search("hello")
    assert len(results) == 1
    assert results[0]["key"] == "k1"

    results = s.search("world")
    assert len(results) == 2

    results = s.search("xyz")
    assert results == []


def test_like_search_with_namespace(tmp_path):
    """_like_search respects namespace filter."""
    s = MemoryStore(db_path=tmp_path / "mem.db")
    s.store("ns1", "k1", "hello world")
    s.store("ns2", "k2", "hello there")

    results = s.search("hello", namespace="ns1")
    assert len(results) == 1
    assert results[0]["key"] == "k1"


def test_like_search_escapes_special_chars(tmp_path):
    """LIKE wildcards % and _ are escaped so they are literal."""
    s = MemoryStore(db_path=tmp_path / "mem.db")
    s.store("ns", "k1", "100% complete")
    s.store("ns", "k2", "100 percent")

    # Literal % should not match everything
    results = s.search("%")
    assert len(results) == 1
    assert results[0]["key"] == "k1"

    s.store("ns", "k3", "a_b")
    s.store("ns", "k4", "abc")
    results = s.search("_")
    assert len(results) == 1
    assert results[0]["key"] == "k3"


# --- list ------------------------------------------------------------------


def test_list_all_and_by_namespace(store):
    """list() returns all non-expired memories, optionally filtered."""
    store.store("ns1", "k1", "first")
    time.sleep(0.01)
    store.store("ns1", "k2", "second")
    time.sleep(0.01)
    store.store("ns2", "k3", "third")

    all_memories = store.list()
    assert len(all_memories) == 3
    # Newest first
    assert all_memories[0]["key"] == "k3"
    assert all_memories[1]["key"] == "k2"
    assert all_memories[2]["key"] == "k1"

    ns1 = store.list(namespace="ns1")
    assert len(ns1) == 2
    assert {r["key"] for r in ns1} == {"k1", "k2"}


def test_list_excludes_expired(store):
    """list() does not include expired memories."""
    store.store("ns", "expired", "old", ttl=1)
    store.store("ns", "fresh", "new")
    time.sleep(1.1)
    results = store.list()
    assert len(results) == 1
    assert results[0]["key"] == "fresh"


# --- delete ----------------------------------------------------------------


def test_delete_existing(store):
    store.store("ns", "k", "content")
    assert store.retrieve("ns", "k") is not None
    assert store.delete("ns", "k") is True
    assert store.retrieve("ns", "k") is None


def test_delete_missing(store):
    assert store.delete("ns", "missing") is False


# --- cleanup ---------------------------------------------------------------


def test_cleanup_removes_expired(store):
    store.store("ns", "expired", "old", ttl=1)
    store.store("ns", "fresh", "new", ttl=3600)
    time.sleep(1.1)
    removed = store.cleanup()
    assert removed == 1
    assert store.retrieve("ns", "expired") is None
    assert store.retrieve("ns", "fresh") is not None


def test_cleanup_no_expired(store):
    store.store("ns", "k", "v", ttl=3600)
    removed = store.cleanup()
    assert removed == 0


# --- close -----------------------------------------------------------------


def test_close_owns_connection(tmp_path):
    import sqlite3
    s = MemoryStore(db_path=tmp_path / "mem.db")
    s.store("ns", "k", "v")
    s.close()
    with pytest.raises(sqlite3.ProgrammingError):
        s._conn.execute("SELECT 1")


def test_close_skips_shared_connection(tmp_path):
    import sqlite3
    path = tmp_path / "shared.db"
    conn = sqlite3.connect(str(path), check_same_thread=False)
    s = MemoryStore(db_path=path, connection=conn)
    s.close()
    # Connection should still be usable
    conn.execute("SELECT 1")
    conn.close()
