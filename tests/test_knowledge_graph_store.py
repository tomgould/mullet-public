import sqlite3
from pathlib import Path

from mullet.knowledge.graph.store import KnowledgeGraph
from mullet.knowledge.graph.models import Entity, Relationship


def _make_graph():
    return KnowledgeGraph(Path("/tmp/fake_index"))


def test_get_entity():
    graph = _make_graph()
    entity = Entity(name="Alice", entity_type="person")
    graph._entities[entity.key] = entity
    assert graph.get_entity(entity.key) is entity
    assert graph.get_entity("missing") is None


def test_get_entities_by_type():
    graph = _make_graph()
    e1 = Entity(name="Alice", entity_type="person")
    e2 = Entity(name="Acme", entity_type="organization")
    e3 = Entity(name="Bob", entity_type="Person")
    graph._entities[e1.key] = e1
    graph._entities[e2.key] = e2
    graph._entities[e3.key] = e3
    result = graph.get_entities_by_type("person")
    assert len(result) == 2
    assert e1 in result
    assert e3 in result


def test_find_entities_by_name():
    graph = _make_graph()
    e1 = Entity(name="Alice Smith", entity_type="person", aliases=["A. Smith"])
    e2 = Entity(name="Bob Jones", entity_type="person")
    graph._entities[e1.key] = e1
    graph._entities[e2.key] = e2
    assert graph.find_entities_by_name("alice") == [e1]
    assert graph.find_entities_by_name("smith") == [e1]
    assert graph.find_entities_by_name("A. Smith") == [e1]
    assert graph.find_entities_by_name("bob") == [e2]
    assert graph.find_entities_by_name("charlie") == []


def test_get_source_chunks():
    graph = _make_graph()
    e1 = Entity(name="Alice", entity_type="person", source_chunk_ids=["c1", "c2"])
    e2 = Entity(name="Bob", entity_type="person", source_chunk_ids=["c2", "c3"])
    r1 = Relationship(source_entity=e1.key, target_entity=e2.key, relationship_type="knows", source_chunk_ids=["c4"])
    graph._entities[e1.key] = e1
    graph._entities[e2.key] = e2
    graph._relationships[r1.key] = r1
    chunks = graph.get_source_chunks([e1.key])
    assert sorted(chunks) == ["c1", "c2", "c4"]


def test_mark_chunks_processed():
    graph = _make_graph()
    graph.mark_chunks_processed(["a", "b"])
    assert graph.processed_chunk_ids == {"a", "b"}
    graph.mark_chunks_processed(["b", "c"])
    assert graph.processed_chunk_ids == {"a", "b", "c"}
    assert graph._dirty_chunks == {"a", "b", "c"}


def test_processed_chunk_ids_property():
    graph = _make_graph()
    assert graph.processed_chunk_ids == set()
    graph._processed_chunk_ids = {"x", "y"}
    assert graph.processed_chunk_ids == {"x", "y"}


def test_stats():
    graph = _make_graph()
    e1 = Entity(name="Alice", entity_type="person", source_chunk_ids=["c1"])
    e2 = Entity(name="Acme", entity_type="organization", source_chunk_ids=["c2"])
    e3 = Entity(name="Bob", entity_type="person", source_chunk_ids=["c3"])
    r1 = Relationship(source_entity=e1.key, target_entity=e2.key, relationship_type="works_at", source_chunk_ids=["c1"])
    r2 = Relationship(source_entity=e3.key, target_entity=e1.key, relationship_type="knows", source_chunk_ids=["c3"])
    graph._entities[e1.key] = e1
    graph._entities[e2.key] = e2
    graph._entities[e3.key] = e3
    graph._relationships[r1.key] = r1
    graph._relationships[r2.key] = r2
    stats = graph.stats
    assert stats["entity_count"] == 3
    assert stats["relationship_count"] == 2
    assert stats["entity_type_distribution"] == {"person": 2, "organization": 1}
    assert stats["source_chunk_count"] == 3
    top = stats["top_entities"]
    assert len(top) == 3
    assert top[0]["name"] == "Alice"
    assert top[0]["connections"] == 2


def test_clear():
    graph = _make_graph()
    e1 = Entity(name="Alice", entity_type="person")
    graph._entities[e1.key] = e1
    graph._relationships["r1"] = Relationship(source_entity=e1.key, target_entity="b", relationship_type="knows")
    graph._processed_chunk_ids = {"c1"}
    graph._dirty_entities = {"x"}
    graph._dirty_relationships = {"y"}
    graph._dirty_chunks = {"z"}
    graph.clear()
    assert graph._entities == {}
    assert graph._relationships == {}
    assert graph._processed_chunk_ids == set()
    assert graph._dirty_entities == set()
    assert graph._dirty_relationships == set()
    assert graph._dirty_chunks == set()


def test_clear_with_db(tmp_path):
    """clear() deletes from SQLite when DB exists (lines 514-518)."""
    graph = KnowledgeGraph(tmp_path)
    e1 = Entity(name="Alice", entity_type="person")
    graph.add_entity(e1)
    graph.save()
    graph.clear()
    # After clear, DB tables should be empty
    db = graph._get_db()
    assert db.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM relationships").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM processed_chunks").fetchone()[0] == 0


def test_rebuild_nx_graph_skips_missing_entities():
    """_rebuild_nx_graph skips edges when source/target missing (lines 103-104)."""
    graph = _make_graph()
    e1 = Entity(name="Alice", entity_type="person")
    graph._entities[e1.key] = e1
    # Relationship references a non-existent target entity
    rel = Relationship(source_entity=e1.key, target_entity="missing", relationship_type="knows")
    graph._relationships[rel.key] = rel
    graph._rebuild_nx_graph()
    assert graph._graph.has_node(e1.key)
    assert not graph._graph.has_edge(e1.key, "missing")


def test_add_relationship_merge():
    """add_relationship merges when key already exists (line 128)."""
    graph = _make_graph()
    e1 = Entity(name="Alice", entity_type="person")
    e2 = Entity(name="Bob", entity_type="person")
    graph.add_entity(e1)
    graph.add_entity(e2)
    r1 = Relationship(source_entity=e1.key, target_entity=e2.key, relationship_type="knows", weight=0.5, source_chunk_ids=["c1"])
    r2 = Relationship(source_entity=e1.key, target_entity=e2.key, relationship_type="knows", weight=0.9, source_chunk_ids=["c2"])
    graph.add_relationship(r1)
    graph.add_relationship(r2)
    stored = graph._relationships[r1.key]
    assert stored.weight == 0.9
    assert sorted(stored.source_chunk_ids) == ["c1", "c2"]


def test_get_neighbors_basic():
    """get_neighbors BFS traversal (lines 167-202)."""
    graph = _make_graph()
    alice = Entity(name="Alice", entity_type="person")
    bob = Entity(name="Bob", entity_type="person")
    charlie = Entity(name="Charlie", entity_type="person")
    dave = Entity(name="Dave", entity_type="person")
    graph.add_entity(alice)
    graph.add_entity(bob)
    graph.add_entity(charlie)
    graph.add_entity(dave)
    graph.add_relationship(Relationship(source_entity=alice.key, target_entity=bob.key, relationship_type="knows"))
    graph.add_relationship(Relationship(source_entity=bob.key, target_entity=charlie.key, relationship_type="knows"))
    graph.add_relationship(Relationship(source_entity=dave.key, target_entity=alice.key, relationship_type="knows"))
    # Depth 1 from Alice
    neighbors = graph.get_neighbors(alice.key, max_depth=1)
    keys = {n[0].key for n in neighbors}
    assert bob.key in keys
    assert dave.key in keys
    assert charlie.key not in keys
    # Depth 2 from Alice reaches Charlie
    neighbors = graph.get_neighbors(alice.key, max_depth=2)
    keys = {n[0].key for n in neighbors}
    assert charlie.key in keys


def test_get_neighbors_unknown_entity():
    """get_neighbors returns empty list for unknown entity."""
    graph = _make_graph()
    assert graph.get_neighbors("nonexistent", max_depth=2) == []


def test_get_neighbors_empty_frontier_break():
    """get_neighbors breaks early when frontier is empty."""
    graph = _make_graph()
    alice = Entity(name="Alice", entity_type="person")
    graph.add_entity(alice)
    # No relationships — frontier becomes empty after first depth iteration
    assert graph.get_neighbors(alice.key, max_depth=2) == []


def test_save_nothing_dirty():
    """save() returns early when nothing is dirty (line 237)."""
    graph = _make_graph()
    # No dirty items; save should be a no-op
    graph.save()
    assert graph._dirty_entities == set()
    assert graph._dirty_relationships == set()
    assert graph._dirty_chunks == set()


def test_load_from_legacy_json(tmp_path):
    """load() migrates legacy JSON to SQLite (lines 302-314)."""
    import json
    json_path = tmp_path / "knowledge_graph.json"
    data = {
        "entities": [
            {"name": "Alice", "entity_type": "person", "description": "", "source_chunk_ids": ["c1"], "aliases": []},
        ],
        "relationships": [
            {"source_entity": "alice::person", "target_entity": "bob::person", "relationship_type": "knows", "description": "", "weight": 1.0, "source_chunk_ids": ["c1"]},
        ],
        "processed_chunk_ids": ["c1"],
    }
    json_path.write_text(json.dumps(data))
    graph = KnowledgeGraph(tmp_path)
    assert graph.load() is True
    assert "alice::person" in graph._entities
    assert len(graph._relationships) == 1
    assert graph.processed_chunk_ids == {"c1"}
    # After migration, SQLite DB should exist and contain data
    db_path = tmp_path / "knowledge_graph.db"
    assert db_path.exists()


def test_load_from_legacy_json_without_processed_chunk_ids(tmp_path):
    """_load_from_json backfills processed_chunk_ids from entities/rels (lines 388-395)."""
    import json
    json_path = tmp_path / "knowledge_graph.json"
    data = {
        "entities": [
            {"name": "Alice", "entity_type": "person", "source_chunk_ids": ["c1"]},
        ],
        "relationships": [
            {"source_entity": "alice::person", "target_entity": "bob::person", "relationship_type": "knows", "source_chunk_ids": ["c2"]},
        ],
    }
    json_path.write_text(json.dumps(data))
    graph = KnowledgeGraph(tmp_path)
    assert graph.load() is True
    assert graph.processed_chunk_ids == {"c1", "c2"}


def test_load_from_legacy_json_bad_data(tmp_path):
    """_load_from_json returns False on JSON decode error (lines 405-407)."""
    import json
    json_path = tmp_path / "knowledge_graph.json"
    json_path.write_text("not json")
    graph = KnowledgeGraph(tmp_path)
    assert graph.load() is False


def test_load_from_sqlite_with_null_weight(tmp_path):
    """_load_from_sqlite handles NULL weight by defaulting to 1.0 (lines 341-349)."""
    import sqlite3
    graph = KnowledgeGraph(tmp_path)
    db = graph._get_db()
    db.execute(
        "INSERT INTO entities (key, name, entity_type, description, source_chunk_ids, aliases) VALUES (?, ?, ?, ?, ?, ?)",
        ("alice::person", "Alice", "person", "", "[]", "[]"),
    )
    db.execute(
        "INSERT INTO relationships (key, source_entity, target_entity, relationship_type, description, weight, source_chunk_ids) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("alice::person--knows--bob::person", "alice::person", "bob::person", "knows", "", None, "[]"),
    )
    db.commit()
    assert graph.load() is True
    rel = graph._relationships["alice::person--knows--bob::person"]
    assert rel.weight == 1.0


def test_load_from_sqlite_exception(tmp_path, monkeypatch):
    """_load_from_sqlite returns False and closes DB on exception (lines 364-367)."""
    graph = KnowledgeGraph(tmp_path)
    # Create a valid DB first
    graph._get_db()
    graph.save()
    graph.close()

    class BadConn:
        def execute(self, *args, **kwargs):
            raise RuntimeError("boom")
        def executescript(self, *args, **kwargs):
            pass
        def commit(self):
            pass
        def close(self):
            pass
        def executemany(self, *args, **kwargs):
            pass

    monkeypatch.setattr(sqlite3, "connect", lambda *a, **kw: BadConn())
    assert graph.load() is False
    assert graph._db is None


def test_load_from_json_keyerror(tmp_path):
    """_load_from_json returns False on KeyError (lines 405-407)."""
    import json
    json_path = tmp_path / "knowledge_graph.json"
    # Missing required keys like "name" in entity dict
    json_path.write_text(json.dumps({"entities": [{"entity_type": "person"}]}))
    graph = KnowledgeGraph(tmp_path)
    assert graph.load() is False


def test_prune_orphaned_chunks_updates_relationships(tmp_path):
    """prune_orphaned_chunks updates relationships with partial stale sources (lines 471-474)."""
    graph = KnowledgeGraph(tmp_path)
    e1 = Entity(name="Alice", entity_type="person", source_chunk_ids=["c1", "c2"])
    e2 = Entity(name="Bob", entity_type="person", source_chunk_ids=["c1"])
    graph.add_entity(e1)
    graph.add_entity(e2)
    r1 = Relationship(source_entity=e1.key, target_entity=e2.key, relationship_type="knows", source_chunk_ids=["c1", "c2"])
    graph.add_relationship(r1)
    graph.mark_chunks_processed(["c1", "c2"])
    graph.save()
    # Only c1 exists now; c2 is stale — relationship should be updated, not removed
    result = graph.prune_orphaned_chunks({"c1"})
    assert result["updated_relationships"] == 1
    assert result["removed_relationships"] == 0
    assert graph._relationships[r1.key].source_chunk_ids == ["c1"]


def test_load_no_files(tmp_path):
    """load() returns False when neither DB nor JSON exists."""
    graph = KnowledgeGraph(tmp_path)
    assert graph.load() is False


def test_close_idempotent():
    """close() is safe to call multiple times."""
    graph = _make_graph()
    graph.close()
    graph.close()
    assert graph._db is None


def test_add_entity_merge():
    """add_entity merges when key already exists (line 116)."""
    graph = _make_graph()
    e1 = Entity(name="Alice", entity_type="person", source_chunk_ids=["c1"])
    e2 = Entity(name="Alice", entity_type="person", source_chunk_ids=["c2"])
    graph.add_entity(e1)
    graph.add_entity(e2)
    stored = graph._entities[e1.key]
    assert sorted(stored.source_chunk_ids) == ["c1", "c2"]


def test_save_and_reload_processed_chunks(tmp_path):
    """_load_from_sqlite restores processed_chunk_ids (line 353)."""
    graph = KnowledgeGraph(tmp_path)
    graph.mark_chunks_processed(["c1", "c2"])
    graph.save()
    graph.close()

    graph2 = KnowledgeGraph(tmp_path)
    assert graph2.load() is True
    assert graph2.processed_chunk_ids == {"c1", "c2"}
    graph2.close()


def test_prune_no_stale_chunks_fast_path(tmp_path):
    """prune_orphaned_chunks returns zeros when nothing is stale (line 424)."""
    graph = KnowledgeGraph(tmp_path)
    e1 = Entity(name="Alice", entity_type="person", source_chunk_ids=["c1"])
    graph.add_entity(e1)
    graph.mark_chunks_processed(["c1"])
    graph.save()
    result = graph.prune_orphaned_chunks({"c1"})
    assert result == {
        "removed_chunks": 0,
        "removed_entities": 0,
        "removed_relationships": 0,
        "updated_entities": 0,
        "updated_relationships": 0,
    }


def test_prune_removes_entity_all_stale_sources(tmp_path):
    """prune_orphaned_chunks deletes entity when all sources are stale (lines 450-451, 458)."""
    graph = KnowledgeGraph(tmp_path)
    e1 = Entity(name="Alice", entity_type="person", source_chunk_ids=["c2"])
    graph.add_entity(e1)
    graph.mark_chunks_processed(["c2"])
    graph.save()
    result = graph.prune_orphaned_chunks({"c1"})
    assert result["removed_entities"] == 1
    assert e1.key not in graph._entities
    # Verify SQLite row is gone
    db = graph._get_db()
    assert db.execute("SELECT COUNT(*) FROM entities WHERE key = ?", (e1.key,)).fetchone()[0] == 0


def test_prune_removes_relationship_all_stale_sources(tmp_path):
    """prune_orphaned_chunks deletes relationship when all sources are stale (lines 469-470, 477)."""
    graph = KnowledgeGraph(tmp_path)
    e1 = Entity(name="Alice", entity_type="person", source_chunk_ids=["c1"])
    e2 = Entity(name="Bob", entity_type="person", source_chunk_ids=["c1"])
    graph.add_entity(e1)
    graph.add_entity(e2)
    r1 = Relationship(source_entity=e1.key, target_entity=e2.key, relationship_type="knows", source_chunk_ids=["c2"])
    graph.add_relationship(r1)
    graph.mark_chunks_processed(["c1", "c2"])
    graph.save()
    result = graph.prune_orphaned_chunks({"c1"})
    assert result["removed_relationships"] == 1
    assert r1.key not in graph._relationships
    # Verify SQLite row is gone
    db = graph._get_db()
    assert db.execute("SELECT COUNT(*) FROM relationships WHERE key = ?", (r1.key,)).fetchone()[0] == 0
