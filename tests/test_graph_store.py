from unittest.mock import MagicMock, patch

import pytest

from mullet.knowledge.graph.models import Entity, Relationship
from mullet.knowledge.graph.store import KnowledgeGraph


def _mock_graph():
    g = MagicMock()
    g.add_node = MagicMock()
    g.add_edge = MagicMock()
    g.out_edges = MagicMock(return_value=[])
    g.in_edges = MagicMock(return_value=[])
    g.__contains__ = MagicMock(return_value=True)
    return g


@pytest.fixture
def kg(tmp_path):
    return KnowledgeGraph(index_path=tmp_path)


def test_add_entity_stores_and_marks_dirty(kg):
    with patch.object(kg, "_ensure_graph", return_value=_mock_graph()):
        e = Entity(name="Foo", entity_type="class")
        kg.add_entity(e)
        assert kg.get_entity(e.key) is not None
        assert e.key in kg._dirty_entities


def test_add_entity_merges_existing(kg):
    with patch.object(kg, "_ensure_graph", return_value=_mock_graph()):
        e1 = Entity(name="Foo", entity_type="class", source_chunk_ids=["c1"])
        e2 = Entity(name="Foo", entity_type="class", source_chunk_ids=["c2"])
        kg.add_entity(e1)
        kg.add_entity(e2)
        assert sorted(kg.get_entity(e1.key).source_chunk_ids) == ["c1", "c2"]


def test_add_relationship_stores_and_marks_dirty(kg):
    with patch.object(kg, "_ensure_graph", return_value=_mock_graph()):
        r = Relationship(source_entity="a", target_entity="b", relationship_type="uses")
        kg.add_relationship(r)
        assert r.key in kg._relationships
        assert r.key in kg._dirty_relationships


def test_get_entities_by_type_filter(kg):
    kg._entities["foo::class"] = Entity(name="Foo", entity_type="class")
    kg._entities["bar::func"] = Entity(name="Bar", entity_type="function")
    results = kg.get_entities_by_type("class")
    assert len(results) == 1
    assert results[0].name == "Foo"


def test_find_entities_by_name_substring(kg):
    kg._entities["foo::class"] = Entity(name="FooBar", entity_type="class")
    kg._entities["bar::func"] = Entity(name="BarBaz", entity_type="function")
    results = kg.find_entities_by_name("bar")
    assert len(results) == 2


def test_find_entities_by_alias(kg):
    e = Entity(name="Foo", entity_type="class", aliases=["AliasFoo"])
    kg._entities[e.key] = e
    results = kg.find_entities_by_name("aliasfoo")
    assert len(results) == 1


def test_get_source_chunks_gathers_from_entities_and_rels(kg):
    e = Entity(name="A", entity_type="class", source_chunk_ids=["c1"])
    r = Relationship(source_entity=e.key, target_entity="b", relationship_type="uses", source_chunk_ids=["c2"])
    kg._entities[e.key] = e
    kg._relationships[r.key] = r
    chunks = kg.get_source_chunks([e.key])
    assert sorted(chunks) == ["c1", "c2"]


def test_mark_chunks_processed_tracks_new_ids(kg):
    kg.mark_chunks_processed(["id1", "id2"])
    assert "id1" in kg.processed_chunk_ids
    assert "id2" in kg.processed_chunk_ids
    assert {"id1", "id2"} <= kg._dirty_chunks


def test_stats_counts_entities_and_rels(kg):
    kg._entities["a::class"] = Entity(name="A", entity_type="class")
    kg._entities["b::func"] = Entity(name="B", entity_type="function")
    kg._relationships["a--uses--b"] = Relationship(source_entity="a::class", target_entity="b::func", relationship_type="uses")
    stats = kg.stats
    assert stats["entity_count"] == 2
    assert stats["relationship_count"] == 1
    assert stats["entity_type_distribution"]["class"] == 1


def test_clear_clears_in_memory_state(kg):
    kg._entities["a::class"] = Entity(name="A", entity_type="class")
    kg._relationships["a--uses--b"] = Relationship(source_entity="a::class", target_entity="b::func", relationship_type="uses")
    kg._processed_chunk_ids = {"c1"}
    with patch.object(kg, "_graph", _mock_graph()):
        kg.clear()
    assert kg._entities == {}
    assert kg._relationships == {}
    assert kg._processed_chunk_ids == set()
    assert kg._dirty_entities == set()
    assert kg._dirty_relationships == set()
    assert kg._dirty_chunks == set()
