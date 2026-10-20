"""Tests for mullet.knowledge.graph — knowledge graph module ported from Trove."""

import pytest
from mullet.knowledge.graph.models import Entity, Relationship
from mullet.knowledge.graph.store import KnowledgeGraph


class TestEntity:
    def test_creation_defaults(self):
        e = Entity(name="CodeParser", entity_type="class")
        assert e.name == "CodeParser"
        assert e.entity_type == "class"
        assert e.description == ""
        assert e.source_chunk_ids == []
        assert e.aliases == []

    def test_key_normalized(self):
        e = Entity(name="  CodeParser ", entity_type=" CLASS ")
        assert e.key == "codeparser::class"

    def test_merge_accumulates_sources(self):
        e1 = Entity(name="Foo", entity_type="func", source_chunk_ids=["c1"])
        e2 = Entity(name="Foo", entity_type="func", source_chunk_ids=["c2", "c1"])
        e1.merge(e2)
        assert "c1" in e1.source_chunk_ids
        assert "c2" in e1.source_chunk_ids
        assert e1.source_chunk_ids.count("c1") == 1

    def test_merge_accumulates_aliases(self):
        e1 = Entity(name="Foo", entity_type="func", aliases=["bar"])
        e2 = Entity(name="Foo", entity_type="func", aliases=["baz", "bar"])
        e1.merge(e2)
        assert "bar" in e1.aliases
        assert "baz" in e1.aliases
        assert e1.aliases.count("bar") == 1

    def test_merge_keeps_longer_description(self):
        e1 = Entity(name="Foo", entity_type="func", description="short")
        e2 = Entity(name="Foo", entity_type="func", description="a much longer description")
        e1.merge(e2)
        assert e1.description == "a much longer description"

    def test_to_dict_from_dict_roundtrip(self):
        e = Entity(
            name="Parser", entity_type="class", description="Parses stuff",
            source_chunk_ids=["c1", "c2"], aliases=["CodeParser"],
        )
        d = e.to_dict()
        e2 = Entity.from_dict(d)
        assert e2.name == e.name
        assert e2.entity_type == e.entity_type
        assert e2.description == e.description
        assert e2.source_chunk_ids == e.source_chunk_ids
        assert e2.aliases == e.aliases


class TestRelationship:
    def test_creation_defaults(self):
        r = Relationship(source_entity="a::class", target_entity="b::func", relationship_type="uses")
        assert r.weight == 1.0
        assert r.description == ""
        assert r.source_chunk_ids == []

    def test_key_format(self):
        r = Relationship(source_entity="a::class", target_entity="b::func", relationship_type="Uses")
        assert r.key == "a::class--uses--b::func"

    def test_merge_takes_max_weight(self):
        r1 = Relationship(source_entity="a", target_entity="b", relationship_type="r", weight=0.5)
        r2 = Relationship(source_entity="a", target_entity="b", relationship_type="r", weight=0.9)
        r1.merge(r2)
        assert r1.weight == 0.9

    def test_merge_accumulates_sources(self):
        r1 = Relationship(source_entity="a", target_entity="b", relationship_type="r", source_chunk_ids=["c1"])
        r2 = Relationship(source_entity="a", target_entity="b", relationship_type="r", source_chunk_ids=["c2"])
        r1.merge(r2)
        assert set(r1.source_chunk_ids) == {"c1", "c2"}

    def test_to_dict_from_dict_roundtrip(self):
        r = Relationship(
            source_entity="a::class", target_entity="b::func",
            relationship_type="calls", description="A calls B",
            weight=0.8, source_chunk_ids=["c1"],
        )
        d = r.to_dict()
        r2 = Relationship.from_dict(d)
        assert r2.source_entity == r.source_entity
        assert r2.target_entity == r.target_entity
        assert r2.relationship_type == r.relationship_type
        assert r2.weight == r.weight


class TestKnowledgeGraph:
    @pytest.fixture
    def index_dir(self, tmp_path):
        return tmp_path

    @pytest.fixture
    def kg(self, index_dir):
        g = KnowledgeGraph(index_dir)
        yield g
        g.close()

    def test_add_and_get_entity(self, kg):
        e = Entity(name="Foo", entity_type="class")
        kg.add_entity(e)
        found = kg.get_entity(e.key)
        assert found is not None
        assert found.name == "Foo"

    def test_get_entity_not_found(self, kg):
        assert kg.get_entity("nonexistent::key") is None

    def test_add_relationship(self, kg):
        e1 = Entity(name="A", entity_type="class")
        e2 = Entity(name="B", entity_type="func")
        kg.add_entity(e1)
        kg.add_entity(e2)
        r = Relationship(source_entity=e1.key, target_entity=e2.key, relationship_type="uses")
        kg.add_relationship(r)
        assert kg.stats["relationship_count"] == 1

    def test_find_entities_by_name(self, kg):
        kg.add_entity(Entity(name="Parser", entity_type="class"))
        kg.add_entity(Entity(name="ParserHelper", entity_type="func"))
        kg.add_entity(Entity(name="Unrelated", entity_type="class"))
        found = kg.find_entities_by_name("parser")
        names = [e.name for e in found]
        assert "Parser" in names
        assert "ParserHelper" in names

    def test_get_entities_by_type(self, kg):
        kg.add_entity(Entity(name="A", entity_type="class"))
        kg.add_entity(Entity(name="B", entity_type="class"))
        kg.add_entity(Entity(name="C", entity_type="func"))
        classes = kg.get_entities_by_type("class")
        assert len(classes) == 2

    def test_stats(self, kg):
        kg.add_entity(Entity(name="E1", entity_type="class"))
        kg.add_entity(Entity(name="E2", entity_type="func"))
        r = Relationship(source_entity="e1::class", target_entity="e2::func", relationship_type="uses")
        kg.add_relationship(r)
        s = kg.stats
        assert s["entity_count"] == 2
        assert s["relationship_count"] == 1

    def test_save_and_reload(self, index_dir):
        kg1 = KnowledgeGraph(index_dir)
        kg1.add_entity(Entity(name="Persistent", entity_type="class", description="survives"))
        kg1.save()
        kg1.close()

        kg2 = KnowledgeGraph(index_dir)
        kg2.load()
        found = kg2.get_entity("persistent::class")
        assert found is not None
        assert found.description == "survives"
        kg2.close()

    def test_clear(self, kg):
        kg.add_entity(Entity(name="Gone", entity_type="class"))
        kg.clear()
        assert kg.get_entity("gone::class") is None
        assert kg.stats["entity_count"] == 0

    def test_mark_chunks_processed(self, kg):
        kg.mark_chunks_processed(["c1", "c2"])
        assert "c1" in kg.processed_chunk_ids
        assert "c2" in kg.processed_chunk_ids
        kg.mark_chunks_processed(["c3"])
        assert len(kg.processed_chunk_ids) == 3

    def test_get_source_chunks(self, kg):
        e = Entity(name="Foo", entity_type="class", source_chunk_ids=["c1", "c2"])
        kg.add_entity(e)


class TestPruneOrphanedChunks:
    def test_prune_removes_stale_chunks(self, tmp_path):
        g = KnowledgeGraph(tmp_path)
        e = Entity(name="Foo", entity_type="class", source_chunk_ids=["c1", "c2"])
        g.add_entity(e)
        g.mark_chunks_processed(["c1", "c2", "c3"])
        g.save()

        result = g.prune_orphaned_chunks({"c1"})

        assert result["removed_chunks"] == 2
        assert g._processed_chunk_ids == {"c1"}
        key = e.key
        assert key in g._entities
        assert g._entities[key].source_chunk_ids == ["c1"]

    def test_prune_drops_entity_with_all_stale_sources(self, tmp_path):
        g = KnowledgeGraph(tmp_path)
        e = Entity(name="Foo", entity_type="class", source_chunk_ids=["c2"])
        g.add_entity(e)
        g.mark_chunks_processed(["c2"])
        g.save()

        result = g.prune_orphaned_chunks({"c1"})

        assert result["removed_entities"] == 1
        assert e.key not in g._entities

    def test_prune_drops_relationship_with_all_stale_sources(self, tmp_path):
        g = KnowledgeGraph(tmp_path)
        a = Entity(name="A", entity_type="class", source_chunk_ids=["c1"])
        b = Entity(name="B", entity_type="class", source_chunk_ids=["c1"])
        g.add_entity(a)
        g.add_entity(b)
        r = Relationship(
            source_entity=a.key,
            target_entity=b.key,
            relationship_type="uses",
            source_chunk_ids=["c2"],
        )
        g.add_relationship(r)
        g.mark_chunks_processed(["c1", "c2"])
        g.save()

        result = g.prune_orphaned_chunks({"c1"})

        assert result["removed_relationships"] == 1
        assert r.key not in g._relationships

    def test_prune_fast_path_nothing_stale(self, tmp_path):
        g = KnowledgeGraph(tmp_path)
        e = Entity(name="Foo", entity_type="class", source_chunk_ids=["c1"])
        g.add_entity(e)
        g.mark_chunks_processed(["c1"])
        g.save()

        result = g.prune_orphaned_chunks({"c1", "c2"})

        assert result["removed_chunks"] == 0
        assert result["removed_entities"] == 0
        assert result["removed_relationships"] == 0
        assert result["updated_entities"] == 0
        assert result["updated_relationships"] == 0

    def test_prune_persists_to_sqlite(self, tmp_path):
        g1 = KnowledgeGraph(tmp_path)
        e = Entity(name="Foo", entity_type="class", source_chunk_ids=["c1", "c2"])
        g1.add_entity(e)
        g1.mark_chunks_processed(["c1", "c2", "c3"])
        g1.save()
        g1.prune_orphaned_chunks({"c1"})
        g1.close()

        g2 = KnowledgeGraph(tmp_path)
        assert g2.load()
        assert g2._processed_chunk_ids == {"c1"}
        assert g2._entities[e.key].source_chunk_ids == ["c1"]
