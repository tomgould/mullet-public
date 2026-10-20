from mullet.knowledge.graph.models import Entity, Relationship


class TestEntity:
    def test_key_normalization(self):
        e = Entity(name="  Alice  ", entity_type=" Person ")
        assert e.key == "alice::person"

    def test_key_case_insensitive(self):
        e1 = Entity(name="Bob", entity_type="Person")
        e2 = Entity(name="bob", entity_type="person")
        assert e1.key == e2.key

    def test_merge_accumulates_sources(self):
        e1 = Entity(name="A", entity_type="t", source_chunk_ids=["c1"], aliases=["a1"])
        e2 = Entity(name="A", entity_type="t", source_chunk_ids=["c2", "c1"], aliases=["a2"])
        e1.merge(e2)
        assert e1.source_chunk_ids == ["c1", "c2"]
        assert e1.aliases == ["a1", "a2"]

    def test_merge_takes_longer_description(self):
        e1 = Entity(name="A", entity_type="t", description="short")
        e2 = Entity(name="A", entity_type="t", description="a much longer description")
        e1.merge(e2)
        assert e1.description == "a much longer description"

    def test_merge_keeps_description_if_not_longer(self):
        e1 = Entity(name="A", entity_type="t", description="already long enough")
        e2 = Entity(name="A", entity_type="t", description="short")
        e1.merge(e2)
        assert e1.description == "already long enough"

    def test_to_dict_roundtrip(self):
        e = Entity(name="X", entity_type="Y", description="desc", source_chunk_ids=["c1"], aliases=["a1"])
        d = e.to_dict()
        assert d == {"name": "X", "entity_type": "Y", "description": "desc", "source_chunk_ids": ["c1"], "aliases": ["a1"]}

    def test_from_dict_defaults(self):
        e = Entity.from_dict({"name": "X", "entity_type": "Y"})
        assert e.description == ""
        assert e.source_chunk_ids == []
        assert e.aliases == []

    def test_from_dict_full(self):
        e = Entity.from_dict({"name": "X", "entity_type": "Y", "description": "d", "source_chunk_ids": ["c1"], "aliases": ["a1"]})
        assert e.name == "X"
        assert e.entity_type == "Y"
        assert e.description == "d"
        assert e.source_chunk_ids == ["c1"]
        assert e.aliases == ["a1"]


class TestRelationship:
    def test_key_format(self):
        r = Relationship(source_entity="a", target_entity="b", relationship_type=" KNOWS ")
        assert r.key == "a--knows--b"

    def test_merge_takes_max_weight(self):
        r1 = Relationship(source_entity="a", target_entity="b", relationship_type="t", weight=0.5)
        r2 = Relationship(source_entity="a", target_entity="b", relationship_type="t", weight=0.9)
        r1.merge(r2)
        assert r1.weight == 0.9

    def test_merge_does_not_lower_weight(self):
        r1 = Relationship(source_entity="a", target_entity="b", relationship_type="t", weight=0.9)
        r2 = Relationship(source_entity="a", target_entity="b", relationship_type="t", weight=0.5)
        r1.merge(r2)
        assert r1.weight == 0.9

    def test_merge_accumulates_chunk_ids(self):
        r1 = Relationship(source_entity="a", target_entity="b", relationship_type="t", source_chunk_ids=["c1"])
        r2 = Relationship(source_entity="a", target_entity="b", relationship_type="t", source_chunk_ids=["c2"])
        r1.merge(r2)
        assert r1.source_chunk_ids == ["c1", "c2"]

    def test_to_dict_roundtrip(self):
        r = Relationship(source_entity="a", target_entity="b", relationship_type="t", description="d", weight=0.7, source_chunk_ids=["c1"])
        d = r.to_dict()
        assert d == {"source_entity": "a", "target_entity": "b", "relationship_type": "t", "description": "d", "weight": 0.7, "source_chunk_ids": ["c1"]}

    def test_from_dict_defaults(self):
        r = Relationship.from_dict({"source_entity": "a", "target_entity": "b", "relationship_type": "t"})
        assert r.description == ""
        assert r.weight == 1.0
        assert r.source_chunk_ids == []

    def test_from_dict_full(self):
        r = Relationship.from_dict({"source_entity": "a", "target_entity": "b", "relationship_type": "t", "description": "d", "weight": 0.3, "source_chunk_ids": ["c1"]})
        assert r.source_entity == "a"
        assert r.target_entity == "b"
        assert r.relationship_type == "t"
        assert r.description == "d"
        assert r.weight == 0.3
        assert r.source_chunk_ids == ["c1"]
