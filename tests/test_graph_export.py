"""Tests for graph stats, export utilities, and progress tracking."""

from pathlib import Path
from mullet.knowledge.graph.models import Entity, Relationship
from mullet.knowledge.graph.store import KnowledgeGraph
from mullet.knowledge.graph.export import graph_stats, export_for_d3
from mullet.knowledge.graph.progress import GraphBuildProgress


def _make_graph(tmp_path: Path) -> KnowledgeGraph:
    """Create an empty in-memory-backed KnowledgeGraph."""
    return KnowledgeGraph(index_path=tmp_path)


def _add_entity(graph: KnowledgeGraph, name: str, etype: str, chunks=None) -> Entity:
    e = Entity(name=name, entity_type=etype, source_chunk_ids=chunks or [])
    graph._entities[e.key] = e
    return e


def _add_relationship(graph: KnowledgeGraph, src_key: str, tgt_key: str, rtype: str) -> Relationship:
    r = Relationship(source_entity=src_key, target_entity=tgt_key, relationship_type=rtype)
    graph._relationships[r.key] = r
    return r


class TestGraphStats:
    def test_empty_graph_has_zero_counts(self, tmp_path):
        g = _make_graph(tmp_path)
        stats = graph_stats(g)
        assert stats["entity_count"] == 0
        assert stats["relationship_count"] == 0
        assert stats["density"] == 0.0

    def test_single_entity_type_distribution(self, tmp_path):
        g = _make_graph(tmp_path)
        _add_entity(g, "Alice", "person")
        stats = graph_stats(g)
        assert stats["entity_count"] == 1
        assert stats["entity_type_distribution"]["person"] == 1

    def test_density_calculated_for_connected_graph(self, tmp_path):
        g = _make_graph(tmp_path)
        e1 = _add_entity(g, "A", "person")
        e2 = _add_entity(g, "B", "person")
        _add_relationship(g, e1.key, e2.key, "knows")
        stats = graph_stats(g)
        assert stats["density"] > 0
        # density = 1 / (2 * 1) = 0.5 for 2 nodes, 1 edge (directed)
        assert abs(stats["density"] - 0.5) < 1e-6

    def test_top_entities_by_connection_count(self, tmp_path):
        g = _make_graph(tmp_path)
        hub = _add_entity(g, "Hub", "concept")
        for i in range(3):
            leaf = _add_entity(g, f"Leaf{i}", "concept")
            _add_relationship(g, hub.key, leaf.key, "links")
        stats = graph_stats(g)
        top = stats["top_entities"]
        assert len(top) >= 1
        assert top[0]["name"] == "Hub"
        assert top[0]["connections"] == 3

    def test_source_chunk_count_unique(self, tmp_path):
        g = _make_graph(tmp_path)
        _add_entity(g, "A", "person", chunks=["c1", "c2"])
        _add_entity(g, "B", "person", chunks=["c2", "c3"])  # c2 shared
        stats = graph_stats(g)
        assert stats["source_chunk_count"] == 3  # c1, c2, c3 unique


class TestExportForD3:
    def test_empty_graph_returns_empty_lists(self, tmp_path):
        g = _make_graph(tmp_path)
        result = export_for_d3(g)
        assert result == {"nodes": [], "links": [], "type_groups": {}}

    def test_nodes_contain_required_fields(self, tmp_path):
        g = _make_graph(tmp_path)
        _add_entity(g, "Alice", "person", chunks=["c1"])
        result = export_for_d3(g)
        assert len(result["nodes"]) == 1
        node = result["nodes"][0]
        assert "id" in node and "name" in node
        assert node["name"] == "Alice"
        assert node["type"] == "person"
        assert node["chunk_count"] == 1

    def test_links_include_relationship_type(self, tmp_path):
        g = _make_graph(tmp_path)
        e1 = _add_entity(g, "A", "person")
        e2 = _add_entity(g, "B", "organization")
        _add_relationship(g, e1.key, e2.key, "works_for")
        result = export_for_d3(g)
        assert len(result["links"]) == 1
        link = result["links"][0]
        assert link["type"] == "works_for"
        assert link["source"] == e1.key
        assert link["target"] == e2.key

    def test_max_nodes_limits_output(self, tmp_path):
        g = _make_graph(tmp_path)
        for i in range(10):
            _add_entity(g, f"E{i}", "concept")
        result = export_for_d3(g, max_nodes=3)
        assert len(result["nodes"]) == 3

    def test_links_excluded_for_nodes_outside_max(self, tmp_path):
        g = _make_graph(tmp_path)
        entities = [_add_entity(g, f"E{i}", "concept") for i in range(5)]
        # Add relationship between last two (will be outside max_nodes=2)
        _add_relationship(g, entities[3].key, entities[4].key, "related")
        result = export_for_d3(g, max_nodes=2)
        assert len(result["links"]) == 0

    def test_type_groups_assigned_incrementally(self, tmp_path):
        g = _make_graph(tmp_path)
        _add_entity(g, "A", "person")
        _add_entity(g, "B", "organization")
        result = export_for_d3(g)
        groups = result["type_groups"]
        assert "person" in groups
        assert "organization" in groups
        assert groups["person"] != groups["organization"]


class TestGraphBuildProgress:
    def test_percent_zero_processed(self):
        p = GraphBuildProgress(total_chunks=100)
        assert p.percent == 0.0

    def test_percent_complete(self):
        p = GraphBuildProgress(total_chunks=100, processed_chunks=100)
        assert p.percent == 100.0

    def test_percent_zero_total_is_100(self):
        p = GraphBuildProgress(total_chunks=0)
        assert p.percent == 100.0

    def test_percent_partial(self):
        p = GraphBuildProgress(total_chunks=200, processed_chunks=50)
        assert abs(p.percent - 25.0) < 1e-6

    def test_eta_none_before_first_chunk(self):
        p = GraphBuildProgress(total_chunks=100)
        assert p.eta_seconds is None

    def test_format_eta_calculating_before_first_chunk(self):
        p = GraphBuildProgress(total_chunks=100)
        assert p.format_eta() == "calculating..."

    def test_format_eta_seconds_only(self):
        import time
        p = GraphBuildProgress(total_chunks=10, processed_chunks=5)
        p.start_time = time.time() - 5  # 5s elapsed, 5 remaining → 5s ETA
        eta_str = p.format_eta()
        assert "s" in eta_str
        assert "m" not in eta_str

    def test_format_eta_minutes_and_seconds(self):
        import time
        # 1 of 100 done in 1s → 99s remaining → 1m 39s
        p = GraphBuildProgress(total_chunks=100, processed_chunks=1)
        p.start_time = time.time() - 1
        eta_str = p.format_eta()
        assert "m" in eta_str
