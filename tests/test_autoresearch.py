"""Tests for mullet.core.orchestration.autoresearch."""

from mullet.core.orchestration.autoresearch import ResearchPipeline, ResearchFinding


def test_pipeline_creation():
    p = ResearchPipeline("How does X work?")
    assert p.question == "How does X work?"
    assert p.status == "pending"
    assert p.should_continue()


def test_add_iteration():
    p = ResearchPipeline("Q")
    f = ResearchFinding(content="Answer", source="web")
    it = p.add_iteration("Q refined", [f], "Summary", 0.5)
    assert it.iteration == 1
    assert len(it.findings) == 1
    assert p.status == "running"


def test_should_continue_below_threshold():
    p = ResearchPipeline("Q", min_score=0.8)
    p.add_iteration("Q", [], "low quality", 0.3)
    assert p.should_continue()


def test_should_continue_above_threshold():
    p = ResearchPipeline("Q", min_score=0.8)
    p.add_iteration("Q", [], "high quality", 0.9)
    assert not p.should_continue()
    assert p.status == "completed"


def test_should_continue_max_iterations():
    p = ResearchPipeline("Q", max_iterations=2)
    p.add_iteration("Q1", [], "s1", 0.3)
    p.add_iteration("Q2", [], "s2", 0.4)
    assert not p.should_continue()


def test_best_iteration():
    p = ResearchPipeline("Q", max_iterations=5)
    p.add_iteration("Q1", [], "s1", 0.3)
    p.add_iteration("Q2", [], "s2", 0.7)
    p.add_iteration("Q3", [], "s3", 0.5)
    assert p.best_iteration().score == 0.7


def test_best_iteration_empty():
    p = ResearchPipeline("Q")
    assert p.best_iteration() is None


def test_get_refined_query():
    p = ResearchPipeline("Original question")
    assert p.get_refined_query() == "Original question"
    p.add_iteration("Q1", [], "gaps here", 0.4)
    refined = p.get_refined_query()
    assert "Original question" in refined
    assert "0.4" in refined


def test_serialization_roundtrip(tmp_path):
    p = ResearchPipeline("Test Q", max_iterations=3, min_score=0.7)
    f = ResearchFinding(content="Data", source="api", relevance=0.9)
    p.add_iteration("Q1", [f], "summary", 0.6)
    path = str(tmp_path / "research.json")
    p.save(path)
    p2 = ResearchPipeline.load(path)
    assert p2.question == "Test Q"
    assert len(p2.iterations) == 1
    assert p2.iterations[0].findings[0].content == "Data"
    assert p2.iterations[0].score == 0.6


def test_finding_defaults():
    f = ResearchFinding(content="X", source="Y")
    assert f.relevance == 1.0
    assert f.timestamp != ""
