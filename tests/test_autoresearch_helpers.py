"""Tests for mullet.core.orchestration.autoresearch — pure state logic."""

from pathlib import Path

from mullet.core.orchestration.autoresearch import (
    ResearchFinding,
    ResearchIteration,
    ResearchPipeline,
)


# ---------------------------------------------------------------------------
# ResearchFinding
# ---------------------------------------------------------------------------

class TestResearchFinding:
    def test_stores_content_and_source(self):
        f = ResearchFinding(content="some text", source="https://example.com")
        assert f.content == "some text"
        assert f.source == "https://example.com"

    def test_default_relevance(self):
        f = ResearchFinding(content="x", source="y")
        assert f.relevance == 1.0

    def test_custom_relevance(self):
        f = ResearchFinding(content="x", source="y", relevance=0.5)
        assert f.relevance == 0.5

    def test_timestamp_auto_set(self):
        f = ResearchFinding(content="x", source="y")
        assert f.timestamp != ""
        assert "T" in f.timestamp  # ISO format

    def test_explicit_timestamp_preserved(self):
        f = ResearchFinding(content="x", source="y", timestamp="2025-01-01T00:00:00")
        assert f.timestamp == "2025-01-01T00:00:00"


# ---------------------------------------------------------------------------
# ResearchIteration
# ---------------------------------------------------------------------------

class TestResearchIteration:
    def test_stores_iteration_and_query(self):
        it = ResearchIteration(iteration=1, query="what is X?")
        assert it.iteration == 1
        assert it.query == "what is X?"

    def test_default_empty_findings(self):
        it = ResearchIteration(iteration=1, query="q")
        assert it.findings == []

    def test_default_score_zero(self):
        it = ResearchIteration(iteration=1, query="q")
        assert it.score == 0.0

    def test_timestamp_auto_set(self):
        it = ResearchIteration(iteration=1, query="q")
        assert it.timestamp != ""

    def test_custom_score(self):
        it = ResearchIteration(iteration=2, query="q", score=0.75)
        assert it.score == 0.75


# ---------------------------------------------------------------------------
# ResearchPipeline — init and state
# ---------------------------------------------------------------------------

class TestResearchPipelineInit:
    def test_stores_question(self):
        p = ResearchPipeline("What is Python?")
        assert p.question == "What is Python?"

    def test_default_max_iterations(self):
        p = ResearchPipeline("q")
        assert p.max_iterations == 5

    def test_default_min_score(self):
        p = ResearchPipeline("q")
        assert p.min_score == 0.8

    def test_initial_status_pending(self):
        p = ResearchPipeline("q")
        assert p.status == "pending"

    def test_initial_iterations_empty(self):
        p = ResearchPipeline("q")
        assert p.iterations == []


# ---------------------------------------------------------------------------
# ResearchPipeline.should_continue
# ---------------------------------------------------------------------------

class TestShouldContinue:
    def test_empty_always_continues(self):
        p = ResearchPipeline("q")
        assert p.should_continue() is True

    def test_high_score_stops(self):
        p = ResearchPipeline("q", min_score=0.8)
        p.add_iteration("q", [], "summary", score=0.9)
        assert p.should_continue() is False

    def test_low_score_continues(self):
        p = ResearchPipeline("q", min_score=0.8)
        p.add_iteration("q", [], "summary", score=0.5)
        assert p.should_continue() is True

    def test_max_iterations_stops(self):
        p = ResearchPipeline("q", max_iterations=2, min_score=0.9)
        p.add_iteration("q", [], "s", score=0.1)
        p.add_iteration("q", [], "s", score=0.1)
        assert p.should_continue() is False

    def test_exactly_at_min_score_stops(self):
        p = ResearchPipeline("q", min_score=0.8)
        p.add_iteration("q", [], "s", score=0.8)
        assert p.should_continue() is False


# ---------------------------------------------------------------------------
# ResearchPipeline.add_iteration
# ---------------------------------------------------------------------------

class TestAddIteration:
    def test_returns_iteration(self):
        p = ResearchPipeline("q")
        it = p.add_iteration("query1", [], "summary", score=0.5)
        assert isinstance(it, ResearchIteration)

    def test_iteration_number_increments(self):
        p = ResearchPipeline("q")
        it1 = p.add_iteration("q1", [], "s", score=0.3)
        it2 = p.add_iteration("q2", [], "s", score=0.3)
        assert it1.iteration == 1
        assert it2.iteration == 2

    def test_status_running_after_add(self):
        p = ResearchPipeline("q", min_score=0.9)
        p.add_iteration("q", [], "s", score=0.3)
        assert p.status == "running"

    def test_status_completed_when_done(self):
        p = ResearchPipeline("q", min_score=0.5)
        p.add_iteration("q", [], "s", score=0.9)
        assert p.status == "completed"

    def test_appends_to_iterations(self):
        p = ResearchPipeline("q")
        p.add_iteration("q", [], "s", score=0.3)
        assert len(p.iterations) == 1


# ---------------------------------------------------------------------------
# ResearchPipeline.best_iteration
# ---------------------------------------------------------------------------

class TestBestIteration:
    def test_empty_returns_none(self):
        p = ResearchPipeline("q")
        assert p.best_iteration() is None

    def test_returns_highest_score(self):
        p = ResearchPipeline("q")
        p.add_iteration("q1", [], "s", score=0.4)
        p.add_iteration("q2", [], "s", score=0.9)
        p.add_iteration("q3", [], "s", score=0.6)
        best = p.best_iteration()
        assert best.score == 0.9

    def test_single_iteration_is_best(self):
        p = ResearchPipeline("q")
        p.add_iteration("only", [], "s", score=0.7)
        assert p.best_iteration().query == "only"


# ---------------------------------------------------------------------------
# ResearchPipeline.get_refined_query
# ---------------------------------------------------------------------------

class TestGetRefinedQuery:
    def test_no_iterations_returns_original(self):
        p = ResearchPipeline("What is Python?")
        assert p.get_refined_query() == "What is Python?"

    def test_with_iteration_includes_original(self):
        p = ResearchPipeline("What is Python?")
        p.add_iteration("q", [], "some gaps here", score=0.4)
        refined = p.get_refined_query()
        assert "What is Python?" in refined

    def test_includes_previous_score(self):
        p = ResearchPipeline("q")
        p.add_iteration("q", [], "summary text", score=0.4)
        refined = p.get_refined_query()
        assert "0.4" in refined

    def test_includes_summary_snippet(self):
        p = ResearchPipeline("q")
        p.add_iteration("q", [], "summary text here", score=0.3)
        refined = p.get_refined_query()
        assert "summary text here" in refined


# ---------------------------------------------------------------------------
# ResearchPipeline.to_dict / from_dict round-trip
# ---------------------------------------------------------------------------

class TestToDictFromDict:
    def test_to_dict_has_question(self):
        p = ResearchPipeline("my question")
        d = p.to_dict()
        assert d["question"] == "my question"

    def test_to_dict_has_status(self):
        p = ResearchPipeline("q")
        assert "status" in p.to_dict()

    def test_to_dict_has_iterations(self):
        p = ResearchPipeline("q")
        p.add_iteration("q1", [], "s", score=0.5)
        d = p.to_dict()
        assert len(d["iterations"]) == 1

    def test_round_trip_question(self):
        p = ResearchPipeline("Test question", max_iterations=3, min_score=0.7)
        p2 = ResearchPipeline.from_dict(p.to_dict())
        assert p2.question == "Test question"
        assert p2.max_iterations == 3
        assert p2.min_score == 0.7

    def test_round_trip_iterations(self):
        p = ResearchPipeline("q")
        f = ResearchFinding(content="data", source="src")
        p.add_iteration("search query", [f], "summary", score=0.6)
        p2 = ResearchPipeline.from_dict(p.to_dict())
        assert len(p2.iterations) == 1
        assert p2.iterations[0].query == "search query"

    def test_round_trip_findings(self):
        p = ResearchPipeline("q")
        f = ResearchFinding(content="finding text", source="http://example.com")
        p.add_iteration("q", [f], "s", score=0.5)
        p2 = ResearchPipeline.from_dict(p.to_dict())
        assert p2.iterations[0].findings[0].content == "finding text"


# ---------------------------------------------------------------------------
# ResearchPipeline.save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_file(self, tmp_path):
        p = ResearchPipeline("q")
        path = str(tmp_path / "research.json")
        p.save(path)
        assert Path(path).exists()

    def test_save_load_round_trip(self, tmp_path):
        p = ResearchPipeline("my question")
        p.add_iteration("q", [], "summary", score=0.7)
        path = str(tmp_path / "research.json")
        p.save(path)
        p2 = ResearchPipeline.load(path)
        assert p2.question == "my question"
        assert len(p2.iterations) == 1

    def test_save_creates_parent_dirs(self, tmp_path):
        p = ResearchPipeline("q")
        path = str(tmp_path / "nested" / "dir" / "research.json")
        p.save(path)
        assert Path(path).exists()
