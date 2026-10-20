import pytest
import yaml

from mullet.knowledge.engine.agents import AgentLoader


def _write_agent(dir_path, slug, data):
    (dir_path / f"{slug}.yaml").write_text(yaml.safe_dump(data))


@pytest.fixture
def agents_dir(tmp_path):
    d = tmp_path / "agents"
    d.mkdir()
    _write_agent(d, "reviewer", {
        "name": "Code Reviewer",
        "description": "Reviews code.",
        "system_prompt": "You are a reviewer.",
        "tags": ["review", "quality"],
    })
    _write_agent(d, "planner", {
        "name": "Planner",
        "description": "Plans tasks.",
        "system_prompt": "You are a planner.",
        "tags": ["planning"],
    })
    _write_agent(d, "researcher", {
        "name": "Researcher",
        "description": "Does research and planning.",
        "system_prompt": "You are a researcher.",
        "tags": ["search"],
    })
    return d


def test_list_all(agents_dir):
    loader = AgentLoader(agents_dir)
    result = loader.list()
    assert len(result) == 3
    names = {r["name"] for r in result}
    assert names == {"Code Reviewer", "Planner", "Researcher"}


def test_list_filter_tag(agents_dir):
    loader = AgentLoader(agents_dir)
    result = loader.list(tag="review")
    assert len(result) == 1
    assert result[0]["name"] == "Code Reviewer"


def test_get_by_slug(agents_dir):
    loader = AgentLoader(agents_dir)
    result = loader.get("reviewer")
    assert result is not None
    assert result["name"] == "Code Reviewer"
    assert "system_prompt" in result


def test_get_by_name(agents_dir):
    loader = AgentLoader(agents_dir)
    result = loader.get("Planner")
    assert result is not None
    assert result["_filename"] == "planner"


def test_get_missing_returns_none(agents_dir):
    loader = AgentLoader(agents_dir)
    assert loader.get("ghost") is None


def test_get_returns_copy(agents_dir):
    loader = AgentLoader(agents_dir)
    first = loader.get("planner")
    first["name"] = "mutated"
    second = loader.get("planner")
    assert second["name"] == "Planner"


def test_search_by_name(agents_dir):
    loader = AgentLoader(agents_dir)
    result = loader.search("reviewer")
    assert len(result) == 1
    assert result[0]["name"] == "Code Reviewer"


def test_search_ranking(agents_dir):
    loader = AgentLoader(agents_dir)
    result = loader.search("plan")
    assert result[0]["name"] == "Planner"
    assert result[1]["name"] == "Researcher"
