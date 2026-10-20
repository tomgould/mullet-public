"""Tests for mullet.knowledge.engine.agents.AgentLoader — YAML template loader."""

from pathlib import Path

import pytest
import yaml

from mullet.knowledge.engine.agents import AgentLoader


def _write_agent(dir_path: Path, slug: str, data: dict) -> None:
    """Helper: write an agent YAML file into dir_path."""
    (dir_path / f"{slug}.yaml").write_text(yaml.safe_dump(data))


@pytest.fixture
def agents_dir(tmp_path):
    """Build a temp agents directory with 3 agent YAML templates."""
    d = tmp_path / "agents"
    d.mkdir()
    _write_agent(d, "code-reviewer", {
        "name": "Code Reviewer",
        "description": "Reviews code for bugs and style issues.",
        "system_prompt": "You are a careful code reviewer.",
        "tags": ["review", "quality"],
    })
    _write_agent(d, "planner", {
        "name": "Planner",
        "description": "Plans implementation approaches.",
        "system_prompt": "You are a software planner.",
        "tags": ["planning", "architecture"],
    })
    _write_agent(d, "researcher", {
        "name": "Researcher",
        "description": "Researches topics via search.",
        "system_prompt": "You are a thorough researcher.",
        "tags": ["search", "research"],
    })
    return d


# --- init ------------------------------------------------------------------


def test_init_with_missing_dir_logs_warning(tmp_path, caplog):
    import logging as _lg
    caplog.set_level(_lg.WARNING, logger="mullet.knowledge.engine.agents")
    AgentLoader(tmp_path / "does-not-exist")
    assert any("does not exist" in r.message for r in caplog.records)


def test_init_with_existing_dir_no_warning(agents_dir, caplog):
    import logging as _lg
    caplog.set_level(_lg.WARNING, logger="mullet.knowledge.engine.agents")
    AgentLoader(agents_dir)
    assert not any("does not exist" in r.message for r in caplog.records)


# --- list ------------------------------------------------------------------


def test_list_all_agents(agents_dir):
    loader = AgentLoader(agents_dir)
    result = loader.list()
    assert len(result) == 3
    names = {r["name"] for r in result}
    assert {"Code Reviewer", "Planner", "Researcher"} == names


def test_list_filter_by_tag(agents_dir):
    loader = AgentLoader(agents_dir)
    result = loader.list(tag="review")
    assert len(result) == 1
    assert result[0]["name"] == "Code Reviewer"


def test_list_filter_by_tag_case_insensitive(agents_dir):
    loader = AgentLoader(agents_dir)
    result = loader.list(tag="PLANNING")
    assert len(result) == 1
    assert result[0]["name"] == "Planner"


def test_list_missing_dir_returns_empty(tmp_path):
    loader = AgentLoader(tmp_path / "absent")
    assert loader.list() == []


# --- get -------------------------------------------------------------------


def test_get_by_slug(agents_dir):
    loader = AgentLoader(agents_dir)
    result = loader.get("code-reviewer")
    assert result is not None
    assert result["name"] == "Code Reviewer"
    assert "system_prompt" in result


def test_get_by_name_field(agents_dir):
    loader = AgentLoader(agents_dir)
    result = loader.get("Researcher")
    assert result is not None
    assert result["_filename"] == "researcher"


def test_get_case_insensitive(agents_dir):
    loader = AgentLoader(agents_dir)
    result = loader.get("PLANNER")
    assert result is not None
    assert result["name"] == "Planner"


def test_get_missing_returns_none(agents_dir):
    loader = AgentLoader(agents_dir)
    assert loader.get("nonexistent") is None


def test_get_returns_copy_not_reference(agents_dir):
    """Mutating the returned dict shouldn't affect subsequent gets."""
    loader = AgentLoader(agents_dir)
    first = loader.get("planner")
    first["name"] = "MUTATED"
    second = loader.get("planner")
    assert second["name"] == "Planner"


# --- search ----------------------------------------------------------------


def test_search_by_name(agents_dir):
    loader = AgentLoader(agents_dir)
    result = loader.search("reviewer")
    assert len(result) == 1
    assert result[0]["name"] == "Code Reviewer"


def test_search_by_description(agents_dir):
    loader = AgentLoader(agents_dir)
    result = loader.search("implementation")
    assert any(r["name"] == "Planner" for r in result)


def test_search_by_tag(agents_dir):
    loader = AgentLoader(agents_dir)
    result = loader.search("architecture")
    assert any(r["name"] == "Planner" for r in result)


def test_search_ranking_name_beats_description(agents_dir):
    """Name match should rank higher than description-only match."""
    loader = AgentLoader(agents_dir)
    # 'search' matches Researcher tag but 'research' is in both name and tag
    result = loader.search("research")
    assert result[0]["name"] == "Researcher"


def test_search_no_match_returns_empty(agents_dir):
    loader = AgentLoader(agents_dir)
    assert loader.search("xyzunfindable") == []


# --- malformed YAML --------------------------------------------------------


def test_malformed_yaml_skipped(tmp_path, caplog):
    import logging as _lg
    d = tmp_path / "agents"
    d.mkdir()
    (d / "broken.yaml").write_text("not-a-mapping: [just, a, list]\nother: valid")  # valid yaml but wrong shape check below
    _write_agent(d, "ok", {"name": "OK", "description": "fine", "tags": []})

    # "non-dict" YAML (e.g. just a string) triggers the "malformed" warning
    (d / "string_only.yaml").write_text('"just a string value"')

    caplog.set_level(_lg.WARNING, logger="mullet.knowledge.engine.agents")
    loader = AgentLoader(d)
    loader._load_all()
    names = {v["name"] for v in loader._cache.values()}
    # "ok" loaded, "string_only" skipped with warning
    assert "OK" in names
    assert any("malformed YAML" in r.message for r in caplog.records)
