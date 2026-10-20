"""Tests for skill invocation and outcome tracking."""

import pytest
from unittest.mock import patch


@pytest.fixture
def tmp_stats(tmp_path):
    """Redirect skill stats file to a temp path."""
    stats_file = tmp_path / "skill_stats.json"
    with patch("mullet.core.skills.stats._STATS_PATH", stats_file):
        yield stats_file


def test_record_invocation_creates_entry(tmp_stats):
    from mullet.core.skills.stats import record_invocation, get_stats
    with patch("mullet.core.skills.stats._STATS_PATH", tmp_stats):
        record_invocation("my-skill")
        s = get_stats("my-skill")
    assert s["invocations"] == 1
    assert s["last_invoked"] is not None


def test_record_invocation_increments(tmp_stats):
    from mullet.core.skills.stats import record_invocation, get_stats
    with patch("mullet.core.skills.stats._STATS_PATH", tmp_stats):
        record_invocation("skill-a")
        record_invocation("skill-a")
        record_invocation("skill-a")
        s = get_stats("skill-a")
    assert s["invocations"] == 3


def test_record_outcome_success(tmp_stats):
    from mullet.core.skills.stats import record_invocation, record_outcome, get_stats
    with patch("mullet.core.skills.stats._STATS_PATH", tmp_stats):
        record_invocation("skill-b")
        record_outcome("skill-b", success=True)
        s = get_stats("skill-b")
    assert s["successes"] == 1
    assert s["failures"] == 0
    assert s["last_outcome"] == "success"


def test_record_outcome_failure(tmp_stats):
    from mullet.core.skills.stats import record_invocation, record_outcome, get_stats
    with patch("mullet.core.skills.stats._STATS_PATH", tmp_stats):
        record_invocation("skill-c")
        record_outcome("skill-c", success=False)
        s = get_stats("skill-c")
    assert s["failures"] == 1
    assert s["last_outcome"] == "failure"


def test_success_rate_none_when_no_outcomes(tmp_stats):
    from mullet.core.skills.stats import record_invocation, success_rate
    with patch("mullet.core.skills.stats._STATS_PATH", tmp_stats):
        record_invocation("skill-d")
        rate = success_rate("skill-d")
    assert rate is None


def test_success_rate_computed(tmp_stats):
    from mullet.core.skills.stats import record_invocation, record_outcome, success_rate
    with patch("mullet.core.skills.stats._STATS_PATH", tmp_stats):
        record_invocation("skill-e")
        record_outcome("skill-e", success=True)
        record_invocation("skill-e")
        record_outcome("skill-e", success=True)
        record_invocation("skill-e")
        record_outcome("skill-e", success=False)
        rate = success_rate("skill-e")
    assert abs(rate - 2/3) < 0.001


def test_top_skills_sorted_by_invocations(tmp_stats):
    from mullet.core.skills.stats import record_invocation, top_skills
    with patch("mullet.core.skills.stats._STATS_PATH", tmp_stats):
        for _ in range(5):
            record_invocation("popular")
        for _ in range(2):
            record_invocation("rare")
        rows = top_skills(10)
    names = [r["name"] for r in rows]
    assert names[0] == "popular"
    assert names[1] == "rare"


def test_get_stats_missing_skill_returns_empty(tmp_stats):
    from mullet.core.skills.stats import get_stats
    with patch("mullet.core.skills.stats._STATS_PATH", tmp_stats):
        s = get_stats("nonexistent")
    assert s == {}
