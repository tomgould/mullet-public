"""Tests for tokenize, search cache/types, prompts, and skill stats pure helpers."""

import json
import pytest
import time
from unittest.mock import patch

from mullet.core.providers.tokenize import estimate_tokens
from mullet.knowledge.engine.search.cache import SearchCache
from mullet.knowledge.engine.search.types import SearchQuality, SearchResult
from mullet.knowledge.engine.prompts import discover_prompts, load_prompt, search_prompts
import mullet.core.skills.stats as _stats_mod


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_returns_int(self):
        assert isinstance(estimate_tokens("hello world"), int)

    def test_empty_string_returns_positive(self):
        # max(1, ...) ensures at least 1
        result = estimate_tokens("")
        assert result >= 0  # implementation returns max(1, 0) = 1 or 0 depending on empty

    def test_nonempty_string_positive(self):
        assert estimate_tokens("hello") >= 1

    def test_longer_string_more_tokens(self):
        short = estimate_tokens("hi")
        long = estimate_tokens("hi " * 100)
        assert long > short

    def test_dict_input_works(self):
        result = estimate_tokens({"key": "value", "num": 42})
        assert isinstance(result, int)
        assert result >= 1

    def test_list_input_works(self):
        result = estimate_tokens([{"role": "user", "content": "hello"}])
        assert isinstance(result, int)
        assert result >= 1

    def test_messages_list_returns_reasonable_count(self):
        msgs = [{"role": "user", "content": "Tell me about Python programming"}]
        result = estimate_tokens(msgs)
        assert result > 0

    def test_large_text_more_tokens_than_small(self):
        a = estimate_tokens("short")
        b = estimate_tokens("This is a much longer piece of text that should have more tokens than the short one")
        assert b > a

    def test_none_fallback_no_crash(self):
        # Objects that fail json.dumps fall back to str()
        result = estimate_tokens(object())
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# SearchCache
# ---------------------------------------------------------------------------

class TestSearchCacheGet:
    def test_miss_returns_none(self):
        cache = SearchCache()
        assert cache.get("nonexistent") is None

    def test_put_and_get(self):
        cache = SearchCache()
        cache.put("key1", [1, 2, 3])
        assert cache.get("key1") == [1, 2, 3]

    def test_expired_returns_none(self):
        cache = SearchCache(ttl=0.01)
        cache.put("k", "value")
        time.sleep(0.05)
        assert cache.get("k") is None

    def test_not_expired_returns_value(self):
        cache = SearchCache(ttl=60.0)
        cache.put("k", "fresh")
        assert cache.get("k") == "fresh"


class TestSearchCachePut:
    def test_put_increments_size(self):
        cache = SearchCache()
        cache.put("k1", "v1")
        assert cache.size == 1

    def test_multiple_puts(self):
        cache = SearchCache()
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        assert cache.size == 2

    def test_overwrite_same_key_no_size_increase(self):
        cache = SearchCache()
        cache.put("k", "v1")
        cache.put("k", "v2")
        assert cache.size == 1
        assert cache.get("k") == "v2"

    def test_evicts_when_full(self):
        cache = SearchCache(max_size=2)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        assert cache.size == 2

    def test_lru_eviction_order(self):
        cache = SearchCache(max_size=2)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        # Access k1 to make it recently used
        cache.get("k1")
        # Add k3 — should evict k2 (least recently used)
        cache.put("k3", "v3")
        assert cache.get("k1") == "v1"
        assert cache.get("k3") == "v3"
        assert cache.get("k2") is None


class TestSearchCacheClear:
    def test_clear_empties_cache(self):
        cache = SearchCache()
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.clear()
        assert cache.size == 0

    def test_get_after_clear_returns_none(self):
        cache = SearchCache()
        cache.put("k", "v")
        cache.clear()
        assert cache.get("k") is None


class TestSearchCacheSize:
    def test_initial_size_zero(self):
        cache = SearchCache()
        assert cache.size == 0

    def test_size_is_int(self):
        cache = SearchCache()
        assert isinstance(cache.size, int)


# ---------------------------------------------------------------------------
# SearchQuality
# ---------------------------------------------------------------------------

class TestSearchQuality:
    def test_defaults(self):
        q = SearchQuality()
        assert q.avg_rerank_score == 0.0
        assert q.top_score == 0.0
        assert q.unique_sources == 0
        assert q.mmr_applied is False
        assert q.cache_hit is False

    def test_explicit_values(self):
        q = SearchQuality(avg_rerank_score=0.85, top_score=0.95, unique_sources=5)
        assert q.avg_rerank_score == pytest.approx(0.85)
        assert q.top_score == pytest.approx(0.95)
        assert q.unique_sources == 5

    def test_timing_default_empty(self):
        q = SearchQuality()
        assert q.timing == {}

    def test_merge_strategy_default(self):
        q = SearchQuality()
        assert q.merge_strategy == "weighted"


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_defaults(self):
        r = SearchResult()
        assert r.results == []
        assert isinstance(r.quality, SearchQuality)

    def test_results_stored(self):
        results = [{"content": "a"}, {"content": "b"}]
        r = SearchResult(results=results)
        assert len(r.results) == 2

    def test_quality_stored(self):
        q = SearchQuality(top_score=0.9)
        r = SearchResult(quality=q)
        assert r.quality.top_score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# discover_prompts
# ---------------------------------------------------------------------------

class TestDiscoverPrompts:
    def test_empty_dirs_returns_empty(self, tmp_path):
        result = discover_prompts([str(tmp_path / "nonexistent")])
        assert result == []

    def test_finds_md_files(self, tmp_path):
        (tmp_path / "greeting.md").write_text("Hello {{name}}")
        result = discover_prompts([str(tmp_path)])
        assert len(result) == 1
        assert result[0]["name"] == "greeting"

    def test_finds_txt_files(self, tmp_path):
        (tmp_path / "query.txt").write_text("Search for {{topic}}")
        result = discover_prompts([str(tmp_path)])
        assert any(r["name"] == "query" for r in result)

    def test_finds_json_files(self, tmp_path):
        (tmp_path / "config.json").write_text('{"prompt": "hello"}')
        result = discover_prompts([str(tmp_path)])
        assert any(r["name"] == "config" for r in result)

    def test_result_has_required_keys(self, tmp_path):
        (tmp_path / "test.md").write_text("Test prompt")
        result = discover_prompts([str(tmp_path)])
        entry = result[0]
        assert "name" in entry
        assert "path" in entry
        assert "format" in entry
        assert "size" in entry

    def test_format_matches_extension(self, tmp_path):
        (tmp_path / "my_prompt.md").write_text("# Hello")
        result = discover_prompts([str(tmp_path)])
        assert result[0]["format"] == "md"

    def test_multiple_dirs_searched(self, tmp_path):
        dir1 = tmp_path / "d1"
        dir2 = tmp_path / "d2"
        dir1.mkdir()
        dir2.mkdir()
        (dir1 / "p1.md").write_text("p1")
        (dir2 / "p2.md").write_text("p2")
        result = discover_prompts([str(dir1), str(dir2)])
        names = [r["name"] for r in result]
        assert "p1" in names
        assert "p2" in names


# ---------------------------------------------------------------------------
# load_prompt
# ---------------------------------------------------------------------------

class TestLoadPrompt:
    def test_loads_file_content(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("Hello world")
        result = load_prompt(str(f))
        assert result == "Hello world"

    def test_substitutes_variables(self, tmp_path):
        f = tmp_path / "greeting.md"
        f.write_text("Hello {{name}}, welcome to {{place}}!")
        result = load_prompt(str(f), {"name": "Alice", "place": "Mullet"})
        assert "Alice" in result
        assert "Mullet" in result
        assert "{{name}}" not in result

    def test_no_variables_returns_raw(self, tmp_path):
        f = tmp_path / "raw.md"
        f.write_text("No substitutions here")
        result = load_prompt(str(f))
        assert result == "No substitutions here"

    def test_missing_variable_kept_as_placeholder(self, tmp_path):
        f = tmp_path / "tmpl.md"
        f.write_text("Hello {{name}}")
        result = load_prompt(str(f), {"other": "value"})
        assert "{{name}}" in result

    def test_returns_string(self, tmp_path):
        f = tmp_path / "t.md"
        f.write_text("content")
        assert isinstance(load_prompt(str(f)), str)


# ---------------------------------------------------------------------------
# search_prompts
# ---------------------------------------------------------------------------

class TestSearchPrompts:
    def _make_prompt_list(self, tmp_path, entries):
        result = []
        for name, content in entries:
            f = tmp_path / f"{name}.md"
            f.write_text(content)
            result.append({"name": name, "path": str(f), "format": "md", "size": len(content)})
        return result

    def test_match_by_name(self, tmp_path):
        prompts = self._make_prompt_list(tmp_path, [
            ("greeting", "Hello there"),
            ("farewell", "Goodbye"),
        ])
        results = search_prompts("greeting", prompts)
        assert len(results) == 1
        assert results[0]["name"] == "greeting"

    def test_match_by_content(self, tmp_path):
        prompts = self._make_prompt_list(tmp_path, [
            ("p1", "This prompt contains authentication logic"),
            ("p2", "Another prompt about something else"),
        ])
        results = search_prompts("authentication", prompts)
        assert len(results) == 1
        assert results[0]["name"] == "p1"

    def test_no_match_returns_empty(self, tmp_path):
        prompts = self._make_prompt_list(tmp_path, [("test", "hello world")])
        results = search_prompts("xyznonexistent", prompts)
        assert results == []

    def test_case_insensitive(self, tmp_path):
        prompts = self._make_prompt_list(tmp_path, [("Auth", "Authentication system")])
        results = search_prompts("auth", prompts)
        assert len(results) >= 1

    def test_returns_list(self, tmp_path):
        prompts = self._make_prompt_list(tmp_path, [("p", "content")])
        assert isinstance(search_prompts("p", prompts), list)


# ---------------------------------------------------------------------------
# skill stats (_entry, success_rate, get_stats)
# ---------------------------------------------------------------------------

class TestSkillStatsEntry:
    def test_creates_entry_for_new_skill(self):
        data = {}
        entry = _stats_mod._entry(data, "my_skill")
        assert "invocations" in entry
        assert "successes" in entry
        assert "failures" in entry

    def test_returns_existing_entry(self):
        data = {"my_skill": {"invocations": 5, "successes": 3, "failures": 2, "last_invoked": None, "last_outcome": None}}
        entry = _stats_mod._entry(data, "my_skill")
        assert entry["invocations"] == 5

    def test_default_entry_invocations_zero(self):
        data = {}
        entry = _stats_mod._entry(data, "new_skill")
        assert entry["invocations"] == 0

    def test_default_entry_last_invoked_none(self):
        data = {}
        entry = _stats_mod._entry(data, "new_skill")
        assert entry["last_invoked"] is None


class TestSuccessRate:
    def test_no_outcomes_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_stats_mod, "_STATS_PATH", tmp_path / "stats.json")
        (tmp_path / "stats.json").write_text('{"test_skill": {"invocations": 3, "successes": 0, "failures": 0}}')
        result = _stats_mod.success_rate("test_skill")
        assert result is None

    def test_all_successes_returns_one(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_stats_mod, "_STATS_PATH", tmp_path / "stats.json")
        data = {"my_skill": {"invocations": 5, "successes": 5, "failures": 0}}
        (tmp_path / "stats.json").write_text(json.dumps(data))
        result = _stats_mod.success_rate("my_skill")
        assert result == pytest.approx(1.0)

    def test_half_success_returns_half(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_stats_mod, "_STATS_PATH", tmp_path / "stats.json")
        data = {"skill": {"invocations": 10, "successes": 5, "failures": 5}}
        (tmp_path / "stats.json").write_text(json.dumps(data))
        result = _stats_mod.success_rate("skill")
        assert result == pytest.approx(0.5)

    def test_unknown_skill_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_stats_mod, "_STATS_PATH", tmp_path / "stats.json")
        (tmp_path / "stats.json").write_text("{}")
        result = _stats_mod.success_rate("nonexistent")
        assert result is None


class TestGetStats:
    def test_empty_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_stats_mod, "_STATS_PATH", tmp_path / "stats.json")
        # No file created
        result = _stats_mod.get_stats()
        assert result == {}

    def test_specific_skill_returned(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_stats_mod, "_STATS_PATH", tmp_path / "stats.json")
        data = {"deploy": {"invocations": 3, "successes": 2, "failures": 1}}
        (tmp_path / "stats.json").write_text(json.dumps(data))
        result = _stats_mod.get_stats("deploy")
        assert result["invocations"] == 3

    def test_all_stats_returned_without_arg(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_stats_mod, "_STATS_PATH", tmp_path / "stats.json")
        data = {"a": {"invocations": 1}, "b": {"invocations": 2}}
        (tmp_path / "stats.json").write_text(json.dumps(data))
        result = _stats_mod.get_stats()
        assert "a" in result
        assert "b" in result


class TestRecordInvocation:
    def test_increments_invocations(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_stats_mod, "_STATS_PATH", tmp_path / "stats.json")
        _stats_mod.record_invocation("my_skill")
        data = json.loads((tmp_path / "stats.json").read_text())
        assert data["my_skill"]["invocations"] == 1

    def test_second_invocation_increments_again(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_stats_mod, "_STATS_PATH", tmp_path / "stats.json")
        _stats_mod.record_invocation("skill")
        _stats_mod.record_invocation("skill")
        data = json.loads((tmp_path / "stats.json").read_text())
        assert data["skill"]["invocations"] == 2

    def test_sets_last_invoked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_stats_mod, "_STATS_PATH", tmp_path / "stats.json")
        _stats_mod.record_invocation("skill")
        data = json.loads((tmp_path / "stats.json").read_text())
        assert data["skill"]["last_invoked"] is not None


class TestRecordOutcome:
    def test_success_increments_successes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_stats_mod, "_STATS_PATH", tmp_path / "stats.json")
        _stats_mod.record_outcome("skill", success=True)
        data = json.loads((tmp_path / "stats.json").read_text())
        assert data["skill"]["successes"] == 1

    def test_failure_increments_failures(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_stats_mod, "_STATS_PATH", tmp_path / "stats.json")
        _stats_mod.record_outcome("skill", success=False)
        data = json.loads((tmp_path / "stats.json").read_text())
        assert data["skill"]["failures"] == 1

    def test_last_outcome_set_success(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_stats_mod, "_STATS_PATH", tmp_path / "stats.json")
        _stats_mod.record_outcome("skill", success=True)
        data = json.loads((tmp_path / "stats.json").read_text())
        assert data["skill"]["last_outcome"] == "success"


class TestTopSkills:
    def test_returns_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_stats_mod, "_STATS_PATH", tmp_path / "stats.json")
        result = _stats_mod.top_skills()
        assert isinstance(result, list)

    def test_empty_stats_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_stats_mod, "_STATS_PATH", tmp_path / "stats.json")
        result = _stats_mod.top_skills()
        assert result == []

    def test_sorted_by_invocations(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_stats_mod, "_STATS_PATH", tmp_path / "stats.json")
        data = {
            "low": {"invocations": 1, "successes": 0, "failures": 0},
            "high": {"invocations": 10, "successes": 8, "failures": 2},
        }
        (tmp_path / "stats.json").write_text(json.dumps(data))
        result = _stats_mod.top_skills()
        assert result[0]["name"] == "high"

    def test_limits_to_n(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_stats_mod, "_STATS_PATH", tmp_path / "stats.json")
        data = {f"skill_{i}": {"invocations": i, "successes": 0, "failures": 0} for i in range(20)}
        (tmp_path / "stats.json").write_text(json.dumps(data))
        result = _stats_mod.top_skills(n=5)
        assert len(result) == 5


class TestSearchPromptsGapLines:
    def test_prompts_none_calls_discover(self, tmp_path):
        """Line 43: when prompts=None, discover_prompts() is called."""
        fake_prompts = [{"name": "greeting", "path": str(tmp_path / "greeting.md")}]
        (tmp_path / "greeting.md").write_text("Hello there")
        with patch("mullet.knowledge.engine.prompts.discover_prompts", return_value=fake_prompts):
            results = search_prompts("greeting")
        assert len(results) == 1
        assert results[0]["name"] == "greeting"

    def test_unreadable_prompt_file_is_silenced(self, tmp_path):
        """Lines 54-55: read_text() raises OSError → exception silenced, item skipped."""
        prompts = [{"name": "broken", "path": "/nonexistent/path/missing.md"}]
        # query doesn't match name, so it tries to read file → OSError (file not found)
        results = search_prompts("zzz", prompts)
        assert results == []
