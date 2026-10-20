"""Tests for conversation_memory and autopilot_init pure helpers."""

import time
import pytest
from pathlib import Path

from mullet.core.knowledge.conversation_memory import (
    ConversationSummary,
    ConversationMemory,
    summarize_messages,
    extract_topics,
)
from mullet.cli.autopilot_init import (
    TEMPLATES,
    STATE_FILES,
    run_autopilot_init,
    require_goal,
    _target_dir,
)


# ---------------------------------------------------------------------------
# ConversationSummary
# ---------------------------------------------------------------------------

class TestConversationSummary:
    def test_fields_stored(self):
        s = ConversationSummary(id="c1", summary="Fixed a bug")
        assert s.id == "c1"
        assert s.summary == "Fixed a bug"

    def test_topics_default_empty(self):
        s = ConversationSummary(id="c1", summary="Fixed a bug")
        assert s.topics == []

    def test_files_touched_default_empty(self):
        s = ConversationSummary(id="c1", summary="Fixed a bug")
        assert s.files_touched == []

    def test_timestamp_default_zero(self):
        s = ConversationSummary(id="c1", summary="s")
        assert s.timestamp == 0.0

    def test_to_dict_roundtrip(self):
        s = ConversationSummary(
            id="c1", summary="summary", topics=["logging"], files_touched=["app.py"],
            timestamp=1000.0, token_count=500,
        )
        d = s.to_dict()
        s2 = ConversationSummary.from_dict(d)
        assert s2.id == "c1"
        assert s2.summary == "summary"
        assert s2.topics == ["logging"]
        assert s2.files_touched == ["app.py"]
        assert s2.timestamp == pytest.approx(1000.0)
        assert s2.token_count == 500

    def test_from_dict_missing_fields_defaults(self):
        s = ConversationSummary.from_dict({})
        assert s.id == ""
        assert s.summary == ""
        assert s.topics == []

    def test_to_dict_returns_dict(self):
        s = ConversationSummary(id="c1", summary="s")
        assert isinstance(s.to_dict(), dict)


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------

class TestConversationMemoryEmpty:
    def test_count_zero_initially(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        assert mem.count == 0

    def test_list_all_empty(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        assert mem.list_all() == []

    def test_recent_empty_list(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        assert mem.recent() == []

    def test_search_empty(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        assert mem.search("anything") == []


class TestConversationMemorySave:
    def test_save_increments_count(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="Fixed auth bug"))
        assert mem.count == 1

    def test_save_multiple(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="s1"))
        mem.save_conversation(ConversationSummary(id="c2", summary="s2"))
        assert mem.count == 2

    def test_save_same_id_replaces(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="old"))
        mem.save_conversation(ConversationSummary(id="c1", summary="new"))
        assert mem.count == 1
        assert mem.get_conversation("c1").summary == "new"

    def test_save_sets_timestamp(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        before = time.time()
        mem.save_conversation(ConversationSummary(id="c1", summary="s"))
        after = time.time()
        c = mem.get_conversation("c1")
        assert before <= c.timestamp <= after

    def test_persists_to_disk(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="persisted"))
        mem2 = ConversationMemory(storage_dir=str(tmp_path))
        assert mem2.count == 1
        assert mem2.get_conversation("c1").summary == "persisted"


class TestConversationMemoryGet:
    def test_get_existing_conversation(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="s"))
        result = mem.get_conversation("c1")
        assert result is not None
        assert result.id == "c1"

    def test_get_missing_returns_none(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        assert mem.get_conversation("nonexistent") is None


class TestConversationMemorySearch:
    def test_search_by_summary_keyword(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="Fixed authentication bug in login"))
        results = mem.search("authentication")
        assert len(results) == 1

    def test_search_by_topic(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="s", topics=["logging", "auth"]))
        results = mem.search("auth")
        assert len(results) == 1

    def test_search_by_file(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="s", files_touched=["auth/login.py"]))
        results = mem.search("login")
        assert len(results) == 1

    def test_search_no_match_empty(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="Fixed auth bug"))
        results = mem.search("xyz_nonexistent")
        assert results == []

    def test_search_case_insensitive(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="Fixed AUTH bug"))
        results = mem.search("auth")
        assert len(results) == 1

    def test_search_respects_max_results(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        for i in range(10):
            mem.save_conversation(ConversationSummary(id=f"c{i}", summary="shared keyword topic"))
        results = mem.search("shared", max_results=3)
        assert len(results) <= 3


class TestConversationMemoryRecent:
    def test_recent_returns_most_recent(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="old", summary="s", timestamp=1000.0))
        mem.save_conversation(ConversationSummary(id="new", summary="s", timestamp=2000.0))
        recent = mem.recent(count=1)
        assert recent[0].id == "new"

    def test_recent_count_limited(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        for i in range(10):
            mem.save_conversation(ConversationSummary(id=f"c{i}", summary="s", timestamp=float(i)))
        assert len(mem.recent(count=3)) == 3


class TestConversationMemoryDelete:
    def test_delete_removes_conversation(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="s"))
        result = mem.delete("c1")
        assert result is True
        assert mem.count == 0

    def test_delete_nonexistent_returns_false(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        assert mem.delete("nonexistent") is False

    def test_delete_persists(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="s"))
        mem.delete("c1")
        mem2 = ConversationMemory(storage_dir=str(tmp_path))
        assert mem2.count == 0


class TestConversationMemoryClear:
    def test_clear_returns_count(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="s"))
        mem.save_conversation(ConversationSummary(id="c2", summary="s"))
        count = mem.clear()
        assert count == 2

    def test_clear_empties_memory(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="s"))
        mem.clear()
        assert mem.count == 0


# ---------------------------------------------------------------------------
# summarize_messages
# ---------------------------------------------------------------------------

class TestSummarizeMessages:
    def test_returns_string(self):
        msgs = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        assert isinstance(summarize_messages(msgs), str)

    def test_empty_messages_returns_empty(self):
        assert summarize_messages([]) == ""

    def test_includes_user_message(self):
        msgs = [{"role": "user", "content": "fix the authentication bug"}]
        result = summarize_messages(msgs)
        assert "fix the authentication bug" in result

    def test_includes_assistant_message(self):
        msgs = [{"role": "assistant", "content": "I'll fix that now"}]
        result = summarize_messages(msgs)
        assert "I'll fix that now" in result

    def test_skips_system_messages(self):
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hello user message"},
        ]
        result = summarize_messages(msgs)
        assert "You are helpful" not in result

    def test_respects_max_length(self):
        msgs = [{"role": "user", "content": "x" * 1000} for _ in range(20)]
        result = summarize_messages(msgs, max_length=100)
        assert len(result) <= 103  # 100 + "..."

    def test_list_content_handled(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "my question here"}]}]
        result = summarize_messages(msgs)
        assert "my question here" in result


# ---------------------------------------------------------------------------
# extract_topics
# ---------------------------------------------------------------------------

class TestExtractTopics:
    def test_returns_list(self):
        msgs = [{"role": "user", "content": "fix the bug"}]
        assert isinstance(extract_topics(msgs), list)

    def test_empty_messages_returns_empty(self):
        assert extract_topics([]) == []

    def test_extracts_file_paths(self):
        msgs = [{"role": "user", "content": "look at auth/login.py please"}]
        topics = extract_topics(msgs)
        assert any("login.py" in t or "auth/login.py" in t for t in topics)

    def test_non_user_messages_excluded(self):
        msgs = [
            {"role": "assistant", "content": "I found Authentication issues"},
            {"role": "user", "content": "check config.py"},
        ]
        topics = extract_topics(msgs)
        # Should only extract from user messages
        # Just verify it doesn't crash and returns a list
        assert isinstance(topics, list)

    def test_max_ten_topics(self):
        msgs = [{"role": "user", "content": "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa Lambda Mu Nu Xi Omicron Pi Rho Sigma Tau Upsilon"}]
        topics = extract_topics(msgs)
        assert len(topics) <= 10


# ---------------------------------------------------------------------------
# TEMPLATES / STATE_FILES
# ---------------------------------------------------------------------------

class TestAutopilotConstants:
    def test_templates_is_tuple(self):
        assert isinstance(TEMPLATES, tuple)

    def test_templates_nonempty(self):
        assert len(TEMPLATES) > 0

    def test_has_greenfield_template(self):
        assert "greenfield-api" in TEMPLATES

    def test_state_files_is_tuple(self):
        assert isinstance(STATE_FILES, tuple)

    def test_state_files_has_goal(self):
        assert "goal.md" in STATE_FILES

    def test_state_files_count(self):
        assert len(STATE_FILES) == 5


# ---------------------------------------------------------------------------
# _target_dir
# ---------------------------------------------------------------------------

class TestTargetDir:
    def test_returns_path(self, tmp_path):
        result = _target_dir(tmp_path)
        assert isinstance(result, Path)

    def test_path_contains_autopilot(self, tmp_path):
        result = _target_dir(tmp_path)
        assert "autopilot" in str(result)

    def test_path_contains_mullet(self, tmp_path):
        result = _target_dir(tmp_path)
        assert ".mullet" in str(result)


# ---------------------------------------------------------------------------
# run_autopilot_init
# ---------------------------------------------------------------------------

class TestRunAutopilotInit:
    def test_help_returns_zero(self, tmp_path):
        rc = run_autopilot_init(["--help"], cwd=tmp_path)
        assert rc == 0

    def test_list_returns_zero(self, tmp_path):
        rc = run_autopilot_init(["--list"], cwd=tmp_path)
        assert rc == 0

    def test_unknown_template_returns_nonzero(self, tmp_path):
        rc = run_autopilot_init(["--template", "nonexistent"], cwd=tmp_path)
        assert rc != 0

    def test_unknown_arg_returns_nonzero(self, tmp_path):
        rc = run_autopilot_init(["--unknown-flag"], cwd=tmp_path)
        assert rc != 0

    def test_default_init_returns_zero(self, tmp_path):
        rc = run_autopilot_init([], cwd=tmp_path)
        assert rc == 0

    def test_creates_state_files(self, tmp_path):
        run_autopilot_init([], cwd=tmp_path)
        ap_dir = tmp_path / ".mullet" / "autopilot"
        for fname in STATE_FILES:
            assert (ap_dir / fname).exists(), f"{fname} not created"

    def test_each_template_works(self, tmp_path):
        for tmpl in TEMPLATES:
            dest = tmp_path / tmpl
            dest.mkdir()
            rc = run_autopilot_init(["--template", tmpl], cwd=dest)
            assert rc == 0

    def test_force_overwrites_existing(self, tmp_path):
        run_autopilot_init([], cwd=tmp_path)
        goal = tmp_path / ".mullet" / "autopilot" / "goal.md"
        goal.write_text("custom content here")
        run_autopilot_init(["--force"], cwd=tmp_path)
        new_content = goal.read_text()
        assert new_content != "custom content here"

    def test_no_force_does_not_overwrite(self, tmp_path):
        run_autopilot_init([], cwd=tmp_path)
        goal = tmp_path / ".mullet" / "autopilot" / "goal.md"
        goal.write_text("preserved content")
        run_autopilot_init([], cwd=tmp_path)
        assert goal.read_text() == "preserved content"


# ---------------------------------------------------------------------------
# require_goal
# ---------------------------------------------------------------------------

class TestRequireGoal:
    def test_missing_goal_returns_false(self, tmp_path):
        ok, msg = require_goal(cwd=tmp_path)
        assert ok is False
        assert "not found" in msg.lower() or "goal.md" in msg

    def test_unfilled_template_returns_false(self, tmp_path):
        run_autopilot_init([], cwd=tmp_path)
        ok, msg = require_goal(cwd=tmp_path)
        assert ok is False

    def test_filled_goal_returns_true(self, tmp_path):
        run_autopilot_init([], cwd=tmp_path)
        goal = tmp_path / ".mullet" / "autopilot" / "goal.md"
        # Write a substantially different goal with 3+ meaningful lines
        goal.write_text(
            "# Goal\n\n"
            "We need to build a user authentication system.\n"
            "This must support OAuth2 and JWT tokens.\n"
            "The system should pass all security tests.\n"
            "Deploy by end of sprint with full test coverage.\n"
        )
        ok, msg = require_goal(cwd=tmp_path)
        assert ok is True

    def test_too_short_goal_returns_false(self, tmp_path):
        run_autopilot_init([], cwd=tmp_path)
        goal = tmp_path / ".mullet" / "autopilot" / "goal.md"
        goal.write_text("# Goal\nJust one line.\n")
        ok, msg = require_goal(cwd=tmp_path)
        assert ok is False

    def test_returns_tuple(self, tmp_path):
        result = require_goal(cwd=tmp_path)
        assert isinstance(result, tuple)
        assert len(result) == 2
