"""Tests for ralphthon dataclasses/state-machine and web_converter pure helpers."""

import json

from mullet.core.orchestration.ralphthon import (
    RalphthonTask,
    RalphthonStory,
    HardeningTask,
    RalphthonConfig,
    RalphthonPRD,
    RalphthonState,
)
from mullet.knowledge.sources.web_converter import (
    clean_markdown,
    extract_title,
)


# ---------------------------------------------------------------------------
# RalphthonTask
# ---------------------------------------------------------------------------

class TestRalphthonTask:
    def test_fields_stored(self):
        t = RalphthonTask(id="t1", title="Fix bug", description="desc")
        assert t.id == "t1"
        assert t.title == "Fix bug"
        assert t.description == "desc"

    def test_defaults(self):
        t = RalphthonTask(id="t1", title="T", description="d")
        assert t.status == "pending"
        assert t.retries == 0
        assert t.notes == ""


# ---------------------------------------------------------------------------
# RalphthonStory.progress
# ---------------------------------------------------------------------------

class TestRalphthonStory:
    def test_progress_empty_tasks(self):
        s = RalphthonStory(id="s1", title="S", description="d")
        assert s.progress == 0.0

    def test_progress_all_pending(self):
        t = RalphthonTask(id="t1", title="T", description="d", status="pending")
        s = RalphthonStory(id="s1", title="S", description="d", tasks=[t])
        assert s.progress == 0.0

    def test_progress_all_done(self):
        t = RalphthonTask(id="t1", title="T", description="d", status="done")
        s = RalphthonStory(id="s1", title="S", description="d", tasks=[t])
        assert s.progress == 1.0

    def test_progress_half_done(self):
        t1 = RalphthonTask(id="t1", title="T1", description="d", status="done")
        t2 = RalphthonTask(id="t2", title="T2", description="d", status="pending")
        s = RalphthonStory(id="s1", title="S", description="d", tasks=[t1, t2])
        assert s.progress == 0.5

    def test_progress_skipped_counts_as_done(self):
        t = RalphthonTask(id="t1", title="T", description="d", status="skipped")
        s = RalphthonStory(id="s1", title="S", description="d", tasks=[t])
        assert s.progress == 1.0

    def test_defaults(self):
        s = RalphthonStory(id="s1", title="S", description="d")
        assert s.priority == "medium"
        assert s.acceptance_criteria == []
        assert s.tasks == []


# ---------------------------------------------------------------------------
# HardeningTask
# ---------------------------------------------------------------------------

class TestHardeningTask:
    def test_fields_stored(self):
        ht = HardeningTask(id="h1", title="HT", description="desc")
        assert ht.id == "h1"
        assert ht.title == "HT"

    def test_defaults(self):
        ht = HardeningTask(id="h1", title="HT", description="d")
        assert ht.category == "quality"
        assert ht.status == "pending"
        assert ht.wave == 1
        assert ht.retries == 0


# ---------------------------------------------------------------------------
# RalphthonConfig
# ---------------------------------------------------------------------------

class TestRalphthonConfig:
    def test_defaults(self):
        cfg = RalphthonConfig()
        assert cfg.max_waves == 10
        assert cfg.clean_waves_for_termination == 3
        assert cfg.max_retries == 3


# ---------------------------------------------------------------------------
# RalphthonPRD.to_dict / from_dict
# ---------------------------------------------------------------------------

class TestRalphthonPRD:
    def _make_prd(self):
        t = RalphthonTask(id="t1", title="Task", description="Do it")
        s = RalphthonStory(id="s1", title="Story", description="d", tasks=[t])
        return RalphthonPRD(project="MyProject", description="Build stuff", stories=[s])

    def test_to_dict_project(self):
        prd = self._make_prd()
        d = prd.to_dict()
        assert d["project"] == "MyProject"
        assert d["description"] == "Build stuff"

    def test_to_dict_stories_list(self):
        prd = self._make_prd()
        d = prd.to_dict()
        assert isinstance(d["stories"], list)
        assert len(d["stories"]) == 1

    def test_from_dict_round_trip_project(self):
        prd = self._make_prd()
        prd2 = RalphthonPRD.from_dict(prd.to_dict())
        assert prd2.project == "MyProject"

    def test_from_dict_round_trip_stories(self):
        prd = self._make_prd()
        prd2 = RalphthonPRD.from_dict(prd.to_dict())
        assert len(prd2.stories) == 1
        assert prd2.stories[0].id == "s1"

    def test_from_dict_round_trip_tasks(self):
        prd = self._make_prd()
        prd2 = RalphthonPRD.from_dict(prd.to_dict())
        assert len(prd2.stories[0].tasks) == 1
        assert prd2.stories[0].tasks[0].id == "t1"

    def test_from_dict_default_config(self):
        prd = RalphthonPRD.from_dict({"project": "p", "description": "d"})
        assert prd.config.max_waves == 10

    def test_to_dict_is_json_serializable(self):
        prd = self._make_prd()
        assert json.dumps(prd.to_dict())  # no exception


# ---------------------------------------------------------------------------
# RalphthonState
# ---------------------------------------------------------------------------

def _make_state_with_tasks(*statuses):
    tasks = [
        RalphthonTask(id=f"t{i}", title=f"T{i}", description="d", status=s)
        for i, s in enumerate(statuses, 1)
    ]
    story = RalphthonStory(id="s1", title="S", description="d", tasks=tasks)
    prd = RalphthonPRD(project="p", description="d", stories=[story])
    return RalphthonState(prd)


class TestRalphthonStateInit:
    def test_initial_phase(self):
        state = _make_state_with_tasks("pending")
        assert state.phase == "execution"

    def test_is_complete_false_initially(self):
        state = _make_state_with_tasks("pending")
        assert state.is_complete is False

    def test_tasks_completed_starts_zero(self):
        state = _make_state_with_tasks("pending")
        assert state.tasks_completed == 0


class TestGetNextTask:
    def test_returns_pending_task(self):
        state = _make_state_with_tasks("pending")
        task = state.get_next_task()
        assert task is not None
        assert task.id == "t1"

    def test_skips_done_tasks(self):
        state = _make_state_with_tasks("done", "pending")
        task = state.get_next_task()
        assert task.id == "t2"

    def test_all_done_transitions_to_hardening(self):
        state = _make_state_with_tasks("done")
        state.get_next_task()
        assert state.phase == "hardening"

    def test_hardening_with_no_tasks_returns_none(self):
        state = _make_state_with_tasks("done")
        result = state.get_next_task()
        assert result is None

    def test_hardening_task_returned_for_current_wave(self):
        prd = RalphthonPRD(project="p", description="d")
        prd.hardening = [HardeningTask(id="h1", title="H", description="d", wave=1)]
        state = RalphthonState(prd)
        state.phase = "hardening"
        state.current_wave = 1
        task = state.get_next_task()
        assert task is not None
        assert task.id == "h1"


class TestCompleteTask:
    def test_success_sets_done(self):
        state = _make_state_with_tasks("pending")
        state.complete_task("t1", success=True)
        assert state.prd.stories[0].tasks[0].status == "done"

    def test_success_increments_completed(self):
        state = _make_state_with_tasks("pending")
        state.complete_task("t1", success=True)
        assert state.tasks_completed == 1

    def test_failure_increments_retries(self):
        state = _make_state_with_tasks("pending")
        state.complete_task("t1", success=False, notes="error")
        assert state.prd.stories[0].tasks[0].retries == 1

    def test_failure_max_retries_skips(self):
        state = _make_state_with_tasks("pending")
        cfg = state.prd.config
        for _ in range(cfg.max_retries):
            state.complete_task("t1", success=False)
        assert state.prd.stories[0].tasks[0].status == "skipped"

    def test_unknown_task_id_noop(self):
        state = _make_state_with_tasks("pending")
        state.complete_task("nonexistent", success=True)
        assert state.tasks_completed == 0


class TestStartHardeningWave:
    def test_increments_wave(self):
        prd = RalphthonPRD(project="p", description="d")
        state = RalphthonState(prd)
        wave = state.start_hardening_wave([])
        assert wave == 1

    def test_empty_wave_increments_consecutive(self):
        prd = RalphthonPRD(project="p", description="d")
        state = RalphthonState(prd)
        state.start_hardening_wave([])
        assert state.consecutive_clean_waves == 1

    def test_nonempty_wave_resets_consecutive(self):
        prd = RalphthonPRD(project="p", description="d")
        state = RalphthonState(prd)
        state.consecutive_clean_waves = 2
        state.start_hardening_wave([HardeningTask(id="h1", title="H", description="d")])
        assert state.consecutive_clean_waves == 0

    def test_three_clean_waves_sets_complete(self):
        prd = RalphthonPRD(project="p", description="d")
        state = RalphthonState(prd)
        for _ in range(3):
            state.start_hardening_wave([])
        assert state.phase == "complete"
        assert state.is_complete is True

    def test_exceeds_max_waves_sets_complete(self):
        prd = RalphthonPRD(project="p", description="d", config=RalphthonConfig(max_waves=2))
        state = RalphthonState(prd)
        for i in range(3):
            state.start_hardening_wave([HardeningTask(id=f"h{i}", title="H", description="d")])
        assert state.phase == "complete"


class TestRalphthonStateToDict:
    def test_to_dict_has_phase(self):
        prd = RalphthonPRD(project="p", description="d")
        state = RalphthonState(prd)
        d = state.to_dict()
        assert "phase" in d
        assert d["phase"] == "execution"

    def test_to_dict_has_prd(self):
        prd = RalphthonPRD(project="proj", description="d")
        state = RalphthonState(prd)
        d = state.to_dict()
        assert d["prd"]["project"] == "proj"

    def test_to_dict_is_json_serializable(self):
        prd = RalphthonPRD(project="p", description="d")
        state = RalphthonState(prd)
        assert json.dumps(state.to_dict())


class TestRalphthonStateSaveLoad:
    def test_save_and_load_round_trip(self, tmp_path):
        t = RalphthonTask(id="t1", title="T", description="d", status="done")
        s = RalphthonStory(id="s1", title="S", description="d", tasks=[t])
        prd = RalphthonPRD(project="proj", description="desc", stories=[s])
        state = RalphthonState(prd)
        state.tasks_completed = 1

        path = str(tmp_path / "state.json")
        state.save(path)

        loaded = RalphthonState.load(path)
        assert loaded.phase == "execution"
        assert loaded.tasks_completed == 1
        assert loaded.prd.project == "proj"


# ---------------------------------------------------------------------------
# clean_markdown
# ---------------------------------------------------------------------------

class TestCleanMarkdown:
    def test_empty_string_returns_none(self):
        assert clean_markdown("") is None

    def test_whitespace_only_returns_none(self):
        assert clean_markdown("   \n\n  ") is None

    def test_plain_text_returned(self):
        assert clean_markdown("Hello world") == "Hello world"

    def test_collapses_excessive_blank_lines(self):
        md = "A\n\n\n\n\nB"
        result = clean_markdown(md)
        assert "\n\n\n\n" not in result
        assert "A" in result and "B" in result

    def test_removes_advertisement_line(self):
        md = "Before\n\n   Advertisement\n\nAfter"
        result = clean_markdown(md)
        assert "Advertisement" not in result

    def test_removes_most_read_section(self):
        md = "# Article\n\nContent here\n\n## Most Read\n\nJunk 1\nJunk 2\n"
        result = clean_markdown(md)
        assert "Most Read" not in result

    def test_removes_subscribe_line(self):
        md = "Text\n\nSubscribe\n\nMore text"
        result = clean_markdown(md)
        assert "Subscribe" not in result.split()

    def test_removes_empty_links(self):
        md = "Hello [](http://example.com) world"
        result = clean_markdown(md)
        assert "[](http://example.com)" not in result

    def test_returns_stripped_text(self):
        md = "   Hello world   "
        result = clean_markdown(md)
        assert result == "Hello world"

    def test_real_content_preserved(self):
        md = "# Title\n\nThis is real content that should be kept."
        result = clean_markdown(md)
        assert "real content" in result


# ---------------------------------------------------------------------------
# extract_title
# ---------------------------------------------------------------------------

class TestExtractTitle:
    def test_h1_extracted(self):
        html = "<html><body><h1>My Title</h1><p>content</p></body></html>"
        assert extract_title(html) == "My Title"

    def test_title_tag_fallback(self):
        html = "<html><head><title>Page Title</title></head><body><p>text</p></body></html>"
        assert extract_title(html) == "Page Title"

    def test_h1_wins_over_title_tag(self):
        html = "<html><head><title>Page Title</title></head><body><h1>H1 Title</h1></body></html>"
        assert extract_title(html) == "H1 Title"

    def test_no_title_returns_empty(self):
        html = "<html><body><p>no title here</p></body></html>"
        assert extract_title(html) == ""

    def test_empty_html_returns_empty(self):
        assert extract_title("") == ""

    def test_strips_whitespace(self):
        html = "<html><body><h1>  Padded  </h1></body></html>"
        assert extract_title(html) == "Padded"
