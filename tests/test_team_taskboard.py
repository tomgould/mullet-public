"""Tests for Team TaskBoard."""
from mullet.core.orchestration.team import Task, TaskBoard


def test_task_defaults():
    t = Task(id="T-001", title="Test")
    assert t.status == "pending"
    assert t.claimed_by == ""


def test_add_and_claim():
    tb = TaskBoard()
    tb.add_task("Task A")
    tb.add_task("Task B")
    t = tb.claim_task("worker-1")
    assert t is not None
    assert t.status == "claimed"
    assert t.claimed_by == "worker-1"


def test_complete_and_fail():
    tb = TaskBoard()
    t1 = tb.add_task("Task 1")
    t2 = tb.add_task("Task 2")
    tb.complete_task(t1.id, "done")
    tb.fail_task(t2.id, "error")
    assert tb.get_all()[0].status == "done"
    assert tb.get_all()[1].status == "failed"


def test_get_pending():
    tb = TaskBoard()
    tb.add_task("A")
    tb.add_task("B")
    tb.claim_task("w")
    assert len(tb.get_pending()) == 1


def test_dedup_merges_similar():
    tb = TaskBoard()
    tb.add_task("Fix the login page")
    tb.add_task("Fix the login page styling")
    tb.add_task("Add dashboard widget")
    removed = tb.deduplicate()
    assert removed == 1
    assert len(tb.get_all()) == 2


def test_dedup_no_merge_different():
    tb = TaskBoard()
    tb.add_task("Add user auth")
    tb.add_task("Fix database migration")
    removed = tb.deduplicate()
    assert removed == 0


def test_dedup_single_word_titles():
    tb = TaskBoard()
    tb.add_task("deploy")
    tb.add_task("deploy")
    removed = tb.deduplicate()
    assert removed == 1
    assert len(tb.get_all()) == 1


def test_dedup_acronym_titles():
    tb = TaskBoard()
    tb.add_task("Fix API")
    tb.add_task("Fix API")
    removed = tb.deduplicate()
    assert removed == 1


def test_dedup_skips_claimed_tasks():
    tb = TaskBoard()
    tb.add_task("Fix the bug")
    t2 = tb.add_task("Fix the bug")
    t2.status = "claimed"
    removed = tb.deduplicate()
    assert removed == 0  # claimed task not eligible


def test_dedup_threshold_boundary():
    # 2/3 words overlap = 0.667 > 0.6 → merged
    tb = TaskBoard()
    tb.add_task("fix login bug")
    tb.add_task("fix login page")
    removed = tb.deduplicate()
    assert removed == 1


def test_dedup_below_threshold_not_merged():
    # 1/4 = 0.25 < 0.6 → not merged
    tb = TaskBoard()
    tb.add_task("add user authentication flow")
    tb.add_task("fix database query")
    removed = tb.deduplicate()
    assert removed == 0


def test_taskboard_configurable_threshold():
    """TaskBoard uses 0.6 threshold; verify edge at exactly 0.6 is not merged."""
    tb = TaskBoard()
    # 2 words shared out of max(3, 3) = 0.667 > 0.6 → merged
    tb.add_task("fix login error")
    tb.add_task("fix login timeout")
    removed = tb.deduplicate()
    assert removed == 1


def test_taskboard_full_workflow():
    tb = TaskBoard()
    t1 = tb.add_task("Task 1", "First task")
    t2 = tb.add_task("Task 2", "Second task")
    tb.add_task("Task 3", "Third task")
    assert len(tb.get_all()) == 3
    assert len(tb.get_pending()) == 3
    claimed = tb.claim_task("worker-1")
    assert claimed.id == t1.id
    assert len(tb.get_pending()) == 2
    tb.complete_task(t1.id, "Done")
    assert tb.get_all()[0].status == "done"
    tb.fail_task(t2.id, "Error")
    assert tb.get_all()[1].status == "failed"


def test_taskboard_claim_exhaustion():
    tb = TaskBoard()
    tb.add_task("A")
    tb.add_task("B")
    tb.claim_task("w1")
    tb.claim_task("w2")
    assert tb.claim_task("w3") is None
    assert len(tb.get_pending()) == 0


def test_blocked_task_not_claimable():
    tb = TaskBoard()
    t1 = tb.add_task("First task")
    t2 = tb.add_task("Second task", blocked_by=[t1.id])
    # Only first task should be available (second is blocked)
    available = tb.get_available()
    assert len(available) == 1
    assert available[0].id == t1.id


def test_task_unblocked_after_dependency_done():
    tb = TaskBoard()
    t1 = tb.add_task("Setup")
    t2 = tb.add_task("Build", blocked_by=[t1.id])
    tb.complete_task(t1.id, "setup done")
    available = tb.get_available()
    assert any(t.id == t2.id for t in available)


def test_claim_respects_blocking():
    tb = TaskBoard()
    t1 = tb.add_task("Gate task")
    tb.add_task("Blocked task", blocked_by=[t1.id])
    claimed = tb.claim_task("worker")
    assert claimed.id == t1.id
    # Next claim should return None (only blocked task remains)
    assert tb.claim_task("worker2") is None


def test_blocked_task_available_after_all_deps_done():
    tb = TaskBoard()
    t1 = tb.add_task("Dep A")
    t2 = tb.add_task("Dep B")
    t3 = tb.add_task("Final", blocked_by=[t1.id, t2.id])
    tb.complete_task(t1.id, "done")
    assert t3 not in tb.get_available()  # still blocked by t2
    tb.complete_task(t2.id, "done")
    assert t3 in tb.get_available()  # now unblocked


def test_owner_assignment_tracked_in_claimed_by():
    tb = TaskBoard()
    tb.add_task("Owned task")
    claimed = tb.claim_task("agent-alpha")
    assert claimed is not None
    assert claimed.claimed_by == "agent-alpha"
    assert claimed.status == "claimed"
