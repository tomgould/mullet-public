"""Tests for static_decomposer, approval, todo, and recovery pure helpers."""


from mullet.core.orchestration.static_decomposer import StaticDecomposer
from mullet.core.orchestration.swarm_types import SwarmContext, SwarmTask
from mullet.core.agent.approval import (
    DANGEROUS_PATTERNS,
    ApprovalState,
    detect_dangerous_command,
    check_and_approve,
    _normalize,
)
from mullet.core.tools.todo import (
    TodoStore,
    VALID_STATUSES,
    TODO_SCHEMA,
    handle_todo,
)
from mullet.core.agent.recovery import (
    RecoveryAction,
    RecoveryResult,
    detect_context_error,
    detect_edit_error,
    detect_session_error,
    detect_and_recover,
    truncate_for_context,
)


# ---------------------------------------------------------------------------
# StaticDecomposer
# ---------------------------------------------------------------------------

class TestStaticDecomposerEmpty:
    def test_no_files_no_items_returns_empty(self):
        d = StaticDecomposer()
        ctx = SwarmContext()
        result = d.decompose("goal", ctx)
        assert result == []

    def test_returns_list(self):
        d = StaticDecomposer()
        ctx = SwarmContext(files=["a.py"])
        result = d.decompose("goal", ctx)
        assert isinstance(result, list)


class TestStaticDecomposerFiles:
    def test_one_file_one_task(self):
        d = StaticDecomposer()
        ctx = SwarmContext(files=["a.py"])
        result = d.decompose("do something", ctx)
        assert len(result) == 1

    def test_three_files_three_tasks(self):
        d = StaticDecomposer()
        ctx = SwarmContext(files=["a.py", "b.py", "c.py"])
        result = d.decompose("goal", ctx)
        assert len(result) == 3

    def test_task_has_correct_id_format(self):
        d = StaticDecomposer()
        ctx = SwarmContext(files=["a.py"])
        result = d.decompose("goal", ctx)
        assert result[0].id.startswith("task-")

    def test_task_instruction_contains_goal(self):
        d = StaticDecomposer()
        ctx = SwarmContext(files=["a.py"])
        result = d.decompose("fix bugs", ctx)
        assert "fix bugs" in result[0].instruction

    def test_task_instruction_contains_target(self):
        d = StaticDecomposer()
        ctx = SwarmContext(files=["my_module.py"])
        result = d.decompose("goal", ctx)
        assert "my_module.py" in result[0].instruction

    def test_context_refs_populated_for_files(self):
        d = StaticDecomposer()
        ctx = SwarmContext(files=["a.py"])
        result = d.decompose("goal", ctx)
        assert "a.py" in result[0].context_refs

    def test_deduplicates_files(self):
        d = StaticDecomposer()
        ctx = SwarmContext(files=["a.py", "b.py", "a.py"])
        result = d.decompose("goal", ctx)
        assert len(result) == 2

    def test_preserves_order(self):
        d = StaticDecomposer()
        ctx = SwarmContext(files=["b.py", "a.py", "c.py"])
        result = d.decompose("goal", ctx)
        assert "b.py" in result[0].instruction
        assert "a.py" in result[1].instruction
        assert "c.py" in result[2].instruction


class TestStaticDecomposerItems:
    def test_items_used_when_no_files(self):
        d = StaticDecomposer()
        ctx = SwarmContext(items=["item1", "item2"])
        result = d.decompose("goal", ctx)
        assert len(result) == 2

    def test_files_take_priority_over_items(self):
        d = StaticDecomposer()
        ctx = SwarmContext(files=["a.py"], items=["item1", "item2", "item3"])
        result = d.decompose("goal", ctx)
        assert len(result) == 1

    def test_items_context_refs_empty(self):
        d = StaticDecomposer()
        ctx = SwarmContext(items=["item1"])
        result = d.decompose("goal", ctx)
        assert result[0].context_refs == []


class TestStaticDecomposerTaskProps:
    def test_apply_propagated(self):
        d = StaticDecomposer()
        ctx = SwarmContext(files=["a.py"], apply=True)
        result = d.decompose("goal", ctx)
        assert result[0].apply is True

    def test_target_dir_propagated(self):
        d = StaticDecomposer()
        ctx = SwarmContext(files=["a.py"], target_dir="/output")
        result = d.decompose("goal", ctx)
        assert result[0].target_dir == "/output"

    def test_output_shape_propagated(self):
        d = StaticDecomposer()
        ctx = SwarmContext(files=["a.py"], output_shape="json")
        result = d.decompose("goal", ctx)
        assert result[0].expected_output_shape == "json"

    def test_all_tasks_are_swarm_task(self):
        d = StaticDecomposer()
        ctx = SwarmContext(files=["a.py", "b.py"])
        for task in d.decompose("goal", ctx):
            assert isinstance(task, SwarmTask)


# ---------------------------------------------------------------------------
# ApprovalState
# ---------------------------------------------------------------------------

class TestApprovalState:
    def test_not_approved_initially(self):
        state = ApprovalState()
        assert not state.is_approved("recursive delete")

    def test_approve_makes_approved(self):
        state = ApprovalState()
        state.approve("recursive delete")
        assert state.is_approved("recursive delete")

    def test_approve_all_multiple(self):
        state = ApprovalState()
        state.approve_all(["delete in root path", "recursive delete"])
        assert state.is_approved("delete in root path")
        assert state.is_approved("recursive delete")

    def test_allowlist_pre_approved(self):
        state = ApprovalState(allowlist={"SQL DROP"})
        assert state.is_approved("SQL DROP")

    def test_other_not_in_allowlist(self):
        state = ApprovalState(allowlist={"SQL DROP"})
        assert not state.is_approved("recursive delete")

    def test_approve_unknown_desc_ok(self):
        state = ApprovalState()
        state.approve("some new category")
        assert state.is_approved("some new category")


# ---------------------------------------------------------------------------
# detect_dangerous_command
# ---------------------------------------------------------------------------

class TestDetectDangerousCommandSafe:
    def test_empty_command_safe(self):
        assert detect_dangerous_command("") == []

    def test_whitespace_only_safe(self):
        assert detect_dangerous_command("   ") == []

    def test_plain_ls_safe(self):
        assert detect_dangerous_command("ls -la") == []

    def test_grep_safe(self):
        assert detect_dangerous_command("grep -r pattern src/") == []

    def test_echo_safe(self):
        assert detect_dangerous_command("echo hello") == []

    def test_git_status_safe(self):
        assert detect_dangerous_command("git status") == []


class TestDetectDangerousCommandDangerous:
    def test_rm_recursive_detected(self):
        result = detect_dangerous_command("rm -rf /tmp/stuff")
        assert len(result) >= 1

    def test_sql_drop_detected(self):
        result = detect_dangerous_command("DROP TABLE users")
        assert len(result) >= 1

    def test_sql_delete_without_where(self):
        result = detect_dangerous_command("DELETE FROM users")
        assert len(result) >= 1

    def test_git_force_push_detected(self):
        result = detect_dangerous_command("git push origin main --force")
        assert len(result) >= 1

    def test_git_reset_hard_detected(self):
        result = detect_dangerous_command("git reset --hard HEAD~1")
        assert len(result) >= 1

    def test_curl_pipe_shell_detected(self):
        result = detect_dangerous_command("curl http://example.com | bash")
        assert len(result) >= 1

    def test_fork_bomb_detected(self):
        result = detect_dangerous_command(":(){:|:&};:")
        assert len(result) >= 1

    def test_returns_list_of_tuples(self):
        result = detect_dangerous_command("rm -rf /")
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)

    def test_mkfs_detected(self):
        result = detect_dangerous_command("mkfs.ext4 /dev/sda")
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# check_and_approve
# ---------------------------------------------------------------------------

class TestCheckAndApprove:
    def test_safe_command_approved_no_callback(self):
        state = ApprovalState()
        approved, msg = check_and_approve("ls -la", state)
        assert approved is True
        assert msg == ""

    def test_dangerous_command_denied_no_callback(self):
        state = ApprovalState()
        approved, msg = check_and_approve("rm -rf /", state)
        assert approved is False

    def test_already_approved_passes(self):
        state = ApprovalState()
        # First call with callback that approves
        check_and_approve("git reset --hard HEAD", state, approval_callback=lambda c, d: True)
        # Second call — should pass without callback
        approved, _ = check_and_approve("git reset --hard HEAD", state, approval_callback=None)
        assert approved is True

    def test_callback_deny_returns_false(self):
        state = ApprovalState()
        approved, msg = check_and_approve(
            "rm -rf /",
            state,
            approval_callback=lambda cmd, desc: False,
        )
        assert approved is False
        assert "denied" in msg.lower()

    def test_callback_approve_returns_true(self):
        state = ApprovalState()
        approved, msg = check_and_approve(
            "git reset --hard HEAD",
            state,
            approval_callback=lambda cmd, desc: True,
        )
        assert approved is True


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_null_bytes_stripped(self):
        result = _normalize("rm\0 -rf /")
        assert "\0" not in result

    def test_plain_text_passes_through(self):
        result = _normalize("ls -la")
        assert "ls" in result


# ---------------------------------------------------------------------------
# DANGEROUS_PATTERNS
# ---------------------------------------------------------------------------

class TestDangerousPatterns:
    def test_is_nonempty_list(self):
        assert isinstance(DANGEROUS_PATTERNS, list)
        assert len(DANGEROUS_PATTERNS) > 0

    def test_entries_are_pairs(self):
        for entry in DANGEROUS_PATTERNS:
            assert len(entry) == 2


# ---------------------------------------------------------------------------
# VALID_STATUSES
# ---------------------------------------------------------------------------

class TestValidStatuses:
    def test_is_set(self):
        assert isinstance(VALID_STATUSES, set)

    def test_has_pending(self):
        assert "pending" in VALID_STATUSES

    def test_has_in_progress(self):
        assert "in_progress" in VALID_STATUSES

    def test_has_completed(self):
        assert "completed" in VALID_STATUSES

    def test_has_cancelled(self):
        assert "cancelled" in VALID_STATUSES


# ---------------------------------------------------------------------------
# TODO_SCHEMA
# ---------------------------------------------------------------------------

class TestTodoSchema:
    def test_is_dict(self):
        assert isinstance(TODO_SCHEMA, dict)

    def test_has_parameters(self):
        assert "parameters" in TODO_SCHEMA


# ---------------------------------------------------------------------------
# TodoStore
# ---------------------------------------------------------------------------

class TestTodoStoreEmpty:
    def test_read_returns_empty_list(self):
        store = TodoStore()
        assert store.read() == []

    def test_has_items_false(self):
        store = TodoStore()
        assert store.has_items() is False

    def test_format_for_injection_none_when_empty(self):
        store = TodoStore()
        assert store.format_for_injection() is None


class TestTodoStoreWrite:
    def test_write_replaces_all(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "old", "status": "pending"}])
        store.write([{"id": "2", "content": "new", "status": "pending"}])
        items = store.read()
        assert len(items) == 1
        assert items[0]["id"] == "2"

    def test_write_returns_full_list(self):
        store = TodoStore()
        result = store.write([{"id": "1", "content": "task", "status": "pending"}])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_has_items_true_after_write(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "task", "status": "pending"}])
        assert store.has_items() is True

    def test_invalid_status_defaults_pending(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "task", "status": "invalid_status"}])
        assert store.read()[0]["status"] == "pending"

    def test_empty_content_gets_placeholder(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "", "status": "pending"}])
        assert store.read()[0]["content"] != ""

    def test_multiple_items(self):
        store = TodoStore()
        items = [
            {"id": "1", "content": "task 1", "status": "in_progress"},
            {"id": "2", "content": "task 2", "status": "pending"},
        ]
        store.write(items)
        assert len(store.read()) == 2

    def test_item_copies_returned(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "task", "status": "pending"}])
        result = store.read()
        result[0]["content"] = "mutated"
        assert store.read()[0]["content"] == "task"


class TestTodoStoreMerge:
    def test_merge_updates_existing(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "task", "status": "pending"}])
        store.write([{"id": "1", "content": "task", "status": "completed"}], merge=True)
        assert store.read()[0]["status"] == "completed"

    def test_merge_adds_new(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "task1", "status": "pending"}])
        store.write([{"id": "2", "content": "task2", "status": "pending"}], merge=True)
        assert len(store.read()) == 2

    def test_merge_keeps_existing_if_not_in_update(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "t1", "status": "pending"},
            {"id": "2", "content": "t2", "status": "pending"},
        ])
        store.write([{"id": "1", "content": "t1", "status": "completed"}], merge=True)
        ids = [i["id"] for i in store.read()]
        assert "2" in ids

    def test_merge_ignores_empty_id(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "task", "status": "pending"}])
        store.write([{"id": "", "content": "x", "status": "pending"}], merge=True)
        ids = [i["id"] for i in store.read()]
        assert "" not in ids


class TestTodoStoreFormatInjection:
    def test_format_includes_pending(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "my task", "status": "pending"}])
        result = store.format_for_injection()
        assert result is not None
        assert "my task" in result

    def test_format_includes_in_progress(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "active task", "status": "in_progress"}])
        result = store.format_for_injection()
        assert result is not None
        assert "active task" in result

    def test_format_excludes_completed(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "done task", "status": "completed"}])
        result = store.format_for_injection()
        assert result is None

    def test_format_none_when_all_completed(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "done", "status": "completed"},
            {"id": "2", "content": "cancelled", "status": "cancelled"},
        ])
        assert store.format_for_injection() is None


# ---------------------------------------------------------------------------
# handle_todo
# ---------------------------------------------------------------------------

class TestHandleTodo:
    def _fresh_store(self):
        return TodoStore()

    def test_read_empty_returns_json(self):
        import json
        store = self._fresh_store()
        result = handle_todo(_store=store)
        data = json.loads(result)
        assert data["todos"] == []

    def test_write_todos_returns_json(self):
        import json
        store = self._fresh_store()
        result = handle_todo(
            todos=[{"id": "1", "content": "task", "status": "pending"}],
            _store=store,
        )
        data = json.loads(result)
        assert len(data["todos"]) == 1

    def test_summary_counts(self):
        import json
        store = self._fresh_store()
        handle_todo(
            todos=[
                {"id": "1", "content": "t1", "status": "pending"},
                {"id": "2", "content": "t2", "status": "in_progress"},
                {"id": "3", "content": "t3", "status": "completed"},
            ],
            _store=store,
        )
        result = json.loads(handle_todo(_store=store))
        assert result["summary"]["pending"] == 1
        assert result["summary"]["in_progress"] == 1
        assert result["summary"]["completed"] == 1


# ---------------------------------------------------------------------------
# RecoveryAction / RecoveryResult
# ---------------------------------------------------------------------------

class TestRecoveryResult:
    def test_fields_stored(self):
        r = RecoveryResult(action=RecoveryAction.NONE, error_type="unknown", message="x", severity=0)
        assert r.action == RecoveryAction.NONE
        assert r.error_type == "unknown"
        assert r.severity == 0

    def test_context_limit_defaults_none(self):
        r = RecoveryResult(action=RecoveryAction.NONE, error_type="x", message="", severity=0)
        assert r.context_limit is None


# ---------------------------------------------------------------------------
# detect_context_error
# ---------------------------------------------------------------------------

class TestDetectContextError:
    def test_none_for_empty_string(self):
        assert detect_context_error("") is None

    def test_context_exceeded_detected(self):
        result = detect_context_error("context length exceeded")
        assert result is not None
        assert result.action == RecoveryAction.TRUNCATE

    def test_token_limit_detected(self):
        result = detect_context_error("token limit exceeded")
        assert result is not None

    def test_too_many_tokens_detected(self):
        result = detect_context_error("too many tokens")
        assert result is not None

    def test_input_too_long_detected(self):
        result = detect_context_error("input is too long")
        assert result is not None

    def test_severity_is_3(self):
        result = detect_context_error("context length exceeded")
        assert result.severity == 3

    def test_unrelated_message_returns_none(self):
        assert detect_context_error("file not found: foo.py") is None

    def test_returns_recovery_result(self):
        result = detect_context_error("context window exceeded")
        assert isinstance(result, RecoveryResult)


# ---------------------------------------------------------------------------
# detect_edit_error
# ---------------------------------------------------------------------------

class TestDetectEditError:
    def test_none_for_empty_string(self):
        assert detect_edit_error("") is None

    def test_string_not_found_detected(self):
        result = detect_edit_error("old_string not found in file")
        assert result is not None
        assert result.action == RecoveryAction.RETRY

    def test_file_not_found_detected(self):
        result = detect_edit_error("file not found: foo.py")
        assert result is not None

    def test_permission_denied_detected(self):
        result = detect_edit_error("permission denied")
        assert result is not None

    def test_severity_is_2(self):
        result = detect_edit_error("old_string not found")
        assert result.severity == 2

    def test_unrelated_returns_none(self):
        assert detect_edit_error("rate limit exceeded") is None

    def test_guidance_in_message(self):
        result = detect_edit_error("old_string not found in file")
        assert len(result.message) > 0


# ---------------------------------------------------------------------------
# detect_session_error
# ---------------------------------------------------------------------------

class TestDetectSessionError:
    def test_none_for_empty(self):
        assert detect_session_error("") is None

    def test_rate_limit_detected(self):
        result = detect_session_error("rate limit exceeded, please wait")
        assert result is not None
        assert result.action == RecoveryAction.RETRY

    def test_connection_refused_detected(self):
        result = detect_session_error("connection refused")
        assert result is not None

    def test_server_error_detected(self):
        result = detect_session_error("500 internal server error")
        assert result is not None

    def test_auth_error_detected(self):
        result = detect_session_error("authentication failed")
        assert result is not None
        assert result.action == RecoveryAction.NONE

    def test_api_key_invalid_detected(self):
        result = detect_session_error("api key is invalid")
        assert result is not None

    def test_unrelated_returns_none(self):
        assert detect_session_error("file not found") is None


# ---------------------------------------------------------------------------
# detect_and_recover
# ---------------------------------------------------------------------------

class TestDetectAndRecover:
    def test_empty_returns_none_action(self):
        result = detect_and_recover("")
        assert result.action == RecoveryAction.NONE

    def test_context_error_prioritized(self):
        result = detect_and_recover("context length exceeded and rate limit hit")
        assert result.action == RecoveryAction.TRUNCATE

    def test_edit_error_returns_retry(self):
        result = detect_and_recover("old_string not found in file")
        assert result.action == RecoveryAction.RETRY

    def test_session_error_returns_retry(self):
        result = detect_and_recover("rate limit exceeded")
        assert result.action == RecoveryAction.RETRY

    def test_unknown_returns_none(self):
        result = detect_and_recover("some totally unknown problem xyz")
        assert result.action == RecoveryAction.NONE

    def test_returns_recovery_result(self):
        result = detect_and_recover("any message here")
        assert isinstance(result, RecoveryResult)


# ---------------------------------------------------------------------------
# truncate_for_context
# ---------------------------------------------------------------------------

class TestTruncateForContext:
    def _msgs(self, n: int, include_system=True) -> list[dict]:
        msgs = []
        if include_system:
            msgs.append({"role": "system", "content": "sys"})
        for i in range(n):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"msg {i}"})
        return msgs

    def test_short_list_unchanged(self):
        msgs = self._msgs(5)
        result = truncate_for_context(msgs, keep_recent=10)
        assert len(result) == len(msgs)

    def test_long_list_truncated(self):
        msgs = self._msgs(30)
        result = truncate_for_context(msgs, keep_recent=5)
        assert len(result) < len(msgs)

    def test_system_prompt_preserved(self):
        msgs = self._msgs(30)
        result = truncate_for_context(msgs, keep_recent=5)
        assert result[0]["role"] == "system"

    def test_recent_messages_at_end(self):
        msgs = self._msgs(20)
        result = truncate_for_context(msgs, keep_recent=5)
        # Last 5 messages of original should be in result
        assert msgs[-1] in result

    def test_returns_list(self):
        result = truncate_for_context([])
        assert isinstance(result, list)
