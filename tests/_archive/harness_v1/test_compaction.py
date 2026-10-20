"""Tests for mullet.harness.engine.compaction."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock


from mullet.harness.engine.compaction import (
    AutoCompactor,
    CompactionResult,
    make_compaction_hook,
)
from mullet.harness.hooks.types import TransformPayload


class TestEstimateTokens:
    def test_empty_messages(self) -> None:
        compactor = AutoCompactor()
        assert compactor.estimate_tokens([]) == 0

    def test_content_only(self) -> None:
        compactor = AutoCompactor()
        # 7 chars // 3.5 = 2.0 -> int(2.0) = 2
        assert compactor.estimate_tokens([{"role": "user", "content": "hello!!"}]) == 2

    def test_tool_calls_counted(self) -> None:
        compactor = AutoCompactor()
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "foo",
                        "arguments": '{"a": 1}',
                    }
                }
            ],
        }
        # content "" + name "foo" (3) + arguments '{"a": 1}' (8) = 11 chars
        # 11 // 3.5 = 3.14 -> int = 3
        assert compactor.estimate_tokens([msg]) == 3

    def test_none_content(self) -> None:
        compactor = AutoCompactor()
        assert compactor.estimate_tokens([{"role": "assistant", "content": None}]) == 0


class TestShouldCompact:
    def test_below_threshold(self) -> None:
        compactor = AutoCompactor(threshold=0.7)
        messages = [{"role": "user", "content": "x" * 10}]
        assert not compactor.should_compact(messages, context_window=10)

    def test_above_threshold(self) -> None:
        compactor = AutoCompactor(threshold=0.7)
        messages = [{"role": "user", "content": "x" * 35}]
        assert compactor.should_compact(messages, context_window=10)

    def test_exactly_at_threshold(self) -> None:
        compactor = AutoCompactor(threshold=0.5)
        # 18 chars // 3.5 = 5.14 -> 5 tokens. 0.5 * 10 = 5. 5 > 5 is False.
        messages = [{"role": "user", "content": "x" * 18}]
        assert not compactor.should_compact(messages, context_window=10)


class TestCompactSummarizesPreRecent:
    def test_summarizes_older_turns(self, monkeypatch) -> None:
        compactor = AutoCompactor(keep_recent_turns=2)

        messages = []
        for i in range(4):
            messages.append({"role": "user", "content": f"user {i}"})
            messages.append({"role": "assistant", "content": f"assistant {i}"})

        mock_provider = MagicMock()
        mock_provider.create.return_value = SimpleNamespace(content="Summary of old stuff")

        monkeypatch.setattr(
            "mullet.harness.engine.loop._resolve_provider",
            lambda name: mock_provider,
        )

        result = compactor.compact(messages, context_window=1000)

        assert isinstance(result, CompactionResult)
        assert result.original_count == 8
        assert result.summarized_count == 4
        assert result.kept_count == 5
        assert result.summary == "Summary of old stuff"
        assert result.new_messages[0]["role"] == "system"
        assert "[Earlier conversation summary]" in result.new_messages[0]["content"]
        assert "Summary of old stuff" in result.new_messages[0]["content"]

    def test_fallback_summary_on_provider_error(self, monkeypatch) -> None:
        compactor = AutoCompactor(keep_recent_turns=1)

        messages = [
            {"role": "user", "content": "old user"},
            {"role": "assistant", "content": "old assistant"},
            {"role": "user", "content": "recent user"},
            {"role": "assistant", "content": "recent assistant"},
        ]

        def _fail(*args, **kwargs):
            raise RuntimeError("provider down")

        monkeypatch.setattr(
            "mullet.harness.engine.loop._resolve_provider",
            lambda name: _fail,
        )

        result = compactor.compact(messages, context_window=1000)

        assert result.summarized_count == 2
        assert result.kept_count == 3
        assert "old user" in result.summary or "old assistant" in result.summary

    def test_no_op_when_not_enough_turns(self) -> None:
        compactor = AutoCompactor(keep_recent_turns=5)

        messages = [
            {"role": "user", "content": "user 0"},
            {"role": "assistant", "content": "assistant 0"},
            {"role": "user", "content": "user 1"},
            {"role": "assistant", "content": "assistant 1"},
        ]

        # Even if threshold is exceeded, there aren't enough turns to compact.
        result = compactor.compact(messages, context_window=1)

        assert result.summarized_count == 0
        assert result.kept_count == 4
        assert result.new_messages == messages


class TestKeepsSystemMessages:
    def test_original_system_messages_preserved(self, monkeypatch) -> None:
        compactor = AutoCompactor(keep_recent_turns=2)

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "very old user"},
            {"role": "assistant", "content": "very old assistant"},
            {"role": "user", "content": "old user"},
            {"role": "assistant", "content": "old assistant"},
            {"role": "user", "content": "recent user"},
            {"role": "assistant", "content": "recent assistant"},
        ]

        mock_provider = MagicMock()
        mock_provider.create.return_value = SimpleNamespace(content="Summary")

        monkeypatch.setattr(
            "mullet.harness.engine.loop._resolve_provider",
            lambda name: mock_provider,
        )

        result = compactor.compact(messages, context_window=1000)

        assert result.new_messages[0]["role"] == "system"
        assert result.new_messages[0]["content"] == "You are helpful"
        assert result.new_messages[1]["role"] == "system"
        assert "Summary" in result.new_messages[1]["content"]
        assert result.new_messages[2]["role"] == "user"
        assert result.new_messages[2]["content"] == "old user"
        assert result.new_messages[3]["role"] == "assistant"
        assert result.new_messages[3]["content"] == "old assistant"
        assert result.new_messages[4]["role"] == "user"
        assert result.new_messages[4]["content"] == "recent user"
        assert result.new_messages[5]["role"] == "assistant"
        assert result.new_messages[5]["content"] == "recent assistant"


class TestKeepsLastNTurns:
    def test_keeps_exactly_n_user_assistant_pairs(self, monkeypatch) -> None:
        compactor = AutoCompactor(keep_recent_turns=2)

        messages = []
        for i in range(5):
            messages.append({"role": "user", "content": f"user {i}"})
            messages.append({"role": "assistant", "content": f"assistant {i}"})

        mock_provider = MagicMock()
        mock_provider.create.return_value = SimpleNamespace(content="Summary")

        monkeypatch.setattr(
            "mullet.harness.engine.loop._resolve_provider",
            lambda name: mock_provider,
        )

        result = compactor.compact(messages, context_window=1000)

        # Kept: summary (1) + last 2 turns (4) = 5
        assert result.kept_count == 5
        assert result.new_messages[1]["content"] == "user 3"
        assert result.new_messages[2]["content"] == "assistant 3"
        assert result.new_messages[3]["content"] == "user 4"
        assert result.new_messages[4]["content"] == "assistant 4"

        contents = [m["content"] for m in result.new_messages]
        assert "user 0" not in contents
        assert "user 1" not in contents
        assert "user 2" not in contents

    def test_tool_messages_included_in_turn(self, monkeypatch) -> None:
        compactor = AutoCompactor(keep_recent_turns=1)

        messages = [
            {"role": "user", "content": "user 0"},
            {"role": "assistant", "content": "assistant 0", "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "content": "tool result", "tool_call_id": "tc1"},
            {"role": "user", "content": "user 1"},
            {"role": "assistant", "content": "assistant 1"},
        ]

        mock_provider = MagicMock()
        mock_provider.create.return_value = SimpleNamespace(content="Summary")

        monkeypatch.setattr(
            "mullet.harness.engine.loop._resolve_provider",
            lambda name: mock_provider,
        )

        result = compactor.compact(messages, context_window=1000)

        # Recent turn starts at user 1 (index 3 in non_system).
        # Pre-recent includes user 0, assistant 0, tool result.
        assert result.summarized_count == 3
        assert result.kept_count == 3  # summary + user 1 + assistant 1
        assert result.new_messages[1]["content"] == "user 1"
        assert result.new_messages[2]["content"] == "assistant 1"


class TestHook:
    def test_fires_when_threshold_exceeded(self, monkeypatch) -> None:
        compactor = AutoCompactor(threshold=0.5, keep_recent_turns=1)

        messages = [
            {"role": "user", "content": "x" * 21},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "content": "recent"},
            {"role": "assistant", "content": "recent reply"},
        ]

        mock_provider = MagicMock()
        mock_provider.create.return_value = SimpleNamespace(content="Compacted")

        monkeypatch.setattr(
            "mullet.harness.engine.loop._resolve_provider",
            lambda name: mock_provider,
        )

        def resolver(provider, model):
            return 10

        hook = make_compaction_hook(compactor, resolver)

        payload = TransformPayload(
            messages=list(messages),
            session_id="sess-1",
            purpose="inject",
            provider="kimi",
            model="kimi-k2.6",
        )

        result = hook(payload)

        assert result is not None
        assert isinstance(result, TransformPayload)
        assert len(result.messages) == 3
        assert result.messages[0]["role"] == "system"
        assert "Compacted" in result.messages[0]["content"]
        assert result.messages[1]["content"] == "recent"
        assert result.messages[2]["content"] == "recent reply"

    def test_passes_through_when_under_threshold(self) -> None:
        compactor = AutoCompactor(threshold=0.5, keep_recent_turns=1)

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

        def resolver(provider, model):
            return 1000

        hook = make_compaction_hook(compactor, resolver)

        payload = TransformPayload(
            messages=list(messages),
            session_id="sess-1",
            purpose="inject",
            provider="kimi",
            model="kimi-k2.6",
        )

        result = hook(payload)

        assert result is None

    def test_ignores_non_inject_purpose(self) -> None:
        compactor = AutoCompactor(threshold=0.0, keep_recent_turns=1)

        messages = [{"role": "user", "content": "x" * 100}]

        def resolver(provider, model):
            return 10

        hook = make_compaction_hook(compactor, resolver)

        payload = TransformPayload(
            messages=list(messages),
            session_id="sess-1",
            purpose="compact",
            provider="kimi",
            model="kimi-k2.6",
        )

        result = hook(payload)

        assert result is None

    def test_ignores_non_transform_payload(self) -> None:
        compactor = AutoCompactor(threshold=0.0, keep_recent_turns=1)

        def resolver(provider, model):
            return 10

        hook = make_compaction_hook(compactor, resolver)

        result = hook({"not": "a transform payload"})

        assert result is None
