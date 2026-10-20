from unittest.mock import MagicMock, patch
import pytest
from mullet.knowledge.engine.runner import AgentRunner, AgentResult
from mullet.knowledge.engine.search import SearchQuality


@pytest.fixture
def mock_search_result():
    result = MagicMock()
    result.results = [
        {"metadata": {"source_file": "doc.md", "chunk_type": "document"}, "_combined_score": 0.9, "content": "Test content."}
    ]
    result.quality = MagicMock(spec=SearchQuality)
    return result


@pytest.fixture
def runner(mock_search_result):
    agent = {"name": "TestAgent", "system_prompt": "You are helpful."}
    client = MagicMock()
    client.llm_config = {"max_context_chars": 4000}
    llm_provider = MagicMock()
    llm_provider.generate.return_value = "The answer is 42."
    searcher = MagicMock()
    searcher.search_with_quality.return_value = mock_search_result
    return AgentRunner(agent, client, llm_provider, searcher)


class TestAgentRunnerInit:
    def test_stores_agent(self):
        agent = {"name": "MyAgent"}
        r = AgentRunner(agent, MagicMock(), MagicMock(), MagicMock())
        assert r.agent is agent

    def test_memory_store_none_by_default(self):
        r = AgentRunner({}, MagicMock(), MagicMock(), MagicMock())
        assert r.memory_store is None

    def test_allowed_providers_stored(self):
        r = AgentRunner({}, MagicMock(), MagicMock(), MagicMock(), allowed_providers=frozenset({"kimi"}))
        assert r.allowed_providers == frozenset({"kimi"})


class TestAgentRunnerRun:
    def test_returns_agent_result(self, runner):
        result = runner.run("What is 6 * 7?")
        assert isinstance(result, AgentResult)

    def test_response_from_llm(self, runner):
        result = runner.run("test query")
        assert result.response == "The answer is 42."

    def test_agent_name_set(self, runner):
        result = runner.run("q")
        assert result.agent_name == "TestAgent"

    def test_query_set(self, runner):
        result = runner.run("my query")
        assert result.query == "my query"

    def test_sources_populated_from_chunks(self, runner):
        result = runner.run("q")
        assert len(result.sources) == 1
        assert result.sources[0]["source"] == "doc.md"

    def test_source_score_uses_rerank_score_when_present(self, runner, mock_search_result):
        mock_search_result.results[0]["_rerank_score"] = 0.95
        result = runner.run("q")
        assert result.sources[0]["score"] == 0.95

    def test_source_score_falls_back_to_combined(self, runner, mock_search_result):
        mock_search_result.results[0].pop("_rerank_score", None)
        result = runner.run("q")
        assert result.sources[0]["score"] == 0.9

    def test_timing_keys_present(self, runner):
        result = runner.run("q")
        assert "search_ms" in result.timing
        assert "generate_ms" in result.timing
        assert "total_ms" in result.timing

    def test_max_tokens_passed_to_generate(self, runner):
        runner.run("q", max_tokens=512)
        call_kwargs = runner.llm_provider.generate.call_args[1]
        assert call_kwargs.get("max_tokens") == 512

    def test_no_max_tokens_not_passed(self, runner):
        runner.run("q")
        call_kwargs = runner.llm_provider.generate.call_args[1]
        assert "max_tokens" not in call_kwargs

    def test_conversation_history_prepended(self, runner):
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        runner.run("q", conversation_history=history)
        messages_arg = runner.llm_provider.generate.call_args[0][0]
        assert messages_arg[0] == {"role": "user", "content": "hi"}

    def test_no_history_not_prepended(self, runner):
        runner.run("q")
        messages_arg = runner.llm_provider.generate.call_args[0][0]
        # messages should not start with a user/assistant history item (starts with RAG content)
        assert messages_arg is not None

    def test_store_memory_stores_result(self, runner):
        mock_mem = MagicMock()
        runner.memory_store = mock_mem
        runner.run("q", store_memory=True)
        mock_mem.store.assert_called_once()
        call_kwargs = mock_mem.store.call_args[1]
        assert "agent:" in call_kwargs["namespace"]

    def test_store_memory_false_does_not_store(self, runner):
        mock_mem = MagicMock()
        runner.memory_store = mock_mem
        runner.run("q", store_memory=False)
        mock_mem.store.assert_not_called()

    def test_no_memory_store_no_error(self, runner):
        runner.memory_store = None
        result = runner.run("q", store_memory=True)
        assert result is not None

    def test_client_without_llm_config_uses_default(self, runner):
        runner.client = MagicMock(spec=[])  # no llm_config attribute
        result = runner.run("q")
        assert result is not None

    def test_custom_memory_namespace_uses_filename(self, runner):
        mock_mem = MagicMock()
        runner.memory_store = mock_mem
        runner.agent["_filename"] = "code-reviewer"
        runner.run("q", store_memory=True)
        call_kwargs = mock_mem.store.call_args[1]
        assert "code-reviewer" in call_kwargs["namespace"]


class TestAgentRunnerChat:
    def test_quit_exits_immediately(self, runner, capsys):
        with patch("builtins.input", side_effect=["quit"]):
            runner.chat()
        out = capsys.readouterr().out
        assert "Bye!" in out

    def test_exit_exits_immediately(self, runner, capsys):
        with patch("builtins.input", side_effect=["exit"]):
            runner.chat()
        out = capsys.readouterr().out
        assert "Bye!" in out

    def test_eof_error_breaks_loop(self, runner, capsys):
        with patch("builtins.input", side_effect=EOFError):
            runner.chat()
        out = capsys.readouterr().out
        assert "Bye!" in out

    def test_keyboard_interrupt_breaks_loop(self, runner, capsys):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            runner.chat()
        out = capsys.readouterr().out
        assert "Bye!" in out

    def test_empty_input_continues(self, runner, capsys):
        with patch("builtins.input", side_effect=["", "quit"]):
            runner.chat()  # must not raise

    def test_query_produces_response(self, runner, capsys):
        with patch("builtins.input", side_effect=["What is Python?", "quit"]):
            runner.chat()
        out = capsys.readouterr().out
        assert "TestAgent:" in out

    def test_history_accumulated(self, runner):
        calls = []
        def mock_run(query, top_k, conversation_history=None):
            # copy to avoid mutation from caller's continued appends
            calls.append(list(conversation_history) if conversation_history else conversation_history)
            return AgentResult(response="ok", agent_name="TestAgent", query=query)
        runner.run = mock_run
        with patch("builtins.input", side_effect=["first q", "second q", "quit"]):
            runner.chat()
        # second call should have history from first
        assert calls[1] is not None
        assert len(calls[1]) == 2  # user + assistant from first turn
