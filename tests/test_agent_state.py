"""Tests for mullet.hud.agent_state module."""
import json
import time
from pathlib import Path
from unittest.mock import patch


from mullet.hud import agent_state


def test_write_agents_empty(tmp_path, monkeypatch):
    """Test write_agents([]) creates file with active_count=0, agents=[]."""
    state_file = tmp_path / "agents.json"
    monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)
    
    agent_state.write_agents([])
    
    assert state_file.exists()
    content = json.loads(state_file.read_text())
    assert content["active_count"] == 0
    assert content["agents"] == []


def test_write_agents_multiple(tmp_path, monkeypatch):
    """Test write_agents with multiple agents sets correct active_count."""
    state_file = tmp_path / "agents.json"
    monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)
    
    agents = [
        {"description": "Test agent 1", "provider": "openai", "model": "gpt-4"},
        {"description": "Test agent 2", "provider": "anthropic", "model": "claude-3"}
    ]
    
    agent_state.write_agents(agents)
    
    content = json.loads(state_file.read_text())
    assert content["active_count"] == 2
    assert content["agents"] == agents


def test_write_agents_valid_json(tmp_path, monkeypatch):
    """Test write_agents result is parseable JSON."""
    state_file = tmp_path / "agents.json"
    monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)
    
    agents = [{"description": "test", "provider": "test", "model": "test"}]
    agent_state.write_agents(agents)
    
    # Will raise json.JSONDecodeError if not valid JSON
    content = json.loads(state_file.read_text())
    assert isinstance(content, dict)
    assert "active_count" in content
    assert "agents" in content


def test_write_agent_strips_newlines(tmp_path, monkeypatch):
    """Test write_agent strips newlines from description."""
    state_file = tmp_path / "agents.json"
    monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)
    
    description = "Test\ndescription\nwith\nnewlines"
    agent_state.write_agent(description, "openai", "gpt-4")
    
    content = json.loads(state_file.read_text())
    agent = content["agents"][0]
    assert "\n" not in agent["description"]
    assert agent["description"] == "Test description with newlines"


def test_write_agent_strips_whitespace(tmp_path, monkeypatch):
    """Test write_agent strips leading/trailing whitespace from description."""
    state_file = tmp_path / "agents.json"
    monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)
    
    description = "   Test description with spaces   "
    agent_state.write_agent(description, "openai", "gpt-4")
    
    content = json.loads(state_file.read_text())
    agent = content["agents"][0]
    assert agent["description"] == "Test description with spaces"


def test_write_agent_fields(tmp_path, monkeypatch):
    """Test write_agent stores provider and model correctly."""
    state_file = tmp_path / "agents.json"
    monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)
    
    agent_state.write_agent("test description", "openai", "gpt-4-turbo")
    
    content = json.loads(state_file.read_text())
    agent = content["agents"][0]
    assert agent["provider"] == "openai"
    assert agent["model"] == "gpt-4-turbo"


def test_clear_agents(tmp_path, monkeypatch):
    """Test clear_agents sets active_count=0."""
    state_file = tmp_path / "agents.json"
    monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)
    
    # First write some agents
    agent_state.write_agents([
        {"description": "Agent 1", "provider": "openai", "model": "gpt-4"},
        {"description": "Agent 2", "provider": "anthropic", "model": "claude-3"}
    ])
    
    # Then clear them
    agent_state.clear_agents()
    
    content = json.loads(state_file.read_text())
    assert content["active_count"] == 0
    assert content["agents"] == []


def test_write_agent_includes_started_at(tmp_path, monkeypatch):
    """Test write_agent stores started_at timestamp."""
    state_file = tmp_path / "agents.json"
    monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)

    before = time.time()
    agent_state.write_agent("test description", "openai", "gpt-4")
    after = time.time()

    content = json.loads(state_file.read_text())
    agent = content["agents"][0]
    assert "started_at" in agent
    assert before <= agent["started_at"] <= after


def test_write_agents_oserror_swallowed(tmp_path, monkeypatch):
    """Test OSError in write_text is swallowed without exception."""
    state_file = tmp_path / "agents.json"
    monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)
    
    # Mock path.write_text to raise OSError
    def mock_write_text(*args, **kwargs):
        raise OSError("Simulated disk full")
    
    with patch.object(Path, 'write_text', mock_write_text):
        # Should not raise an exception
        agent_state.write_agents([{"description": "test", "provider": "test", "model": "test"}])


def test_update_agent_tokens_accumulates(tmp_path, monkeypatch):
    """Test update_agent_tokens accumulates tokens and cost for an existing agent."""
    state_file = tmp_path / "agents.json"
    monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)

    agent_state.add_agent("swarm-abc", "test goal", "swarm", "kimi")
    agent_state.update_agent_tokens("swarm-abc", 100, 50, 0.02)
    agent_state.update_agent_tokens("swarm-abc", 200, 75, 0.03)

    content = json.loads(state_file.read_text())
    agent = content["agents"][0]
    assert agent["tokens_in"] == 300
    assert agent["tokens_out"] == 125
    assert agent["cost_usd"] == 0.05


def test_update_agent_tokens_missing_agent_is_noop(tmp_path, monkeypatch):
    """Test update_agent_tokens is a no-op when the agent id does not exist."""
    state_file = tmp_path / "agents.json"
    monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)

    agent_state.add_agent("swarm-abc", "test goal", "swarm", "kimi")
    agent_state.update_agent_tokens("swarm-xyz", 100, 50, 0.02)

    content = json.loads(state_file.read_text())
    agent = content["agents"][0]
    assert "tokens_in" not in agent
    assert "tokens_out" not in agent
    assert "cost_usd" not in agent


class TestAgentStateMissingLines:
    """Cover lines 22, 33-34, 92-98 in agent_state.py."""

    def test_read_current_bad_json_returns_empty(self, tmp_path, monkeypatch):
        """_read_current returns [] when file contains invalid JSON (lines 33-34)."""
        state_file = tmp_path / "agents.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text("{not json")
        monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)
        assert agent_state._read_current() == []

    def test_read_current_oserror_returns_empty(self, tmp_path, monkeypatch):
        """_read_current returns [] when read_text raises OSError (lines 33-34)."""
        state_file = tmp_path / "agents.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text("{}")
        monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)
        from unittest.mock import patch
        with patch.object(state_file.__class__, "read_text", side_effect=OSError("disk error")):
            assert agent_state._read_current() == []

    def test_state_path_is_under_the_state_dir(self):
        """_state_path returns state/agents.json in the state directory (line 22)."""
        from mullet.core.paths import state_dir
        assert agent_state._state_path() == state_dir() / "state" / "agents.json"

    def test_update_agent_status(self, tmp_path, monkeypatch):
        """update_agent_status changes the status field (lines 92-98)."""
        state_file = tmp_path / "agents.json"
        monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)
        agent_state.add_agent("agent-1", "desc", "openai", "gpt-4")
        agent_state.update_agent_status("agent-1", "completed")
        content = json.loads(state_file.read_text())
        assert content["agents"][0]["status"] == "completed"

    def test_update_agent_status_missing_id_is_noop(self, tmp_path, monkeypatch):
        """update_agent_status is a no-op when agent id is missing (lines 92-98)."""
        state_file = tmp_path / "agents.json"
        monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)
        agent_state.add_agent("agent-1", "desc", "openai", "gpt-4")
        agent_state.update_agent_status("agent-missing", "completed")
        content = json.loads(state_file.read_text())
        # add_agent sets status="running"; missing id should leave it unchanged
        assert content["agents"][0]["status"] == "running"

    def test_update_agent_model_corrects_provider_and_model(self, tmp_path, monkeypatch):
        """update_agent_model overwrites provider/model for a running agent by id."""
        state_file = tmp_path / "agents.json"
        monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)
        agent_state.add_agent("agent-1", "desc", "opencode", "openrouter/deepseek/deepseek-r1")
        agent_state.update_agent_model("agent-1", "kimi", "moonshot/kimi-k2.7-code")
        content = json.loads(state_file.read_text())
        agent = content["agents"][0]
        assert agent["provider"] == "kimi"
        assert agent["model"] == "moonshot/kimi-k2.7-code"

    def test_update_agent_model_missing_id_is_noop(self, tmp_path, monkeypatch):
        """update_agent_model is a no-op when the agent id does not exist."""
        state_file = tmp_path / "agents.json"
        monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)
        agent_state.add_agent("agent-1", "desc", "opencode", "openrouter/deepseek/deepseek-r1")
        agent_state.update_agent_model("agent-missing", "kimi", "moonshot/kimi-k2.7-code")
        content = json.loads(state_file.read_text())
        assert content["agents"][0]["model"] == "openrouter/deepseek/deepseek-r1"