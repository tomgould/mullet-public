"""Integration tests against real LLM (LM Studio on localhost:1234).

These tests require a running local LLM. They are skipped if no
local provider is detected. Run with: pytest tests/test_integration.py -v

These are NOT unit tests — they make real API calls and verify
end-to-end behavior.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Skip all tests if no local LLM is available
try:
    from mullet.cli.main import detect_local_provider, get_local_loaded_context
    # Require 512+ context as basic sanity check
    HAS_LOCAL_LLM = detect_local_provider() is not None and get_local_loaded_context() >= 512
except Exception:
    HAS_LOCAL_LLM = False

# Tests that require actual tool use need enough context to include tool schemas (~4K tokens)
HAS_LLM_TOOLS = HAS_LOCAL_LLM and get_local_loaded_context() >= 6144

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not HAS_LOCAL_LLM, reason="No local LLM running"),
]


@pytest.fixture
def provider():
    from mullet.cli.main import detect_local_provider
    return detect_local_provider()


@pytest.fixture
def minimal_agent(provider):
    from mullet.core.agent.runner import AgentRunner
    from mullet.core.tools.registry import ToolRegistry
    return AgentRunner(
        provider=provider,
        tool_registry=ToolRegistry(),
        toolsets=[],
        system_prompt="You are a helpful assistant.",
        max_iterations=3,
    )


@pytest.fixture
def agent(provider):
    from mullet.core.agent.runner import AgentRunner
    from mullet.cli.main import build_tool_registry
    from mullet.core.agent.prompts import build_base_prompt
    registry = build_tool_registry()
    return AgentRunner(
        provider=provider,
        tool_registry=registry,
        toolsets=["terminal", "file", "web", "code"],
        system_prompt=build_base_prompt(),
        max_iterations=10,
    )


def test_simple_question(minimal_agent):
    """Agent answers a simple factual question — uses no tools to stay within small context windows."""
    result = minimal_agent.run("What is 2 + 2? Answer with just the number.")
    assert result.completed
    assert "4" in result.final_response


@pytest.mark.skipif(not HAS_LLM_TOOLS, reason="Tool use requires ≥6K context model")
def test_terminal_tool_use(agent):
    """Agent uses terminal to run a command."""
    result = agent.run("Run the command 'echo hello_mullet' and tell me the output.")
    assert result.completed
    assert "hello_mullet" in result.final_response
    assert result.tool_calls_made >= 1


@pytest.mark.skipif(not HAS_LLM_TOOLS, reason="Tool use requires ≥6K context model")
def test_file_read(agent):
    """Agent reads a real file using read_file tool."""
    result = agent.run(
        f"Use the read_file tool to read {REPO_ROOT / 'pyproject.toml'} "
        "and tell me what the project name is."
    )
    assert result.completed
    # The agent should either use the tool or at least mention mullet
    assert "mullet" in result.final_response.lower() or result.tool_calls_made >= 1


@pytest.mark.skipif(not HAS_LLM_TOOLS, reason="Tool use requires ≥6K context model")
def test_code_execution(agent):
    """Agent runs Python code."""
    result = agent.run("Use the code_exec tool to run: print(sum(range(10)))")
    assert result.completed
    # sum(range(10)) = 45
    assert "45" in result.final_response


def test_iteration_budget_respected(provider):
    """Agent stops when budget is exhausted."""
    from mullet.core.agent.runner import AgentRunner
    from mullet.cli.main import build_tool_registry
    registry = build_tool_registry()
    agent = AgentRunner(
        provider=provider,
        tool_registry=registry,
        max_iterations=1,
    )
    # With only 1 iteration, should complete quickly
    result = agent.run("hello")
    assert result.iterations_used <= 1


@pytest.mark.skipif(not HAS_LLM_TOOLS, reason="Tool use requires ≥6K context model")
def test_web_search(agent):
    """Agent performs a web search."""
    result = agent.run("Search the web for 'Python programming language' and tell me one fact.")
    assert result.completed
    assert len(result.final_response) > 20


def test_router_complexity_scoring():
    """Router correctly scores task complexity."""
    from mullet.core.router.complexity import extract_signals, score_complexity, score_to_tier, Tier

    # Simple
    signals = extract_signals("list files")
    assert score_to_tier(score_complexity(signals)) == Tier.LOW

    # Complex
    signals = extract_signals(
        "refactor the entire authentication system to support OAuth2 "
        "with JWT token refresh across all microservices in production"
    )
    assert score_to_tier(score_complexity(signals)) == Tier.ADVANCED


def test_skill_loading():
    """Built-in skills load correctly."""
    from mullet.core.skills.loader import SkillLoader
    from mullet.core.skills.registry import SkillRegistry

    loader = SkillLoader()
    registry = SkillRegistry(loader=loader)
    registry.load_all()

    assert registry.get("autopilot") is not None
    assert registry.get("ralph") is not None
    assert registry.get("code-review") is not None

    # Trigger matching
    match = registry.match("/code-review")
    assert match is not None
    assert match.skill.metadata.name == "code-review"

    # Informational intent should NOT trigger
    match = registry.match("what is autopilot?")
    assert match is None


def test_mcp_server_responds():
    """MCP server handles tool listing."""
    from mullet.server.mcp import handle_request

    resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    assert resp["result"]["tools"]
    tool_names = [t["name"] for t in resp["result"]["tools"]]
    assert "mullet_run" in tool_names
    assert "mullet_status" in tool_names
