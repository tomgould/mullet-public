import pytest
from mullet.hud.types import HudRenderContext, HudConfig, HudElementConfig
from mullet.hud.elements import (
    render_context, render_agents, render_mode, render_todos,
    render_session, render_git, render_model, render_tokens, render_rate_limits,
)
from mullet.hud.render import render_hud
from mullet.hud.colors import strip_ansi, green, red, RESET

# Test element renderers
@pytest.fixture
def empty_context():
    return HudRenderContext()

@pytest.fixture
def populated_context():
    return HudRenderContext(
        cwd="/home/user/project",
        model_name="gpt-4",
        context_pct=75.0,
        agent_count=3,
        ralph_active=True,
        ralph_iter=2,
        ralph_max=5,
        todos_done=1,
        todos_total=2,
        session_minutes=45.0,
        git_branch="main",
        git_dirty=True,
        total_tokens=1500,
        last_in=500,
        last_out=1000,
        five_hour_pct=70.0,
        weekly_pct=50.0,
    )

def test_render_context_empty(empty_context):
    assert render_context(empty_context) is None

def test_render_context_populated(populated_context):
    output = render_context(populated_context)
    assert output is not None
    assert "Ctx" in output
    assert "75%" in output

def test_render_agents_empty(empty_context):
    assert render_agents(empty_context) is None

def test_render_agents_populated(populated_context):
    output = render_agents(populated_context)
    assert output is not None
    assert "Agents: 3" in strip_ansi(output)

def test_render_mode_empty(empty_context):
    assert render_mode(empty_context) is None

def test_render_mode_populated_ralph(populated_context):
    output = render_mode(populated_context)
    assert output is not None
    assert "Ralph: 2/5" in strip_ansi(output)

def test_render_mode_populated_autopilot():
    ctx = HudRenderContext(autopilot_active=True, autopilot_phase="planning")
    output = render_mode(ctx)
    assert output is not None
    assert "Autopilot: planning" in strip_ansi(output)

def test_render_mode_populated_ultrawork():
    ctx = HudRenderContext(ultrawork_active=True, ultrawork_tasks=4)
    output = render_mode(ctx)
    assert output is not None
    assert "Ultrawork: 4 tasks" in strip_ansi(output)

def test_render_todos_empty(empty_context):
    assert render_todos(empty_context) is None

def test_render_todos_populated(populated_context):
    output = render_todos(populated_context)
    assert output is not None
    assert "Todos: 1/2" in strip_ansi(output)

def test_render_session_empty(empty_context):
    assert render_session(empty_context) is None

def test_render_session_populated(populated_context):
    output = render_session(populated_context)
    assert output is not None
    assert "Session: 45m" in strip_ansi(output)

def test_render_git_empty(empty_context):
    assert render_git(empty_context) is None

def test_render_git_populated(populated_context):
    output = render_git(populated_context)
    assert output is not None
    assert "Git: main" in strip_ansi(output)

def test_render_model_empty(empty_context):
    assert render_model(empty_context) is None

def test_render_model_populated(populated_context):
    output = render_model(populated_context)
    assert output is not None
    assert "Model: gpt-4" in strip_ansi(output)

def test_render_tokens_empty(empty_context):
    assert render_tokens(empty_context) is None

def test_render_tokens_populated(populated_context):
    output = render_tokens(populated_context)
    assert output is not None
    assert "Tok: 1500 (500/1000)" in strip_ansi(output)

def test_render_rate_limits_empty(empty_context):
    # When usage data is unavailable, HUD shows "?" placeholders (not hidden)
    output = render_rate_limits(empty_context)
    assert output is not None
    assert "5h" in strip_ansi(output)
    assert "?" in strip_ansi(output)

def test_render_rate_limits_populated(populated_context):
    output = render_rate_limits(populated_context)
    assert output is not None
    assert "5h: 70%" in strip_ansi(output)
    assert "7d: 50%" in strip_ansi(output)

# Test render_hud
def test_render_hud_produces_output(populated_context):
    output = render_hud(populated_context)
    assert isinstance(output, str)
    assert len(output) > 0
    plain = strip_ansi(output)
    assert "Ctx" in plain
    assert "Agents" in plain
    assert "Ralph" in plain

def test_render_hud_with_config(populated_context):
    config = HudConfig(preset="minimal")
    output = render_hud(populated_context, config)
    assert isinstance(output, str)
    assert len(output) > 0
    plain = strip_ansi(output)
    # Minimal preset hides agents, todos, session, git, model, tokens
    assert "Agents" not in plain
    assert "Todos" not in plain
    assert "Session" not in plain
    assert "Model" not in plain
    assert "Tok" not in plain

def test_render_hud_custom_elements():
    ctx = HudRenderContext(context_pct=50, agent_count=1)
    config = HudConfig(elements=HudElementConfig(context_bar=True, agents=True, todos=False))
    output = render_hud(ctx, config)
    assert "Ctx" in output
    assert "Agents" in output
    assert "Todos" not in output

def test_render_hud_max_width():
    ctx = HudRenderContext(context_pct=50, model_name="gpt-4")
    config = HudConfig(max_width=20)
    output = render_hud(ctx, config)
    assert len(strip_ansi(output.splitlines()[0])) <= 20

# Test strip_ansi
def test_strip_ansi_removes_codes():
    text_with_ansi = f"{green('hello')} {red('world')}{RESET}"
    stripped_text = strip_ansi(text_with_ansi)
    assert stripped_text == "hello world"

def test_strip_ansi_no_codes():
    text_no_ansi = "hello world"
    stripped_text = strip_ansi(text_no_ansi)
    assert stripped_text == "hello world"

def test_strip_ansi_empty_string():
    assert strip_ansi("") == ""


def test_render_speed_badge_economy_hidden():
    from mullet.hud.elements import render_speed
    ctx = HudRenderContext(speed_mode="economy")
    assert render_speed(ctx) is None


def test_render_speed_badge_fast():
    from mullet.hud.elements import render_speed
    from mullet.hud.colors import strip_ansi
    ctx = HudRenderContext(speed_mode="fast")
    result = render_speed(ctx)
    assert result is not None
    assert "[fast]" in strip_ansi(result)


def test_render_speed_badge_max():
    from mullet.hud.elements import render_speed
    from mullet.hud.colors import strip_ansi
    ctx = HudRenderContext(speed_mode="max")
    result = render_speed(ctx)
    assert result is not None
    assert "[max]" in strip_ansi(result)


def test_render_hud_includes_speed_badge():
    ctx = HudRenderContext(speed_mode="fast", context_pct=10)
    output = render_hud(ctx)
    assert "[fast]" in strip_ansi(output)
