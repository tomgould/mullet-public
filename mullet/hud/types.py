"""HUD type definitions."""
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal

HudPreset = Literal["minimal", "focused", "full", "dense"]
HudLayout = Literal["minimal", "full"]

@dataclass
class HudElementConfig:
    """Configuration for which HUD elements to display."""
    context_bar: bool = True
    agents: bool = True
    mode_state: bool = True
    todos: bool = True
    session: bool = True
    git: bool = True
    model: bool = True
    tokens: bool = True
    rate_limits: bool = True
    show_git: bool = True
    max_lines: int = 4
    compact_usage: bool = False

@dataclass
class HudConfig:
    """Top-level HUD configuration."""
    preset: HudPreset = "focused"
    elements: HudElementConfig = field(default_factory=HudElementConfig)
    max_width: Optional[int] = None
    layout: HudLayout = "full"

@dataclass
class HudRenderContext:
    """Runtime context used when rendering the HUD."""
    cwd: str = ""
    model_name: str = ""
    # provider inferred from model_name (claude-*→anthropic, kimi-*→kimi, …).
    # Set alongside model_name so the "Thinking…" row can render provider/model.
    main_provider: str = ""
    context_pct: float = 0.0
    agent_count: int = 0
    ralph_active: bool = False
    ralph_iter: int = 0
    ralph_max: int = 0
    autopilot_active: bool = False
    autopilot_phase: str = ""
    ultrawork_active: bool = False
    ultrawork_tasks: int = 0
    todos_done: int = 0
    todos_total: int = 0
    session_minutes: float = 0.0
    git_branch: str = ""
    git_dirty: bool = False
    total_tokens: int = 0
    last_in: int = 0
    last_out: int = 0
    five_hour_pct: Optional[float] = None
    weekly_pct: Optional[float] = None
    sonnet_weekly_pct: Optional[float] = None
    running_agents: list = field(default_factory=list)  # [{description, provider, model}]
    running_tools: list = field(default_factory=list)  # [{name, target, started_at, provider, model}]
    completed_tools: list = field(default_factory=list)  # [{name, target, duration_ms, provider, model}]
    session_tokens_in: int = 0
    session_tokens_out: int = 0
    rate_limit_reset_s: int = 0  # seconds until 5h reset
    thinking_active: bool = False
    speed_mode: str = "economy"  # economy | balanced | fast | max
    pressure_5h: float = 0.0    # paced 5h budget pressure (0-1)
    pressure_7d: float = 0.0    # paced 7d budget pressure (0-1)
    provider_costs_5h: dict = field(default_factory=dict)  # {provider: cost_usd} last 5h
    budget_status: str = ""     # go | throttle | pause — from check_budget()

PRESETS: Dict[str, HudElementConfig] = {
    "minimal": HudElementConfig(agents=False, todos=False, session=False, git=False, model=False, tokens=False, max_lines=1),
    "compact": HudElementConfig(agents=False, todos=False, mode_state=False, model=False, tokens=False, max_lines=1),
    "focused": HudElementConfig(max_lines=2),
    "full": HudElementConfig(max_lines=4),
    "dense": HudElementConfig(max_lines=1),
    # Claude Code statusLine preset: no project line (CC shows model/branch natively in its
    # own right-side box, consuming ~40 chars of the shared container budget).  Omitting
    # the 36-char project line means the budget/agent info fills the available space.
    "statusline": HudElementConfig(model=False, git=False, max_lines=4, compact_usage=True),
}
