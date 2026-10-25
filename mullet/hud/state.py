"""Read Mullet state for HUD rendering."""
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from mullet.core.paths import state_dir as _user_state_dir
from mullet.hud.types import HudRenderContext

logger = logging.getLogger(__name__)

_STATE_MAX_AGE = 1800  # 30 minutes — ignore state files older than this
_BUDGET_CACHE_TTL = 300  # 5 minutes — check_budget result cached in-process
_budget_cache: tuple[float, object] | None = None  # (ts, BudgetCheck)
_SPEND_CACHE_TTL = 60   # 1 minute — routing history is local, cheap to read
_spend_cache: tuple[float, dict] | None = None  # (ts, {provider: cost_usd})

# Git info memoisation
_GIT_CACHE: dict[tuple[str, int], tuple[float, tuple[Optional[str], bool]]] = {}
_GIT_TTL = 2.0  # seconds


def _boot_time() -> float:
    """Return system boot time as a Unix timestamp."""
    try:
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        return time.time() - uptime
    except (OSError, ValueError):
        return 0.0


def read_state_file(path: Path, max_age: float = _STATE_MAX_AGE) -> dict:
    """Read a state JSON file, returning {} if missing, stale, or pre-boot."""
    if not path.exists():
        return {}
    try:
        mtime = path.stat().st_mtime
        # Reject files older than max_age
        if time.time() - mtime > max_age:
            return {}
        # Reject files written before the last system boot — pre-reboot state is always stale
        if mtime < _boot_time():
            return {}
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

def get_git_info(cwd: str) -> tuple[Optional[str], bool]:
    """Memoised: return (branch, dirty). Cache expires every 2s or when .git/HEAD mtime changes."""
    # Get .git/HEAD mtime as cache invalidation sentinel
    try:
        head_mtime = int(Path(cwd, ".git", "HEAD").stat().st_mtime)
    except OSError:
        head_mtime = 0

    key = (cwd, head_mtime)
    now = time.time()
    cached = _GIT_CACHE.get(key)

    # Return cached result if within TTL
    if cached is not None and now - cached[0] < _GIT_TTL:
        return cached[1]

    # Cache miss or expired — recompute
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=cwd, timeout=2,
        ).stdout.strip() or None
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=cwd, timeout=2,
        ).stdout.strip())
        result = (branch, dirty)
    except (subprocess.SubprocessError, FileNotFoundError):
        result = (None, False)

    _GIT_CACHE[key] = (now, result)
    return result

def build_context(cwd: str = ".") -> HudRenderContext:
    """Assemble a ``HudRenderContext`` from on-disk state files under ``cwd/.mullet/state``."""
    state_dir = Path(cwd) / ".mullet" / "state"
    ralph = read_state_file(state_dir / "ralph.json")
    autopilot = read_state_file(state_dir / "autopilot.json")
    ultrawork = read_state_file(state_dir / "ultrawork.json")
    session = read_state_file(state_dir / "session.json")
    tokens = read_state_file(state_dir / "tokens.json")
    agents = read_state_file(_user_state_dir() / "state" / "agents.json")
    context = read_state_file(state_dir / "context.json")
    mode = read_state_file(state_dir / "mode.json")
    model = read_state_file(state_dir / "model.json")
    todos = read_state_file(state_dir / "todos.json")
    branch, dirty = get_git_info(cwd)

    # Session duration — handle both "duration" (seconds) and "duration_minutes"
    duration_raw = session.get("duration_minutes") or session.get("duration", 0)
    # If value > 1000, it's likely seconds — convert to minutes
    session_minutes = duration_raw / 60 if duration_raw > 1000 else duration_raw

    # Total tokens — check tokens.json then session.json
    total_tokens = tokens.get("usage", 0) or tokens.get("total_tokens", 0) or session.get("total_tokens", 0)

    # Ralph — check both state file formats
    ralph_active = ralph.get("active", False) or bool(ralph.get("prd"))
    ralph_iter = ralph.get("iteration", mode.get("iteration", 0))
    ralph_max = ralph.get("max_iterations", mode.get("total_iterations", 0))

    # Wire in Anthropic usage stats + reset timer (cached 5 min in-process)
    global _budget_cache, _spend_cache
    five_hour_pct = None
    weekly_pct = None
    rate_limit_reset_s = 0
    pressure_5h = 0.0
    pressure_7d = 0.0
    provider_costs_5h: dict = {}
    budget_status = ""
    try:
        now_ts = time.time()
        if _budget_cache is None or now_ts - _budget_cache[0] > _BUDGET_CACHE_TTL:
            from mullet.hud.guard import check_budget
            _budget_cache = (now_ts, check_budget())
        budget = _budget_cache[1]
        five_hour_pct = budget.five_hour_pct
        weekly_pct = budget.weekly_pct
        pressure_5h = budget.session_pressure
        pressure_7d = budget.weekly_pressure
        budget_status = budget.status.value
        from mullet.hud.usage import parse_resets_at
        reset_dt = parse_resets_at(budget.reset_time)
        if reset_dt is not None:
            delta = reset_dt - datetime.now(timezone.utc)
            rate_limit_reset_s = max(0, int(delta.total_seconds()))
    except Exception as _exc:
        logger.warning("hud_state: failed to fetch budget: %s", _exc)

    # Per-provider spend in last 5h (cached 1 min — local file read)
    try:
        now_ts = time.time()
        if _spend_cache is None or now_ts - _spend_cache[0] > _SPEND_CACHE_TTL:
            from mullet.core.routing_history import provider_spend_5h
            _spend_cache = (now_ts, provider_spend_5h(Path(cwd)))
        provider_costs_5h = _spend_cache[1]
    except Exception as _exc:
        logger.warning("hud_state: failed to fetch provider spend: %s", _exc)

    running_agents_list = agents.get("agents", [])
    agent_count = len(running_agents_list) if running_agents_list else agents.get("active_count", 0)

    return HudRenderContext(
        cwd=cwd,
        git_branch=branch or "",
        git_dirty=dirty,
        model_name=model.get("name", ""),
        # Values 0.0-1.0 are fractions; multiply to get percentage. Values >1.0 are already percentages.
        context_pct=context.get("percent", 0.0) * 100 if context.get("percent", 0.0) <= 1.0 else context.get("percent", 0.0),
        agent_count=agent_count,
        ralph_active=ralph_active,
        ralph_iter=ralph_iter,
        ralph_max=ralph_max,
        autopilot_active=autopilot.get("active", False) or bool(autopilot.get("phase")),
        autopilot_phase=autopilot.get("phase", ""),
        ultrawork_active=ultrawork.get("active", False),
        ultrawork_tasks=ultrawork.get("task_count", 0),
        todos_done=todos.get("completed", 0),
        todos_total=todos.get("total", 0),
        session_minutes=session_minutes,
        total_tokens=total_tokens,
        five_hour_pct=five_hour_pct,
        weekly_pct=weekly_pct,
        rate_limit_reset_s=rate_limit_reset_s,
        pressure_5h=pressure_5h,
        pressure_7d=pressure_7d,
        provider_costs_5h=provider_costs_5h,
        budget_status=budget_status,
        running_agents=running_agents_list,
    )
