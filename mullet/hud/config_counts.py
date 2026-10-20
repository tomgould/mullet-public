"""Config file counting for HUD: MCP servers, hooks, rules, CLAUDE.md files."""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CONFIG_CACHE: dict = {}


@dataclass
class ConfigCounts:
    """Counts of configuration items."""

    mcp_count: int = 0
    hooks_count: int = 0
    rules_count: int = 0
    claude_md_count: int = 0
    output_style: Optional[str] = None


def _sentinel_key(path: Path) -> tuple:
    """Return (mtime, size) sentinel for a file."""
    try:
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None


def _read_json(path: Path) -> dict:
    """Read and parse JSON file, return {} on any error."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}


def _get_mcp_servers(path: Path) -> set[str]:
    """Extract MCP server names from mcpServers dict."""
    cfg = _read_json(path)
    mcp_servers = cfg.get("mcpServers", {})
    if isinstance(mcp_servers, dict):
        return set(mcp_servers.keys())
    return set()


def _get_disabled_mcp_servers(path: Path, key: str) -> set[str]:
    """Extract disabled MCP server names from config array."""
    cfg = _read_json(path)
    disabled = cfg.get(key, [])
    if isinstance(disabled, list):
        return {name for name in disabled if isinstance(name, str)}
    return set()


def _count_hooks(path: Path) -> int:
    """Count number of top-level keys under config['hooks']."""
    cfg = _read_json(path)
    hooks = cfg.get("hooks", {})
    if isinstance(hooks, dict):
        return len(hooks)
    return 0


def _count_rules_recursive(dir_path: Path) -> int:
    """Recursively count .md files under a directory."""
    if not dir_path.is_dir():
        return 0
    count = 0
    try:
        for item in dir_path.rglob("*.md"):
            if item.is_file():
                count += 1
    except OSError:
        pass
    return count


def _get_output_style(path: Path) -> Optional[str]:
    """Read outputStyle string from config."""
    cfg = _read_json(path)
    style = cfg.get("outputStyle", "")
    if isinstance(style, str):
        style = style.strip()
        return style if style else None
    return None


def _paths_equal(a: Path, b: Path) -> bool:
    """Check if two paths refer to the same location."""
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def get_config_counts(cwd: str, home_dir: Optional[str] = None) -> ConfigCounts:
    """Return MCP / hooks / rules / CLAUDE.md counts for the given cwd.

    Reads the user-level Claude config (~/.claude/) merged with
    the project-level config (<cwd>/.claude/).
    MCP servers are deduped across scopes and minus disabledMcpServers.
    Uses an in-module sentinel cache keyed on (file path, mtime, size).
    All IO errors return a zeroed ConfigCounts; never raise.

    Args:
        cwd: Current working directory (project root)
        home_dir: Optional override for home directory (for testing)
    """
    home = Path(home_dir) if home_dir else Path.home()
    cwd_path = Path(cwd)
    claude_dir = home / ".claude"
    project_claude_dir = cwd_path / ".claude"

    # Build cache key from sentinels
    sentinel_paths = [
        claude_dir / "CLAUDE.md",
        claude_dir / "rules",
        claude_dir / "settings.json",
        claude_dir / "settings.local.json",
        home / ".claude.json",
        cwd_path / ".mcp.json",
        cwd_path / "CLAUDE.md",
        cwd_path / ".claude" / "CLAUDE.md",
        cwd_path / ".claude" / "settings.json",
        cwd_path / ".claude" / "settings.local.json",
    ]

    cache_key = tuple(_sentinel_key(p) for p in sentinel_paths)
    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]

    # === Count MCPs ===
    user_mcp_servers = set()
    user_mcp_servers.update(_get_mcp_servers(claude_dir / "settings.json"))
    user_mcp_servers.update(_get_mcp_servers(claude_dir / "settings.local.json"))
    user_mcp_servers.update(_get_mcp_servers(home / ".claude.json"))

    # Remove disabled user MCPs
    disabled_user = _get_disabled_mcp_servers(home / ".claude.json", "disabledMcpServers")
    user_mcp_servers -= disabled_user

    project_mcp_servers = set()
    project_claude_overlaps_user = _paths_equal(project_claude_dir, claude_dir)

    if not project_claude_overlaps_user:
        project_mcp_servers.update(_get_mcp_servers(project_claude_dir / "settings.json"))
        project_mcp_servers.update(_get_mcp_servers(project_claude_dir / "settings.local.json"))

    # .mcp.json servers
    mcp_json_servers = _get_mcp_servers(cwd_path / ".mcp.json")
    disabled_mcp_json = _get_disabled_mcp_servers(
        cwd_path / ".claude" / "settings.local.json", "disabledMcpjsonServers"
    )
    mcp_json_servers -= disabled_mcp_json
    project_mcp_servers.update(mcp_json_servers)

    mcp_count = len(user_mcp_servers) + len(project_mcp_servers)

    # === Count hooks ===
    hooks_count = 0
    hooks_count += _count_hooks(claude_dir / "settings.json")
    hooks_count += _count_hooks(claude_dir / "settings.local.json")

    if not project_claude_overlaps_user:
        hooks_count += _count_hooks(project_claude_dir / "settings.json")

    hooks_count += _count_hooks(project_claude_dir / "settings.local.json")

    # === Count rules ===
    rules_count = _count_rules_recursive(claude_dir / "rules")
    if not project_claude_overlaps_user:
        rules_count += _count_rules_recursive(project_claude_dir / "rules")

    # === Count CLAUDE.md ===
    claude_md_count = 0
    if (claude_dir / "CLAUDE.md").exists():
        claude_md_count += 1
    if (cwd_path / "CLAUDE.md").exists():
        claude_md_count += 1
    if not project_claude_overlaps_user and (project_claude_dir / "CLAUDE.md").exists():
        claude_md_count += 1

    # === Get output style (prefer later files) ===
    output_style = None
    output_style = _get_output_style(claude_dir / "settings.json") or output_style
    output_style = _get_output_style(claude_dir / "settings.local.json") or output_style

    if not project_claude_overlaps_user:
        output_style = _get_output_style(project_claude_dir / "settings.json") or output_style

    output_style = _get_output_style(project_claude_dir / "settings.local.json") or output_style

    result = ConfigCounts(
        mcp_count=mcp_count,
        hooks_count=hooks_count,
        rules_count=rules_count,
        claude_md_count=claude_md_count,
        output_style=output_style,
    )

    _CONFIG_CACHE[cache_key] = result
    return result
