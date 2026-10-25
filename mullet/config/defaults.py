"""Default configuration for Mullet."""

from dataclasses import dataclass, field
from pathlib import Path

from mullet.core import paths


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""
    name: str
    type: str = "openai_compatible"
    base_url: str = ""
    api_key_env: str = ""
    models: dict = field(default_factory=dict)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


SPEED_MODES = ("economy", "balanced", "fast", "max")
"""Speed modes controlling provider routing aggressiveness.

economy  — local LLM first for everything (default, cheapest)
balanced — skip local for M/L tasks, use Kimi/Gemini directly
fast     — always use Kimi K2 (skip local, still cheap)
max      — use Sonnet for all tasks (fastest, most capable, expensive)
"""


@dataclass
class RoutingConfig:
    """Configuration for the intelligent router."""
    enabled: bool = True
    default_tier: str = "STANDARD"
    local_only: bool = False
    speed_mode: str = "economy"  # economy | balanced | fast | max
    thresholds: dict = field(default_factory=lambda: {"low_max": 20, "medium_max": 50})
    budget: dict = field(default_factory=lambda: {
        "per_task_max_tokens": 100_000,
        "session_max_tokens": 1_000_000,
        "daily_max_dollars": 10.0,
    })


@dataclass
class MulletConfig:
    """Top-level Mullet configuration."""
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    skills_dirs: list[str] = field(default_factory=list)
    disabled_skills: list[str] = field(default_factory=list)
    max_iterations: int = 90
    default_toolsets: list[str] = field(default_factory=lambda: ["terminal", "file", "web"])
    docs_sources: list[str] = field(default_factory=lambda: ["context7", "trawl", "web"])
    opencode_timeout_seconds: int = 300

    @staticmethod
    def user_config_dir() -> Path:
        """Return the user-level config directory (see ``mullet.core.paths``)."""
        return paths.config_dir()

    @staticmethod
    def user_skills_dir() -> Path:
        """Return the directory where user-defined skills live."""
        return paths.data_dir() / "skills"


def load_config(path: str | None = None) -> MulletConfig:
    """Load configuration from YAML file or defaults."""
    if path and Path(path).exists():
        return _load_yaml_config(path)

    # Try default locations
    for candidate in ["mullet.yaml", str(paths.config_file("config.yaml"))]:
        if Path(candidate).exists():
            return _load_yaml_config(candidate)

    return MulletConfig()


def _load_yaml_config(path: str) -> MulletConfig:
    """Parse a mullet.yaml file into MulletConfig."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    config = MulletConfig()

    if "providers" in data:
        for name, pdata in data["providers"].items():
            config.providers[name] = ProviderConfig(name=name, **{
                k: v for k, v in pdata.items() if k in ProviderConfig.__dataclass_fields__
            })

    if "routing" in data:
        rdata = data["routing"]
        config.routing = RoutingConfig(**{
            k: v for k, v in rdata.items() if k in RoutingConfig.__dataclass_fields__
        })

    if "skills" in data:
        sdata = data["skills"]
        config.skills_dirs = sdata.get("external_dirs", [])
        config.disabled_skills = sdata.get("disabled", [])

    config.max_iterations = data.get("max_iterations", 90)
    config.default_toolsets = data.get("default_toolsets", ["terminal", "file", "web"])
    config.opencode_timeout_seconds = data.get("opencode_timeout_seconds", 300)

    return config
