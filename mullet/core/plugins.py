"""Plugin system — extend Mullet with custom tools and providers.

Plugins are Python packages that register tools, providers, or skills
via entry points. They can be installed with pip and discovered automatically.

Usage:
  pip install mullet-plugin-docker
  mullet --tools  # Shows docker tools

Plugin structure:
  mullet_plugin_docker/
  ├── __init__.py         # register(registry) function
  └── tools.py            # Tool implementations

Entry point in pyproject.toml:
  [project.entry-points."mullet.plugins"]
  docker = "mullet_plugin_docker:register"
"""

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Registry of loaded plugins
_loaded_plugins: dict[str, dict] = {}


def discover_plugins() -> list[dict]:
    """Discover installed plugins via entry points."""
    plugins = []

    try:
        if hasattr(__import__("importlib.metadata", fromlist=["entry_points"]), "entry_points"):
            from importlib.metadata import entry_points

            eps = entry_points()
            # Python 3.12+ returns SelectableGroups, older returns dict
            if hasattr(eps, "select"):
                mullet_eps = eps.select(group="mullet.plugins")
            elif isinstance(eps, dict):
                mullet_eps = eps.get("mullet.plugins", [])
            else:
                mullet_eps = []

            for ep in mullet_eps:
                plugins.append({
                    "name": ep.name,
                    "module": ep.value,
                    "entry_point": ep,
                })
    except Exception as e:
        logger.debug("Plugin discovery failed: %s", e)

    return plugins


def load_plugins(tool_registry=None, provider_registry=None) -> int:
    """Load all discovered plugins and register their extensions.

    Loads from two sources:
    1. Entry points (pip-installed packages)
    2. Directory plugins (plugins/*.py in the data directory and .mullet/plugins/*.py)

    Returns the number of plugins loaded.
    """
    loaded = 0

    # Source 1: Entry point plugins
    plugins = discover_plugins()
    for plugin in plugins:
        try:
            register_fn = plugin["entry_point"].load()
            if callable(register_fn):
                register_fn(
                    tool_registry=tool_registry,
                    provider_registry=provider_registry,
                )
                _loaded_plugins[plugin["name"]] = plugin
                loaded += 1
                logger.info("Loaded plugin: %s", plugin["name"])
        except Exception as e:
            logger.warning("Failed to load plugin %s: %s", plugin["name"], e)

    # Source 2: Directory plugins
    loaded += _load_directory_plugins(tool_registry, provider_registry)

    return loaded


def _load_directory_plugins(tool_registry=None, provider_registry=None) -> int:
    """Load plugins from plugins/ in the data directory and the project's .mullet/plugins/.

    Each .py file should have a register(tool_registry=None, provider_registry=None) function.
    """
    import importlib.util
    from pathlib import Path

    from mullet.core.paths import data_dir

    plugin_dirs = [
        data_dir() / "plugins",
        Path(".mullet") / "plugins",
    ]

    loaded = 0
    for plugin_dir in plugin_dirs:
        if not plugin_dir.exists():
            continue

        for plugin_file in sorted(plugin_dir.glob("*.py")):
            if plugin_file.name.startswith("_"):
                continue

            name = plugin_file.stem
            if name in _loaded_plugins:
                continue  # Already loaded via entry point

            try:
                spec = importlib.util.spec_from_file_location(f"mullet_plugin_{name}", plugin_file)
                if not spec or not spec.loader:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "register") and callable(module.register):
                    module.register(
                        tool_registry=tool_registry,
                        provider_registry=provider_registry,
                    )
                    _loaded_plugins[name] = {"name": name, "module": str(plugin_file), "source": "directory"}
                    loaded += 1
                    logger.info("Loaded directory plugin: %s from %s", name, plugin_file)
            except Exception as e:
                logger.warning("Failed to load plugin %s: %s", name, e)

    return loaded


def list_plugins() -> list[dict]:
    """List all loaded plugins."""
    return [
        {"name": name, "module": info.get("module", ""), "source": info.get("source", "entry-point")}
        for name, info in _loaded_plugins.items()
    ]


# --- Plugin helpers for plugin authors ---

def register_tool(registry, name: str, func: Callable, schema: dict, toolset: str = "plugin"):
    """Helper for plugins to register tools."""
    registry.register(name=name, func=func, schema=schema, toolset=toolset)


def register_provider(registry, name: str, provider):
    """Helper for plugins to register providers."""
    if hasattr(registry, "providers"):
        registry.providers[name] = provider


# ================================================================
# Plugin System Depth — extension points, discovery, per-client config
# ================================================================

_extension_points: dict[str, type] = {}


def register_extension_point(name: str, interface: type) -> None:
    """Register a named extension point with its interface type."""
    _extension_points[name] = interface


def get_extension_point(name: str) -> type | None:
    """Get the interface type for an extension point."""
    return _extension_points.get(name)


def scan_plugin_dirs(search_dirs: list[str] | None = None) -> list[str]:
    """Auto-discover plugin .py files from search directories."""
    import importlib.util
    from pathlib import Path

    from mullet.core.paths import data_dir

    dirs = search_dirs or [
        str(data_dir() / "plugins"),
        str(Path.cwd() / ".mullet" / "plugins"),
    ]
    found = []
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in p.glob("*.py"):
            if f.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(f.stem, f)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    found.append(str(f))
            except Exception as _exc:
                logger.warning("plugins: failed to scan directory plugin: %s", _exc)
    return found


def per_client_plugin_config(client_name: str) -> dict:
    """Load plugin configuration for a specific client from config.yaml."""
    import yaml

    from mullet.core.env import get_pools_dir

    clients_dir = get_pools_dir()
    config_path = clients_dir / client_name / "config.yaml"
    if not config_path.exists():
        config_path = clients_dir / client_name / "client.yaml"
    if not config_path.exists():
        return {}
    try:
        cfg = yaml.safe_load(config_path.read_text())
        return cfg.get("plugins", {})
    except Exception:
        return {}
