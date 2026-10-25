"""``mullet config show``: every active setting, its default and where it came from.

Values come from the real loaders (``mullet.jobs.ladder.load_config`` and
``mullet.config.defaults.load_config``); the user's raw config file only says
which keys the user set. Secret values are never printed: a setting whose name
contains key, token, secret or password shows ``***``, and a provider's API key
shows only ``set`` or ``unset``.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

SECTIONS = ("ladder", "budget", "delegation", "routing", "providers", "skills", "other", "unrecognised")
MASK = "***"
_SECRET_WORDS = ("key", "token", "secret", "password")
LEGACY_BUDGET_KEYS = ("per_task", "per_session", "daily")
LEGACY_NOTE = "legacy delegation path, not used by gated jobs"

# Top-level keys the loaders and the validator know about.
KNOWN_TOP_LEVEL = frozenset({
    "providers", "routing", "budget", "max_iterations", "server", "knowledge", "toolsets", "plugins",
    "skills", "default_toolsets", "ladder", "opencode_timeout_seconds", "delegation",
})
_OTHER_SCALARS = ("max_iterations", "default_toolsets", "opencode_timeout_seconds")
_OTHER_BLOCKS = ("server", "knowledge", "toolsets", "plugins")

# LadderConfig fields whose key in the ``ladder:`` block has a different name.
_LADDER_YAML_KEY = {
    "kimi_model": "kimi",
    "openrouter_models": "openrouter",
    "free_model": "free",
    "local_model": "local",
    "local_url": "local",
}
_LADDER_INTERNAL = frozenset({"configured", "source", "problems"})
_SKILLS_KEYS = {"external_dirs": "skills_dirs", "disabled": "disabled_skills"}

Row = dict[str, Any]


def is_secret_name(name: str) -> bool:
    """True when a setting's name says its value is a secret."""
    lowered = name.lower()
    return any(word in lowered for word in _SECRET_WORDS)


def _safe(name: str, value: Any) -> Any:
    """The value to print: masked when the name is secret, and inside mappings by key name."""
    if is_secret_name(name.rsplit(".", 1)[-1]):
        return MASK
    if isinstance(value, dict):
        return {k: _safe(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(name, v) for v in value]
    return value


def _row(name: str, value: Any, default: Any, source: str, note: str = "") -> Row:
    return {"setting": name, "value": _safe(name, value), "default": _safe(name, default),
            "source": source, "note": note}


def _flatten(prefix: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict) and value:
        out: list[tuple[str, Any]] = []
        for key, sub in value.items():
            out += _flatten(f"{prefix}.{key}", sub)
        return out
    return [(prefix, value)]


def _plain(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def _block(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    return value if isinstance(value, dict) else {}


def _ladder_rows(raw: dict[str, Any], path: Path, problems: list[str], unknown: list[str]) -> list[Row]:
    from mullet.jobs.ladder import LadderConfig, load_config

    active = load_config(path)
    default = LadderConfig()
    problems.extend(active.problems)
    block = _block(raw, "ladder")
    names = [f.name for f in fields(LadderConfig) if f.name not in _LADDER_INTERNAL]
    valid = {_LADDER_YAML_KEY.get(n, n) for n in names}
    unknown += [f"ladder.{k}" for k in block if k not in valid]
    return [
        _row(f"ladder.{n}", _plain(getattr(active, n)), _plain(getattr(default, n)),
             str(path) if _LADDER_YAML_KEY.get(n, n) in block else "default")
        for n in names
    ]


def _routing_rows(raw: dict[str, Any], config: Any, path: Path, unknown: list[str]) -> list[Row]:
    from mullet.config.defaults import RoutingConfig

    block = _block(raw, "routing")
    unknown += [f"routing.{k}" for k in block if k not in {f.name for f in fields(RoutingConfig)}]
    default = RoutingConfig()
    rows = []
    for f in fields(RoutingConfig):
        value, base = getattr(config.routing, f.name), getattr(default, f.name)
        if isinstance(base, dict):
            user = _block(block, f.name)
            for key in list(base) + [k for k in user if k not in base]:
                rows.append(_row(f"routing.{f.name}.{key}", value.get(key), base.get(key),
                                 str(path) if key in user else "default"))
        else:
            rows.append(_row(f"routing.{f.name}", value, base, str(path) if f.name in block else "default"))
    return rows


def _budget_rows(raw: dict[str, Any], path: Path) -> list[Row]:
    block = _block(raw, "budget")
    rows = []
    for name, value in _flatten("budget", block) if block else []:
        legacy = name.removeprefix("budget.") in LEGACY_BUDGET_KEYS
        rows.append(_row(name, value, None, str(path), LEGACY_NOTE if legacy else ""))
    shown = {row["setting"] for row in rows}
    for window in ("daily", "weekly"):
        name = f"budget.envelope.{window}"
        if name not in shown:
            rows.append(_row(name, None, None, "default", "no limit: set dollars per UTC day or week across all jobs"))
    return rows


def _delegation_rows(raw: dict[str, Any], path: Path) -> list[Row]:
    """Which toolsets each provider may be offered: the shipped default, then the user's entries."""
    from mullet.core.toolset_policy import SAFE_DEFAULT, TRUSTED_PROVIDERS, parse_policy

    block = _block(raw, "delegation")
    given = block.get("toolsets") if isinstance(block.get("toolsets"), dict) else {}
    policy = parse_policy(given)
    rows = [_row("delegation.toolsets.default", policy.default, list(SAFE_DEFAULT),
                 str(path) if "default" in given else "default",
                 "any provider not listed below; `all` is every toolset")]
    for name in sorted(policy.providers):
        from_user = name in {str(k).lower() for k in given}
        rows.append(_row(f"delegation.toolsets.{name}", policy.providers[name],
                         "all" if name in TRUSTED_PROVIDERS else None,
                         str(path) if from_user else "default",
                         "Claude's own provider" if name in TRUSTED_PROVIDERS else ""))
    return rows


def _provider_rows(raw: dict[str, Any], config: Any, path: Path) -> list[Row]:
    block = _block(raw, "providers")
    rows = []
    for name, provider in config.providers.items():
        entry = _block(block, name)
        prefix = f"providers.{name}"
        rows.append(_row(f"{prefix}.type", provider.type, "openai_compatible",
                         str(path) if "type" in entry else "default"))
        rows.append(_row(f"{prefix}.base_url", provider.base_url, "",
                         str(path) if "base_url" in entry else "default"))
        env_name = provider.api_key_env
        has_key = bool(entry.get("api_key")) or bool(env_name and os.environ.get(env_name))
        user_set = "api_key" in entry or "api_key_env" in entry
        rows.append({"setting": f"{prefix}.api_key", "value": "set" if has_key else "unset",
                     "default": "unset", "source": str(path) if user_set else "default", "note": ""})
    return rows


def _skills_rows(raw: dict[str, Any], config: Any, path: Path, unknown: list[str]) -> list[Row]:
    block = _block(raw, "skills")
    unknown += [f"skills.{k}" for k in block if k not in _SKILLS_KEYS]
    return [_row(f"skills.{key}", getattr(config, attr), [], str(path) if key in block else "default")
            for key, attr in _SKILLS_KEYS.items()]


def _other_rows(raw: dict[str, Any], config: Any, path: Path) -> list[Row]:
    from mullet.config.defaults import MulletConfig

    default = MulletConfig()
    rows = [_row(key, getattr(config, key), getattr(default, key), str(path) if key in raw else "default")
            for key in _OTHER_SCALARS]
    for key in _OTHER_BLOCKS:
        if key in raw:
            rows += [_row(n, v, None, str(path)) for n, v in _flatten(key, raw[key])]
    return rows


def collect(path: Path | None = None) -> dict[str, Any]:
    """Every setting as ``{"config_file", "exists", "sections", "unrecognised", "problems"}``."""
    from mullet.config.defaults import MulletConfig, load_config
    from mullet.config.loader import load_from_yaml
    from mullet.core.paths import config_file

    target = path or config_file("config.yaml")
    exists = target.exists()
    raw: Any = load_from_yaml(str(target)) if exists else {}
    raw = raw if isinstance(raw, dict) else {}
    config = load_config(str(target)) if exists else MulletConfig()
    problems: list[str] = []
    unknown: list[str] = [str(k) for k in raw if k not in KNOWN_TOP_LEVEL and not str(k).startswith("#")]
    sections = {
        "ladder": _ladder_rows(raw, target, problems, unknown),
        "budget": _budget_rows(raw, target),
        "delegation": _delegation_rows(raw, target),
        "routing": _routing_rows(raw, config, target, unknown),
        "providers": _provider_rows(raw, config, target),
        "skills": _skills_rows(raw, config, target, unknown),
        "other": _other_rows(raw, config, target),
    }
    return {"config_file": str(target), "exists": exists, "sections": sections,
            "unrecognised": sorted(unknown), "problems": problems}


def _text(value: Any) -> str:
    return "-" if value is None else json.dumps(value, default=str)


def render(data: dict[str, Any], section: str | None = None) -> str:
    """The plain-text table for `collect()` output, optionally limited to one section."""
    missing = "" if data["exists"] else " (not found; all defaults)"
    lines = [f"Config file: {data['config_file']}{missing}"]
    for name, rows in data["sections"].items():
        if section not in (None, name) or not rows:
            continue
        table = [("setting", "value", "default", "source")] + [
            (r["setting"], _text(r["value"]), _text(r["default"]),
             r["source"] + (f" ({r['note']})" if r["note"] else "")) for r in rows]
        widths = [max(len(row[i]) for row in table) for i in range(3)]
        lines += ["", f"[{name}]"]
        for row in table:
            lines.append("  ".join(row[i].ljust(widths[i]) for i in range(3)) + "  " + row[3])
    if section in (None, "unrecognised") and data["unrecognised"]:
        lines += ["", "[unrecognised]  (not read by Mullet; check for typos)"]
        lines += [f"  {key}" for key in data["unrecognised"]]
    if section in (None, "ladder") and data["problems"]:
        lines += ["", "[problems]"] + [f"  {p}" for p in data["problems"]]
    return "\n".join(lines)


def cmd_show(args: list[str]) -> int:
    """Handle ``mullet config show [--json] [section]``."""
    as_json = "--json" in args
    rest = [a for a in args if a != "--json"]
    if len(rest) > 1 or (rest and rest[0] not in SECTIONS):
        print(f"Usage: mullet config show [--json] [{'|'.join(SECTIONS)}]", file=sys.stderr)
        return 1
    section = rest[0] if rest else None
    data = collect()
    if as_json:
        if section:
            data["sections"] = {k: v for k, v in data["sections"].items() if k == section}
            if section != "unrecognised":
                data["unrecognised"] = []
            if section != "ladder":
                data["problems"] = []
        print(json.dumps(data, indent=2, default=str))
    else:
        print(render(data, section))
    return 0
