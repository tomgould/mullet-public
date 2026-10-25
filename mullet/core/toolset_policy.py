"""Which toolsets a delegated model may use, per provider, set in the Mullet config file.

Tools like ``terminal`` and ``file`` act on the user's machine with the user's
rights, so a model that reads a hostile web page or file could be talked into
misusing them. Which providers get which toolsets is the user's call::

    delegation:
      toolsets:
        default: [web, browser, session, vision]   # every provider not listed below
        kimi: all                                  # `all` is every toolset
        local: [terminal, file, web, code]

With no ``delegation:`` block, Claude's own providers get every toolset and any
other provider gets the safe default. Nothing is removed from Mullet: a toolset
is only left out of the list a given provider is offered.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

ALL = "all"
# Claude's own providers, reached through the user's own login or key.
TRUSTED_PROVIDERS = ("anthropic", "claude-cli", "claude_max")
# Offered to any other provider until the user says otherwise: nothing that runs commands,
# runs code, or reads and writes the user's files.
SAFE_DEFAULT = ("web", "browser", "session", "vision", "trove", "lsp")
# Every toolset name the delegation entry points offer, so `all` can be spelled out.
KNOWN_TOOLSETS = ("terminal", "file", "web", "code", "browser", "delegation", "session", "git", "testing",
                  "vision", "docker", "lsp", "trove")
# Toolsets that act on the user's machine; `mullet doctor` warns when an untrusted provider has them.
RISKY = ("terminal", "file", "code", "docker", "git", "testing")


@dataclass
class ToolsetPolicy:
    """The toolsets each provider may be offered. A value is ``ALL`` or a list of toolset names."""

    default: Any = field(default_factory=lambda: list(SAFE_DEFAULT))
    providers: dict[str, Any] = field(default_factory=lambda: {p: ALL for p in TRUSTED_PROVIDERS})
    problems: list[str] = field(default_factory=list)

    def allowed(self, provider: str | None) -> Any:
        """``ALL`` or the list of toolsets `provider` may be offered."""
        name = provider.lower() if isinstance(provider, str) else ""
        return self.providers.get(name, self.default)


def _entry(value: Any, where: str, problems: list[str]) -> Any:
    if isinstance(value, str) and value.strip().lower() == ALL:
        return ALL
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return [v.strip() for v in value if v.strip()]
    problems.append(f"{where}: expected `all` or a list of toolset names; ignored")
    return None


def parse_policy(data: Any, source: str = "delegation.toolsets") -> ToolsetPolicy:
    """Read the ``delegation.toolsets`` mapping; anything unreadable keeps the shipped default."""
    policy = ToolsetPolicy()
    if data is None:
        return policy
    if not isinstance(data, dict):
        policy.problems.append(f"{source}: expected a mapping of provider to toolsets; ignored")
        return policy
    for name, value in data.items():
        entry = _entry(value, f"{source}.{name}", policy.problems)
        if entry is None:
            continue
        if str(name).lower() == "default":
            policy.default = entry
        else:
            policy.providers[str(name).lower()] = entry
    return policy


def load_policy(path: Path | None = None) -> ToolsetPolicy:
    """The policy from the user's Mullet config file; the shipped default when there is none."""
    if path is None:
        from mullet.core.paths import config_file

        path = config_file("config.yaml")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return ToolsetPolicy()
    except (OSError, yaml.YAMLError) as exc:
        policy = ToolsetPolicy()
        policy.problems.append(f"{path}: unreadable ({exc}); the shipped toolset default is used")
        return policy
    block = data.get("delegation") if isinstance(data, dict) else None
    return parse_policy(block.get("toolsets") if isinstance(block, dict) else None, f"{path} delegation.toolsets")


def apply_policy(toolsets: Iterable[str], provider: str | None, policy: ToolsetPolicy | None = None) -> list[str]:
    """`toolsets` cut down to what `provider` may be offered, in their original order."""
    policy = policy or load_policy()
    allowed = policy.allowed(provider)
    wanted = list(toolsets)
    if allowed == ALL:
        return wanted
    kept = [t for t in wanted if t in allowed]
    if len(kept) != len(wanted):
        logger.info("toolsets for %s limited to %s (delegation.toolsets in the Mullet config)", provider, kept)
    return kept


def risky_grants(policy: ToolsetPolicy, known: Iterable[str]) -> list[tuple[str, list[str]]]:
    """(provider, risky toolsets) for each non-Claude provider the policy lets act on the machine.

    `known` is every toolset that exists, so ``all`` can be spelled out. The entry
    named ``default`` stands for any provider not listed.
    """
    known = list(known)
    found = []
    for name, allowed in [("default", policy.default), *policy.providers.items()]:
        if name in TRUSTED_PROVIDERS:
            continue
        granted = [t for t in (known if allowed == ALL else allowed) if t in RISKY]
        if granted:
            found.append((name, granted))
    return found
