"""Provider-preset system for Mullet.

Defines a ``Preset`` dataclass that captures routing preferences
(allowed providers, speed mode, opencode model) and a ``PresetManager``
that persists presets to ``presets.yaml`` under the Mullet config
directory.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ANTHROPIC_MODEL_MARKERS = ("claude", "anthropic", "sonnet", "haiku", "opus", "fable")
"""Substrings that identify Anthropic models. Used to reject them as opencode targets."""

_DEFAULT_ACTIVE = "balanced-mix"


@dataclass
class Preset:
    """A single provider preset.

    Attributes:
        name: Stable slug (e.g. ``local-only``).
        label: Human-readable display label.
        allowed_providers: Concrete provider names allowed by this preset,
            or ``None`` for unrestricted budget-aware routing.
        speed_mode: Routing speed mode (economy, balanced, fast, max, premium).
        opencode_model: Model identifier for opencode edits, or ``None``
            to disable opencode for this preset.
        native_only: When ``True``, delegation is disabled entirely and
            Claude Code handles everything natively.
        description: Optional longer description.
    """

    name: str
    label: str
    allowed_providers: list[str] | None
    speed_mode: str
    opencode_model: str | None
    native_only: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if self.opencode_model is not None:
            lowered = self.opencode_model.lower()
            if any(marker in lowered for marker in ANTHROPIC_MODEL_MARKERS):
                raise ValueError(
                    f"Anthropic models are not allowed as opencode_model: {self.opencode_model!r}"
                )

    def mcp_allowlist(self) -> set[str] | None:
        """Return the concrete provider allowlist for the MCP path.

        ``None`` means no restriction.
        """
        if self.allowed_providers is None:
            return None
        return set(self.allowed_providers)

    def router_allowlist(self) -> frozenset[str] | None:
        """Return the provider allowlist for the core router.

        ``None`` means unrestricted budget-aware routing. When
        ``allowed_providers`` is set and contains ``kimi`` or ``gemini``,
        the alias ``cheap_cloud`` is added because the router
        routes the cheap tier via that alias.
        """
        if self.allowed_providers is None:
            return None
        result = set(self.allowed_providers)
        if "kimi" in result or "gemini" in result:
            result.add("cheap_cloud")
        return frozenset(result)


BUILTIN_PRESETS: list[Preset] = [
    Preset(
        name="balanced-mix",
        label="Balanced mix (cost-aware default)",
        allowed_providers=None,
        speed_mode="balanced",
        opencode_model="moonshot/kimi-k2.6",
        native_only=False,
        description="Local for simple, cloud for hard — budget-aware routing.",
    ),
    Preset(
        name="local-only",
        label="Local only (free/private)",
        allowed_providers=["local"],
        speed_mode="economy",
        opencode_model="lmstudio/qwen2.5-coder-14b-instruct",
        native_only=False,
        description="Everything on local LM Studio coders. No cloud.",
    ),
    Preset(
        name="claude-only",
        label="Claude Code only (native)",
        allowed_providers=["sonnet", "haiku", "opus-4-8"],
        speed_mode="balanced",
        opencode_model=None,
        native_only=True,
        description="Anthropic models used natively by Claude Code only — never via opencode or Mullet delegation.",
    ),
    Preset(
        name="cheap-cloud",
        label="Cheap cloud (no Anthropic)",
        allowed_providers=["kimi", "gemini"],
        speed_mode="balanced",
        opencode_model="moonshot/kimi-k2.6",
        native_only=False,
        description="OpenRouter/Kimi/Gemini cloud. No local, no Anthropic quota.",
    ),
]

BUILTIN_PRESET_NAMES: frozenset[str] = frozenset(p.name for p in BUILTIN_PRESETS)


@dataclass
class PresetConfig:
    """On-disk shape for ``presets.yaml``."""

    active: str | None
    presets: list[Preset]


def _preset_from_dict(data: dict[str, Any]) -> Preset:
    return Preset(
        name=data["name"],
        label=data["label"],
        allowed_providers=data.get("allowed_providers"),
        speed_mode=data["speed_mode"],
        opencode_model=data.get("opencode_model"),
        native_only=data.get("native_only", False),
        description=data.get("description", ""),
    )


class PresetManager:
    """Manage named provider presets persisted to ``presets.yaml``."""

    def __init__(self, config_dir: Path | None = None) -> None:
        if config_dir is None:
            env_override = os.environ.get("MULLET_CONFIG_DIR")
            if env_override:
                config_dir = Path(env_override)
            else:
                config_dir = Path.home() / ".config" / "mullet"
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.presets_path = self.config_dir / "presets.yaml"

    def load(self) -> PresetConfig:
        """Read ``presets.yaml``; seed built-ins when missing or empty."""
        config: PresetConfig | None = None

        if self.presets_path.exists():
            try:
                raw = self.presets_path.read_text(encoding="utf-8")
                if raw.strip():
                    data: Any = yaml.safe_load(raw)
                    if isinstance(data, dict):
                        presets_data = data.get("presets", [])
                        presets = [
                            _preset_from_dict(p)
                            for p in presets_data
                            if isinstance(p, dict)
                        ]
                        config = PresetConfig(
                            active=data.get("active"),
                            presets=presets,
                        )
            except Exception:
                config = None

        if config is None:
            config = self._default_config()
            self.save(config)
            return config

        # Rebuild: code-owned built-ins first, then user presets from disk
        merged_presets: list[Preset] = [
            dataclasses.replace(p) for p in BUILTIN_PRESETS
        ]
        for p in config.presets:
            if p.name not in BUILTIN_PRESET_NAMES:
                merged_presets.append(p)

        merged_names = {p.name for p in merged_presets}
        merged_active = config.active
        if merged_active is None or merged_active not in merged_names:
            merged_active = _DEFAULT_ACTIVE

        needs_save = (merged_presets != config.presets) or (
            merged_active != config.active
        )
        config.presets = merged_presets
        config.active = merged_active

        if needs_save:
            self.save(config)

        return config

    def save(self, config: PresetConfig) -> None:
        """Atomically write *config* to ``presets.yaml`` (tmp + os.replace)."""
        payload = {
            "active": config.active,
            "presets": [dataclasses.asdict(p) for p in config.presets],
        }
        text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix="presets.", suffix=".yaml.tmp", dir=str(self.config_dir)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp_path, self.presets_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def list(self) -> list[Preset]:
        """Return all configured presets."""
        return list(self.load().presets)

    def get(self, name: str) -> Preset | None:
        """Return the preset named *name*, or ``None`` if not found."""
        return next((p for p in self.load().presets if p.name == name), None)

    def active(self) -> Preset:
        """Return the currently-active preset."""
        config = self.load()
        target = config.active or _DEFAULT_ACTIVE
        preset = next((p for p in config.presets if p.name == target), None)
        if preset is None:
            preset = next(
                (p for p in BUILTIN_PRESETS if p.name == _DEFAULT_ACTIVE), None
            )
            if preset is None:
                raise RuntimeError("default preset not found")
        return preset

    def use(self, name: str) -> None:
        """Set *name* as the active preset; raise if unknown."""
        config = self.load()
        if not any(p.name == name for p in config.presets):
            raise ValueError(f"unknown preset: {name!r}")
        config.active = name
        self.save(config)

    def add(self, preset: Preset) -> None:
        """Append *preset* to the config; raise on duplicate names."""
        config = self.load()
        if any(p.name == preset.name for p in config.presets):
            raise ValueError(f"preset {preset.name!r} already exists")
        config.presets.append(preset)
        self.save(config)

    def remove(self, name: str, force: bool = False) -> None:
        """Remove a preset by name.

        Refuses to remove the active preset or a built-in unless ``force``
        is set.
        """
        config = self.load()
        if not any(p.name == name for p in config.presets):
            raise ValueError(f"unknown preset: {name!r}")
        if name == config.active and not force:
            raise ValueError(
                f"refusing to remove active preset {name!r}; pass force=True"
            )
        if self.is_builtin(name) and not force:
            raise ValueError(
                f"refusing to remove built-in preset {name!r}; pass force=True"
            )
        config.presets = [p for p in config.presets if p.name != name]
        if config.active == name:
            config.active = None
        self.save(config)

    def is_builtin(self, name: str) -> bool:
        """Return True when *name* is a Mullet-shipped built-in."""
        return name in BUILTIN_PRESET_NAMES

    def _default_config(self) -> PresetConfig:
        return PresetConfig(
            active=_DEFAULT_ACTIVE,
            presets=[dataclasses.replace(p) for p in BUILTIN_PRESETS],
        )
