"""Tests for mullet.core.presets."""

import pytest

from mullet.core.presets import (
    BUILTIN_PRESETS,
    BUILTIN_PRESET_NAMES,
    Preset,
    PresetManager,
    _DEFAULT_ACTIVE,
)


class TestPreset:
    def test_mcp_allowlist_none(self):
        preset = Preset(
            name="balanced-mix",
            label="Balanced",
            allowed_providers=None,
            speed_mode="balanced",
            opencode_model=None,
        )
        assert preset.mcp_allowlist() is None

    def test_mcp_allowlist_set(self):
        preset = Preset(
            name="local-only",
            label="Local",
            allowed_providers=["local"],
            speed_mode="economy",
            opencode_model=None,
        )
        assert preset.mcp_allowlist() == {"local"}

    def test_router_allowlist_none(self):
        preset = Preset(
            name="balanced-mix",
            label="Balanced",
            allowed_providers=None,
            speed_mode="balanced",
            opencode_model=None,
        )
        assert preset.router_allowlist() is None

    def test_router_allowlist_adds_cheap_cloud_for_kimi(self):
        preset = Preset(
            name="cheap-cloud",
            label="Cheap",
            allowed_providers=["kimi"],
            speed_mode="balanced",
            opencode_model=None,
        )
        result = preset.router_allowlist()
        assert result is not None
        assert "cheap_cloud" in result
        assert "kimi" in result

    def test_router_allowlist_adds_cheap_cloud_for_gemini(self):
        preset = Preset(
            name="cheap-cloud",
            label="Cheap",
            allowed_providers=["gemini"],
            speed_mode="balanced",
            opencode_model=None,
        )
        result = preset.router_allowlist()
        assert result is not None
        assert "cheap_cloud" in result
        assert "gemini" in result

    def test_router_allowlist_no_cheap_cloud_without_kimi_or_gemini(self):
        preset = Preset(
            name="local-only",
            label="Local",
            allowed_providers=["local"],
            speed_mode="economy",
            opencode_model=None,
        )
        result = preset.router_allowlist()
        assert result is not None
        assert "cheap_cloud" not in result

    def test_anthropic_opencode_model_rejected(self):
        with pytest.raises(ValueError):
            Preset(
                name="bad",
                label="Bad",
                allowed_providers=None,
                speed_mode="balanced",
                opencode_model="anthropic/claude-sonnet-4-6",
            )

    def test_anthropic_opencode_model_rejected_case_insensitive(self):
        with pytest.raises(ValueError):
            Preset(
                name="bad",
                label="Bad",
                allowed_providers=None,
                speed_mode="balanced",
                opencode_model="Claude-3-opus",
            )

    def test_non_anthropic_opencode_model_accepted(self):
        preset = Preset(
            name="good",
            label="Good",
            allowed_providers=None,
            speed_mode="balanced",
            opencode_model="moonshot/kimi-k2.6",
        )
        assert preset.opencode_model == "moonshot/kimi-k2.6"


class TestPresetManager:
    def test_fresh_manager_seeds_builtins(self, tmp_path):
        mgr = PresetManager(config_dir=tmp_path)
        config = mgr.load()
        names = {p.name for p in config.presets}
        assert names == BUILTIN_PRESET_NAMES
        assert config.active == _DEFAULT_ACTIVE

    def test_save_load_roundtrip_custom_preset(self, tmp_path):
        mgr = PresetManager(config_dir=tmp_path)
        mgr.load()  # seed
        custom = Preset(
            name="custom",
            label="Custom",
            allowed_providers=["openrouter"],
            speed_mode="fast",
            opencode_model="openrouter/deepseek-v3",
            description="My custom preset.",
        )
        mgr.add(custom)

        # Create a new manager to force re-read from disk
        mgr2 = PresetManager(config_dir=tmp_path)
        config = mgr2.load()
        names = {p.name for p in config.presets}
        assert "custom" in names
        retrieved = mgr2.get("custom")
        assert retrieved is not None
        assert retrieved.opencode_model == "openrouter/deepseek-v3"
        assert retrieved.description == "My custom preset."

    def test_use_sets_active(self, tmp_path):
        mgr = PresetManager(config_dir=tmp_path)
        mgr.load()
        mgr.use("local-only")
        assert mgr.load().active == "local-only"

    def test_use_unknown_raises(self, tmp_path):
        mgr = PresetManager(config_dir=tmp_path)
        mgr.load()
        with pytest.raises(ValueError, match="unknown preset"):
            mgr.use("nonexistent")

    def test_remove_refuses_builtin(self, tmp_path):
        mgr = PresetManager(config_dir=tmp_path)
        mgr.load()
        # Switch active away from balanced-mix so the built-in check fires first
        mgr.use("local-only")
        with pytest.raises(ValueError, match="refusing to remove built-in"):
            mgr.remove("balanced-mix")

    def test_remove_refuses_active_without_force(self, tmp_path):
        mgr = PresetManager(config_dir=tmp_path)
        mgr.load()
        mgr.use("local-only")
        with pytest.raises(ValueError, match="refusing to remove active"):
            mgr.remove("local-only")

    def test_remove_succeeds_with_force_on_non_active_custom(self, tmp_path):
        mgr = PresetManager(config_dir=tmp_path)
        mgr.load()
        custom = Preset(
            name="custom",
            label="Custom",
            allowed_providers=["openrouter"],
            speed_mode="fast",
            opencode_model="openrouter/deepseek-v3",
        )
        mgr.add(custom)
        mgr.use("balanced-mix")
        mgr.remove("custom", force=True)
        assert mgr.get("custom") is None

    def test_active_returns_correct_preset(self, tmp_path):
        mgr = PresetManager(config_dir=tmp_path)
        mgr.load()
        active = mgr.active()
        assert active.name == _DEFAULT_ACTIVE

    def test_active_returns_claude_only_with_native_only(self, tmp_path):
        mgr = PresetManager(config_dir=tmp_path)
        mgr.load()
        mgr.use("claude-only")
        active = mgr.active()
        assert active.name == "claude-only"
        assert active.allowed_providers == ["sonnet", "haiku", "opus-4-8"]
        assert active.mcp_allowlist() == {"sonnet", "haiku", "opus-4-8"}
        assert active.router_allowlist() == frozenset({"sonnet", "haiku", "opus-4-8"})
        assert active.native_only is True

    def test_get_none_for_unknown(self, tmp_path):
        mgr = PresetManager(config_dir=tmp_path)
        mgr.load()
        assert mgr.get("unknown") is None

    def test_list_returns_all(self, tmp_path):
        mgr = PresetManager(config_dir=tmp_path)
        mgr.load()
        assert len(mgr.list()) == len(BUILTIN_PRESETS)

    def test_is_builtin(self, tmp_path):
        mgr = PresetManager(config_dir=tmp_path)
        assert mgr.is_builtin("balanced-mix")
        assert not mgr.is_builtin("custom")

    def test_add_duplicate_raises(self, tmp_path):
        mgr = PresetManager(config_dir=tmp_path)
        mgr.load()
        preset = Preset(
            name="balanced-mix",
            label="Duplicate",
            allowed_providers=None,
            speed_mode="balanced",
            opencode_model=None,
        )
        with pytest.raises(ValueError, match="already exists"):
            mgr.add(preset)

    def test_remove_unknown_raises(self, tmp_path):
        mgr = PresetManager(config_dir=tmp_path)
        mgr.load()
        with pytest.raises(ValueError, match="unknown preset"):
            mgr.remove("unknown")

    def test_load_re_adds_missing_builtin(self, tmp_path):
        mgr = PresetManager(config_dir=tmp_path)
        mgr.load()
        # Remove a built-in with force (must switch active away first)
        mgr.use("balanced-mix")
        mgr.remove("local-only", force=True)

        # Verify the file on disk no longer contains it
        mgr3 = PresetManager(config_dir=tmp_path)
        raw = mgr3.presets_path.read_text(encoding="utf-8")
        assert "local-only" not in raw

        # Reloading should re-add the built-in
        mgr2 = PresetManager(config_dir=tmp_path)
        config = mgr2.load()
        names = {p.name for p in config.presets}
        assert "local-only" in names

    def test_load_refreshes_stale_builtin(self, tmp_path):
        """Built-in presets must always reflect the in-code definition."""
        presets_yaml = tmp_path / "presets.yaml"
        presets_yaml.write_text(
            "active: my-mix\n"
            "presets:\n"
            "  - name: claude-only\n"
            "    label: Claude Code only (native)\n"
            "    allowed_providers: null\n"
            "    speed_mode: balanced\n"
            "    opencode_model: null\n"
            "    native_only: true\n"
            "    description: Stale built-in\n"
            "  - name: my-mix\n"
            "    label: My Mix\n"
            "    allowed_providers: [local, kimi]\n"
            "    speed_mode: fast\n"
            "    opencode_model: moonshot/kimi-k2.6\n"
            "    native_only: false\n"
            "    description: My custom preset.\n",
            encoding="utf-8",
        )
        mgr = PresetManager(config_dir=tmp_path)
        config = mgr.load()

        # Stale built-in refreshed from code
        claude = next((p for p in config.presets if p.name == "claude-only"), None)
        assert claude is not None
        assert claude.allowed_providers == ["sonnet", "haiku", "opus-4-8"]

        # Custom user preset preserved
        my_mix = next((p for p in config.presets if p.name == "my-mix"), None)
        assert my_mix is not None
        assert my_mix.allowed_providers == ["local", "kimi"]

        # Active pointer preserved
        assert config.active == "my-mix"

        # File self-healed
        assert "sonnet" in mgr.presets_path.read_text(encoding="utf-8")

    def test_load_fallback_active_for_missing_preset(self, tmp_path):
        """Active must fall back to default when it names a missing preset."""
        presets_yaml = tmp_path / "presets.yaml"
        presets_yaml.write_text(
            "active: missing-preset\n"
            "presets:\n"
            "  - name: my-mix\n"
            "    label: My Mix\n"
            "    allowed_providers: [local, kimi]\n"
            "    speed_mode: fast\n"
            "    opencode_model: moonshot/kimi-k2.6\n"
            "    native_only: false\n"
            "    description: Custom preset.\n",
            encoding="utf-8",
        )
        mgr = PresetManager(config_dir=tmp_path)
        config = mgr.load()
        assert config.active == _DEFAULT_ACTIVE
        assert "missing-preset" not in {p.name for p in config.presets}
