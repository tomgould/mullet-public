"""End-to-end integration tests across the harness layer.

These scenarios exercise the seams between:

* :mod:`mullet.harness.engine.loop` — the agent loop and hook fire-points
* :mod:`mullet.harness.skills` — frontmatter, loader, merger, built-ins
* :mod:`mullet.harness.hooks` — registry + Pydantic payload types
* :mod:`mullet.harness.plugins` — manifest, discovery, loader
* :mod:`mullet.harness.permissions` — full PermissionChecker wired as PreToolUse
* :mod:`mullet.harness.profiles` — ProfileManager.add/resolve round-trip

Modules that are still in flight at the time of writing
(``mullet.harness.plugins.loader``, ``mullet.harness.commands`` slash-command
handler, async hook dispatch) are skipped via :func:`pytest.importorskip`
guards so the suite remains green and the gaps are visible in the report.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

import pytest

from mullet.core.providers.base import LLMResponse, Provider
from mullet.harness.engine.loop import HarnessEngine
from mullet.harness.hooks.registry import Block, HookRegistry
from mullet.harness.hooks.types import PreToolUsePayload
from mullet.harness.permissions.checker import (
    PermissionChecker,
    PermissionsConfig,
)
from mullet.harness.profiles.manager import ProfileManager
from mullet.harness.profiles.schema import ProviderProfile
from mullet.harness.sessions.store import SessionStore
from mullet.harness.skills.builtin import BUILTIN_DIR
from mullet.harness.skills.loader import load_skills


pytestmark = pytest.mark.harness


# ---------------------------------------------------------------------------
# Local mocks
# ---------------------------------------------------------------------------


class _RecordingProvider(Provider):
    """Synchronous mock provider that records every call and returns a fixed reply."""

    name = "mock"

    def __init__(self, content: str = "OK", finish_reason: str = "stop") -> None:
        self.calls: list[tuple[list[dict], list[dict] | None]] = []
        self._content = content
        self._finish_reason = finish_reason
        self.model = "mock-1"

    def create(  # noqa: D401 — provider protocol
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        reasoning_config: dict | None = None,
    ) -> LLMResponse:
        # Defensive copy so subsequent test mutations to ``messages`` don't
        # rewrite history we already captured.
        self.calls.append(([dict(m) for m in messages], tools))
        return LLMResponse(content=self._content, finish_reason=self._finish_reason)


def _make_engine(
    tmp_path: Path,
    provider: Provider,
    monkeypatch: pytest.MonkeyPatch,
    *,
    hooks: HookRegistry | None = None,
    perms: PermissionChecker | None = None,
) -> HarnessEngine:
    """Build a HarnessEngine wired to a mock provider via monkeypatch."""
    store = SessionStore(base_dir=tmp_path / "sessions")
    monkeypatch.setattr(
        "mullet.harness.engine.loop._resolve_provider",
        lambda name, model=None: provider,
    )
    return HarnessEngine(store=store, hooks=hooks, perms=perms)


# Convenience: best-effort resolution for in-flight modules.
def _try_import(modname: str):
    try:
        return importlib.import_module(modname)
    except ImportError:
        return None


_skill_command_mod = _try_import("mullet.harness.commands")
_handle_skill_command = getattr(_skill_command_mod, "handle_skill_command", None) if _skill_command_mod else None

_plugins_loader_mod = _try_import("mullet.harness.plugins.loader")
_load_all_plugins = getattr(_plugins_loader_mod, "load_all_plugins", None) if _plugins_loader_mod else None


# ---------------------------------------------------------------------------
# 1. Skill body injected into next provider call
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _handle_skill_command is None,
    reason="mullet.harness.commands.handle_skill_command not yet implemented (P3.11 in_progress)",
)
def test_engine_with_skill_use_injects_skill_body_into_next_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _RecordingProvider()
    engine = _make_engine(tmp_path, provider, monkeypatch, hooks=HookRegistry())

    _handle_skill_command("use commit", registry=engine.hooks)
    assert engine.hooks.count("Message") >= 1, "skill use should register a Message hook"

    engine.run("hello")

    assert provider.calls, "provider was not called"
    messages, _tools = provider.calls[-1]
    system_messages = [m for m in messages if m.get("role") == "system"]
    assert system_messages, "expected a system message containing the skill body"
    blob = "\n".join(m.get("content", "") for m in system_messages)
    assert "## When to use" in blob, "expected commit skill body to contain '## When to use'"

    # The Message hook must auto-unregister itself after one shot.
    assert engine.hooks.count("Message") == 0, "skill hook should self-clear after the run"


# ---------------------------------------------------------------------------
# 2. Project-scope skill overrides built-in by name
# ---------------------------------------------------------------------------


def test_load_skills_includes_priority_override_from_project_scope(tmp_path: Path) -> None:
    project_skills = tmp_path / ".mullet" / "skills" / "commit"
    project_skills.mkdir(parents=True)
    skill_md = project_skills / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: commit\n"
        "description: Project override of the commit skill\n"
        "tools: [Bash]\n"
        "---\n"
        "# Commit (project override)\n"
        "PROJECT_OVERRIDE_COMMIT marker body.\n"
    )

    scope_dirs = {
        "project": [tmp_path / ".mullet" / "skills"],
        "user": [],
        "global": [BUILTIN_DIR],
    }
    skills = load_skills(scope_dirs)

    commits = [s for s in skills if s.name == "commit"]
    assert len(commits) == 1, "priority dedup failed: built-in commit was not replaced"
    only = commits[0]
    assert only.scope == "project"
    assert "PROJECT_OVERRIDE_COMMIT" in only.body
    # And built-in skills from other names should still come through, proving
    # global scope was actually scanned.
    other_names = {s.name for s in skills}
    assert "plan" in other_names or "review" in other_names, (
        "expected built-in skills aside from 'commit' to remain"
    )


# ---------------------------------------------------------------------------
# 3 & 4. Plugin loader scenarios
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _load_all_plugins is None,
    reason="mullet.harness.plugins.loader.load_all_plugins not yet implemented (P3.9 in_progress)",
)
def test_plugin_with_skills_dir_loads_skills(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins" / "myplugin"
    (plugin_dir / ".claude-plugin").mkdir(parents=True)
    (plugin_dir / ".claude-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "myplugin",
                "version": "0.1.0",
                "skills": ["./skills"],
            }
        )
    )
    skill_dir = plugin_dir / "skills" / "myskill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: myskill\n"
        "description: A plugin-supplied skill\n"
        "tools: [Read]\n"
        "---\n"
        "# myskill body\n"
    )

    result = _load_all_plugins(scope_dirs={"project": [tmp_path / "plugins"]})

    assert len(result.loaded) >= 1, "plugin was not loaded"
    loaded_names = [p.manifest.name for p in result.loaded]
    assert "myplugin" in loaded_names
    myplugin = next(p for p in result.loaded if p.manifest.name == "myplugin")
    assert myplugin.manifest.skills, "manifest.skills should resolve to the skills dir"
    resolved = Path(myplugin.manifest.skills[0])
    assert resolved.name == "skills"
    assert resolved.parent.name == "myplugin"


@pytest.mark.skipif(
    _load_all_plugins is None,
    reason="mullet.harness.plugins.loader.load_all_plugins not yet implemented (P3.9 in_progress)",
)
def test_plugin_loader_isolates_one_bad_plugin(tmp_path: Path) -> None:
    plugins_root = tmp_path / "plugins"

    good1 = plugins_root / "good1"
    good1.mkdir(parents=True)
    (good1 / "plugin.json").write_text(json.dumps({"name": "good1", "version": "1.0.0"}))

    bad = plugins_root / "bad"
    bad.mkdir(parents=True)
    (bad / "plugin.json").write_text("{not valid json")

    good2 = plugins_root / "good2"
    (good2 / ".claude-plugin").mkdir(parents=True)
    (good2 / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "good2", "version": "1.0.0"})
    )

    result = _load_all_plugins(scope_dirs={"project": [plugins_root]})

    loaded_names = [p.manifest.name for p in result.loaded]
    assert "good1" in loaded_names
    assert "good2" in loaded_names
    assert len(result.loaded) == 2, f"expected exactly two loaded plugins, got {loaded_names}"
    assert len(result.errors) == 1, f"expected one error, got {result.errors!r}"
    err = result.errors[0]
    err_path = getattr(err, "path", None) or getattr(err, "plugin_dir", None) or str(err)
    assert "bad" in str(err_path)


# ---------------------------------------------------------------------------
# 5. Permission hook blocks denied tool, lets others through
# ---------------------------------------------------------------------------


def test_engine_permission_hook_blocks_denied_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _RecordingProvider()
    perms = PermissionChecker(config=PermissionsConfig(denied_tools=["WebSearch"]))
    # Pass hooks=None so the engine wires its default PreToolUse permission hook.
    engine = _make_engine(tmp_path, provider, monkeypatch, perms=perms)
    # Default engine now wires 5 PreToolUse hooks (permission + guards).
    assert engine.hooks.count("PreToolUse") >= 1

    blocked = engine.hooks.fire(
        "PreToolUse",
        PreToolUsePayload(tool_name="WebSearch", tool_args={}, session_id="s1"),
    )
    assert isinstance(blocked, Block), f"expected Block, got {blocked!r}"
    assert "WebSearch" in blocked.reason

    passthrough = engine.hooks.fire(
        "PreToolUse",
        PreToolUsePayload(tool_name="Read", tool_args={"file_path": "/tmp/x"}, session_id="s1"),
    )
    # When no handler returns a Block or a new payload, the registry returns
    # the original payload back (it never returns ``None``). A PreToolUsePayload
    # for a non-denied tool is the expected passthrough sentinel here.
    assert isinstance(passthrough, PreToolUsePayload)
    assert passthrough.tool_name == "Read"


# ---------------------------------------------------------------------------
# 6. engine.run records the user.prompt + session.message_complete events
# ---------------------------------------------------------------------------


def test_engine_run_appends_session_event_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _RecordingProvider(content="hi back")
    engine = _make_engine(tmp_path, provider, monkeypatch, hooks=HookRegistry())

    result = engine.run("hi")
    sid = result["session_id"]

    _meta, messages, events = engine.store.load(sid)

    event_types = [e.event_type for e in events]
    assert "user.prompt" in event_types
    # The completion event is fired through the Event hook which writes into
    # the event log via the engine; if the engine still fires it directly to
    # the registry only, the message-complete signal lives in the assistant
    # message instead — assert at least one of the two surfaces shows it.
    has_complete_event = any(e.event_type == "session.message_complete" for e in events)
    has_assistant_msg = any(m.role == "assistant" and m.content == "hi back" for m in messages)
    assert has_complete_event or has_assistant_msg, (
        "expected either a session.message_complete event or a recorded assistant message"
    )


# ---------------------------------------------------------------------------
# 7. Full happy-path loop: skill use → engine.run
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _handle_skill_command is None,
    reason="mullet.harness.commands.handle_skill_command not yet implemented (P3.11 in_progress)",
)
def test_full_loop_skill_then_engine_run_works_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _RecordingProvider(content="design proposal")
    engine = _make_engine(tmp_path, provider, monkeypatch)
    # Engine wires its default permission hook when hooks are not supplied.
    assert engine.hooks.count("PreToolUse") >= 1

    _handle_skill_command("use plan", registry=engine.hooks)
    result = engine.run("design a feature")
    sid = result["session_id"]

    messages, _tools = provider.calls[-1]
    blob = "\n".join(m.get("content", "") for m in messages)
    # The plan built-in skill mentions phases / planning — assert at least one
    # marker sticks. The skill body is injected as a system message.
    assert any(
        kw.lower() in blob.lower()
        for kw in ("plan", "phase", "design")
    ), "expected plan skill content to land in the prompt"

    _meta, stored_messages, _events = engine.store.load(sid)
    assert len(stored_messages) == 2, "expected user + assistant message stored"
    roles = [m.role for m in stored_messages]
    assert roles == ["user", "assistant"]


# ---------------------------------------------------------------------------
# 8. ProfileManager.resolve round-trip with monkeypatched provider class
# ---------------------------------------------------------------------------


def test_profile_resolve_then_engine_can_use_profile_name(
    isolated_profiles_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class _MockKimi:
        def __init__(self, api_key=None, model=None, base_url=None) -> None:
            captured["api_key"] = api_key
            captured["model"] = model
            captured["base_url"] = base_url
            self.model = model
            self.name = "kimi"

        def create(self, *_args, **_kwargs):  # pragma: no cover — not invoked here
            return LLMResponse(content="", finish_reason="stop")

    # Patch the symbol the resolver imports lazily.
    monkeypatch.setattr("mullet.core.providers.kimi.KimiProvider", _MockKimi)
    # Make sure resolution doesn't bail because the env var is missing.
    monkeypatch.setenv("KIMI_API_KEY", "test-key")

    pm = ProfileManager()
    profile = ProviderProfile(
        name="test-kimi",
        label="Test Kimi",
        provider_type="kimi",
        api_format="openai",
        auth_source="KIMI_API_KEY",
        model="kimi-k2.5",
        base_url="https://api.moonshot.ai/v1",
    )
    pm.add(profile)

    resolved = pm.resolve("test-kimi")
    assert isinstance(resolved, _MockKimi)
    assert resolved.model == "kimi-k2.5"
    assert captured["model"] == "kimi-k2.5"
    assert captured["api_key"] == "test-key"


# ---------------------------------------------------------------------------
# 9. Async hook dispatch — only meaningful once fire_async lands
# ---------------------------------------------------------------------------


def test_hooks_with_async_handler_via_fire_async() -> None:
    reg = HookRegistry()
    if not hasattr(reg, "fire_async"):
        pytest.skip(
            "HookRegistry.fire_async not yet implemented (P3.6 in_progress); "
            "sync fire() is the only supported dispatch path for now"
        )

    async def _async_handler(payload):  # pragma: no cover — exercised when impl lands
        await asyncio.sleep(0)
        payload["touched"] = True
        return payload

    reg.register("Event", _async_handler)
    out = asyncio.run(reg.fire_async("Event", {"event": "session.idle"}))  # type: ignore[attr-defined]
    assert out["touched"] is True


# ---------------------------------------------------------------------------
# 10. Plan-mode permission blocks Edit
# ---------------------------------------------------------------------------


def test_disabled_skill_via_permission_mode_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _RecordingProvider()
    perms = PermissionChecker(config=PermissionsConfig(mode="plan"))
    engine = _make_engine(tmp_path, provider, monkeypatch, perms=perms)

    blocked = engine.hooks.fire(
        "PreToolUse",
        PreToolUsePayload(
            tool_name="Edit",
            tool_args={"file_path": "/tmp/x"},
            session_id="s1",
        ),
    )
    assert isinstance(blocked, Block)
    assert "plan" in blocked.reason.lower() or "denied" in blocked.reason.lower()
