"""Tests for scripts/hooks/skill_activator.py"""
import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

_HOOK = Path(__file__).resolve().parent.parent / "scripts" / "hooks" / "skill_activator.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("skill_activator", _HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(prompt: str) -> str:
    """Invoke the hook main() and return captured stdout."""
    hook = _load_hook()
    payload = json.dumps({"prompt": prompt})
    buf = io.StringIO()
    with (
        patch.object(sys, "stdin", io.StringIO(payload)),
        patch.object(sys, "exit"),
        patch("sys.stdout", buf),
    ):
        hook.main()
    return buf.getvalue()


class TestSkillActivatorHook:
    def test_autopilot_keyword_triggers_hint(self):
        out = _run("can you use autopilot to implement the auth feature?")
        assert "autopilot" in out.lower()
        assert "/autopilot" in out

    def test_ralph_keyword_triggers_hint(self):
        out = _run("ralph this until everything passes")
        assert "ralph" in out.lower()
        assert "/ralph" in out

    def test_ultrawork_keyword_triggers_hint(self):
        out = _run("ultrawork the test refactor across all modules")
        assert "ultrawork" in out.lower()
        assert "/ultrawork" in out

    def test_eco_keyword_triggers_hint(self):
        out = _run("eco mode — save tokens on this")
        assert "eco" in out.lower()

    def test_short_prompt_no_output(self):
        out = _run("ok")
        assert out.strip() == ""

    def test_empty_prompt_no_output(self):
        out = _run("")
        assert out.strip() == ""

    def test_unrelated_prompt_no_output(self):
        out = _run("what is the capital of France?")
        assert "skill hint" not in out.lower()

    def test_hint_includes_slash_command(self):
        out = _run("I want to autopilot the whole feature branch")
        assert "/autopilot" in out

    def test_code_review_keyword(self):
        out = _run("please review this pull request for security issues")
        # Should suggest code-review skill
        assert "code-review" in out or "review" in out.lower()
