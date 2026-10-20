"""Tests for system prompt generation."""

from mullet.core.agent.prompts import build_base_prompt, _get_model_guidance


def test_base_prompt_contains_identity():
    prompt = build_base_prompt()
    assert "Mullet" in prompt
    assert "direct, efficient" in prompt


def test_base_prompt_contains_tool_rules():
    prompt = build_base_prompt()
    assert "terminal" in prompt
    assert "read_file" in prompt
    assert "absolute paths" in prompt


def test_base_prompt_contains_cwd():
    prompt = build_base_prompt(cwd="/test/dir")
    assert "/test/dir" in prompt


def test_base_prompt_with_project():
    prompt = build_base_prompt(project_name="backend")
    assert "backend" in prompt


def test_model_guidance_gemini():
    guidance = _get_model_guidance("gemini-2.5-flash")
    assert "absolute" in guidance and "file paths" in guidance
    assert "sequentially" in guidance.lower() or "single response" in guidance.lower()


def test_model_guidance_gpt():
    guidance = _get_model_guidance("gpt-4o")
    assert "MUST use your tools" in guidance


def test_model_guidance_mistral():
    guidance = _get_model_guidance("mistral-nemo")
    assert "tool" in guidance.lower()


def test_model_guidance_unknown():
    guidance = _get_model_guidance("some-random-model")
    assert guidance == ""


def test_model_guidance_included_in_prompt():
    prompt = build_base_prompt(model="gemini-2.5-flash")
    assert "absolute" in prompt and "file paths" in prompt


def test_prompt_no_model():
    prompt = build_base_prompt(model=None)
    # Should still work without model
    assert "Mullet" in prompt
