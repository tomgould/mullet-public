"""Tests for OllamaConfig/Provider construction, skill_invoke schema/handler, and TrawlDocsLookup."""

import json

from mullet.core.providers.ollama import OllamaConfig, OllamaProvider
from mullet.core.tools.skill_invoke import (
    skill_invoke_handler,
    SKILL_INVOKE_SCHEMA,
)


# ---------------------------------------------------------------------------
# OllamaConfig defaults
# ---------------------------------------------------------------------------

class TestOllamaConfig:
    def test_default_base_url(self):
        assert OllamaConfig().base_url == "http://localhost:11434"

    def test_default_model(self):
        assert OllamaConfig().model == "llama2"

    def test_default_temperature(self):
        assert OllamaConfig().temperature == 0.7

    def test_default_top_p(self):
        assert OllamaConfig().top_p == 0.9

    def test_default_max_tokens(self):
        assert OllamaConfig().max_tokens == 4096

    def test_default_request_timeout_positive(self):
        assert OllamaConfig().request_timeout > 0

    def test_custom_model(self):
        cfg = OllamaConfig(model="mistral")
        assert cfg.model == "mistral"

    def test_custom_temperature(self):
        cfg = OllamaConfig(temperature=0.3)
        assert cfg.temperature == 0.3

    def test_custom_base_url(self):
        cfg = OllamaConfig(base_url="http://remote:11434")
        assert cfg.base_url == "http://remote:11434"

    def test_custom_max_tokens(self):
        cfg = OllamaConfig(max_tokens=2048)
        assert cfg.max_tokens == 2048


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------

class TestOllamaProvider:
    def test_config_stored(self):
        cfg = OllamaConfig()
        provider = OllamaProvider(cfg)
        assert provider.config is cfg

    def test_custom_config_accessible(self):
        cfg = OllamaConfig(model="phi3")
        provider = OllamaProvider(cfg)
        assert provider.config.model == "phi3"

    def test_instantiates(self):
        provider = OllamaProvider(OllamaConfig())
        assert provider is not None


# ---------------------------------------------------------------------------
# SKILL_INVOKE_SCHEMA
# ---------------------------------------------------------------------------

class TestSkillInvokeSchema:
    def test_has_description(self):
        assert "description" in SKILL_INVOKE_SCHEMA

    def test_has_parameters(self):
        assert "parameters" in SKILL_INVOKE_SCHEMA

    def test_action_required(self):
        assert "action" in SKILL_INVOKE_SCHEMA["parameters"]["required"]

    def test_action_param_exists(self):
        props = SKILL_INVOKE_SCHEMA["parameters"]["properties"]
        assert "action" in props

    def test_action_enum_has_invoke(self):
        enum = SKILL_INVOKE_SCHEMA["parameters"]["properties"]["action"]["enum"]
        assert "invoke" in enum

    def test_action_enum_has_list(self):
        enum = SKILL_INVOKE_SCHEMA["parameters"]["properties"]["action"]["enum"]
        assert "list" in enum

    def test_action_enum_has_match(self):
        enum = SKILL_INVOKE_SCHEMA["parameters"]["properties"]["action"]["enum"]
        assert "match" in enum

    def test_skill_name_param_exists(self):
        props = SKILL_INVOKE_SCHEMA["parameters"]["properties"]
        assert "skill_name" in props

    def test_text_param_exists(self):
        props = SKILL_INVOKE_SCHEMA["parameters"]["properties"]
        assert "text" in props


# ---------------------------------------------------------------------------
# skill_invoke_handler — pure error/no-network paths
# ---------------------------------------------------------------------------

class TestSkillInvokeHandler:
    def test_list_returns_json_string(self):
        result = skill_invoke_handler("list")
        assert isinstance(result, str)
        data = json.loads(result)
        assert "skills" in data

    def test_list_skills_is_list(self):
        data = json.loads(skill_invoke_handler("list"))
        assert isinstance(data["skills"], list)

    def test_invoke_no_skill_name_returns_error(self):
        data = json.loads(skill_invoke_handler("invoke"))
        assert "error" in data

    def test_invoke_no_skill_name_error_message(self):
        data = json.loads(skill_invoke_handler("invoke"))
        assert "skill_name" in data["error"]

    def test_match_no_text_returns_error(self):
        data = json.loads(skill_invoke_handler("match"))
        assert "error" in data

    def test_match_no_text_error_message(self):
        data = json.loads(skill_invoke_handler("match"))
        assert "text" in data["error"]

    def test_invalid_action_returns_error(self):
        data = json.loads(skill_invoke_handler("invalid"))
        assert "error" in data

    def test_invalid_action_mentions_action(self):
        data = json.loads(skill_invoke_handler("invalid"))
        assert "invalid" in data["error"].lower() or "Invalid" in data["error"]

    def test_match_no_match_returns_matched_false(self):
        data = json.loads(skill_invoke_handler("match", text="xyzzy_no_such_skill_ever"))
        assert data["matched"] is False

    def test_returns_valid_json(self):
        result = skill_invoke_handler("list")
        json.loads(result)  # should not raise


