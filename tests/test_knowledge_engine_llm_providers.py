"""Tests for provider classes and factory functions in mullet/knowledge/engine/llm.py."""
import pytest
from unittest.mock import MagicMock, patch
import requests

from mullet.knowledge.engine.llm import (
    LMStudioNativeProvider,
    LMStudioAnthropicProvider,
    LMStudioProvider,
    ClaudeProvider,
    get_provider,
    create_provider_from_config,
    DEFAULT_LMSTUDIO_HOST,
    DEFAULT_LMSTUDIO_MODEL,
)


# Helpers
def _ok_response(json_data):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = json_data
    return resp

def _error_response(exc=None):
    resp = MagicMock()
    resp.raise_for_status.side_effect = exc or requests.HTTPError("500")
    return resp


# ---------------------------------------------------------------------------
# LMStudioNativeProvider
# ---------------------------------------------------------------------------

class TestLMStudioNativeProviderInit:
    def test_default_host_set(self):
        p = LMStudioNativeProvider()
        assert p.host == DEFAULT_LMSTUDIO_HOST

    def test_trailing_slash_stripped(self):
        p = LMStudioNativeProvider(host="http://localhost:1234/")
        assert p.host == "http://localhost:1234"

    def test_api_base_uses_api_v1(self):
        p = LMStudioNativeProvider()
        assert p.api_base == "http://localhost:1234/api/v1"

    def test_api_token_added_to_headers(self):
        p = LMStudioNativeProvider(api_token="secret")
        assert p.headers.get("Authorization") == "Bearer secret"

    def test_no_api_token_no_auth_header(self):
        p = LMStudioNativeProvider()
        assert "Authorization" not in p.headers

    def test_model_stored(self):
        p = LMStudioNativeProvider(model="my-model")
        assert p.model == "my-model"

    def test_context_length_stored(self):
        p = LMStudioNativeProvider(context_length=4096)
        assert p.context_length == 4096


class TestLMStudioNativeListModels:
    def test_returns_models_list(self):
        p = LMStudioNativeProvider()
        with patch("requests.get", return_value=_ok_response({"models": [{"id": "m1"}]})):
            result = p.list_models()
        assert result == [{"id": "m1"}]

    def test_returns_empty_on_exception(self):
        p = LMStudioNativeProvider()
        with patch("requests.get", side_effect=requests.ConnectionError("err")):
            result = p.list_models()
        assert result == []

    def test_missing_models_key_returns_empty(self):
        p = LMStudioNativeProvider()
        with patch("requests.get", return_value=_ok_response({})):
            result = p.list_models()
        assert result == []


class TestLMStudioNativeGetLoadedModel:
    def test_returns_model_with_loaded_instances(self):
        p = LMStudioNativeProvider()
        models = [
            {"id": "m1", "loaded_instances": []},
            {"id": "m2", "loaded_instances": [{"id": "inst1"}]},
        ]
        with patch.object(p, "list_models", return_value=models):
            result = p.get_loaded_model()
        assert result["id"] == "m2"

    def test_returns_none_when_no_loaded(self):
        p = LMStudioNativeProvider()
        with patch.object(p, "list_models", return_value=[{"id": "m1", "loaded_instances": []}]):
            assert p.get_loaded_model() is None

    def test_returns_none_when_no_models(self):
        p = LMStudioNativeProvider()
        with patch.object(p, "list_models", return_value=[]):
            assert p.get_loaded_model() is None


class TestLMStudioNativeLoadModel:
    def test_returns_true_on_success(self):
        p = LMStudioNativeProvider()
        with patch("requests.post", return_value=_ok_response({})):
            assert p.load_model("my/model") is True

    def test_returns_false_on_exception(self):
        p = LMStudioNativeProvider()
        with patch("requests.post", side_effect=Exception("err")):
            assert p.load_model("bad") is False

    def test_context_length_included_in_payload(self):
        p = LMStudioNativeProvider()
        captured = {}
        def fake_post(url, json=None, headers=None, **kw):
            captured["json"] = json
            return _ok_response({})
        with patch("requests.post", side_effect=fake_post):
            p.load_model("m", context_length=8192)
        assert captured["json"]["context_length"] == 8192

    def test_gpu_offload_included_in_payload(self):
        p = LMStudioNativeProvider()
        captured = {}
        def fake_post(url, json=None, headers=None, **kw):
            captured["json"] = json
            return _ok_response({})
        with patch("requests.post", side_effect=fake_post):
            p.load_model("m", gpu_offload=0.5)
        assert captured["json"]["gpu_offload"] == 0.5


class TestLMStudioNativeUnloadModel:
    def test_returns_true_on_success(self):
        p = LMStudioNativeProvider()
        with patch("requests.post", return_value=_ok_response({})):
            assert p.unload_model() is True

    def test_model_path_in_payload_when_specified(self):
        p = LMStudioNativeProvider()
        captured = {}
        def fake_post(url, json=None, headers=None, **kw):
            captured["json"] = json
            return _ok_response({})
        with patch("requests.post", side_effect=fake_post):
            p.unload_model("my/model")
        assert captured["json"]["model"] == "my/model"

    def test_returns_false_on_exception(self):
        p = LMStudioNativeProvider()
        with patch("requests.post", side_effect=Exception("err")):
            assert p.unload_model() is False


class TestLMStudioNativeGenerate:
    def _post_ok(self, content="response text"):
        return _ok_response({"output": [{"type": "message", "content": content}]})

    def test_basic_generate(self):
        p = LMStudioNativeProvider(model="test-model")
        with patch("requests.post", return_value=self._post_ok("hello")):
            result = p.generate([{"role": "user", "content": "Q"}])
        assert result == "hello"

    def test_system_prompt_prepended(self):
        p = LMStudioNativeProvider(model="m")
        captured = {}
        def fake_post(url, json=None, headers=None, **kw):
            captured["json"] = json
            return self._post_ok()
        with patch("requests.post", side_effect=fake_post):
            p.generate([{"role": "user", "content": "Q"}], system_prompt="Be helpful")
        assert "System: Be helpful" in captured["json"]["input"]

    def test_no_model_returns_error(self):
        p = LMStudioNativeProvider(model=None)
        with patch.object(p, "get_loaded_model", return_value=None):
            result = p.generate([{"role": "user", "content": "Q"}])
        assert "Error:" in result

    def test_uses_loaded_model_id_from_instance(self):
        p = LMStudioNativeProvider(model=None)
        loaded = {"loaded_instances": [{"id": "inst-abc"}], "key": "key-abc"}
        with patch.object(p, "get_loaded_model", return_value=loaded):
            captured = {}
            def fake_post(url, json=None, headers=None, **kw):
                captured["json"] = json
                return self._post_ok()
            with patch("requests.post", side_effect=fake_post):
                p.generate([{"role": "user", "content": "Q"}])
        assert captured["json"]["model"] == "inst-abc"

    def test_uses_loaded_model_key_when_no_instances(self):
        p = LMStudioNativeProvider(model=None)
        loaded = {"loaded_instances": [], "key": "key-xyz"}
        with patch.object(p, "get_loaded_model", return_value=loaded):
            captured = {}
            def fake_post(url, json=None, headers=None, **kw):
                captured["json"] = json
                return self._post_ok()
            with patch("requests.post", side_effect=fake_post):
                p.generate([{"role": "user", "content": "Q"}])
        assert captured["json"]["model"] == "key-xyz"

    def test_context_length_in_config(self):
        p = LMStudioNativeProvider(model="m", context_length=4096)
        captured = {}
        def fake_post(url, json=None, headers=None, **kw):
            captured["json"] = json
            return self._post_ok()
        with patch("requests.post", side_effect=fake_post):
            p.generate([{"role": "user", "content": "Q"}])
        assert captured["json"]["config"]["context_length"] == 4096

    def test_per_request_context_length_overrides_provider(self):
        p = LMStudioNativeProvider(model="m", context_length=2048)
        captured = {}
        def fake_post(url, json=None, headers=None, **kw):
            captured["json"] = json
            return self._post_ok()
        with patch("requests.post", side_effect=fake_post):
            p.generate([{"role": "user", "content": "Q"}], context_length=8192)
        assert captured["json"]["config"]["context_length"] == 8192

    def test_connection_error_returns_error_string(self):
        p = LMStudioNativeProvider(model="m")
        with patch("requests.post", side_effect=requests.exceptions.ConnectionError()):
            result = p.generate([{"role": "user", "content": "Q"}])
        assert "Cannot connect" in result

    def test_timeout_returns_error_string(self):
        p = LMStudioNativeProvider(model="m")
        with patch("requests.post", side_effect=requests.exceptions.Timeout()):
            result = p.generate([{"role": "user", "content": "Q"}])
        assert "timed out" in result

    def test_context_error_returns_context_error_string(self):
        p = LMStudioNativeProvider(model="m")
        with patch("requests.post", side_effect=Exception("context window exceeded")):
            result = p.generate([{"role": "user", "content": "Q"}])
        assert "Context too long" in result

    def test_generic_error_returns_error_string(self):
        p = LMStudioNativeProvider(model="m")
        with patch("requests.post", side_effect=Exception("something else")):
            result = p.generate([{"role": "user", "content": "Q"}])
        assert "Error:" in result

    def test_empty_output_array_returns_empty_string(self):
        p = LMStudioNativeProvider(model="m")
        with patch("requests.post", return_value=_ok_response({"output": []})):
            result = p.generate([{"role": "user", "content": "Q"}])
        assert result == ""

    def test_non_message_output_items_skipped(self):
        p = LMStudioNativeProvider(model="m")
        data = {"output": [{"type": "token", "content": "x"}, {"type": "message", "content": "answer"}]}
        with patch("requests.post", return_value=_ok_response(data)):
            result = p.generate([{"role": "user", "content": "Q"}])
        assert result == "answer"


# ---------------------------------------------------------------------------
# LMStudioAnthropicProvider
# ---------------------------------------------------------------------------

class TestLMStudioAnthropicProviderInit:
    def test_default_init(self):
        p = LMStudioAnthropicProvider()
        assert p.host == DEFAULT_LMSTUDIO_HOST
        assert p.model == DEFAULT_LMSTUDIO_MODEL

    def test_trailing_slash_stripped(self):
        p = LMStudioAnthropicProvider(host="http://localhost:1234/")
        assert p.host == "http://localhost:1234"

    def test_api_base_uses_v1(self):
        p = LMStudioAnthropicProvider(host="http://localhost:5678")
        assert p.api_base == "http://localhost:5678/v1"

    def test_headers_include_anthropic_version(self):
        p = LMStudioAnthropicProvider()
        assert p.headers["anthropic-version"] == "2023-06-01"


class TestLMStudioAnthropicGenerate:
    def _post_ok(self, text="reply"):
        return _ok_response({"content": [{"type": "text", "text": text}]})

    def test_basic_generate(self):
        p = LMStudioAnthropicProvider()
        with patch("requests.post", return_value=self._post_ok("hi")):
            result = p.generate([{"role": "user", "content": "Q"}])
        assert result == "hi"

    def test_system_prompt_added_to_payload(self):
        p = LMStudioAnthropicProvider()
        captured = {}
        def fake_post(url, json=None, headers=None, **kw):
            captured["json"] = json
            return self._post_ok()
        with patch("requests.post", side_effect=fake_post):
            p.generate([{"role": "user", "content": "Q"}], system_prompt="sys")
        assert captured["json"]["system"] == "sys"

    def test_no_system_prompt_not_in_payload(self):
        p = LMStudioAnthropicProvider()
        captured = {}
        def fake_post(url, json=None, headers=None, **kw):
            captured["json"] = json
            return self._post_ok()
        with patch("requests.post", side_effect=fake_post):
            p.generate([{"role": "user", "content": "Q"}])
        assert "system" not in captured["json"]

    def test_empty_content_returns_empty_string(self):
        p = LMStudioAnthropicProvider()
        with patch("requests.post", return_value=_ok_response({"content": []})):
            result = p.generate([{"role": "user", "content": "Q"}])
        assert result == ""

    def test_non_text_blocks_skipped(self):
        p = LMStudioAnthropicProvider()
        data = {"content": [{"type": "image", "data": "x"}, {"type": "text", "text": "answer"}]}
        with patch("requests.post", return_value=_ok_response(data)):
            result = p.generate([{"role": "user", "content": "Q"}])
        assert result == "answer"

    def test_connection_error_returns_error_string(self):
        p = LMStudioAnthropicProvider()
        with patch("requests.post", side_effect=requests.exceptions.ConnectionError()):
            result = p.generate([{"role": "user", "content": "Q"}])
        assert "Cannot connect" in result

    def test_context_error_returns_context_string(self):
        p = LMStudioAnthropicProvider()
        with patch("requests.post", side_effect=Exception("token limit exceeded")):
            result = p.generate([{"role": "user", "content": "Q"}])
        assert "Context too long" in result

    def test_generic_error_returns_error_string(self):
        p = LMStudioAnthropicProvider()
        with patch("requests.post", side_effect=Exception("boom")):
            result = p.generate([{"role": "user", "content": "Q"}])
        assert "Error:" in result


# ---------------------------------------------------------------------------
# LMStudioProvider (OpenAI-compatible)
# ---------------------------------------------------------------------------

class TestLMStudioProviderInit:
    def test_model_stored(self):
        with patch("mullet.knowledge.engine.llm.OpenAI"):
            p = LMStudioProvider(model="my-model")
        assert p.model == "my-model"

    def test_openai_client_created(self):
        with patch("mullet.knowledge.engine.llm.OpenAI") as mock_openai:
            LMStudioProvider(base_url="http://localhost:1234/v1")
        mock_openai.assert_called_once_with(base_url="http://localhost:1234/v1", api_key="not-needed")


class TestLMStudioProviderGenerate:
    def _provider(self):
        with patch("mullet.knowledge.engine.llm.OpenAI"):
            p = LMStudioProvider()
        return p

    def _mock_completion(self, text="reply"):
        choice = MagicMock()
        choice.message.content = text
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_basic_generate(self):
        p = self._provider()
        p.client = MagicMock()
        p.client.chat.completions.create.return_value = self._mock_completion("answer")
        result = p.generate([{"role": "user", "content": "Q"}])
        assert result == "answer"

    def test_system_prompt_prepended(self):
        p = self._provider()
        p.client = MagicMock()
        p.client.chat.completions.create.return_value = self._mock_completion()
        captured = {}
        def fake_create(**kwargs):
            captured["messages"] = kwargs["messages"]
            return self._mock_completion()
        p.client.chat.completions.create = fake_create
        p.generate([{"role": "user", "content": "Q"}], system_prompt="sys")
        assert captured["messages"][0] == {"role": "system", "content": "sys"}

    def test_empty_choices_returns_error(self):
        p = self._provider()
        p.client = MagicMock()
        resp = MagicMock()
        resp.choices = []
        p.client.chat.completions.create.return_value = resp
        result = p.generate([{"role": "user", "content": "Q"}])
        assert "Error:" in result

    def test_connection_refused_returns_error(self):
        p = self._provider()
        p.client = MagicMock()
        p.client.chat.completions.create.side_effect = Exception("Connection refused")
        result = p.generate([{"role": "user", "content": "Q"}])
        assert "Cannot connect" in result

    def test_context_error_returns_context_string(self):
        p = self._provider()
        p.client = MagicMock()
        p.client.chat.completions.create.side_effect = Exception("token limit exceeded")
        result = p.generate([{"role": "user", "content": "Q"}])
        assert "Context too long" in result

    def test_generic_error_returns_error(self):
        p = self._provider()
        p.client = MagicMock()
        p.client.chat.completions.create.side_effect = Exception("boom")
        result = p.generate([{"role": "user", "content": "Q"}])
        assert "Error:" in result


# ---------------------------------------------------------------------------
# ClaudeProvider
# ---------------------------------------------------------------------------

class TestClaudeProviderInit:
    def test_raises_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            ClaudeProvider()

    def test_uses_env_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        with patch("mullet.knowledge.engine.llm.Anthropic") as mock_ant:
            p = ClaudeProvider()
        mock_ant.assert_called_once_with(api_key="env-key")

    def test_explicit_api_key_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        with patch("mullet.knowledge.engine.llm.Anthropic") as mock_ant:
            ClaudeProvider(api_key="explicit-key")
        mock_ant.assert_called_once_with(api_key="explicit-key")

    def test_model_stored(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        with patch("mullet.knowledge.engine.llm.Anthropic"):
            p = ClaudeProvider(api_key="k", model="claude-haiku")
        assert p.model == "claude-haiku"


class TestClaudeProviderGenerate:
    def _provider(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch("mullet.knowledge.engine.llm.Anthropic"):
            p = ClaudeProvider(api_key="test-key")
        p.client = MagicMock()
        return p

    def _mock_response(self, text="claude reply"):
        resp = MagicMock()
        content_block = MagicMock()
        content_block.text = text
        resp.content = [content_block]
        return resp

    def test_basic_generate(self, monkeypatch):
        p = self._provider(monkeypatch)
        p.client.messages.create.return_value = self._mock_response("answer")
        result = p.generate([{"role": "user", "content": "Q"}])
        assert result == "answer"

    def test_system_prompt_forwarded(self, monkeypatch):
        p = self._provider(monkeypatch)
        p.client.messages.create.return_value = self._mock_response()
        captured = {}
        def fake_create(**kwargs):
            captured.update(kwargs)
            return self._mock_response()
        p.client.messages.create = fake_create
        p.generate([{"role": "user", "content": "Q"}], system_prompt="Be concise")
        assert captured["system"] == "Be concise"

    def test_empty_content_returns_error(self, monkeypatch):
        p = self._provider(monkeypatch)
        resp = MagicMock()
        resp.content = []
        p.client.messages.create.return_value = resp
        result = p.generate([{"role": "user", "content": "Q"}])
        assert "Error:" in result

    def test_exception_returns_error(self, monkeypatch):
        p = self._provider(monkeypatch)
        p.client.messages.create.side_effect = Exception("API error")
        result = p.generate([{"role": "user", "content": "Q"}])
        assert "Error:" in result


# ---------------------------------------------------------------------------
# get_provider
# ---------------------------------------------------------------------------

class TestGetProvider:
    def test_lmstudio_default(self):
        with patch("mullet.knowledge.engine.llm.OpenAI"):
            p = get_provider("lmstudio")
        assert isinstance(p, LMStudioProvider)

    def test_lmstudio_openai_alias(self):
        with patch("mullet.knowledge.engine.llm.OpenAI"):
            p = get_provider("lmstudio_openai")
        assert isinstance(p, LMStudioProvider)

    def test_lmstudio_anthropic(self):
        p = get_provider("lmstudio_anthropic")
        assert isinstance(p, LMStudioAnthropicProvider)

    def test_lmstudio_native(self):
        p = get_provider("lmstudio_native")
        assert isinstance(p, LMStudioNativeProvider)

    def test_claude(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        with patch("mullet.knowledge.engine.llm.Anthropic"):
            p = get_provider("claude", api_key="k")
        assert isinstance(p, ClaudeProvider)

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown-provider")


# ---------------------------------------------------------------------------
# create_provider_from_config
# ---------------------------------------------------------------------------

class TestCreateProviderFromConfig:
    def test_lmstudio_anthropic_from_config(self):
        p = create_provider_from_config({"provider": "lmstudio_anthropic"})
        assert isinstance(p, LMStudioAnthropicProvider)

    def test_lmstudio_native_from_config(self):
        p = create_provider_from_config({"provider": "lmstudio_native"})
        assert isinstance(p, LMStudioNativeProvider)

    def test_lmstudio_from_config(self):
        with patch("mullet.knowledge.engine.llm.OpenAI"):
            p = create_provider_from_config({"provider": "lmstudio"})
        assert isinstance(p, LMStudioProvider)

    def test_lmstudio_openai_alias_from_config(self):
        with patch("mullet.knowledge.engine.llm.OpenAI"):
            p = create_provider_from_config({"provider": "lmstudio_openai"})
        assert isinstance(p, LMStudioProvider)

    def test_claude_from_config(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        with patch("mullet.knowledge.engine.llm.Anthropic"):
            p = create_provider_from_config({"provider": "claude", "api_key": "k"})
        assert isinstance(p, ClaudeProvider)

    def test_claude_from_config_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            create_provider_from_config({"provider": "claude"})

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider_from_config({"provider": "bad-provider"})

    def test_default_provider_is_lmstudio(self):
        with patch("mullet.knowledge.engine.llm.OpenAI"):
            p = create_provider_from_config({})
        assert isinstance(p, LMStudioProvider)

    def test_host_forwarded_to_provider(self):
        p = create_provider_from_config({"provider": "lmstudio_anthropic", "lmstudio_host": "http://gpu-server:1234"})
        assert p.host == "http://gpu-server:1234"

    def test_model_forwarded_to_provider(self):
        p = create_provider_from_config({"provider": "lmstudio_native", "lmstudio_model": "llama3"})
        assert p.model == "llama3"
