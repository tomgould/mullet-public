"""Coverage-gap tests for mullet.core.providers.gemini."""

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def provider():
    """Create GeminiProvider with mocked SDK."""
    # Pre-import gemini before patch.dict so sys.modules snapshot includes it.
    # Without this, patch.dict removes the module on exit, causing @patch in
    # subsequent tests to patch a different module object than the one the
    # resolver reimports, breaking isolation for other test files.
    import importlib
    if "mullet.core.providers.gemini" not in sys.modules:
        importlib.import_module("mullet.core.providers.gemini")

    mock_genai = MagicMock()
    mock_types = MagicMock()
    mock_genai.types = mock_types
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client

    mock_types.Content = MagicMock()
    mock_types.Part = MagicMock()
    mock_types.FunctionCall = MagicMock()
    mock_types.FunctionResponse = MagicMock()
    mock_types.FunctionDeclaration = MagicMock()
    mock_types.Tool = MagicMock()
    mock_types.GenerateContentConfig = MagicMock()

    with patch.dict(sys.modules, {
        "google": MagicMock(),
        "google.genai": mock_genai,
        "google.genai.types": mock_types,
    }):
        from mullet.core.providers.gemini import GeminiProvider
        p = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")
        yield p, mock_client


# ---------------------------------------------------------------------------
# Lines 31-32: ImportError on google-genai
# ---------------------------------------------------------------------------

class FakeGoogle:
    def __getattr__(self, name):
        if name == "genai":
            raise ImportError("No module named 'google.genai'")
        return MagicMock()


def test_init_import_error():
    import mullet.core.providers as _providers_pkg
    orig_gemini_attr = getattr(_providers_pkg, "gemini", None)
    orig_gemini_mod = sys.modules.get("mullet.core.providers.gemini")
    try:
        with patch.dict(sys.modules, {"google": FakeGoogle()}):
            sys.modules.pop("google.genai", None)
            # Force re-import of the gemini module so the local import runs
            # inside the patched sys.modules context.
            for key in list(sys.modules):
                if key == "mullet.core.providers.gemini" or key.startswith("mullet.core.providers.gemini."):
                    del sys.modules[key]
            with pytest.raises(ImportError, match="google-genai required"):
                from mullet.core.providers.gemini import GeminiProvider
                GeminiProvider(api_key="k")
    finally:
        # Restore parent-package attribute — patch.dict restores sys.modules but
        # Python's import system also sets the attribute on the parent package when
        # a submodule is reimported, and that change is NOT undone by patch.dict.
        if orig_gemini_mod is not None:
            sys.modules["mullet.core.providers.gemini"] = orig_gemini_mod
            _providers_pkg.gemini = orig_gemini_attr
        elif hasattr(_providers_pkg, "gemini"):
            delattr(_providers_pkg, "gemini")


# ---------------------------------------------------------------------------
# Lines 70, 72: config.system_instruction and config.tools assignments
# ---------------------------------------------------------------------------

def test_create_sets_system_instruction_and_tools(provider):
    p, _client = provider
    p.client.models.generate_content.return_value = MagicMock(
        candidates=[MagicMock(
            content=MagicMock(parts=[MagicMock(text="hi", function_call=None)]),
            finish_reason="STOP",
        )],
        usage_metadata=MagicMock(prompt_token_count=1, candidates_token_count=1),
    )

    messages = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "hello"},
    ]
    tools = [
        {
            "type": "function",
            "function": {"name": "fn", "description": "d", "parameters": {"type": "object"}},
        }
    ]
    p.create(messages=messages, tools=tools)

    config = p.client.models.generate_content.call_args.kwargs["config"]
    assert config.system_instruction == "sys prompt"
    assert config.tools is not None


# ---------------------------------------------------------------------------
# Lines 91-92, 94-111: log_interaction exception paths
# ---------------------------------------------------------------------------

def test_create_log_interaction_failure_in_success_path(provider):
    """log_interaction raising on success should be swallowed (lines 91-92)."""
    p, _client = provider
    p.client.models.generate_content.return_value = MagicMock(
        candidates=[MagicMock(
            content=MagicMock(parts=[MagicMock(text="ok", function_call=None)]),
            finish_reason="STOP",
        )],
        usage_metadata=MagicMock(prompt_token_count=1, candidates_token_count=1),
    )
    with patch("mullet.core.logging.interaction_log.log_interaction", side_effect=RuntimeError("boom")):
        result = p.create(messages=[{"role": "user", "content": "hi"}])
    assert result.content == "ok"


def test_create_exception_path_logs_and_re_raises(provider):
    """Lines 94-111: exception in create logs failure then re-raises."""
    p, _client = provider
    p.client.models.generate_content.side_effect = RuntimeError("api fail")
    with patch("mullet.core.logging.interaction_log.log_interaction", side_effect=RuntimeError("log boom")):
        with pytest.raises(RuntimeError, match="api fail"):
            p.create(messages=[{"role": "user", "content": "hi"}])


def test_create_exception_path_log_interaction_silent(provider):
    """Lines 109-110: log_interaction failure in exception handler is silent."""
    p, _client = provider
    p.client.models.generate_content.side_effect = RuntimeError("api fail")
    with patch("mullet.core.logging.interaction_log.log_interaction", side_effect=RuntimeError("log boom")):
        with pytest.raises(RuntimeError, match="api fail"):
            p.create(messages=[{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Lines 154-155: config.to_dict() exception in retry
# ---------------------------------------------------------------------------

def test_call_with_retry_config_to_dict_exception(provider):
    """Constraint self-healing when config.to_dict() raises (lines 154-155)."""
    p, _client = provider
    success = MagicMock(
        candidates=[MagicMock(
            content=MagicMock(parts=[MagicMock(text="ok", function_call=None)]),
            finish_reason="STOP",
        )],
        usage_metadata=MagicMock(prompt_token_count=1, candidates_token_count=1),
    )

    call_count = 0
    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("invalid field 'temperature'")
        return success

    p.client.models.generate_content.side_effect = side_effect
    mock_config = MagicMock()
    mock_config.to_dict.side_effect = AttributeError("no to_dict")

    with patch.object(type(p), "parse_error", return_value=("temperature", "test reason")):
        with patch("mullet.core.providers.gemini.get_constraint_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            with patch("time.sleep"):
                result = p._call_with_retry([], mock_config)
    assert result.content == "ok"


# ---------------------------------------------------------------------------
# Lines 164-165: retry config build exception after constraint healing
# ---------------------------------------------------------------------------

def test_call_with_retry_rebuild_config_raises(provider):
    """If rebuilding config after constraint healing fails, last_error updated (164-165).

    Note: the inner except catches the config-build error, but the outer ``raise``
    re-raises the *original* exception, not the config-build error.
    """
    p, _client = provider
    mock_config = MagicMock()
    mock_config.to_dict.return_value = {"temperature": 0.5}

    p.client.models.generate_content.side_effect = Exception("invalid field 'temperature'")

    with patch.object(type(p), "parse_error", return_value=("temperature", "test reason")):
        with patch("mullet.core.providers.gemini.get_constraint_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            with patch("time.sleep"):
                with patch("google.genai.types.GenerateContentConfig", side_effect=ValueError("bad config")):
                    with pytest.raises(Exception, match="invalid field 'temperature'"):
                        p._call_with_retry([], mock_config)


# ---------------------------------------------------------------------------
# Lines 225-226: json.JSONDecodeError in tool args parsing
# ---------------------------------------------------------------------------

def test_convert_messages_tool_args_decode_error(provider):
    p, _ = provider
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tc1", "type": "function",
             "function": {"name": "fn", "arguments": "not-json"}},
        ]},
    ]
    system, contents = p._convert_messages(messages)
    assert system == ""
    assert len(contents) == 2


# ---------------------------------------------------------------------------
# Lines 232-236: role == "tool" message conversion
# ---------------------------------------------------------------------------

def test_convert_messages_tool_role(provider):
    p, _ = provider
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tc1", "type": "function",
             "function": {"name": "search", "arguments": "{}"}},
        ]},
        {"role": "tool", "content": "result data", "tool_call_id": "tc1"},
    ]
    system, contents = p._convert_messages(messages)
    assert len(contents) == 3

    # Inspect the mock types to verify the third Content call used role="user"
    mock_types = sys.modules["google.genai.types"]
    third_call = mock_types.Content.call_args_list[2]
    assert third_call.kwargs["role"] == "user"
    # Verify Part was called with function_response for the tool message
    part_calls = mock_types.Part.call_args_list
    assert any("function_response" in (call.kwargs or {}) for call in part_calls)


# ---------------------------------------------------------------------------
# Line 298: finish_reason = "length" when MAX_TOKENS in finish_reason
# ---------------------------------------------------------------------------

def test_to_response_length_finish_reason(provider):
    p, _ = provider
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [MagicMock(text="trunc", function_call=None)]
    mock_candidate.finish_reason = "MAX_TOKENS"
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = None

    result = p._to_response(mock_response)
    assert result.finish_reason == "length"
