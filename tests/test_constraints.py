"""Tests for mullet.core.providers.constraints."""

import json
import time
import threading



# ---------------------------------------------------------------------------
# shape_request tests
# ---------------------------------------------------------------------------

def test_shape_strips_temperature_moonshot_v1():
    """moonshot-v1-128k has temperature in omit_params — must be stripped."""
    from mullet.core.providers.constraints import shape_request
    raw = {"model": "moonshot-v1-128k", "messages": [], "max_tokens": 1000, "temperature": 0.7}
    shaped = shape_request("moonshot-v1-128k", raw)
    assert "temperature" not in shaped


def test_shape_clamps_temperature_kimi_k2():
    """kimi-k2.5 temperature_range=(0.0,2.0) — 3.0 must be clamped to 2.0."""
    from mullet.core.providers.constraints import shape_request
    raw = {"model": "kimi-k2.5", "messages": [], "max_tokens": 1000, "temperature": 3.0}
    shaped = shape_request("kimi-k2.5", raw)
    assert shaped["temperature"] == 2.0


def test_shape_noop_gpt4o():
    """gpt-4o with valid params — shape_request must not modify them."""
    from mullet.core.providers.constraints import shape_request
    raw = {"model": "gpt-4o", "messages": [], "max_tokens": 1000, "temperature": 0.7}
    shaped = shape_request("gpt-4o", raw)
    assert shaped["temperature"] == 0.7
    assert shaped["max_tokens"] == 1000


def test_shape_clamps_max_output_tokens(monkeypatch):
    """Model with max_output_tokens=4096 and request max_tokens=8192 → clamped."""
    import mullet.core.providers.constraints as constraints_mod
    from mullet.core.providers.constraints import ModelConstraints, ConstraintRegistry

    # Build a registry with a tight-ceiling model
    reg = ConstraintRegistry.__new__(ConstraintRegistry)
    reg._lock = threading.Lock()
    reg._learned = {}
    reg._static = {"test-model-tight": ModelConstraints(max_output_tokens=4096)}
    monkeypatch.setattr(constraints_mod, "_registry", reg)

    from mullet.core.providers.constraints import shape_request
    raw = {"model": "test-model-tight", "messages": [], "max_tokens": 8192}
    shaped = shape_request("test-model-tight", raw)
    assert shaped["max_tokens"] == 4096


def test_shape_does_not_mutate_input():
    """shape_request must return a new dict — never mutate the input."""
    from mullet.core.providers.constraints import shape_request
    raw = {"model": "moonshot-v1-8k", "messages": [], "max_tokens": 1000, "temperature": 0.7}
    before = dict(raw)
    shape_request("moonshot-v1-8k", raw)
    assert raw == before


def test_shape_strips_temperature_o1_mini():
    """o1-mini has temperature in omit_params — must be stripped."""
    from mullet.core.providers.constraints import shape_request
    raw = {"model": "o1-mini", "messages": [], "max_tokens": 1000, "temperature": 0.5, "top_p": 0.9}
    shaped = shape_request("o1-mini", raw)
    assert "temperature" not in shaped
    assert "top_p" not in shaped


def test_shape_low_temperature_clamped_to_zero():
    """Negative temperature must be clamped to range minimum."""
    from mullet.core.providers.constraints import shape_request
    raw = {"model": "kimi-k2.5", "messages": [], "temperature": -0.5}
    shaped = shape_request("kimi-k2.5", raw)
    assert shaped["temperature"] == 0.0


# ---------------------------------------------------------------------------
# ConstraintRegistry tests
# ---------------------------------------------------------------------------

def test_registry_get_moonshot():
    """moonshot-v1-128k must have temperature in omit_params and correct context."""
    from mullet.core.providers.constraints import get_constraint_registry
    reg = get_constraint_registry()
    c = reg.get("moonshot-v1-128k")
    assert "temperature" in c.omit_params
    assert c.max_context == 131_072
    assert not c.supports_tools


def test_registry_get_kimi_k2():
    """kimi-k2.5 must have full capabilities."""
    from mullet.core.providers.constraints import get_constraint_registry
    reg = get_constraint_registry()
    c = reg.get("kimi-k2.5")
    assert "temperature" not in c.omit_params
    assert c.supports_tools
    assert c.temperature_range == (0.0, 2.0)
    assert c.max_context == 262_144


def test_registry_prefix_match():
    """Any moonshot-v1-* model not in static should hit the prefix fallback."""
    from mullet.core.providers.constraints import ConstraintRegistry
    reg = ConstraintRegistry()
    c = reg.get("moonshot-v1-8k-custom-variant")
    # Prefix match should return moonshot-v1-* constraints (temperature in omit_params)
    assert "temperature" in c.omit_params


def test_registry_default_is_permissive():
    """Unknown model should return a permissive default."""
    from mullet.core.providers.constraints import get_constraint_registry
    reg = get_constraint_registry()
    c = reg.get("some-unknown-model-xyz-12345")
    assert c.can_serve({"context_tokens": 1000, "requires_tools": True, "requires_streaming": False})


def test_registry_learn_persists(tmp_path, monkeypatch):
    """learn() must mutate in-memory and persist to disk; reload picks it up."""
    import mullet.core.providers.constraints as constraints_mod
    learned_path = tmp_path / "learned_constraints.json"
    monkeypatch.setattr(constraints_mod, "LEARNED_CONSTRAINTS_PATH", learned_path)
    monkeypatch.setattr(constraints_mod, "_registry", None)

    from mullet.core.providers.constraints import get_constraint_registry
    reg = get_constraint_registry()
    reg.learn("some-future-model", "top_p", "rejected by API")

    # In-memory: get() must reflect the learned constraint
    c = reg.get("some-future-model")
    assert "top_p" in c.omit_params

    # On disk: file must exist and contain the entry
    assert learned_path.exists()
    data = json.loads(learned_path.read_text())
    assert "some-future-model" in data
    assert "top_p" in data["some-future-model"]["omit_params"]

    # Re-create registry from scratch — must reload from disk
    monkeypatch.setattr(constraints_mod, "_registry", None)
    reg2 = get_constraint_registry()
    c2 = reg2.get("some-future-model")
    assert "top_p" in c2.omit_params


def test_registry_learn_ttl_expired(tmp_path, monkeypatch):
    """Learned entries older than 30 days must be ignored on load."""
    import mullet.core.providers.constraints as constraints_mod
    learned_path = tmp_path / "learned_constraints.json"
    # Timestamp 31 days in the past
    old_ts = time.time() - (31 * 24 * 3600)
    stale_data = {
        "stale-model": {
            "omit_params": ["top_p"],
            "reason": "old constraint",
            "ts": old_ts,
        }
    }
    learned_path.write_text(json.dumps(stale_data))

    monkeypatch.setattr(constraints_mod, "LEARNED_CONSTRAINTS_PATH", learned_path)
    monkeypatch.setattr(constraints_mod, "_registry", None)

    from mullet.core.providers.constraints import get_constraint_registry
    reg = get_constraint_registry()
    c = reg.get("stale-model")
    # Expired entry must NOT appear in omit_params
    assert "top_p" not in c.omit_params


def test_registry_learn_multiple_params(tmp_path, monkeypatch):
    """learn() called twice for same model accumulates both params."""
    import mullet.core.providers.constraints as constraints_mod
    learned_path = tmp_path / "lc.json"
    monkeypatch.setattr(constraints_mod, "LEARNED_CONSTRAINTS_PATH", learned_path)
    monkeypatch.setattr(constraints_mod, "_registry", None)

    from mullet.core.providers.constraints import get_constraint_registry
    reg = get_constraint_registry()
    reg.learn("multi-model", "top_p", "reason1")
    reg.learn("multi-model", "frequency_penalty", "reason2")

    c = reg.get("multi-model")
    assert "top_p" in c.omit_params
    assert "frequency_penalty" in c.omit_params


# ---------------------------------------------------------------------------
# parse_constraint_error tests
# ---------------------------------------------------------------------------

def test_parse_error_moonshot_temperature():
    """Moonshot temperature error maps to ('temperature', ...)."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("invalid temperature: only 1 is allowed for this model")
    result = parse_constraint_error("kimi", exc)
    assert result is not None
    param, reason = result
    assert param == "temperature"
    assert "moonshot" in reason.lower() or "temperature" in reason.lower()


def test_parse_error_unknown_returns_none():
    """Unrecognized errors must return None — no spurious learning."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("something completely unrelated and random")
    assert parse_constraint_error("kimi", exc) is None
    assert parse_constraint_error("openai", exc) is None
    assert parse_constraint_error("anthropic", exc) is None


def test_parse_error_o1_unsupported_param():
    """OpenAI 'Unsupported parameter' error returns a (param, reason) tuple."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("Unsupported parameter: temperature")
    result = parse_constraint_error("openai", exc)
    assert result is not None
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)


def test_parse_error_moonshot_provider_name_variants():
    """parse_constraint_error recognizes both 'kimi' and 'moonshot' as provider names."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("invalid temperature: only 1 is allowed for this model")
    assert parse_constraint_error("moonshot", exc) is not None
    assert parse_constraint_error("custom", exc) is not None


# --- Anthropic error parser -------------------------------------------------

def test_parse_error_anthropic_overloaded_not_constraint():
    """Anthropic overloaded_error is transient, not a constraint — return None."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("overloaded_error: Service is temporarily unavailable")
    assert parse_constraint_error("anthropic", exc) is None


def test_parse_error_anthropic_invalid_value_for_param():
    """Anthropic 'Invalid value for temperature' returns ('temperature', reason)."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("invalid_request_error: Invalid value for 'temperature'")
    result = parse_constraint_error("anthropic", exc)
    assert result is not None
    param, reason = result
    assert param == "temperature"
    assert "anthropic" in reason.lower()

def test_parse_error_anthropic_top_p_out_of_range():
    """Anthropic 'top_p must be between 0 and 1' returns ('top_p', reason)."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("invalid_request_error: top_p must be between 0 and 1")
    result = parse_constraint_error("anthropic", exc)
    assert result is not None
    param, reason = result
    assert param == "top_p"
    assert "top_p out of range" in reason.lower()

def test_parse_error_anthropic_max_tokens_too_large():
    """Anthropic 'max_tokens is too large' returns ('max_tokens', reason)."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("invalid_request_error: max_tokens is too large")
    result = parse_constraint_error("anthropic", exc)
    assert result is not None
    param, reason = result
    assert param == "max_tokens"
    assert "max_tokens exceeds limit" in reason.lower()

def test_parse_error_anthropic_invalid_role_not_constraint():
    """Anthropic 'messages.N.role' error is not a constraint — return None."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("invalid_request_error: messages.0.role must be 'user' or 'assistant'")
    assert parse_constraint_error("anthropic", exc) is None

def test_parse_error_anthropic_could_not_parse_json_not_constraint():
    """Anthropic 'Could not parse JSON body' error is not a constraint — return None."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("could not parse json body")
    assert parse_constraint_error("anthropic", exc) is None


# --- Gemini error parser ----------------------------------------------------

def test_parse_error_gemini_quota_not_constraint():
    """Gemini resource_exhausted / quota errors are transient — return None."""
    from mullet.core.providers.constraints import parse_constraint_error
    for msg in (
        "429 RESOURCE_EXHAUSTED: quota exceeded",
        "quota exceeded for project",
    ):
        exc = Exception(msg)
        assert parse_constraint_error("gemini", exc) is None

def test_parse_error_gemini_api_key_invalid_not_constraint():
    """Gemini 'API key not valid' error is not a constraint — return None."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("API key not valid. Please ensure that the API key is correct.")
    assert parse_constraint_error("gemini", exc) is None


def test_parse_error_gemini_safety_not_constraint():
    """Gemini safety-filter blocks are not parameter issues — return None."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("Content blocked by safety filter")
    assert parse_constraint_error("gemini", exc) is None


def test_parse_error_gemini_invalid_argument_extracts_param():
    """Gemini 400 INVALID_ARGUMENT with field info returns (param, reason)."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("400 INVALID_ARGUMENT: unknown field 'frequency_penalty'")
    result = parse_constraint_error("gemini", exc)
    assert result is not None
    param, _ = result
    assert param == "frequency_penalty"


# ---------------------------------------------------------------------------
# ModelConstraints.can_serve tests
# ---------------------------------------------------------------------------

def test_can_serve_context_overflow():
    """moonshot-v1-8k with context_tokens=50_000 must return False."""
    from mullet.core.providers.constraints import get_constraint_registry
    c = get_constraint_registry().get("moonshot-v1-8k")
    assert not c.can_serve({"context_tokens": 50_000})


def test_can_serve_tools_not_supported():
    """moonshot-v1-8k with requires_tools=True must return False."""
    from mullet.core.providers.constraints import get_constraint_registry
    c = get_constraint_registry().get("moonshot-v1-8k")
    assert not c.can_serve({"context_tokens": 100, "requires_tools": True})


def test_can_serve_ok():
    """kimi-k2.5 with reasonable context and tools=True must return True."""
    from mullet.core.providers.constraints import get_constraint_registry
    c = get_constraint_registry().get("kimi-k2.5")
    assert c.can_serve({"context_tokens": 1000, "requires_tools": True, "requires_streaming": False})


def test_can_serve_streaming_not_supported():
    """o1-mini with requires_streaming=True must return False."""
    from mullet.core.providers.constraints import get_constraint_registry
    c = get_constraint_registry().get("o1-mini")
    assert not c.can_serve({"requires_streaming": True})


def test_can_serve_moonshot_128k_exceeds_context():
    """moonshot-v1-128k must reject a 200k token prompt."""
    from mullet.core.providers.constraints import get_constraint_registry
    c = get_constraint_registry().get("moonshot-v1-128k")
    assert not c.can_serve({"context_tokens": 200_000})


def test_can_serve_kimi_k2_large_prompt():
    """kimi-k2.5 (262k context) must accept a 200k token prompt."""
    from mullet.core.providers.constraints import get_constraint_registry
    c = get_constraint_registry().get("kimi-k2.5")
    assert c.can_serve({"context_tokens": 200_000, "requires_tools": True})


def test_registry_load_learned_corrupt_json(tmp_path, monkeypatch):
    """Cover lines 407-408: _load_learned exception handler."""
    import mullet.core.providers.constraints as constraints_mod
    learned_path = tmp_path / "learned_constraints.json"
    learned_path.write_text("not valid json {")
    monkeypatch.setattr(constraints_mod, "LEARNED_CONSTRAINTS_PATH", learned_path)
    monkeypatch.setattr(constraints_mod, "_registry", None)

    from mullet.core.providers.constraints import get_constraint_registry
    reg = get_constraint_registry()
    assert reg._learned == {}


def test_registry_save_learned_write_fails_cleanup_fails(tmp_path, monkeypatch):
    """Cover lines 420-424: _save_learned exception handler."""
    import mullet.core.providers.constraints as constraints_mod
    from mullet.core.providers.constraints import ConstraintRegistry

    learned_path = tmp_path / "learned_constraints.json"
    monkeypatch.setattr(constraints_mod, "LEARNED_CONSTRAINTS_PATH", learned_path)

    reg = ConstraintRegistry.__new__(ConstraintRegistry)
    reg._lock = threading.Lock()
    reg._learned = {"model-x": {"omit_params": ["top_p"], "reason": "r", "ts": 1.0}}

    monkeypatch.setattr("os.fdopen", lambda *a, **k: (_ for _ in ()).throw(OSError("write fail")))
    monkeypatch.setattr("os.unlink", lambda p: (_ for _ in ()).throw(OSError("unlink fail")))

    reg._save_learned()


def test_shape_clamps_max_tokens_to_max_output_tokens(monkeypatch):
    """Cover lines 458-460: max_tokens clamping in shape_request."""
    import mullet.core.providers.constraints as constraints_mod
    from mullet.core.providers.constraints import ModelConstraints, ConstraintRegistry, shape_request

    reg = ConstraintRegistry.__new__(ConstraintRegistry)
    reg._lock = threading.Lock()
    reg._learned = {}
    reg._static = {"test-clamp": ModelConstraints(max_output_tokens=2048)}
    monkeypatch.setattr(constraints_mod, "_registry", reg)

    raw = {"model": "test-clamp", "messages": [], "max_tokens": 4096}
    shaped = shape_request("test-clamp", raw)
    assert shaped["max_tokens"] == 2048


def test_shape_strips_temperature_when_range_none(monkeypatch):
    """Cover lines 472-473: temperature stripped when temperature_range is None."""
    import mullet.core.providers.constraints as constraints_mod
    from mullet.core.providers.constraints import ModelConstraints, ConstraintRegistry, shape_request

    reg = ConstraintRegistry.__new__(ConstraintRegistry)
    reg._lock = threading.Lock()
    reg._learned = {}
    reg._static = {"test-no-range": ModelConstraints(omit_params=frozenset(), temperature_range=None)}
    monkeypatch.setattr(constraints_mod, "_registry", reg)

    raw = {"model": "test-no-range", "messages": [], "temperature": 0.5}
    shaped = shape_request("test-no-range", raw)
    assert "temperature" not in shaped


def test_shape_strips_enable_thinking_when_thinking_required(monkeypatch):
    """Cover lines 484-486: thinking_required strips enable_thinking=False."""
    import mullet.core.providers.constraints as constraints_mod
    from mullet.core.providers.constraints import ModelConstraints, ConstraintRegistry, shape_request

    reg = ConstraintRegistry.__new__(ConstraintRegistry)
    reg._lock = threading.Lock()
    reg._learned = {}
    reg._static = {"test-think": ModelConstraints(thinking_required=True)}
    monkeypatch.setattr(constraints_mod, "_registry", reg)

    raw = {"model": "test-think", "messages": [], "enable_thinking": False}
    shaped = shape_request("test-think", raw)
    assert "enable_thinking" not in shaped


def test_parse_error_openai_unsupported_param_regex_match():
    """Cover line 521: openai unsupported parameter with regex match."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("Unsupported parameter: 'top_p'")
    result = parse_constraint_error("openai", exc)
    assert result == ("top_p", "openai model rejected parameter: top_p")


def test_parse_error_openai_unsupported_param_no_regex():
    """Cover line 523: openai unsupported parameter without regex match."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("unsupported parameter")
    result = parse_constraint_error("openai", exc)
    assert result == ("temperature", "openai model rejected parameter")


def test_parse_error_anthropic_max_tokens_too_large():
    """Cover line 552: anthropic max_tokens too large branch."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("invalid_request_error: max_tokens is too large")
    result = parse_constraint_error("anthropic", exc)
    assert result == ("max_tokens", "anthropic: max_tokens exceeds limit")


def test_parse_error_anthropic_deprecated_parameter():
    """Cover line 561: anthropic deprecated parameter match."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("invalid_request_error: `temperature` is deprecated for this model.")
    result = parse_constraint_error("anthropic", exc)
    assert result == ("temperature", "anthropic: parameter deprecated for this model")


def test_parse_error_gemini_response_generation_config_field():
    """Cover line 586: gemini invalid_argument with response generation config field match."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("400 INVALID_ARGUMENT: invalid response generation configuration for field 'temperature'")
    result = parse_constraint_error("gemini", exc)
    assert result == ("temperature", "gemini rejected generation configuration field: temperature")


def test_parse_error_gemini_invalid_field_regex():
    """Cover lines 591-592: gemini 'invalid field X' pattern."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("400 INVALID_ARGUMENT: invalid field temperature")
    result = parse_constraint_error("gemini", exc)
    assert result == ("temperature", "gemini rejected parameter: temperature")


def test_parse_error_gemini_is_not_recognized():
    """Cover lines 593-595: gemini 'param is not a valid' pattern."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("400 INVALID_ARGUMENT: 'top_p' is not a valid generation parameter")
    result = parse_constraint_error("gemini", exc)
    assert result == ("top_p", "gemini rejected parameter: top_p")


def test_parse_error_openai_does_not_support_temperature():
    """Cover line 523: openai 'does not support temperature' branch."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("this model does not support temperature parameter")
    result = parse_constraint_error("openai", exc)
    assert result == ("temperature", "model does not support temperature")


def test_shape_rename_params(monkeypatch):
    """Cover lines 458-460: rename_params step in shape_request."""
    import mullet.core.providers.constraints as constraints_mod
    from mullet.core.providers.constraints import ModelConstraints, ConstraintRegistry, shape_request
    reg = ConstraintRegistry.__new__(ConstraintRegistry)
    reg._lock = __import__("threading").Lock()
    reg._learned = {}
    reg._static = {"test-rename": ModelConstraints(rename_params={"old_param": "new_param"})}
    monkeypatch.setattr(constraints_mod, "_registry", reg)
    raw = {"model": "test-rename", "old_param": "value123"}
    shaped = shape_request("test-rename", raw)
    assert "old_param" not in shaped
    assert shaped.get("new_param") == "value123"


def test_parse_error_anthropic_is_not_allowed():
    """Cover line 561: anthropic 'unsupported_value' first regex match."""
    from mullet.core.providers.constraints import parse_constraint_error
    exc = Exception("unsupported_value: top_p is not allowed for this model")
    result = parse_constraint_error("anthropic", exc)
    assert result is not None
    assert result[0] == "top_p"
