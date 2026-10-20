"""Tests for config validation."""

from mullet.config.validate import validate_config


def test_empty_config_valid():
    result = validate_config({})
    assert result.valid


def test_valid_full_config():
    result = validate_config({
        "providers": {
            "local": {
                "type": "openai_compatible",
                "base_url": "http://localhost:1234/v1/",
                "tier": "free",
                "models": {"default": "mistral"},
            },
        },
        "routing": {"enabled": True, "default_tier": "MEDIUM"},
        "max_iterations": 90,
        "budget": {"per_task": 1.0, "daily": 10.0},
    })
    assert result.valid


def test_invalid_provider_type():
    result = validate_config({
        "providers": {"x": {"type": "invalid_type"}},
    })
    assert not result.valid
    assert any("Unknown type" in i.message for i in result.issues)


def test_missing_provider_type():
    result = validate_config({
        "providers": {"x": {"base_url": "http://localhost"}},
    })
    assert not result.valid
    assert any("Required" in i.message for i in result.issues)


def test_invalid_tier():
    result = validate_config({
        "providers": {"x": {"type": "anthropic", "tier": "ultra"}},
    })
    assert not result.valid
    assert any("tier" in i.path for i in result.issues)


def test_invalid_max_iterations():
    result = validate_config({"max_iterations": 0})
    assert not result.valid

    result = validate_config({"max_iterations": 999})
    assert not result.valid


def test_valid_max_iterations():
    result = validate_config({"max_iterations": 50})
    assert result.valid


def test_invalid_budget_negative():
    result = validate_config({"budget": {"daily": -5}})
    assert not result.valid


def test_invalid_routing_tier():
    result = validate_config({"routing": {"default_tier": "ULTRA"}})
    assert not result.valid


def test_unknown_keys_warning():
    result = validate_config({"random_key": "value"})
    assert result.valid  # Warnings don't fail
    assert len(result.issues) == 1
    assert result.issues[0].severity == "warning"


def test_invalid_port():
    result = validate_config({"server": {"port": 99999}})
    assert not result.valid


def test_non_dict_root():
    result = validate_config("not a dict")
    assert not result.valid


def test_format_output():
    result = validate_config({"max_iterations": -1})
    output = result.format()
    assert "ERROR" in output
    assert "max_iterations" in output


def test_provider_models_must_be_dict():
    result = validate_config({
        "providers": {"x": {"type": "anthropic", "models": "sonnet"}},
    })
    assert not result.valid
    assert any("models" in i.path for i in result.issues)


def test_gemini_provider_type_valid():
    """GeminiProvider is a real class in core.providers.gemini — accept it."""
    result = validate_config({
        "providers": {
            "gemini": {
                "type": "gemini",
                "api_key_env": "GOOGLE_API_KEY",
                "tier": "cheap",
            },
        },
    })
    assert result.valid, [i.message for i in result.issues]


def test_standard_tier_valid():
    """`standard` sits between cheap and premium (e.g. Claude Sonnet)."""
    result = validate_config({
        "providers": {
            "sonnet": {"type": "anthropic", "tier": "standard"},
        },
    })
    assert result.valid, [i.message for i in result.issues]


def test_invalid_tier_still_rejected():
    """Unknown tier strings should still be rejected — regression for the
    extended `_VALID_TIERS` set."""
    result = validate_config({
        "providers": {
            "bad": {"type": "anthropic", "tier": "expensive-as-fuck"},
        },
    })
    assert not result.valid
    assert any("tier" in i.path for i in result.issues)


def test_bedrock_provider_type_valid():
    """Bedrock has been valid since v1 — keep regression coverage."""
    result = validate_config({
        "providers": {"aws": {"type": "bedrock"}},
    })
    assert result.valid, [i.message for i in result.issues]


# ---------------------------------------------------------------------------
# ValidationResult — add_error, add_warning, format
# ---------------------------------------------------------------------------

from mullet.config.validate import ValidationResult, ValidationIssue


class TestValidationResult:
    def test_initially_valid(self):
        result = ValidationResult()
        assert result.valid
        assert result.issues == []

    def test_add_error_flips_valid(self):
        result = ValidationResult()
        result.add_error("providers.kimi", "missing api_key")
        assert not result.valid

    def test_add_error_creates_error_issue(self):
        result = ValidationResult()
        result.add_error("providers.kimi", "missing api_key")
        assert result.issues[0].severity == "error"
        assert result.issues[0].path == "providers.kimi"

    def test_add_warning_does_not_flip_valid(self):
        result = ValidationResult()
        result.add_warning("providers.gemini", "optional field missing")
        assert result.valid  # warnings do not invalidate

    def test_add_warning_creates_warning_issue(self):
        result = ValidationResult()
        result.add_warning("providers.gemini", "optional field missing")
        assert result.issues[0].severity == "warning"

    def test_format_no_issues(self):
        result = ValidationResult()
        assert result.format() == "Config valid."

    def test_format_error_prefixed(self):
        result = ValidationResult()
        result.add_error("providers.kimi", "bad key")
        text = result.format()
        assert "[ERROR]" in text
        assert "providers.kimi" in text
        assert "bad key" in text

    def test_format_warning_prefixed(self):
        result = ValidationResult()
        result.add_warning("providers.gemini", "opt missing")
        text = result.format()
        assert "[WARN]" in text
        assert "providers.gemini" in text

    def test_multiple_issues_each_on_own_line(self):
        result = ValidationResult()
        result.add_error("a", "err1")
        result.add_warning("b", "warn1")
        lines = result.format().splitlines()
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# Missing-line coverage — sections that existing tests didn't reach
# ---------------------------------------------------------------------------


class TestProvidersSectionNotDict:
    """Line 75: providers value is not a dict."""

    def test_providers_not_dict(self):
        result = validate_config({"providers": "local"})
        assert not result.valid
        assert any(i.path == "providers" for i in result.issues)


class TestRoutingSectionNotDict:
    """Line 84: routing value is not a dict."""

    def test_routing_not_dict(self):
        result = validate_config({"routing": ["round_robin"]})
        assert not result.valid
        assert any(i.path == "routing" for i in result.issues)


class TestBudgetSectionNotDict:
    """Line 96: budget value is not a dict."""

    def test_budget_not_dict(self):
        result = validate_config({"budget": 50})
        assert not result.valid
        assert any(i.path == "budget" for i in result.issues)


class TestServerSectionNotDict:
    """Line 114: server value is not a dict."""

    def test_server_not_dict(self):
        result = validate_config({"server": "localhost:8000"})
        assert not result.valid
        assert any(i.path == "server" for i in result.issues)


class TestToolsets:
    """Lines 123-124: toolsets must be a list."""

    def test_toolsets_not_list(self):
        result = validate_config({"toolsets": "all"})
        assert not result.valid
        assert any(i.path == "toolsets" for i in result.issues)

    def test_toolsets_as_list_valid(self):
        result = validate_config({"toolsets": ["search", "shell"]})
        assert result.valid


class TestProviderNotDict:
    """Lines 143-144: individual provider config must be a dict."""

    def test_provider_config_not_dict(self):
        result = validate_config({"providers": {"local": "openai_compatible"}})
        assert not result.valid
        assert any("providers.local" in i.path for i in result.issues)


class TestOpenAICompatibleBaseUrlWarning:
    """Line 160: openai_compatible without base_url emits a warning."""

    def test_openai_compatible_no_base_url_warns(self):
        result = validate_config({
            "providers": {
                "lmstudio": {
                    "type": "openai_compatible",
                    # base_url deliberately omitted
                },
            },
        })
        # Warning only — should still be valid
        assert result.valid
        assert any(
            "base_url" in i.path and i.severity == "warning"
            for i in result.issues
        )


class TestApiKeyEnvMustBeString:
    """Line 165: api_key_env must be a string."""

    def test_api_key_env_not_string(self):
        result = validate_config({
            "providers": {
                "kimi": {
                    "type": "openai_compatible",
                    "api_key_env": 123,
                },
            },
        })
        assert not result.valid
        assert any("api_key_env" in i.path for i in result.issues)
