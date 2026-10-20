"""Tests for url_safety, HTMLStripper, gemini_cli, and live_test pure helpers."""


from mullet.core.tools.url_safety import is_safe_url
from mullet.core.tools.web_fetch import HTMLStripper
from mullet.core.providers.gemini_cli import GeminiCLIProvider, is_gemini_cli_available
from mullet.cli.live_test import (
    Target,
    _parse_args,
    _DEFAULT_SECONDS,
    _DEFAULT_SWARM_WORKERS,
    _KIMI_MODELS,
    _GEMINI_MODELS,
    _ANTHROPIC_MODELS,
)


# ---------------------------------------------------------------------------
# is_safe_url
# ---------------------------------------------------------------------------

class TestIsSafeUrlSafe:
    def test_https_url_safe(self):
        assert is_safe_url("https://google.com") is True

    def test_http_url_safe(self):
        assert is_safe_url("http://example.com") is True

    def test_returns_bool(self):
        assert isinstance(is_safe_url("https://example.com"), bool)

    def test_public_ip_safe(self):
        assert is_safe_url("https://8.8.8.8") is True


class TestIsSafeUrlBlocked:
    def test_file_scheme_blocked(self):
        assert is_safe_url("file:///etc/passwd") is False

    def test_ftp_scheme_blocked(self):
        assert is_safe_url("ftp://example.com") is False

    def test_gopher_scheme_blocked(self):
        assert is_safe_url("gopher://example.com") is False

    def test_loopback_ip_blocked(self):
        assert is_safe_url("http://127.0.0.1") is False

    def test_private_ip_10_blocked(self):
        assert is_safe_url("http://10.0.0.1") is False

    def test_private_ip_192_blocked(self):
        assert is_safe_url("http://192.168.1.1") is False

    def test_invalid_url_does_not_crash(self):
        result = is_safe_url("")
        assert isinstance(result, bool)

    def test_dict_scheme_blocked(self):
        assert is_safe_url("dict://example.com") is False


# ---------------------------------------------------------------------------
# HTMLStripper
# ---------------------------------------------------------------------------

class TestHTMLStripper:
    def test_extracts_text_from_html(self):
        stripper = HTMLStripper()
        stripper.feed("<html><body><p>Hello world</p></body></html>")
        assert "Hello world" in stripper.get_data()

    def test_strips_tags(self):
        stripper = HTMLStripper()
        stripper.feed("<b>Bold</b> text")
        result = stripper.get_data()
        assert "<b>" not in result
        assert "Bold" in result

    def test_plain_text_passes_through(self):
        stripper = HTMLStripper()
        stripper.feed("no html here")
        assert stripper.get_data() == "no html here"

    def test_script_content_excluded(self):
        stripper = HTMLStripper()
        stripper.feed("<p>Visible</p><script>alert('hidden')</script>")
        result = stripper.get_data()
        assert "alert" not in result
        assert "Visible" in result

    def test_multiple_elements(self):
        stripper = HTMLStripper()
        stripper.feed("<h1>Title</h1><p>Content</p>")
        result = stripper.get_data()
        assert "Title" in result
        assert "Content" in result

    def test_empty_html_returns_empty(self):
        stripper = HTMLStripper()
        stripper.feed("")
        assert stripper.get_data() == ""

    def test_get_data_returns_string(self):
        stripper = HTMLStripper()
        stripper.feed("<p>test</p>")
        assert isinstance(stripper.get_data(), str)

    def test_nested_tags_stripped(self):
        stripper = HTMLStripper()
        stripper.feed("<div><span><a href='x'>link text</a></span></div>")
        result = stripper.get_data()
        assert "<div>" not in result
        assert "link text" in result

    def test_attributes_stripped(self):
        stripper = HTMLStripper()
        stripper.feed('<p class="big" id="main">content here</p>')
        result = stripper.get_data()
        assert 'class=' not in result
        assert 'content here' in result


# ---------------------------------------------------------------------------
# GeminiCLIProvider
# ---------------------------------------------------------------------------

class TestGeminiCLIProvider:
    def test_default_model(self):
        p = GeminiCLIProvider()
        assert isinstance(p.model, str)
        assert len(p.model) > 0

    def test_default_name(self):
        p = GeminiCLIProvider()
        assert p.name == "gemini-cli"

    def test_custom_model(self):
        p = GeminiCLIProvider(model="gemini-2.0-flash")
        assert p.model == "gemini-2.0-flash"

    def test_messages_to_prompt_user_role(self):
        p = GeminiCLIProvider()
        msgs = [{"role": "user", "content": "hello world"}]
        result = p._messages_to_prompt(msgs)
        assert "hello world" in result

    def test_messages_to_prompt_system_role(self):
        p = GeminiCLIProvider()
        msgs = [{"role": "system", "content": "you are helpful"}]
        result = p._messages_to_prompt(msgs)
        assert "you are helpful" in result
        assert "Instructions" in result

    def test_messages_to_prompt_assistant_role(self):
        p = GeminiCLIProvider()
        msgs = [{"role": "assistant", "content": "here is my answer"}]
        result = p._messages_to_prompt(msgs)
        assert "here is my answer" in result

    def test_messages_to_prompt_returns_string(self):
        p = GeminiCLIProvider()
        result = p._messages_to_prompt([{"role": "user", "content": "test"}])
        assert isinstance(result, str)

    def test_messages_to_prompt_empty_list(self):
        p = GeminiCLIProvider()
        result = p._messages_to_prompt([])
        assert isinstance(result, str)

    def test_messages_to_prompt_multimodal_content(self):
        p = GeminiCLIProvider()
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "analyze this"},
            {"type": "image", "url": "http://example.com/img.png"},
        ]}]
        result = p._messages_to_prompt(msgs)
        assert "analyze this" in result

    def test_messages_to_prompt_multi_turn(self):
        p = GeminiCLIProvider()
        msgs = [
            {"role": "user", "content": "question one"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "question two"},
        ]
        result = p._messages_to_prompt(msgs)
        assert "question one" in result
        assert "answer one" in result
        assert "question two" in result


class TestIsGeminiCliAvailable:
    def test_returns_bool(self):
        result = is_gemini_cli_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# live_test constants
# ---------------------------------------------------------------------------

class TestLiveTestConstants:
    def test_default_seconds_positive(self):
        assert _DEFAULT_SECONDS > 0

    def test_default_swarm_workers_positive(self):
        assert _DEFAULT_SWARM_WORKERS > 0

    def test_kimi_models_nonempty(self):
        assert len(_KIMI_MODELS) > 0

    def test_gemini_models_nonempty(self):
        assert len(_GEMINI_MODELS) > 0

    def test_anthropic_models_nonempty(self):
        assert len(_ANTHROPIC_MODELS) > 0

    def test_kimi_models_are_strings(self):
        assert all(isinstance(m, str) for m in _KIMI_MODELS)

    def test_gemini_models_are_strings(self):
        assert all(isinstance(m, str) for m in _GEMINI_MODELS)


# ---------------------------------------------------------------------------
# Target dataclass
# ---------------------------------------------------------------------------

class TestTarget:
    def test_fields_stored(self):
        factory = lambda: None
        t = Target(provider_label="kimi", model_id="kimi-k2.6", build=factory)
        assert t.provider_label == "kimi"
        assert t.model_id == "kimi-k2.6"
        assert t.build is factory

    def test_build_callable(self):
        called = []
        def factory():
            called.append(1)
            return "provider"
        t = Target(provider_label="test", model_id="m1", build=factory)
        result = t.build()
        assert result == "provider"
        assert len(called) == 1


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_default_seconds(self):
        args = _parse_args([])
        assert args.seconds == _DEFAULT_SECONDS

    def test_custom_seconds(self):
        args = _parse_args(["30"])
        assert args.seconds == 30

    def test_no_swarm_default_false(self):
        args = _parse_args([])
        assert args.no_swarm is False

    def test_no_swarm_flag(self):
        args = _parse_args(["--no-swarm"])
        assert args.no_swarm is True

    def test_no_local_default_false(self):
        args = _parse_args([])
        assert args.no_local is False

    def test_no_local_flag(self):
        args = _parse_args(["--no-local"])
        assert args.no_local is True

    def test_swarm_size_default(self):
        args = _parse_args([])
        assert args.swarm_size == _DEFAULT_SWARM_WORKERS

    def test_swarm_size_custom(self):
        args = _parse_args(["--swarm-size", "5"])
        assert args.swarm_size == 5

    def test_combined_flags(self):
        args = _parse_args(["10", "--no-swarm", "--swarm-size", "2"])
        assert args.seconds == 10
        assert args.no_swarm is True
        assert args.swarm_size == 2
