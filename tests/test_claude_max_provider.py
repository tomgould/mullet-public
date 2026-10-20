"""Tests for ClaudeMaxProvider — OAuth bearer-token Anthropic access.

Covers init paths, wire-call shape, HUD tracking, router fallback logic,
and health_check.  No real LLM calls are made — all wire calls are mocked.
"""

import json
from unittest.mock import MagicMock, call, patch

import pytest

from mullet.hud import agent_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def creds_file(tmp_path):
    """Write a fake ~/.claude/.credentials.json and return its path."""
    creds = {
        "claudeAiOauth": {
            "accessToken": "test-oauth-token-abc123",
            "expiresAt": 9999999999000,
        }
    }
    f = tmp_path / ".credentials.json"
    f.write_text(json.dumps(creds))
    return f


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Redirect HUD agent_state writes to a tmp file."""
    state_file = tmp_path / "agents.json"
    monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)
    yield state_file


@pytest.fixture
def mock_anthropic_client(monkeypatch):
    """Patch anthropic.Anthropic so no real SDK init / HTTP call happens."""
    fake_client = MagicMock()
    fake_response = _make_sdk_response("Hello from Max")
    fake_client.messages.create.return_value = fake_response
    monkeypatch.setattr(
        "mullet.core.providers.claude_max.Anthropic",
        lambda **kw: fake_client,
        raising=False,
    )
    # Also patch via the anthropic module imported inside the provider
    with patch("anthropic.Anthropic", return_value=fake_client):
        yield fake_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sdk_response(text: str = "OK"):
    """Build a minimal fake Anthropic SDK message response."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    response.usage = MagicMock(
        input_tokens=10, output_tokens=5,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
    )
    response.model = "claude-haiku-4-5"
    return response


def _make_provider(token: str = "tok", model: str = "claude-haiku-4-5"):
    """Instantiate ClaudeMaxProvider with a patched SDK client."""
    with patch("anthropic.Anthropic"):
        from mullet.core.providers.claude_max import ClaudeMaxProvider
        return ClaudeMaxProvider(model=model, access_token=token)


# ---------------------------------------------------------------------------
# 1. Init from OAuth token file (mocked creds path)
# ---------------------------------------------------------------------------

def test_init_from_creds_file(tmp_path, monkeypatch):
    """Provider reads the OAuth token from the credentials file when no explicit token given."""
    creds = {"claudeAiOauth": {"accessToken": "file-token-xyz"}}
    creds_path = tmp_path / ".credentials.json"
    creds_path.write_text(json.dumps(creds))

    with patch("mullet.core.providers.claude_max.Path") as mock_path_cls, \
         patch("anthropic.Anthropic"):
        # Make Path.home() / ".claude" / ".credentials.json" resolve to our tmp file
        mock_home = MagicMock()
        mock_path_cls.home.return_value = mock_home
        # Build the chain: Path.home() / ".claude" / ".credentials.json"
        mock_home.__truediv__ = lambda self, other: (
            creds_path.parent if other == ".claude" else MagicMock()
        )
        creds_parent = MagicMock()
        creds_parent.__truediv__ = lambda self, other: creds_path
        mock_home.__truediv__ = lambda self, other: creds_parent

        # Simpler approach: patch _read_oauth_token directly
        from mullet.core.providers.claude_max import ClaudeMaxProvider
        with patch("mullet.core.providers.claude_max._read_oauth_token", return_value="file-token-xyz"):
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            assert p.name == "claude_max"
            assert p.model == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# 2. Init with explicit access_token=
# ---------------------------------------------------------------------------

def test_init_explicit_token():
    """Provider accepts an explicit access_token= without reading any file."""
    with patch("anthropic.Anthropic"):
        from mullet.core.providers.claude_max import ClaudeMaxProvider
        p = ClaudeMaxProvider(model="claude-sonnet-4-6", access_token="explicit-token")
        assert p.name == "claude_max"
        assert p.model == "claude-sonnet-4-6"
        assert p._token == "explicit-token"


# ---------------------------------------------------------------------------
# 3. Raises when neither token nor creds file present
# ---------------------------------------------------------------------------

def test_raises_when_no_token():
    """RuntimeError raised when no token is available from any source."""
    with patch("mullet.core.providers.claude_max._read_oauth_token", return_value=None):
        from mullet.core.providers.claude_max import ClaudeMaxProvider
        with pytest.raises(RuntimeError, match="no OAuth token"):
            ClaudeMaxProvider()


# ---------------------------------------------------------------------------
# 4. name and model attributes
# ---------------------------------------------------------------------------

def test_name_and_model_attributes():
    """name == 'claude_max', model matches what was passed."""
    with patch("anthropic.Anthropic"), \
         patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"):
        from mullet.core.providers.claude_max import ClaudeMaxProvider
        p = ClaudeMaxProvider(model="claude-opus-4-6")
        assert p.name == "claude_max"
        assert p.model == "claude-opus-4-6"


# ---------------------------------------------------------------------------
# 5. create() sends bearer token in Authorization header (SDK auth_token path)
# ---------------------------------------------------------------------------

def test_create_uses_bearer_token():
    """The Anthropic SDK is initialised with auth_token= so bearer auth is used."""
    captured_kwargs: dict = {}

    def fake_anthropic(**kwargs):
        captured_kwargs.update(kwargs)
        client = MagicMock()
        client.messages.create.return_value = _make_sdk_response()
        return client

    with patch("anthropic.Anthropic", side_effect=fake_anthropic), \
         patch("mullet.core.providers.claude_max._read_oauth_token", return_value="bearer-xyz"):
        from mullet.core.providers.claude_max import ClaudeMaxProvider
        p = ClaudeMaxProvider(model="claude-haiku-4-5")
        # auth_token= is how the SDK sets the Authorization: Bearer header
        assert captured_kwargs.get("auth_token") == "bearer-xyz"
        # The oauth beta header should also be present
        headers = captured_kwargs.get("default_headers", {})
        assert "anthropic-beta" in headers
        assert "oauth" in headers["anthropic-beta"]


# ---------------------------------------------------------------------------
# 6. create() returns a well-formed LLMResponse
# ---------------------------------------------------------------------------

def test_create_returns_llm_response():
    """create() returns an LLMResponse with content, usage, provider fields set."""
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _make_sdk_response("Test response")

    with patch("anthropic.Anthropic", return_value=fake_client), \
         patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"), \
         patch("mullet.core.providers.context_gate.assert_context_fits"):
        from mullet.core.providers.claude_max import ClaudeMaxProvider
        from mullet.core.providers.base import LLMResponse
        p = ClaudeMaxProvider(model="claude-haiku-4-5")
        resp = p.create(messages=[{"role": "user", "content": "hello"}])
        assert isinstance(resp, LLMResponse)
        assert resp.content == "Test response"
        assert resp.provider == "claude_max"
        assert resp.model == "claude-haiku-4-5"
        assert resp.usage is not None


# ---------------------------------------------------------------------------
# 7. HUD hook fires during create() and clears after
# ---------------------------------------------------------------------------

def test_hud_hook_fires_during_create(isolated_state):
    """agents.json is populated during create() and cleared after it returns."""
    seen_during: list[dict] = []
    fake_client = MagicMock()

    def fake_create(**kwargs):
        # Snapshot HUD state mid-call
        try:
            seen_during.append(json.loads(isolated_state.read_text()))
        except (OSError, json.JSONDecodeError):
            seen_during.append({})
        return _make_sdk_response("hi")

    fake_client.messages.create.side_effect = fake_create

    with patch("anthropic.Anthropic", return_value=fake_client), \
         patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"), \
         patch("mullet.core.providers.context_gate.assert_context_fits"):
        from mullet.core.providers.claude_max import ClaudeMaxProvider
        p = ClaudeMaxProvider(model="claude-haiku-4-5")
        p.create(messages=[{"role": "user", "content": "HUD probe"}])

    # During the call: HUD showed the claude_max agent
    assert seen_during, "No snapshot taken during create()"
    assert seen_during[0].get("active_count", 0) == 1
    assert seen_during[0]["agents"][0]["provider"] == "claude_max"

    # After the call: HUD is cleared
    after = json.loads(isolated_state.read_text())
    assert after["active_count"] == 0


# ---------------------------------------------------------------------------
# 8. Router never falls back to ClaudeMaxProvider when no ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------

def test_router_returns_none_when_no_api_key(monkeypatch):
    """ProviderResolver._create_provider returns None when API key absent — it no
    longer falls back to ClaudeMaxProvider (reusing the Claude Max/Pro/Free OAuth
    token in a third-party tool violates Anthropic's ToS, so that fallback was
    removed). This holds even if a real OAuth token is available on disk."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from mullet.core.router.decision import ProviderEntry
    from mullet.core.router.resolver import ProviderResolver

    with patch("mullet.core.providers.claude_max._read_oauth_token", return_value="oauth-tok"), \
         patch("anthropic.Anthropic"):
        entry = ProviderEntry(
            name="haiku", tier="cheap",
            models={"default": "claude-haiku-4-5-20251001"},
        )
        resolver = ProviderResolver(router=MagicMock())
        provider = resolver._create_provider(entry, model="claude-haiku-4-5-20251001")
        assert provider is None


# ---------------------------------------------------------------------------
# 9. Router prefers AnthropicProvider when ANTHROPIC_API_KEY is set (no regression)
# ---------------------------------------------------------------------------

def test_router_prefers_anthropic_when_api_key_set(monkeypatch):
    """AnthropicProvider is used when ANTHROPIC_API_KEY is present — Max is not tried."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    from mullet.core.router.decision import ProviderEntry
    from mullet.core.router.resolver import ProviderResolver
    from mullet.core.providers.anthropic import AnthropicProvider

    entry = ProviderEntry(
        name="haiku", tier="cheap",
        models={"default": "claude-haiku-4-5-20251001"},
    )
    resolver = ProviderResolver(router=MagicMock())
    provider = resolver._create_provider(entry, model="claude-haiku-4-5-20251001")
    assert isinstance(provider, AnthropicProvider)


# ---------------------------------------------------------------------------
# 10. health_check success and failure
# ---------------------------------------------------------------------------

def test_health_check_success():
    """health_check returns True when the ping create() call succeeds."""
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _make_sdk_response("pong")

    with patch("anthropic.Anthropic", return_value=fake_client), \
         patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"):
        from mullet.core.providers.claude_max import ClaudeMaxProvider
        p = ClaudeMaxProvider(model="claude-haiku-4-5")
        assert p.health_check() is True


def test_health_check_failure():
    """health_check returns False when the ping create() call raises."""
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = Exception("connection refused")

    with patch("anthropic.Anthropic", return_value=fake_client), \
         patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"):
        from mullet.core.providers.claude_max import ClaudeMaxProvider
        p = ClaudeMaxProvider(model="claude-haiku-4-5")
        assert p.health_check() is False


# ---------------------------------------------------------------------------
# 11. Missing line coverage
# ---------------------------------------------------------------------------

class TestClaudeMaxMissingLines:
    """Tests covering previously-uncovered branches in claude_max.py."""

    # -- _read_oauth_token -------------------------------------------------

    def test_read_oauth_token_missing_file(self):
        """_read_oauth_token returns None when the credentials file does not exist."""
        home = MagicMock()
        home.__truediv__ = lambda self, other: MagicMock(
            __truediv__=lambda self, other: MagicMock(exists=lambda: False)
        )
        with patch("mullet.core.providers.claude_max.Path.home", return_value=home):
            from mullet.core.providers.claude_max import _read_oauth_token
            assert _read_oauth_token() is None

    def test_read_oauth_token_json_decode_error(self):
        """_read_oauth_token returns None when the file contains invalid JSON."""
        final = MagicMock()
        final.exists.return_value = True
        final.read_text.return_value = "not-json{"
        mid = MagicMock()
        mid.__truediv__ = lambda self, other: final
        home = MagicMock()
        home.__truediv__ = lambda self, other: mid
        with patch("mullet.core.providers.claude_max.Path.home", return_value=home):
            from mullet.core.providers.claude_max import _read_oauth_token
            assert _read_oauth_token() is None

    def test_read_oauth_token_os_error(self):
        """_read_oauth_token returns None when reading the file raises OSError."""
        final = MagicMock()
        final.exists.return_value = True
        final.read_text.side_effect = OSError("permission denied")
        mid = MagicMock()
        mid.__truediv__ = lambda self, other: final
        home = MagicMock()
        home.__truediv__ = lambda self, other: mid
        with patch("mullet.core.providers.claude_max.Path.home", return_value=home):
            from mullet.core.providers.claude_max import _read_oauth_token
            assert _read_oauth_token() is None

    # -- create() kwargs shaping -------------------------------------------

    def test_create_with_system_message(self):
        """create() passes system= when a system message is present."""
        fake_client = MagicMock()
        fake_client.messages.create.return_value = _make_sdk_response("ok")
        with patch("anthropic.Anthropic", return_value=fake_client), \
             patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"), \
             patch("mullet.core.providers.context_gate.assert_context_fits"):
            from mullet.core.providers.claude_max import ClaudeMaxProvider
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            p.create(messages=[
                {"role": "system", "content": "Be nice"},
                {"role": "user", "content": "hi"},
            ])
            kwargs = fake_client.messages.create.call_args.kwargs
            assert kwargs.get("system") == "Be nice"

    def test_create_with_tools(self):
        """create() passes tools= when tools are provided."""
        fake_client = MagicMock()
        fake_client.messages.create.return_value = _make_sdk_response("ok")
        with patch("anthropic.Anthropic", return_value=fake_client), \
             patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"), \
             patch("mullet.core.providers.context_gate.assert_context_fits"):
            from mullet.core.providers.claude_max import ClaudeMaxProvider
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            p.create(
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"function": {"name": "calc", "description": "adds", "parameters": {"type": "object"}}}],
            )
            kwargs = fake_client.messages.create.call_args.kwargs
            assert "tools" in kwargs
            assert kwargs["tools"][0]["name"] == "calc"

    def test_create_with_reasoning_config(self):
        """create() passes thinking= when reasoning_config is provided."""
        fake_client = MagicMock()
        fake_client.messages.create.return_value = _make_sdk_response("ok")
        with patch("anthropic.Anthropic", return_value=fake_client), \
             patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"), \
             patch("mullet.core.providers.context_gate.assert_context_fits"):
            from mullet.core.providers.claude_max import ClaudeMaxProvider
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            p.create(
                messages=[{"role": "user", "content": "hi"}],
                reasoning_config={"type": "enabled", "budget_tokens": 1024},
            )
            kwargs = fake_client.messages.create.call_args.kwargs
            assert kwargs.get("thinking") == {"type": "enabled", "budget_tokens": 1024}

    # -- create_stream() ---------------------------------------------------

    def _stream_cm(self, events):
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.__iter__ = lambda self: iter(events)
        return cm

    def test_stream_content_block_delta_text(self):
        """Stream yields text from content_block_delta events."""
        event = MagicMock()
        event.type = "content_block_delta"
        delta = MagicMock()
        delta.text = "hello"
        event.delta = delta
        fake_client = MagicMock()
        fake_client.messages.stream.return_value = self._stream_cm([event])
        with patch("anthropic.Anthropic", return_value=fake_client), \
             patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"):
            from mullet.core.providers.claude_max import ClaudeMaxProvider
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            chunks = list(p.create_stream(messages=[{"role": "user", "content": "hi"}]))
            assert len(chunks) == 1
            assert chunks[0].content == "hello"

    def test_stream_content_block_delta_partial_json(self):
        """Stream yields tool_call_delta from partial_json events."""
        event = MagicMock()
        event.type = "content_block_delta"
        delta = MagicMock(spec=["partial_json"])
        delta.partial_json = '{"a":1}'
        event.delta = delta
        fake_client = MagicMock()
        fake_client.messages.stream.return_value = self._stream_cm([event])
        with patch("anthropic.Anthropic", return_value=fake_client), \
             patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"):
            from mullet.core.providers.claude_max import ClaudeMaxProvider
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            chunks = list(p.create_stream(messages=[{"role": "user", "content": "hi"}]))
            assert chunks[0].tool_call_delta == {"partial_json": '{"a":1}'}

    def test_stream_content_block_start_tool_use(self):
        """Stream yields tool_call_delta from content_block_start tool_use events."""
        event = MagicMock()
        event.type = "content_block_start"
        block = MagicMock()
        block.type = "tool_use"
        block.id = "tu_1"
        block.name = "calc"
        event.content_block = block
        fake_client = MagicMock()
        fake_client.messages.stream.return_value = self._stream_cm([event])
        with patch("anthropic.Anthropic", return_value=fake_client), \
             patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"):
            from mullet.core.providers.claude_max import ClaudeMaxProvider
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            chunks = list(p.create_stream(messages=[{"role": "user", "content": "hi"}]))
            assert chunks[0].tool_call_delta == {
                "id": "tu_1",
                "type": "function",
                "function": {"name": "calc", "arguments": ""},
            }

    def test_stream_message_delta_stop_reason(self):
        """Stream yields finish_reason from message_delta events."""
        event = MagicMock()
        event.type = "message_delta"
        delta = MagicMock()
        delta.stop_reason = "end_turn"
        event.delta = delta
        fake_client = MagicMock()
        fake_client.messages.stream.return_value = self._stream_cm([event])
        with patch("anthropic.Anthropic", return_value=fake_client), \
             patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"):
            from mullet.core.providers.claude_max import ClaudeMaxProvider
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            chunks = list(p.create_stream(messages=[{"role": "user", "content": "hi"}]))
            assert chunks[0].finish_reason == "end_turn"

    def test_stream_with_system(self):
        """create_stream passes system= when a system message is present."""
        fake_client = MagicMock()
        fake_client.messages.stream.return_value = self._stream_cm([])
        with patch("anthropic.Anthropic", return_value=fake_client), \
             patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"):
            from mullet.core.providers.claude_max import ClaudeMaxProvider
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            list(p.create_stream(messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ]))
            kwargs = fake_client.messages.stream.call_args.kwargs
            assert kwargs.get("system") == "sys"

    def test_stream_with_tools(self):
        """create_stream passes tools= when tools are provided."""
        fake_client = MagicMock()
        fake_client.messages.stream.return_value = self._stream_cm([])
        with patch("anthropic.Anthropic", return_value=fake_client), \
             patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"):
            from mullet.core.providers.claude_max import ClaudeMaxProvider
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            list(p.create_stream(
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"function": {"name": "calc", "description": "adds", "parameters": {"type": "object"}}}],
            ))
            kwargs = fake_client.messages.stream.call_args.kwargs
            assert "tools" in kwargs
            assert kwargs["tools"][0]["name"] == "calc"

    def test_stream_temperature_not_one(self):
        """create_stream passes temperature when it is not 1.0."""
        fake_client = MagicMock()
        fake_client.messages.stream.return_value = self._stream_cm([])
        with patch("anthropic.Anthropic", return_value=fake_client), \
             patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"):
            from mullet.core.providers.claude_max import ClaudeMaxProvider
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            list(p.create_stream(
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.5,
            ))
            kwargs = fake_client.messages.stream.call_args.kwargs
            assert kwargs.get("temperature") == 0.5

    # -- _call_with_retry --------------------------------------------------

    def test_retry_auth_error_401(self):
        """401 auth errors raise RuntimeError immediately without retry."""
        fake_client = MagicMock()
        fake_client.messages.create.side_effect = Exception("401 Unauthorized")
        with patch("anthropic.Anthropic", return_value=fake_client), \
             patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"), \
             patch("mullet.core.providers.context_gate.assert_context_fits"):
            from mullet.core.providers.claude_max import ClaudeMaxProvider
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            with pytest.raises(RuntimeError, match="authentication failed"):
                p.create(messages=[{"role": "user", "content": "hi"}])

    def test_retry_429_then_success(self):
        """429 errors sleep and retry, then succeed."""
        fake_client = MagicMock()
        fake_client.messages.create.side_effect = [
            Exception("429 rate_limit_error"),
            _make_sdk_response("ok"),
        ]
        with patch("anthropic.Anthropic", return_value=fake_client), \
             patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"), \
             patch("mullet.core.providers.context_gate.assert_context_fits"), \
             patch("mullet.core.providers.claude_max._time.sleep") as mock_sleep:
            from mullet.core.providers.claude_max import ClaudeMaxProvider
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            resp = p.create(messages=[{"role": "user", "content": "hi"}])
            assert resp.content == "ok"
            mock_sleep.assert_called_once()

    def test_retry_constraint_self_heal_success(self):
        """Constraint error is fixed by parse_constraint_error and retry succeeds."""
        fake_client = MagicMock()
        fake_client.messages.create.side_effect = [
            Exception("unsupported parameter: temperature"),
            _make_sdk_response("healed"),
        ]
        with patch("anthropic.Anthropic", return_value=fake_client), \
             patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"), \
             patch("mullet.core.providers.context_gate.assert_context_fits"), \
             patch("mullet.core.providers.claude_max._time.sleep"), \
             patch("mullet.core.providers.claude_max.parse_constraint_error", return_value=("temperature", "not supported")), \
             patch("mullet.core.providers.claude_max.get_constraint_registry") as mock_reg:
            from mullet.core.providers.claude_max import ClaudeMaxProvider
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            resp = p.create(messages=[{"role": "user", "content": "hi"}])
            assert resp.content == "healed"
            mock_reg.return_value.learn.assert_called_once_with("claude-haiku-4-5", "temperature", "not supported")
            assert fake_client.messages.create.call_count == 2

    def test_retry_constraint_self_heal_then_fails(self):
        """Constraint fix is applied but the retry still fails — exception propagates."""
        fake_client = MagicMock()
        fake_client.messages.create.side_effect = [
            Exception("unsupported parameter: temperature"),
            Exception("another error"),
        ]
        with patch("anthropic.Anthropic", return_value=fake_client), \
             patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"), \
             patch("mullet.core.providers.context_gate.assert_context_fits"), \
             patch("mullet.core.providers.claude_max._time.sleep"), \
             patch("mullet.core.providers.claude_max.parse_constraint_error", return_value=("temperature", "not supported")), \
             patch("mullet.core.providers.claude_max.get_constraint_registry"):
            from mullet.core.providers.claude_max import ClaudeMaxProvider
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            with pytest.raises(Exception, match="another error"):
                p.create(messages=[{"role": "user", "content": "hi"}])

    def test_retry_generic_error_raises_immediately(self):
        """Generic non-retriable errors raise immediately."""
        fake_client = MagicMock()
        fake_client.messages.create.side_effect = Exception("generic failure")
        with patch("anthropic.Anthropic", return_value=fake_client), \
             patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"), \
             patch("mullet.core.providers.context_gate.assert_context_fits"), \
             patch("mullet.core.providers.claude_max.parse_constraint_error", return_value=None):
            from mullet.core.providers.claude_max import ClaudeMaxProvider
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            with pytest.raises(Exception, match="generic failure"):
                p.create(messages=[{"role": "user", "content": "hi"}])

    def test_retry_exhausted_raises_last_error(self):
        """After max_retries, the last error is raised."""
        fake_client = MagicMock()
        fake_client.messages.create.side_effect = [
            Exception("500 server error"),
            Exception("500 server error"),
            Exception("500 server error"),
        ]
        with patch("anthropic.Anthropic", return_value=fake_client), \
             patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"), \
             patch("mullet.core.providers.context_gate.assert_context_fits"), \
             patch("mullet.core.providers.claude_max._time.sleep") as mock_sleep:
            from mullet.core.providers.claude_max import ClaudeMaxProvider
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            with pytest.raises(Exception, match="500 server error"):
                p.create(messages=[{"role": "user", "content": "hi"}])
            assert mock_sleep.call_count == 3

    # -- _log exception swallowing -----------------------------------------

    def test_log_exception_swallowed(self):
        """Exceptions inside _log are swallowed and do not propagate."""
        with patch("anthropic.Anthropic"), \
             patch("mullet.core.providers.claude_max._read_oauth_token", return_value="tok"), \
             patch("mullet.core.logging.interaction_log.log_interaction", side_effect=Exception("boom")):
            from mullet.core.providers.claude_max import ClaudeMaxProvider
            p = ClaudeMaxProvider(model="claude-haiku-4-5")
            p._log({}, None, 0, success=True)  # should not raise

    # -- _convert_messages -------------------------------------------------

    def test_convert_messages_flush_tools_before_next_user(self):
        """Tool results are flushed as a user content block before the next user message."""
        p = _make_provider()
        messages = [
            {"role": "user", "content": "first"},
            {"role": "tool", "tool_call_id": "t1", "content": "res1"},
            {"role": "user", "content": "second"},
        ]
        system, anthropic_messages = p._convert_messages(messages)
        assert system == ""
        assert anthropic_messages == [
            {"role": "user", "content": "first"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "res1"}]},
            {"role": "user", "content": "second"},
        ]

    def test_convert_messages_extracts_system(self):
        """System messages are extracted and returned separately."""
        p = _make_provider()
        messages = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "hi"},
        ]
        system, anthropic_messages = p._convert_messages(messages)
        assert system == "sys prompt"
        assert anthropic_messages == [{"role": "user", "content": "hi"}]

    def test_convert_messages_user_role(self):
        """User messages are passed through."""
        p = _make_provider()
        messages = [{"role": "user", "content": "hello"}]
        system, anthropic_messages = p._convert_messages(messages)
        assert system == ""
        assert anthropic_messages == [{"role": "user", "content": "hello"}]

    def test_convert_messages_assistant_tool_calls_json_string(self):
        """Assistant tool_calls with JSON-string arguments are parsed."""
        p = _make_provider()
        messages = [{
            "role": "assistant",
            "content": "Using tool",
            "tool_calls": [{
                "id": "tc1",
                "function": {"name": "calc", "arguments": '{"a":1}'},
            }],
        }]
        system, anthropic_messages = p._convert_messages(messages)
        assert anthropic_messages[0]["role"] == "assistant"
        assert anthropic_messages[0]["content"] == [
            {"type": "text", "text": "Using tool"},
            {"type": "tool_use", "id": "tc1", "name": "calc", "input": {"a": 1}},
        ]

    def test_convert_messages_assistant_tool_calls_invalid_json(self):
        """Assistant tool_calls with invalid JSON arguments fall back to {}."""
        p = _make_provider()
        messages = [{
            "role": "assistant",
            "content": "Using tool",
            "tool_calls": [{
                "id": "tc1",
                "function": {"name": "calc", "arguments": "not json"},
            }],
        }]
        system, anthropic_messages = p._convert_messages(messages)
        assert anthropic_messages[0]["content"] == [
            {"type": "text", "text": "Using tool"},
            {"type": "tool_use", "id": "tc1", "name": "calc", "input": {}},
        ]

    def test_convert_messages_tool_role(self):
        """Tool results are batched and flushed at the end."""
        p = _make_provider()
        messages = [
            {"role": "user", "content": "call tool"},
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
        ]
        system, anthropic_messages = p._convert_messages(messages)
        assert anthropic_messages == [
            {"role": "user", "content": "call tool"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tc1", "content": "result"}]},
        ]

    # -- _convert_tools ----------------------------------------------------

    def test_convert_tools(self):
        """OpenAI-format tools are converted to Anthropic format."""
        p = _make_provider()
        tools = [{"function": {"name": "calc", "description": "adds", "parameters": {"type": "object", "properties": {}}}}]
        result = p._convert_tools(tools)
        assert result == [{
            "name": "calc",
            "description": "adds",
            "input_schema": {"type": "object", "properties": {}},
        }]

    # -- _to_response ------------------------------------------------------

    def test_to_response_with_tool_use(self):
        """tool_use blocks are converted to tool_calls in the LLMResponse."""
        p = _make_provider()
        response = MagicMock()
        response.content = []
        block = MagicMock()
        block.type = "tool_use"
        block.id = "tu1"
        block.name = "calc"
        block.input = {"x": 1}
        response.content.append(block)
        response.stop_reason = "end_turn"
        response.usage = MagicMock(input_tokens=5, output_tokens=3, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        response.model = "claude-haiku-4-5"
        resp = p._to_response(response)
        assert resp.tool_calls == [{
            "id": "tu1",
            "type": "function",
            "function": {"name": "calc", "arguments": '{"x": 1}'},
        }]

    def test_to_response_stop_reason_tool_use(self):
        """stop_reason 'tool_use' maps to finish_reason 'tool_calls'."""
        p = _make_provider()
        response = MagicMock()
        response.content = []
        response.stop_reason = "tool_use"
        response.usage = MagicMock(input_tokens=5, output_tokens=3, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        response.model = "claude-haiku-4-5"
        resp = p._to_response(response)
        assert resp.finish_reason == "tool_calls"

    def test_to_response_stop_reason_max_tokens(self):
        """stop_reason 'max_tokens' maps to finish_reason 'length'."""
        p = _make_provider()
        response = MagicMock()
        response.content = []
        response.stop_reason = "max_tokens"
        response.usage = MagicMock(input_tokens=5, output_tokens=3, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        response.model = "claude-haiku-4-5"
        resp = p._to_response(response)
        assert resp.finish_reason == "length"

    def test_read_oauth_token_valid_json_missing_key(self):
        """_read_oauth_token returns None when the JSON is valid but lacks the accessToken key."""
        final = MagicMock()
        final.exists.return_value = True
        final.read_text.return_value = '{"otherKey": true}'
        mid = MagicMock()
        mid.__truediv__ = lambda self, other: final
        home = MagicMock()
        home.__truediv__ = lambda self, other: mid
        with patch("mullet.core.providers.claude_max.Path.home", return_value=home):
            from mullet.core.providers.claude_max import _read_oauth_token
            assert _read_oauth_token() is None

    def test_supports_tool_calling(self):
        """supports_tool_calling property returns True."""
        p = _make_provider()
        assert p.supports_tool_calling is True
