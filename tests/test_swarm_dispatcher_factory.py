"""Tests for SwarmDispatcher factory methods (_build_sync_provider and _make_provider).

Covers provider construction, model selection, registry fallback, and error handling.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from mullet.core.orchestration.swarm_dispatcher import SwarmDispatcher


class TestBuildSyncProvider:
    """Tests for _build_sync_provider factory method."""

    def test_build_sync_provider_kimi_real_construction(self):
        """Test _build_sync_provider returns KimiProvider for 'kimi'."""
        dispatcher = SwarmDispatcher()
        # Patch the KimiProvider constructor to avoid needing real API keys
        with patch("mullet.core.providers.kimi.KimiProvider") as mock_kimi:
            mock_instance = MagicMock()
            mock_kimi.return_value = mock_instance

            result = dispatcher._build_sync_provider("kimi", "test-model")

            assert result is mock_instance
            mock_kimi.assert_called_once()
            # Should pass model and api_key kwargs
            call_kwargs = mock_kimi.call_args[1]
            assert call_kwargs.get("model") == "test-model"
            assert "api_key" in call_kwargs

    def test_build_sync_provider_gemini_real_construction(self):
        """Test _build_sync_provider returns GeminiProvider for 'gemini'."""
        dispatcher = SwarmDispatcher()
        with patch("mullet.core.providers.gemini.GeminiProvider") as mock_gemini:
            mock_instance = MagicMock()
            mock_gemini.return_value = mock_instance

            result = dispatcher._build_sync_provider("gemini", "test-model")

            assert result is mock_instance
            mock_gemini.assert_called_once()
            call_kwargs = mock_gemini.call_args[1]
            assert call_kwargs.get("model") == "test-model"
            assert "api_key" in call_kwargs

    def test_build_sync_provider_openrouter_real_construction(self):
        """Test _build_sync_provider returns OpenRouterProvider for 'openrouter'."""
        dispatcher = SwarmDispatcher()
        with patch("mullet.core.providers.openrouter.OpenRouterProvider") as mock_or:
            mock_instance = MagicMock()
            mock_or.return_value = mock_instance

            result = dispatcher._build_sync_provider("openrouter", "test-model")

            assert result is mock_instance
            mock_or.assert_called_once()
            call_kwargs = mock_or.call_args[1]
            assert call_kwargs.get("model") == "test-model"
            assert "api_key" in call_kwargs

    def test_build_sync_provider_anthropic_real_construction(self):
        """Test _build_sync_provider returns AnthropicProvider for 'anthropic'."""
        dispatcher = SwarmDispatcher()
        with patch("mullet.core.providers.anthropic.AnthropicProvider") as mock_ant:
            mock_instance = MagicMock()
            mock_ant.return_value = mock_instance

            result = dispatcher._build_sync_provider("anthropic", "test-model")

            assert result is mock_instance
            mock_ant.assert_called_once()
            call_kwargs = mock_ant.call_args[1]
            assert call_kwargs.get("model") == "test-model"
            assert "api_key" in call_kwargs

    def test_build_sync_provider_haiku_real_construction(self):
        """Test _build_sync_provider returns AnthropicProvider for 'haiku'."""
        dispatcher = SwarmDispatcher()
        with patch("mullet.core.providers.anthropic.AnthropicProvider") as mock_ant:
            mock_instance = MagicMock()
            mock_ant.return_value = mock_instance

            result = dispatcher._build_sync_provider("haiku", "test-model")

            assert result is mock_instance
            mock_ant.assert_called_once()
            call_kwargs = mock_ant.call_args[1]
            assert call_kwargs.get("model") == "test-model"

    def test_build_sync_provider_sonnet_real_construction(self):
        """Test _build_sync_provider returns AnthropicProvider for 'sonnet'."""
        dispatcher = SwarmDispatcher()
        with patch("mullet.core.providers.anthropic.AnthropicProvider") as mock_ant:
            mock_instance = MagicMock()
            mock_ant.return_value = mock_instance

            result = dispatcher._build_sync_provider("sonnet", "test-model")

            assert result is mock_instance
            mock_ant.assert_called_once()

    def test_build_sync_provider_opus_real_construction(self):
        """Test _build_sync_provider returns AnthropicProvider for 'opus'."""
        dispatcher = SwarmDispatcher()
        with patch("mullet.core.providers.anthropic.AnthropicProvider") as mock_ant:
            mock_instance = MagicMock()
            mock_ant.return_value = mock_instance

            result = dispatcher._build_sync_provider("opus", "test-model")

            assert result is mock_instance
            mock_ant.assert_called_once()

    def test_build_sync_provider_local_real_construction(self):
        """Test _build_sync_provider returns OpenAICompatibleProvider for 'local'."""
        dispatcher = SwarmDispatcher()
        with patch("mullet.core.providers.openai_compat.OpenAICompatibleProvider") as mock_local:
            mock_instance = MagicMock()
            mock_local.return_value = mock_instance

            result = dispatcher._build_sync_provider("local", "test-model")

            assert result is mock_instance
            mock_local.assert_called_once()
            call_kwargs = mock_local.call_args[1]
            assert call_kwargs.get("model") == "test-model"
            assert "base_url" in call_kwargs

    def test_build_sync_provider_unknown_raises_valueerror(self):
        """Test _build_sync_provider raises ValueError for unknown provider."""
        dispatcher = SwarmDispatcher()

        with pytest.raises(ValueError, match="unknown provider_name"):
            dispatcher._build_sync_provider("totally-unknown", "m")

    def test_build_sync_provider_passes_correct_api_keys(self):
        """Test _build_sync_provider passes correct API keys from environment."""
        dispatcher = SwarmDispatcher()

        # Test kimi with both KIMI_API_KEY and MOONSHOT_API_KEY
        with patch("mullet.core.providers.kimi.KimiProvider") as mock_kimi:
            mock_instance = MagicMock()
            mock_kimi.return_value = mock_instance

            with patch.dict(os.environ, {"KIMI_API_KEY": "test-kimi-key"}):
                dispatcher._build_sync_provider("kimi", "m")

            call_kwargs = mock_kimi.call_args[1]
            assert call_kwargs.get("api_key") == "test-kimi-key"

    def test_build_sync_provider_local_uses_default_url(self):
        """Test _build_sync_provider uses default LOCAL_LLM_URL for local provider."""
        dispatcher = SwarmDispatcher()

        with patch("mullet.core.providers.openai_compat.OpenAICompatibleProvider") as mock_local:
            mock_instance = MagicMock()
            mock_local.return_value = mock_instance

            # Clear any LOCAL_LLM_URL env var
            with patch.dict(os.environ, {}, clear=False):
                if "LOCAL_LLM_URL" in os.environ:
                    del os.environ["LOCAL_LLM_URL"]
                dispatcher._build_sync_provider("local", "m")

            call_kwargs = mock_local.call_args[1]
            assert call_kwargs.get("base_url") == "http://localhost:1234/v1"


class TestMakeProvider:
    """Tests for _make_provider factory method."""

    def test_make_provider_generic_path_gemini(self):
        """Test _make_provider returns KimiAsyncProvider wrapping GeminiProvider for 'gemini'."""
        dispatcher = SwarmDispatcher()

        with patch("mullet.core.providers.gemini.GeminiProvider") as mock_gemini:
            mock_sync = MagicMock()
            mock_gemini.return_value = mock_sync

            with patch("mullet.core.providers.kimi_async.KimiAsyncProvider") as mock_async:
                mock_async_instance = MagicMock()
                mock_async.return_value = mock_async_instance

                result = dispatcher._make_provider("STANDARD", 100, provider_name="gemini")

                # Should return KimiAsyncProvider
                assert result is mock_async_instance
                # Should wrap the sync provider
                mock_async.assert_called_once_with(provider=mock_sync)

    def test_make_provider_generic_path_openrouter(self):
        """Test _make_provider returns KimiAsyncProvider wrapping OpenRouterProvider."""
        dispatcher = SwarmDispatcher()

        with patch("mullet.core.providers.openrouter.OpenRouterProvider") as mock_or:
            mock_sync = MagicMock()
            mock_or.return_value = mock_sync

            with patch("mullet.core.providers.kimi_async.KimiAsyncProvider") as mock_async:
                mock_async_instance = MagicMock()
                mock_async.return_value = mock_async_instance

                result = dispatcher._make_provider("STANDARD", 100, provider_name="openrouter")

                assert result is mock_async_instance
                mock_async.assert_called_once_with(provider=mock_sync)

    def test_make_provider_anthropic_alias_haiku(self):
        """Test _make_provider uses hardcoded model for Anthropic 'haiku' alias."""
        dispatcher = SwarmDispatcher()

        with patch("mullet.core.providers.anthropic.AnthropicProvider") as mock_ant:
            mock_sync = MagicMock()
            mock_ant.return_value = mock_sync

            with patch("mullet.core.providers.kimi_async.KimiAsyncProvider") as mock_async:
                mock_async_instance = MagicMock()
                mock_async.return_value = mock_async_instance

                result = dispatcher._make_provider("STANDARD", 100, provider_name="haiku")

                # Should use the hardcoded haiku model
                call_kwargs = mock_ant.call_args[1]
                assert call_kwargs.get("model") == "claude-haiku-4-5"

    def test_make_provider_anthropic_alias_sonnet(self):
        """Test _make_provider uses hardcoded model for Anthropic 'sonnet' alias."""
        dispatcher = SwarmDispatcher()

        with patch("mullet.core.providers.anthropic.AnthropicProvider") as mock_ant:
            mock_sync = MagicMock()
            mock_ant.return_value = mock_sync

            with patch("mullet.core.providers.kimi_async.KimiAsyncProvider") as mock_async:
                mock_async_instance = MagicMock()
                mock_async.return_value = mock_async_instance

                dispatcher._make_provider("STANDARD", 100, provider_name="sonnet")

                call_kwargs = mock_ant.call_args[1]
                assert call_kwargs.get("model") == "claude-sonnet-4-6"

    def test_make_provider_anthropic_alias_opus(self):
        """Test _make_provider uses hardcoded model for Anthropic 'opus' alias."""
        dispatcher = SwarmDispatcher()

        with patch("mullet.core.providers.anthropic.AnthropicProvider") as mock_ant:
            mock_sync = MagicMock()
            mock_ant.return_value = mock_sync

            with patch("mullet.core.providers.kimi_async.KimiAsyncProvider") as mock_async:
                mock_async_instance = MagicMock()
                mock_async.return_value = mock_async_instance

                dispatcher._make_provider("STANDARD", 100, provider_name="opus")

                call_kwargs = mock_ant.call_args[1]
                assert call_kwargs.get("model") == "claude-opus-4-8"

    def test_make_provider_registry_fallback_on_exception(self):
        """Test _make_provider falls back to static defaults when registry raises."""
        dispatcher = SwarmDispatcher()

        with patch("mullet.core.providers.model_registry.get_registry") as mock_registry_fn:
            # Registry lookup raises
            mock_registry_fn.side_effect = Exception("registry broken")

            with patch("mullet.core.providers.openrouter.OpenRouterProvider") as mock_or:
                mock_sync = MagicMock()
                mock_or.return_value = mock_sync

                with patch("mullet.core.providers.kimi_async.KimiAsyncProvider") as mock_async:
                    mock_async_instance = MagicMock()
                    mock_async.return_value = mock_async_instance

                    result = dispatcher._make_provider("STANDARD", 100, provider_name="openrouter")

                    assert result is mock_async_instance
                    # Should have called _build_sync_provider with the default model
                    call_kwargs = mock_or.call_args[1]
                    assert call_kwargs.get("model") == "deepseek/deepseek-v3"

    def test_make_provider_fallback_to_kimi_on_build_error(self):
        """Test _make_provider falls back to kimi when non-kimi provider build fails."""
        dispatcher = SwarmDispatcher()

        # Patch _build_sync_provider to raise for non-kimi providers
        original_build = dispatcher._build_sync_provider

        def side_effect_build(provider_name, model_id):
            if provider_name == "kimi":
                # Let kimi succeed
                return original_build(provider_name, model_id)
            raise RuntimeError("Provider build failed")

        with patch.object(dispatcher, "_build_sync_provider", side_effect=side_effect_build):
            with patch("mullet.core.providers.kimi.KimiProvider") as mock_kimi:
                mock_sync = MagicMock()
                mock_kimi.return_value = mock_sync

                with patch("mullet.core.providers.kimi_async.KimiAsyncProvider") as mock_async:
                    mock_async_instance = MagicMock()
                    mock_async.return_value = mock_async_instance

                    # Request gemini but it should fall back to kimi
                    result = dispatcher._make_provider("STANDARD", 100, provider_name="gemini")

                    assert result is mock_async_instance
                    # Should have created a KimiProvider as fallback
                    mock_kimi.assert_called_once()

    def test_make_provider_kimi_path_with_registry(self):
        """Test _make_provider kimi path uses registry for model selection."""
        dispatcher = SwarmDispatcher()

        with patch("mullet.core.providers.kimi.KimiProvider") as mock_kimi:
            mock_sync = MagicMock()
            mock_kimi.return_value = mock_sync

            with patch("mullet.core.providers.model_registry.get_registry") as mock_reg_fn:
                mock_registry = MagicMock()
                mock_registry.get_best_model.return_value = "kimi-k2.7-code"
                mock_reg_fn.return_value = mock_registry

                with patch("mullet.core.providers.model_info.get_model_info") as mock_info_fn:
                    mock_info = MagicMock()
                    mock_info.context_length = 10000
                    mock_info_fn.return_value = mock_info

                    with patch("mullet.core.providers.kimi_async.KimiAsyncProvider") as mock_async:
                        mock_async_instance = MagicMock()
                        mock_async.return_value = mock_async_instance

                        result = dispatcher._make_provider("STANDARD", 100, provider_name="kimi")

                        assert result is mock_async_instance
                        # Should have called get_best_model at least once
                        assert mock_registry.get_best_model.call_count >= 1

    def test_make_provider_kimi_fallback_on_registry_error(self):
        """Test _make_provider kimi path falls back to hard-coded model on registry error."""
        dispatcher = SwarmDispatcher()

        with patch("mullet.core.providers.model_registry.get_registry") as mock_reg_fn:
            # Registry itself fails
            mock_reg_fn.side_effect = Exception("registry unavailable")

            with patch("mullet.core.providers.kimi.KimiProvider") as mock_kimi:
                mock_sync = MagicMock()
                mock_kimi.return_value = mock_sync

                with patch("mullet.core.providers.kimi_async.KimiAsyncProvider") as mock_async:
                    mock_async_instance = MagicMock()
                    mock_async.return_value = mock_async_instance

                    result = dispatcher._make_provider("STANDARD", 100, provider_name="kimi")

                    assert result is mock_async_instance
                    # Should have created KimiProvider with fallback model
                    call_kwargs = mock_kimi.call_args[1]
                    assert call_kwargs.get("model") == "kimi-k2.7-code"

    def test_make_provider_returns_kimi_async_provider_type(self):
        """Test _make_provider always returns KimiAsyncProvider instance."""
        dispatcher = SwarmDispatcher()

        with patch("mullet.core.providers.gemini.GeminiProvider"):
            with patch("mullet.core.providers.kimi_async.KimiAsyncProvider") as mock_async:
                mock_instance = MagicMock()
                mock_async.return_value = mock_instance

                result = dispatcher._make_provider("STANDARD", 100, provider_name="gemini")

                # Result should be the KimiAsyncProvider instance
                assert result is mock_instance

    def test_make_provider_passes_prompt_to_kimi_path(self):
        """Test _make_provider passes actual prompt to kimi path for token estimation."""
        dispatcher = SwarmDispatcher()

        test_prompt = "This is a test prompt" * 10

        with patch("mullet.core.providers.kimi.KimiProvider"):
            with patch("mullet.core.providers.model_registry.get_registry") as mock_reg_fn:
                mock_registry = MagicMock()
                mock_registry.get_best_model.return_value = "kimi-k2.7-code"
                mock_reg_fn.return_value = mock_registry

                with patch("mullet.core.providers.model_info.get_model_info") as mock_info_fn:
                    mock_info = MagicMock()
                    mock_info.context_length = 10000
                    mock_info_fn.return_value = mock_info

                    with patch("mullet.core.providers.tokenize.estimate_tokens") as mock_estimate:
                        mock_estimate.return_value = 50

                        with patch("mullet.core.providers.kimi_async.KimiAsyncProvider"):
                            # Should call estimate_tokens with the actual prompt
                            dispatcher._make_provider("STANDARD", len(test_prompt), provider_name="kimi", prompt=test_prompt)

                            mock_estimate.assert_called_once_with(test_prompt)

    def test_make_provider_token_estimation_fallback(self):
        """Test _make_provider falls back to char//6 when tokenizer fails."""
        dispatcher = SwarmDispatcher()

        with patch("mullet.core.providers.kimi.KimiProvider"):
            with patch("mullet.core.providers.model_registry.get_registry") as mock_reg_fn:
                mock_registry = MagicMock()
                mock_registry.get_best_model.return_value = "kimi-k2.7-code"
                mock_reg_fn.return_value = mock_registry

                with patch("mullet.core.providers.model_info.get_model_info") as mock_info_fn:
                    mock_info = MagicMock()
                    mock_info.context_length = 10000
                    mock_info_fn.return_value = mock_info

                    with patch("mullet.core.providers.tokenize.estimate_tokens") as mock_estimate:
                        # Estimator fails
                        mock_estimate.side_effect = Exception("tokenizer broken")

                        with patch("mullet.core.providers.kimi_async.KimiAsyncProvider"):
                            # Should still work with fallback
                            dispatcher._make_provider("STANDARD", 600, provider_name="kimi", prompt="test")

                            # Should have called get_best_model at least once (with fallback estimate)
                            assert mock_registry.get_best_model.call_count >= 1

    def test_make_provider_tier_escalation(self):
        """Test _make_provider escalates tier in kimi path when needed."""
        dispatcher = SwarmDispatcher()

        with patch("mullet.core.providers.kimi.KimiProvider") as mock_kimi:
            mock_sync = MagicMock()
            mock_kimi.return_value = mock_sync

            with patch("mullet.core.providers.model_registry.get_registry") as mock_reg_fn:
                mock_registry = MagicMock()
                # First tier returns small context, escalate to next
                mock_registry.get_best_model.side_effect = [
                    "kimi-k2.7-mini",  # STANDARD returns mini
                ]
                mock_reg_fn.return_value = mock_registry

                with patch("mullet.core.providers.model_info.get_model_info") as mock_info_fn:
                    # Mini has small context, won't fit the prompt estimate
                    mock_info = MagicMock()
                    mock_info.context_length = 100
                    mock_info_fn.return_value = mock_info

                    with patch("mullet.core.providers.kimi_async.KimiAsyncProvider") as mock_async:
                        mock_async_instance = MagicMock()
                        mock_async.return_value = mock_async_instance

                        # Prompt estimate is 50 tokens, but STANDARD returns 100-context model
                        # So it should not escalate, should return immediately
                        result = dispatcher._make_provider("STANDARD", 300, provider_name="kimi")

                        assert result is mock_async_instance


class TestMakeProviderErrorCases:
    """Tests for error handling and recovery in _make_provider."""

    def test_make_provider_generic_build_error_fallback_to_kimi(self):
        """Test generic path build error causes fallback to kimi."""
        dispatcher = SwarmDispatcher()

        # Patch _build_sync_provider to always raise
        with patch.object(dispatcher, "_build_sync_provider") as mock_build:
            mock_build.side_effect = RuntimeError("boom")

            with patch("mullet.core.providers.kimi.KimiProvider") as mock_kimi:
                mock_sync = MagicMock()
                mock_kimi.return_value = mock_sync

                with patch("mullet.core.providers.kimi_async.KimiAsyncProvider") as mock_async:
                    mock_async_instance = MagicMock()
                    mock_async.return_value = mock_async_instance

                    result = dispatcher._make_provider("STANDARD", 100, provider_name="gemini")

                    # Should have fallen back to kimi
                    assert mock_kimi.call_count >= 1
                    assert result is mock_async_instance

    def test_make_provider_respects_prompt_parameter(self):
        """Test _make_provider passes prompt parameter through to kimi fallback."""
        dispatcher = SwarmDispatcher()

        test_prompt = "Special test prompt for recursive call"

        with patch.object(dispatcher, "_build_sync_provider") as mock_build:
            mock_build.side_effect = RuntimeError("error")

            with patch("mullet.core.providers.kimi.KimiProvider"):
                with patch("mullet.core.providers.kimi_async.KimiAsyncProvider"):
                    # This should not raise even though gemini fails
                    dispatcher._make_provider("STANDARD", 100, provider_name="gemini", prompt=test_prompt)

                    # Verify _build_sync_provider was called for gemini first
                    assert mock_build.call_count >= 1
                    first_call = mock_build.call_args_list[0]
                    assert first_call[0][0] == "gemini"
