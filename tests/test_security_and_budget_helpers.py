"""Tests for credential_guard, predigest pure methods, and budget_advisor helpers."""

from mullet.core.tools.credential_guard import (
    CREDENTIAL_FILENAMES,
    CREDENTIAL_EXTENSIONS,
    CREDENTIAL_PATTERNS,
    CredentialGuard,
)
from mullet.core.knowledge.predigest import PreDigestPipeline, PREDIGEST_THRESHOLD_CHARS
from mullet.core.budget_advisor import (
    _estimate_tokens,
    _estimate_cost,
    should_preflight,
    format_preflight_message,
    LaneRecommendation,
)


# ---------------------------------------------------------------------------
# CredentialGuard — constants
# ---------------------------------------------------------------------------

class TestCredentialConstants:
    def test_filenames_is_frozenset(self):
        assert isinstance(CREDENTIAL_FILENAMES, frozenset)

    def test_extensions_is_frozenset(self):
        assert isinstance(CREDENTIAL_EXTENSIONS, frozenset)

    def test_patterns_is_list(self):
        assert isinstance(CREDENTIAL_PATTERNS, list)

    def test_env_in_filenames(self):
        assert ".env" in CREDENTIAL_FILENAMES

    def test_pem_in_extensions(self):
        assert ".pem" in CREDENTIAL_EXTENSIONS

    def test_patterns_are_compiled(self):
        for p in CREDENTIAL_PATTERNS:
            assert hasattr(p, "search")


# ---------------------------------------------------------------------------
# CredentialGuard.is_credential_file
# ---------------------------------------------------------------------------

class TestIsCredentialFile:
    def setup_method(self):
        self.guard = CredentialGuard()

    def test_dotenv_is_credential(self):
        assert self.guard.is_credential_file("/project/.env") is True

    def test_env_production_is_credential(self):
        assert self.guard.is_credential_file("/app/.env.production") is True

    def test_credentials_json_is_credential(self):
        assert self.guard.is_credential_file("/home/user/credentials.json") is True

    def test_id_rsa_is_credential(self):
        assert self.guard.is_credential_file("/home/user/.ssh/id_rsa") is True

    def test_pem_extension_is_credential(self):
        assert self.guard.is_credential_file("/certs/server.pem") is True

    def test_key_extension_is_credential(self):
        assert self.guard.is_credential_file("/certs/private.key") is True

    def test_normal_python_file_not_credential(self):
        assert self.guard.is_credential_file("/app/main.py") is False

    def test_readme_not_credential(self):
        assert self.guard.is_credential_file("/app/README.md") is False

    def test_case_insensitive(self):
        assert self.guard.is_credential_file("/home/user/.ENV") is True

    def test_service_account_json(self):
        assert self.guard.is_credential_file("/home/user/service_account.json") is True


# ---------------------------------------------------------------------------
# CredentialGuard.scan_content
# ---------------------------------------------------------------------------

class TestScanContent:
    def setup_method(self):
        self.guard = CredentialGuard()

    def test_clean_text_returns_empty(self):
        assert self.guard.scan_content("Hello, world!") == []

    def test_openai_key_detected(self):
        text = "OPENAI_KEY=sk-abcdefghijklmnopqrstuvwxyz0123456789"
        matches = self.guard.scan_content(text)
        assert len(matches) > 0

    def test_github_pat_detected(self):
        text = "ghp_" + "A" * 36
        matches = self.guard.scan_content(text)
        assert len(matches) > 0

    def test_aws_access_key_detected(self):
        text = "AKIA" + "A" * 16
        matches = self.guard.scan_content(text)
        assert len(matches) > 0

    def test_pem_header_detected(self):
        text = "-----BEGIN RSA PRIVATE KEY-----"
        matches = self.guard.scan_content(text)
        assert len(matches) > 0

    def test_jwt_detected(self):
        # JWT: three base64 segments
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIzIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        matches = self.guard.scan_content(jwt)
        assert len(matches) > 0

    def test_api_key_env_var_detected(self):
        text = "api_key=super_secret_key_value_here_1234567890"
        matches = self.guard.scan_content(text)
        assert len(matches) > 0

    def test_returns_pattern_strings(self):
        text = "ghp_" + "B" * 36
        matches = self.guard.scan_content(text)
        assert all(isinstance(m, str) for m in matches)


# ---------------------------------------------------------------------------
# CredentialGuard.warn_if_credential
# ---------------------------------------------------------------------------

class TestWarnIfCredential:
    def setup_method(self):
        self.guard = CredentialGuard()

    def test_credential_file_returns_warning(self):
        result = self.guard.warn_if_credential("/home/user/.env")
        assert result is not None
        assert "WARNING" in result

    def test_normal_file_returns_none(self):
        assert self.guard.warn_if_credential("/app/utils.py") is None

    def test_warning_includes_filename(self):
        result = self.guard.warn_if_credential("/secrets/id_rsa")
        assert "id_rsa" in result


# ---------------------------------------------------------------------------
# PreDigestPipeline — pure methods
# ---------------------------------------------------------------------------

class TestPreDigestThreshold:
    def test_constant_is_50k(self):
        assert PREDIGEST_THRESHOLD_CHARS == 50_000


class TestShouldPredigest:
    def test_no_provider_always_false(self):
        pipeline = PreDigestPipeline(local_provider=None)
        assert pipeline.should_predigest("x" * 100_000) is False

    def test_with_provider_short_content_false(self):
        pipeline = PreDigestPipeline(local_provider=object())
        assert pipeline.should_predigest("short content") is False

    def test_with_provider_long_content_true(self):
        pipeline = PreDigestPipeline(local_provider=object())
        assert pipeline.should_predigest("x" * 60_000) is True

    def test_exactly_at_threshold_false(self):
        pipeline = PreDigestPipeline(local_provider=object())
        assert pipeline.should_predigest("x" * PREDIGEST_THRESHOLD_CHARS) is False

    def test_one_over_threshold_true(self):
        pipeline = PreDigestPipeline(local_provider=object())
        assert pipeline.should_predigest("x" * (PREDIGEST_THRESHOLD_CHARS + 1)) is True


class TestBuildCloudPrompt:
    def test_contains_summary(self):
        pipeline = PreDigestPipeline()
        result = pipeline.build_cloud_prompt("explain the code", "this is a summary")
        assert "this is a summary" in result

    def test_contains_task(self):
        pipeline = PreDigestPipeline()
        result = pipeline.build_cloud_prompt("explain the code", "summary here")
        assert "explain the code" in result

    def test_returns_string(self):
        pipeline = PreDigestPipeline()
        result = pipeline.build_cloud_prompt("task", "summary")
        assert isinstance(result, str)

    def test_mentions_predigested(self):
        pipeline = PreDigestPipeline()
        result = pipeline.build_cloud_prompt("task", "summary")
        assert "pre-processed" in result or "pre-digested" in result or "summary" in result


# ---------------------------------------------------------------------------
# budget_advisor helpers
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_basic_calculation(self):
        inp, out = _estimate_tokens(4000, 1)
        assert inp == 1000
        assert out == 300  # 30% of input

    def test_multiple_iterations(self):
        inp, out = _estimate_tokens(4000, 3)
        assert inp == 3000
        assert out == 900

    def test_zero_chars(self):
        inp, out = _estimate_tokens(0, 5)
        assert inp == 0
        assert out == 0


class TestEstimateCost:
    def test_local_is_zero(self):
        in_cost, out_cost = _estimate_cost("local", 1_000_000, 500_000)
        assert in_cost == 0.0
        assert out_cost == 0.0

    def test_known_provider_nonzero(self):
        in_cost, out_cost = _estimate_cost("opus", 1_000_000, 500_000)
        assert in_cost > 0

    def test_unknown_provider_falls_back_to_kimi(self):
        in_cost_unknown, _ = _estimate_cost("nonexistent", 1_000_000, 0)
        in_cost_kimi, _ = _estimate_cost("kimi", 1_000_000, 0)
        assert in_cost_unknown == in_cost_kimi

    def test_kimi_cheaper_than_opus(self):
        in_kimi, _ = _estimate_cost("kimi", 1_000_000, 0)
        in_opus, _ = _estimate_cost("opus", 1_000_000, 0)
        assert in_kimi < in_opus


class TestShouldPreflight:
    def test_large_task_triggers(self):
        assert should_preflight(task_size_chars=25_000, loop_iterations=1) is True

    def test_many_iterations_triggers(self):
        assert should_preflight(task_size_chars=100, loop_iterations=3) is True

    def test_orchestration_always_triggers(self):
        assert should_preflight(task_size_chars=0, loop_iterations=1, is_orchestration=True) is True

    def test_small_single_no_trigger(self):
        assert should_preflight(task_size_chars=1000, loop_iterations=1) is False

    def test_exactly_20k_no_trigger(self):
        assert should_preflight(task_size_chars=20_000, loop_iterations=1) is False

    def test_one_over_20k_triggers(self):
        assert should_preflight(task_size_chars=20_001, loop_iterations=1) is True


class TestFormatPreflightMessage:
    def _make_rec(self, lane="economy", primary="kimi", fallback="local"):
        return LaneRecommendation(
            lane=lane,
            primary_provider=primary,
            fallback_provider=fallback,
            rationale="Testing rationale",
            est_input_cost_usd=0.01,
            est_output_cost_usd=0.02,
            budget_snapshot="5h: 10% | weekly: 30% | pressure: 0.10",
        )

    def test_returns_string(self):
        rec = self._make_rec()
        result = format_preflight_message(rec, [])
        assert isinstance(result, str)

    def test_contains_budget_snapshot(self):
        rec = self._make_rec()
        result = format_preflight_message(rec, [])
        assert "10%" in result

    def test_contains_lane(self):
        rec = self._make_rec(lane="premium", primary="opus")
        result = format_preflight_message(rec, [])
        assert "premium" in result

    def test_contains_provider(self):
        rec = self._make_rec(primary="kimi")
        result = format_preflight_message(rec, [])
        assert "kimi" in result

    def test_no_alternatives_no_alt_section(self):
        rec = self._make_rec()
        result = format_preflight_message(rec, [])
        assert "Alternatives" not in result

    def test_with_alternatives_shows_section(self):
        rec = self._make_rec()
        alt = self._make_rec(lane="fast", primary="haiku")
        result = format_preflight_message(rec, [alt])
        assert "Alternatives" in result

    def test_box_drawing_chars_present(self):
        rec = self._make_rec()
        result = format_preflight_message(rec, [])
        assert "┌" in result or "│" in result
