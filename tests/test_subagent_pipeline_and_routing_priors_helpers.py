"""Tests for subagent_pipeline pure helpers and routing_priors fail-open behavior."""


from mullet.core.orchestration.subagent_pipeline import (
    ReviewFinding,
    ReviewResult,
    spec_compliance_review,
    code_quality_review,
    run_three_role_review,
    format_review_report,
)
from mullet.core.routing_priors import (
    pick_better_provider,
    summarize_priors,
)


# ---------------------------------------------------------------------------
# ReviewFinding dataclass
# ---------------------------------------------------------------------------

class TestReviewFinding:
    def test_fields_stored(self):
        f = ReviewFinding(
            role="spec-compliance",
            severity="critical",
            description="Missing X",
        )
        assert f.role == "spec-compliance"
        assert f.severity == "critical"
        assert f.description == "Missing X"

    def test_file_defaults_empty(self):
        f = ReviewFinding(role="r", severity="minor", description="d")
        assert f.file == ""

    def test_line_defaults_zero(self):
        f = ReviewFinding(role="r", severity="minor", description="d")
        assert f.line == 0


# ---------------------------------------------------------------------------
# ReviewResult.__post_init__
# ---------------------------------------------------------------------------

class TestReviewResult:
    def test_approved_when_no_critical_or_major(self):
        r = ReviewResult(
            findings=[ReviewFinding(role="r", severity="minor", description="d")],
            summary="",
        )
        assert r.approved is True

    def test_rejected_when_critical(self):
        r = ReviewResult(
            findings=[ReviewFinding(role="r", severity="critical", description="x")],
            summary="",
        )
        assert r.approved is False

    def test_rejected_when_major(self):
        r = ReviewResult(
            findings=[ReviewFinding(role="r", severity="major", description="x")],
            summary="",
        )
        assert r.approved is False

    def test_approved_empty_findings(self):
        r = ReviewResult(findings=[], summary="")
        assert r.approved is True

    def test_summary_auto_generated_when_empty(self):
        r = ReviewResult(findings=[], summary="")
        assert len(r.summary) > 0

    def test_custom_summary_preserved(self):
        r = ReviewResult(findings=[], summary="My custom summary")
        assert r.summary == "My custom summary"


# ---------------------------------------------------------------------------
# spec_compliance_review
# ---------------------------------------------------------------------------

class TestSpecComplianceReview:
    def test_all_criteria_met_returns_approved(self):
        result = spec_compliance_review(
            step_title="Login",
            step_description="Implement login",
            acceptance_criteria=["login works", "token returned"],
            implementation_summary="The login now works and the token returned successfully",
        )
        assert result.approved is True
        assert len(result.findings) == 0

    def test_missing_criterion_returns_critical_finding(self):
        result = spec_compliance_review(
            step_title="Login",
            step_description="Implement login",
            acceptance_criteria=["database connection must be encrypted"],
            implementation_summary="login implemented",
        )
        assert result.approved is False
        assert any(f.severity == "critical" for f in result.findings)

    def test_empty_criteria_passes(self):
        result = spec_compliance_review("t", "d", [], "summary")
        assert result.approved is True

    def test_returns_review_result(self):
        result = spec_compliance_review("t", "d", [], "")
        assert isinstance(result, ReviewResult)


# ---------------------------------------------------------------------------
# code_quality_review
# ---------------------------------------------------------------------------

class TestCodeQualityReview:
    def test_clean_code_approved(self):
        result = code_quality_review("def main(): return True")
        assert result.approved is True

    def test_rm_rf_detected_as_critical(self):
        result = code_quality_review("running rm -rf /tmp/dir to cleanup")
        critical = [f for f in result.findings if f.severity == "critical"]
        assert len(critical) >= 1

    def test_eval_detected_as_critical(self):
        result = code_quality_review("using eval(user_input) for dynamic code")
        critical = [f for f in result.findings if f.severity == "critical"]
        assert len(critical) >= 1

    def test_todo_detected_as_major(self):
        result = code_quality_review("TODO: implement this properly later")
        major = [f for f in result.findings if f.severity == "major"]
        assert len(major) >= 1

    def test_not_implemented_detected(self):
        result = code_quality_review("raise NotImplementedError here")
        major = [f for f in result.findings if f.severity == "major"]
        assert len(major) >= 1

    def test_returns_review_result(self):
        assert isinstance(code_quality_review("clean code"), ReviewResult)


# ---------------------------------------------------------------------------
# run_three_role_review
# ---------------------------------------------------------------------------

class TestRunThreeRoleReview:
    def test_returns_tuple_of_two_results(self):
        spec, quality = run_three_role_review("t", "d", [], "clean implementation")
        assert isinstance(spec, ReviewResult)
        assert isinstance(quality, ReviewResult)

    def test_clean_implementation_both_approved(self):
        spec, quality = run_three_role_review(
            "step", "desc", ["users login successfully"],
            "Users login successfully with token"
        )
        assert spec.approved is True
        assert quality.approved is True


# ---------------------------------------------------------------------------
# format_review_report
# ---------------------------------------------------------------------------

class TestFormatReviewReport:
    def _make_ok_result(self) -> ReviewResult:
        return ReviewResult(findings=[], summary="")

    def test_returns_string(self):
        r = self._make_ok_result()
        assert isinstance(format_review_report(r, r), str)

    def test_header_present(self):
        r = self._make_ok_result()
        result = format_review_report(r, r)
        assert "Review Report" in result

    def test_findings_in_output(self):
        r = ReviewResult(
            findings=[ReviewFinding(role="spec-compliance", severity="critical", description="Missing auth")],
            summary="",
        )
        result = format_review_report(r, self._make_ok_result())
        assert "Missing auth" in result


# ---------------------------------------------------------------------------
# routing_priors — fail-open behavior (no history data → returns picked unchanged)
# ---------------------------------------------------------------------------

class TestPickBetterProviderFailOpen:
    def test_no_history_returns_picked(self, tmp_path):
        # tmp_path has no routing history → min_samples not met → returns picked
        result, reason = pick_better_provider(
            task="do something",
            tier="fast",
            picked_provider="kimi",
            candidates=["gemini", "local"],
            base=tmp_path,
        )
        assert result == "kimi"
        assert reason is None

    def test_returns_tuple(self, tmp_path):
        result = pick_better_provider("task", "fast", "kimi", [], base=tmp_path)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_empty_candidates_returns_picked(self, tmp_path):
        result, reason = pick_better_provider("task", "fast", "kimi", [], base=tmp_path)
        assert result == "kimi"

    def test_returns_string_provider(self, tmp_path):
        provider, _ = pick_better_provider("task", "fast", "kimi", ["gemini"], base=tmp_path)
        assert isinstance(provider, str)


class TestSummarizePriorsFailOpen:
    def test_no_history_returns_empty_dict(self, tmp_path):
        result = summarize_priors("task", "fast", base=tmp_path)
        assert isinstance(result, dict)

    def test_returns_dict(self, tmp_path):
        result = summarize_priors("some task", "balanced", base=tmp_path)
        assert isinstance(result, dict)
