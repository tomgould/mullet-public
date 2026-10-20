"""Tests for mullet.core.orchestration.subagent_pipeline — review pipeline helpers."""

from mullet.core.orchestration.subagent_pipeline import (
    ReviewFinding,
    ReviewResult,
    spec_compliance_review,
    code_quality_review,
    run_three_role_review,
    format_review_report,
)


# ---------------------------------------------------------------------------
# ReviewFinding
# ---------------------------------------------------------------------------

class TestReviewFinding:
    def test_stores_role_severity_description(self):
        f = ReviewFinding(role="spec-compliance", severity="critical", description="Missing X")
        assert f.role == "spec-compliance"
        assert f.severity == "critical"
        assert f.description == "Missing X"

    def test_defaults_for_file_and_line(self):
        f = ReviewFinding(role="code-quality", severity="minor", description="Style issue")
        assert f.file == ""
        assert f.line == 0

    def test_custom_file_and_line(self):
        f = ReviewFinding(role="code-quality", severity="major", description="Bug", file="foo.py", line=42)
        assert f.file == "foo.py"
        assert f.line == 42


# ---------------------------------------------------------------------------
# ReviewResult — auto-approval logic
# ---------------------------------------------------------------------------

class TestReviewResult:
    def test_no_findings_approved(self):
        r = ReviewResult(findings=[], summary="All good")
        assert r.approved is True

    def test_critical_finding_not_approved(self):
        f = ReviewFinding(role="spec-compliance", severity="critical", description="Missing X")
        r = ReviewResult(findings=[f], summary="")
        assert r.approved is False

    def test_major_finding_not_approved(self):
        f = ReviewFinding(role="code-quality", severity="major", description="TODO left in")
        r = ReviewResult(findings=[f], summary="")
        assert r.approved is False

    def test_minor_finding_still_approved(self):
        f = ReviewFinding(role="code-quality", severity="minor", description="Style issue")
        r = ReviewResult(findings=[f], summary="Minor stuff")
        assert r.approved is True

    def test_suggestion_still_approved(self):
        f = ReviewFinding(role="code-quality", severity="suggestion", description="Consider X")
        r = ReviewResult(findings=[f], summary="Suggestion only")
        assert r.approved is True

    def test_empty_summary_auto_filled_approved(self):
        r = ReviewResult(findings=[], summary="")
        assert "Approved" in r.summary

    def test_empty_summary_auto_filled_rejected(self):
        f = ReviewFinding(role="spec-compliance", severity="critical", description="Fail")
        r = ReviewResult(findings=[f], summary="")
        assert "Rejected" in r.summary

    def test_rejected_summary_counts_critical(self):
        f1 = ReviewFinding(role="x", severity="critical", description="A")
        f2 = ReviewFinding(role="x", severity="critical", description="B")
        r = ReviewResult(findings=[f1, f2], summary="")
        assert "2 critical" in r.summary

    def test_provided_summary_not_overwritten(self):
        r = ReviewResult(findings=[], summary="My custom summary")
        assert r.summary == "My custom summary"


# ---------------------------------------------------------------------------
# spec_compliance_review
# ---------------------------------------------------------------------------

class TestSpecComplianceReview:
    def test_all_criteria_met_no_findings(self):
        summary = "The user authentication was implemented with JWT tokens and session management."
        result = spec_compliance_review(
            "Auth", "Implement auth", ["authentication JWT tokens", "session management"], summary
        )
        assert result.findings == []
        assert result.approved is True

    def test_missing_criterion_adds_critical_finding(self):
        result = spec_compliance_review(
            "Auth", "Implement auth", ["missing feature xyz"], "No mention here"
        )
        assert any(f.severity == "critical" for f in result.findings)

    def test_empty_criteria_no_findings(self):
        result = spec_compliance_review("Title", "Desc", [], "Implementation done")
        assert result.findings == []

    def test_partial_criteria_some_findings(self):
        result = spec_compliance_review(
            "T", "D",
            ["authentication done", "tests written"],
            "Authentication done, but no tests yet"
        )
        # "tests written" keywords not all present
        findings_descs = [f.description for f in result.findings]
        assert any("tests" in d.lower() for d in findings_descs)

    def test_case_insensitive_matching(self):
        result = spec_compliance_review(
            "T", "D", ["authentication"], "AUTHENTICATION was handled"
        )
        assert result.findings == []

    def test_result_type_is_review_result(self):
        result = spec_compliance_review("T", "D", [], "done")
        assert isinstance(result, ReviewResult)


# ---------------------------------------------------------------------------
# code_quality_review
# ---------------------------------------------------------------------------

class TestCodeQualityReview:
    def test_clean_summary_no_findings(self):
        result = code_quality_review("The implementation uses standard patterns and is complete.")
        assert result.findings == []
        assert result.approved is True

    def test_rm_rf_is_critical(self):
        result = code_quality_review("ran rm -rf /tmp/cache to clean up")
        assert any(f.severity == "critical" for f in result.findings)

    def test_eval_is_critical(self):
        result = code_quality_review("used eval(user_input) for dynamic evaluation")
        assert any(f.severity == "critical" for f in result.findings)

    def test_exec_is_critical(self):
        result = code_quality_review("called exec(code_string) to run dynamic code")
        assert any(f.severity == "critical" for f in result.findings)

    def test_hardcoded_credentials_critical(self):
        result = code_quality_review('password = "secretpass123"')
        assert any(f.severity == "critical" for f in result.findings)

    def test_todo_is_major(self):
        result = code_quality_review("TODO: finish the implementation later")
        assert any(f.severity == "major" for f in result.findings)

    def test_fixme_is_major(self):
        result = code_quality_review("FIXME: this is broken")
        assert any(f.severity == "major" for f in result.findings)

    def test_not_implemented_error_is_major(self):
        result = code_quality_review("raise NotImplementedError in the stub")
        assert any(f.severity == "major" for f in result.findings)

    def test_findings_have_code_quality_role(self):
        result = code_quality_review("TODO: fix this")
        assert all(f.role == "code-quality" for f in result.findings)

    def test_returns_review_result(self):
        result = code_quality_review("clean code")
        assert isinstance(result, ReviewResult)


# ---------------------------------------------------------------------------
# run_three_role_review
# ---------------------------------------------------------------------------

class TestRunThreeRoleReview:
    def test_returns_two_results(self):
        spec_r, qual_r = run_three_role_review("T", "D", [], "clean implementation")
        assert isinstance(spec_r, ReviewResult)
        assert isinstance(qual_r, ReviewResult)

    def test_clean_implementation_both_approved(self):
        spec_r, qual_r = run_three_role_review(
            "T", "D",
            ["authentication", "logging"],
            "Implemented authentication and logging in the module",
        )
        assert spec_r.approved is True
        assert qual_r.approved is True

    def test_bad_quality_code_qual_not_approved(self):
        spec_r, qual_r = run_three_role_review(
            "T", "D", [], 'eval(user_input) and TODO: fix later'
        )
        assert qual_r.approved is False


# ---------------------------------------------------------------------------
# format_review_report
# ---------------------------------------------------------------------------

class TestFormatReviewReport:
    def _make_clean(self):
        return ReviewResult(findings=[], summary="Approved")

    def test_returns_string(self):
        result = format_review_report(self._make_clean(), self._make_clean())
        assert isinstance(result, str)

    def test_contains_review_report_header(self):
        result = format_review_report(self._make_clean(), self._make_clean())
        assert "Review Report" in result

    def test_no_findings_shows_all_criteria_met(self):
        result = format_review_report(self._make_clean(), self._make_clean())
        assert "criteria met" in result.lower() or "✅" in result

    def test_findings_appear_in_report(self):
        f = ReviewFinding(role="spec-compliance", severity="critical", description="Missing auth")
        bad = ReviewResult(findings=[f], summary="Rejected")
        result = format_review_report(bad, self._make_clean())
        assert "Missing auth" in result

    def test_severity_appears_in_report(self):
        f = ReviewFinding(role="spec-compliance", severity="critical", description="Issue")
        bad = ReviewResult(findings=[f], summary="Rejected")
        result = format_review_report(bad, self._make_clean())
        assert "CRITICAL" in result
