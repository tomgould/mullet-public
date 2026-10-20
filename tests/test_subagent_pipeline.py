"""Tests for three-role review pipeline."""

from mullet.core.orchestration.subagent_pipeline import (
    spec_compliance_review, code_quality_review,
    run_three_role_review, format_review_report,
    ReviewResult,
)


class TestSpecComplianceReview:
    def test_all_criteria_met(self):
        result = spec_compliance_review(
            "Login endpoint",
            "POST /login",
            ["Returns 200 on success"],
            "Implemented POST /login. Returns 200 on success with JWT token.",
        )
        assert result.approved is True
        assert not any(f.severity == "critical" for f in result.findings)

    def test_missing_criterion_flagged(self):
        result = spec_compliance_review(
            "Login endpoint",
            "POST /login",
            ["Returns 401 on invalid creds"],
            "Implemented login. Does not handle invalid credentials yet.",
        )
        assert result.approved is False
        assert any(f.severity in ("critical", "major") for f in result.findings)

    def test_no_criteria_always_approved(self):
        result = spec_compliance_review("Task", "Do something", [], "Done.")
        assert result.approved is True


class TestCodeQualityReview:
    def test_clean_implementation_approved(self):
        result = code_quality_review("Added login route with proper error handling.")
        assert result.approved is True

    def test_exec_call_detected(self):
        result = code_quality_review("exec(user_input) to process commands.")
        assert result.approved is False
        assert any(f.severity == "critical" for f in result.findings)

    def test_todo_in_implementation_flagged(self):
        result = code_quality_review("Added feature. TODO: add validation later.")
        assert result.approved is False
        assert any(f.severity == "major" for f in result.findings)

    def test_rm_rf_is_critical(self):
        result = code_quality_review("Script runs rm -rf /tmp/cache to clean up.")
        assert result.approved is False
        assert any(f.severity == "critical" for f in result.findings)


class TestRunThreeRoleReview:
    def test_returns_two_results(self):
        spec_r, qual_r = run_three_role_review(
            "Task", "Description", ["criterion"], "implementation"
        )
        assert isinstance(spec_r, ReviewResult)
        assert isinstance(qual_r, ReviewResult)

    def test_both_pass_on_clean_input(self):
        spec_r, qual_r = run_three_role_review(
            "Add endpoint",
            "POST /api/endpoint",
            [],
            "Added POST /api/endpoint with proper error handling and tests.",
        )
        assert spec_r.approved is True
        assert qual_r.approved is True


class TestFormatReviewReport:
    def test_report_contains_severity_emojis(self):
        spec_r, qual_r = run_three_role_review(
            "Task", "Desc", ["crit"], "exec(input) todo fix"
        )
        report = format_review_report(spec_r, qual_r)
        assert "🔴" in report or "CRITICAL" in report

    def test_empty_findings_produce_short_report(self):
        spec_r, qual_r = run_three_role_review(
            "Task", "Desc", [], "Clean implementation."
        )
        report = format_review_report(spec_r, qual_r)
        assert isinstance(report, str)
        assert len(report) > 0
