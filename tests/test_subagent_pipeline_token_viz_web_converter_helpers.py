"""Tests for subagent_pipeline review helpers, token_viz, and web_converter pure helpers."""


from mullet.core.orchestration.subagent_pipeline import (
    ReviewFinding,
    ReviewResult,
    spec_compliance_review,
    code_quality_review,
    run_three_role_review,
    format_review_report,
)
from mullet.core.agent.token_viz import TokenUsageViz
from mullet.knowledge.sources.web_converter import (
    clean_markdown,
    extract_title,
)


# ---------------------------------------------------------------------------
# ReviewFinding
# ---------------------------------------------------------------------------

class TestReviewFinding:
    def test_fields_stored(self):
        rf = ReviewFinding(role="spec-compliance", severity="critical", description="Missing")
        assert rf.role == "spec-compliance"
        assert rf.severity == "critical"
        assert rf.description == "Missing"

    def test_file_defaults_empty(self):
        rf = ReviewFinding(role="code-quality", severity="minor", description="Style")
        assert rf.file == ""

    def test_line_defaults_zero(self):
        rf = ReviewFinding(role="code-quality", severity="minor", description="Style")
        assert rf.line == 0

    def test_explicit_file_and_line(self):
        rf = ReviewFinding(role="r", severity="s", description="d", file="app.py", line=42)
        assert rf.file == "app.py"
        assert rf.line == 42


# ---------------------------------------------------------------------------
# ReviewResult — approved logic
# ---------------------------------------------------------------------------

class TestReviewResultApproved:
    def test_no_findings_is_approved(self):
        rr = ReviewResult(findings=[], summary="")
        assert rr.approved is True

    def test_critical_finding_not_approved(self):
        rr = ReviewResult(findings=[ReviewFinding("spec", "critical", "Missing feature")], summary="")
        assert rr.approved is False

    def test_major_finding_not_approved(self):
        rr = ReviewResult(findings=[ReviewFinding("code", "major", "TODO left in")], summary="")
        assert rr.approved is False

    def test_minor_finding_still_approved(self):
        rr = ReviewResult(findings=[ReviewFinding("code", "minor", "Style issue")], summary="")
        assert rr.approved is True

    def test_suggestion_still_approved(self):
        rr = ReviewResult(findings=[ReviewFinding("code", "suggestion", "Consider refactor")], summary="")
        assert rr.approved is True

    def test_mixed_critical_and_minor_not_approved(self):
        findings = [
            ReviewFinding("spec", "critical", "Missing"),
            ReviewFinding("code", "minor", "Style"),
        ]
        rr = ReviewResult(findings=findings, summary="")
        assert rr.approved is False


class TestReviewResultDefaultSummary:
    def test_empty_findings_approved_summary(self):
        rr = ReviewResult(findings=[], summary="")
        assert "Approved" in rr.summary

    def test_critical_rejected_summary(self):
        rr = ReviewResult(findings=[ReviewFinding("spec", "critical", "Missing")], summary="")
        assert "Rejected" in rr.summary

    def test_critical_count_in_summary(self):
        findings = [
            ReviewFinding("spec", "critical", "c1"),
            ReviewFinding("spec", "critical", "c2"),
        ]
        rr = ReviewResult(findings=findings, summary="")
        assert "2 critical" in rr.summary

    def test_explicit_summary_preserved(self):
        rr = ReviewResult(findings=[], summary="Custom summary")
        assert rr.summary == "Custom summary"


# ---------------------------------------------------------------------------
# spec_compliance_review
# ---------------------------------------------------------------------------

class TestSpecComplianceReview:
    def test_all_criteria_met_approved(self):
        result = spec_compliance_review(
            "Login", "User login", ["user can log in"],
            "the user can log in successfully"
        )
        assert result.approved is True

    def test_criterion_not_met_not_approved(self):
        result = spec_compliance_review(
            "Login", "User login", ["user can log in"],
            "authentication not implemented"
        )
        assert result.approved is False

    def test_missing_criterion_adds_finding(self):
        result = spec_compliance_review(
            "Login", "Desc", ["user can log in"],
            "no relevant content here"
        )
        assert len(result.findings) == 1

    def test_finding_role_is_spec_compliance(self):
        result = spec_compliance_review(
            "T", "D", ["missing feature"], "something else"
        )
        assert result.findings[0].role == "spec-compliance"

    def test_finding_severity_is_critical(self):
        result = spec_compliance_review(
            "T", "D", ["missing feature"], "unrelated text"
        )
        assert result.findings[0].severity == "critical"

    def test_empty_criteria_no_findings(self):
        result = spec_compliance_review("T", "D", [], "any content")
        assert result.findings == []
        assert result.approved is True

    def test_multiple_criteria_all_met(self):
        result = spec_compliance_review(
            "T", "D", ["login works", "logout works"],
            "login works and logout works too"
        )
        assert result.approved is True

    def test_multiple_criteria_one_missing(self):
        result = spec_compliance_review(
            "T", "D", ["login works", "logout works"],
            "login works"
        )
        assert len(result.findings) == 1


# ---------------------------------------------------------------------------
# code_quality_review
# ---------------------------------------------------------------------------

class TestCodeQualityReview:
    def test_clean_code_approved(self):
        result = code_quality_review("Implemented the function cleanly with proper error handling.")
        assert result.approved is True

    def test_todo_is_major(self):
        result = code_quality_review("TODO: implement this later")
        findings = [f for f in result.findings if f.severity == "major"]
        assert len(findings) >= 1

    def test_fixme_is_major(self):
        result = code_quality_review("FIXME: this is broken")
        findings = [f for f in result.findings if f.severity == "major"]
        assert len(findings) >= 1

    def test_hardcoded_password_is_critical(self):
        result = code_quality_review('password = "secret123"')
        findings = [f for f in result.findings if f.severity == "critical"]
        assert len(findings) >= 1

    def test_eval_is_critical(self):
        result = code_quality_review("eval(user_input)")
        findings = [f for f in result.findings if f.severity == "critical"]
        assert len(findings) >= 1

    def test_not_implemented_error_is_major(self):
        result = code_quality_review("raise NotImplementedError")
        findings = [f for f in result.findings if f.severity == "major"]
        assert len(findings) >= 1

    def test_returns_review_result(self):
        result = code_quality_review("clean code")
        assert isinstance(result, ReviewResult)

    def test_findings_have_code_quality_role(self):
        result = code_quality_review("TODO: fix this")
        assert all(f.role == "code-quality" for f in result.findings)


# ---------------------------------------------------------------------------
# run_three_role_review
# ---------------------------------------------------------------------------

class TestRunThreeRoleReview:
    def test_returns_tuple_of_two(self):
        result = run_three_role_review("T", "D", [], "clean implementation")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_both_results_are_review_result(self):
        spec_r, qual_r = run_three_role_review("T", "D", [], "clean")
        assert isinstance(spec_r, ReviewResult)
        assert isinstance(qual_r, ReviewResult)

    def test_clean_impl_both_approved(self):
        spec_r, qual_r = run_three_role_review("T", "D", ["feature works"], "feature works correctly")
        assert spec_r.approved is True
        assert qual_r.approved is True


# ---------------------------------------------------------------------------
# format_review_report
# ---------------------------------------------------------------------------

class TestFormatReviewReport:
    def test_returns_string(self):
        spec_r = ReviewResult(findings=[], summary="")
        qual_r = ReviewResult(findings=[], summary="")
        result = format_review_report(spec_r, qual_r)
        assert isinstance(result, str)

    def test_contains_review_report_header(self):
        spec_r = ReviewResult(findings=[], summary="")
        qual_r = ReviewResult(findings=[], summary="")
        result = format_review_report(spec_r, qual_r)
        assert "Review Report" in result

    def test_no_findings_shows_criteria_met(self):
        spec_r = ReviewResult(findings=[], summary="")
        qual_r = ReviewResult(findings=[], summary="")
        result = format_review_report(spec_r, qual_r)
        assert "criteria met" in result.lower() or "✅" in result

    def test_spec_finding_in_report(self):
        spec_r = ReviewResult(findings=[ReviewFinding("spec-compliance", "critical", "Missing X")], summary="")
        qual_r = ReviewResult(findings=[], summary="")
        result = format_review_report(spec_r, qual_r)
        assert "Missing X" in result

    def test_quality_finding_in_report(self):
        spec_r = ReviewResult(findings=[], summary="")
        qual_r = ReviewResult(findings=[ReviewFinding("code-quality", "major", "TODO found")], summary="")
        result = format_review_report(spec_r, qual_r)
        assert "TODO found" in result


# ---------------------------------------------------------------------------
# TokenUsageViz
# ---------------------------------------------------------------------------

class TestTokenUsageVizEmpty:
    def test_no_turns_table_message(self):
        viz = TokenUsageViz()
        table = viz.format_table()
        assert "No token usage" in table

    def test_total_all_zeros_when_empty(self):
        viz = TokenUsageViz()
        t = viz.total()
        assert t["total_tokens_in"] == 0
        assert t["total_tokens_out"] == 0
        assert t["total_tokens_combined"] == 0

    def test_total_returns_dict(self):
        viz = TokenUsageViz()
        assert isinstance(viz.total(), dict)


class TestTokenUsageVizRecordTurn:
    def test_record_single_turn(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert len(viz.turns) == 1

    def test_record_multiple_turns(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        viz.record_turn(200, 75)
        assert len(viz.turns) == 2

    def test_cached_defaults_false(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert viz.turns[0]["cached"] is False

    def test_cached_true_stored(self):
        viz = TokenUsageViz()
        viz.record_turn(200, 75, cached=True)
        assert viz.turns[0]["cached"] is True


class TestTokenUsageVizTotal:
    def _make_viz(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        viz.record_turn(200, 75, cached=True)
        viz.record_turn(150, 60)
        return viz

    def test_total_tokens_in(self):
        assert self._make_viz().total()["total_tokens_in"] == 450

    def test_total_tokens_out(self):
        assert self._make_viz().total()["total_tokens_out"] == 185

    def test_total_combined(self):
        assert self._make_viz().total()["total_tokens_combined"] == 635

    def test_cached_tokens_in(self):
        assert self._make_viz().total()["total_cached_tokens_in"] == 200

    def test_cached_tokens_out(self):
        assert self._make_viz().total()["total_cached_tokens_out"] == 75

    def test_cached_combined(self):
        assert self._make_viz().total()["total_cached_tokens_combined"] == 275

    def test_has_all_keys(self):
        t = self._make_viz().total()
        for key in (
            "total_tokens_in", "total_tokens_out", "total_tokens_combined",
            "total_cached_tokens_in", "total_cached_tokens_out", "total_cached_tokens_combined"
        ):
            assert key in t


class TestTokenUsageVizFormatTable:
    def test_returns_string(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert isinstance(viz.format_table(), str)

    def test_has_turn_header(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert "Turn" in viz.format_table()

    def test_has_tokens_in_header(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert "Tokens In" in viz.format_table()

    def test_has_tokens_out_header(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert "Tokens Out" in viz.format_table()

    def test_has_cached_header(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert "Cached" in viz.format_table()

    def test_cached_yes_for_cached_turn(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50, cached=True)
        assert "Yes" in viz.format_table()

    def test_cached_no_for_uncached_turn(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50, cached=False)
        assert "No" in viz.format_table()

    def test_token_values_in_table(self):
        viz = TokenUsageViz()
        viz.record_turn(12345, 6789)
        table = viz.format_table()
        assert "12345" in table
        assert "6789" in table


# ---------------------------------------------------------------------------
# clean_markdown
# ---------------------------------------------------------------------------

class TestCleanMarkdown:
    def test_returns_none_for_empty(self):
        assert clean_markdown("") is None

    def test_returns_none_for_none(self):
        assert clean_markdown(None) is None

    def test_normal_text_returned(self):
        result = clean_markdown("Hello world, this is some content.")
        assert result is not None
        assert "Hello world" in result

    def test_removes_advertisement_line(self):
        text = "Content here\n\nArticle continues below\n\nMore content"
        result = clean_markdown(text)
        assert "Article continues below" not in result

    def test_removes_advertisement_marker(self):
        text = "Content\n\nAdvertisement\n\nMore"
        result = clean_markdown(text)
        assert result is not None
        assert "\nAdvertisement\n" not in result

    def test_collapses_excessive_blank_lines(self):
        text = "line1\n\n\n\n\nline2"
        result = clean_markdown(text)
        assert "\n\n\n\n" not in result

    def test_removes_empty_links(self):
        text = "Hello [](http://example.com) world, some more content here."
        result = clean_markdown(text)
        assert "[](http://example.com)" not in result

    def test_removes_empty_images(self):
        text = "Hello ![](http://example.com/img.png) world content here."
        result = clean_markdown(text)
        assert "![](http://example.com/img.png)" not in result

    def test_strips_whitespace(self):
        result = clean_markdown("   content here   ")
        assert result == result.strip()

    def test_whitespace_only_returns_none(self):
        result = clean_markdown("   \n\n   \n")
        assert result is None

    def test_subscribe_standalone_removed(self):
        text = "Real content here\n\nSign up\n\nMore content here"
        result = clean_markdown(text)
        assert result is not None
        # Sign up standalone line should be removed
        lines = result.split("\n")
        stripped = [l.strip() for l in lines]
        assert "Sign up" not in stripped


# ---------------------------------------------------------------------------
# extract_title
# ---------------------------------------------------------------------------

class TestExtractTitle:
    def test_title_from_h1(self):
        html = "<html><body><h1>Page Title</h1><p>content</p></body></html>"
        assert extract_title(html) == "Page Title"

    def test_title_from_title_tag(self):
        html = "<html><head><title>Site Title</title></head><body></body></html>"
        assert extract_title(html) == "Site Title"

    def test_h1_takes_precedence_over_title_tag(self):
        html = "<html><head><title>Page Title</title></head><body><h1>H1 Title</h1></body></html>"
        assert extract_title(html) == "H1 Title"

    def test_no_title_returns_empty(self):
        html = "<html><body><p>Just content</p></body></html>"
        assert extract_title(html) == ""

    def test_returns_string(self):
        assert isinstance(extract_title("<html></html>"), str)

    def test_strips_whitespace(self):
        html = "<html><body><h1>  Padded Title  </h1></body></html>"
        assert extract_title(html) == "Padded Title"
