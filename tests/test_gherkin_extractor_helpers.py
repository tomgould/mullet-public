"""Tests for mullet.knowledge.code.extractors.gherkin_extractor — pure helpers."""

from mullet.knowledge.code.extractors.gherkin_extractor import GherkinExtractor

SIMPLE_FEATURE = """\
Feature: User Login
  As a user I want to log in

  Background:
    Given the app is running

  @smoke
  Scenario: Successful login
    Given I am on the login page
    When I enter valid credentials
    Then I should be logged in

  @wip
  Scenario Outline: Failed login
    Given I am on the login page
    When I enter <username> and <password>
    Then I should see an error

    Examples:
      | username | password |
      | bad      | wrong    |
"""

EXTRACTOR = GherkinExtractor()


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

class TestGherkinPatterns:
    def test_feature_pattern_matches(self):
        m = EXTRACTOR.FEATURE_PATTERN.search("Feature: My Feature\n")
        assert m is not None
        assert m.group(1) == "My Feature"

    def test_scenario_pattern_matches_scenario(self):
        m = EXTRACTOR.SCENARIO_PATTERN.search("  Scenario: Login test\n")
        assert m is not None
        assert m.group(3) == "Login test"

    def test_scenario_outline_pattern_matches(self):
        m = EXTRACTOR.SCENARIO_PATTERN.search("  Scenario Outline: Parametric\n")
        assert m is not None

    def test_background_pattern_matches(self):
        m = EXTRACTOR.BACKGROUND_PATTERN.search("  Background:\n")
        assert m is not None

    def test_tag_pattern_finds_tags(self):
        tags = EXTRACTOR.TAG_PATTERN.findall("@smoke @regression @wip")
        assert "@smoke" in tags
        assert "@wip" in tags

    def test_tag_pattern_no_tags(self):
        assert EXTRACTOR.TAG_PATTERN.findall("no tags here") == []


# ---------------------------------------------------------------------------
# _extract_feature_description
# ---------------------------------------------------------------------------

class TestExtractFeatureDescription:
    def test_extracts_description(self):
        source = "Feature: Login\n  As a user I want to log in\n\nScenario: test\n"
        desc = EXTRACTOR._extract_feature_description(source)
        assert "As a user" in desc

    def test_no_description_returns_empty(self):
        source = "Feature: Login\n\nScenario: test\n"
        desc = EXTRACTOR._extract_feature_description(source)
        assert desc == ""

    def test_stops_at_scenario(self):
        source = "Feature: F\n  Description line\nScenario: S\n  More here"
        desc = EXTRACTOR._extract_feature_description(source)
        assert "More here" not in desc

    def test_stops_at_tag(self):
        source = "Feature: F\n  Description\n@smoke\nScenario: S\n"
        desc = EXTRACTOR._extract_feature_description(source)
        assert "@smoke" not in desc


# ---------------------------------------------------------------------------
# _extract_steps
# ---------------------------------------------------------------------------

class TestExtractSteps:
    def test_given_extracted(self):
        content = "Scenario: test\n  Given something is true\n"
        steps = EXTRACTOR._extract_steps(content)
        assert any("Given something is true" in s for s in steps)

    def test_when_extracted(self):
        content = "  When I do something\n"
        steps = EXTRACTOR._extract_steps(content)
        assert any("When" in s for s in steps)

    def test_then_extracted(self):
        content = "  Then I see the result\n"
        steps = EXTRACTOR._extract_steps(content)
        assert any("Then" in s for s in steps)

    def test_and_extracted(self):
        content = "  And another condition\n"
        steps = EXTRACTOR._extract_steps(content)
        assert any("And" in s for s in steps)

    def test_no_steps_returns_empty(self):
        content = "Some content\nNo gherkin keywords\n"
        assert EXTRACTOR._extract_steps(content) == []

    def test_multiple_steps(self):
        content = "  Given a\n  When b\n  Then c\n"
        steps = EXTRACTOR._extract_steps(content)
        assert len(steps) == 3


# ---------------------------------------------------------------------------
# extract — full pipeline
# ---------------------------------------------------------------------------

class TestGherkinExtractorExtract:
    def test_returns_list(self):
        result = EXTRACTOR.extract(SIMPLE_FEATURE, "login.feature")
        assert isinstance(result, list)

    def test_feature_chunk_present(self):
        chunks = EXTRACTOR.extract(SIMPLE_FEATURE, "login.feature")
        feature_chunks = [c for c in chunks if c.symbol_type == "feature"]
        assert len(feature_chunks) == 1

    def test_feature_name_correct(self):
        chunks = EXTRACTOR.extract(SIMPLE_FEATURE, "login.feature")
        feature = next(c for c in chunks if c.symbol_type == "feature")
        assert "Login" in feature.symbol_name

    def test_scenario_chunks_present(self):
        chunks = EXTRACTOR.extract(SIMPLE_FEATURE, "login.feature")
        scenarios = [c for c in chunks if c.symbol_type in ("scenario", "scenario_outline")]
        assert len(scenarios) >= 1

    def test_background_chunk_present(self):
        chunks = EXTRACTOR.extract(SIMPLE_FEATURE, "login.feature")
        bg = [c for c in chunks if c.symbol_type == "background"]
        assert len(bg) == 1

    def test_scenario_has_steps(self):
        chunks = EXTRACTOR.extract(SIMPLE_FEATURE, "login.feature")
        scenarios = [c for c in chunks if c.symbol_type == "scenario"]
        assert any(len(c.extras.get("steps", [])) > 0 for c in scenarios)

    def test_source_file_set(self):
        chunks = EXTRACTOR.extract(SIMPLE_FEATURE, "features/login.feature")
        assert all(c.source_file == "features/login.feature" for c in chunks)

    def test_empty_source_returns_feature_chunk(self):
        chunks = EXTRACTOR.extract("", "empty.feature")
        # Even empty source gets a feature chunk using filename
        assert len(chunks) >= 1

    def test_feature_chunk_has_scenario_count(self):
        chunks = EXTRACTOR.extract(SIMPLE_FEATURE, "login.feature")
        feature = next(c for c in chunks if c.symbol_type == "feature")
        assert feature.extras.get("scenario_count", 0) >= 1

    def test_tagged_scenario_has_tags(self):
        chunks = EXTRACTOR.extract(SIMPLE_FEATURE, "login.feature")
        scenarios = [c for c in chunks if c.symbol_type in ("scenario", "scenario_outline")]
        all_tags = [tag for c in scenarios for tag in c.extras.get("tags", [])]
        assert "@smoke" in all_tags or "@wip" in all_tags
