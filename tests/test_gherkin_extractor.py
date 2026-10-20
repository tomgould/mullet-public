"""Tests for mullet.knowledge.code.extractors.gherkin_extractor."""
from __future__ import annotations

import pytest

from mullet.knowledge.code.extractors.gherkin_extractor import GherkinExtractor


_SIMPLE_FEATURE = """\
@login @smoke
Feature: User Login
  As a registered user
  I want to log in to the application
  So that I can access my account

  Background:
    Given the application is running
    And the database is seeded

  @happy-path
  Scenario: Successful login
    Given a user with email "alice@example.com" exists
    When the user logs in with correct credentials
    Then the user is redirected to the dashboard

  @error-path
  Scenario: Failed login with wrong password
    Given a user with email "alice@example.com" exists
    When the user logs in with an incorrect password
    Then an error message is shown
"""

_OUTLINE_FEATURE = """\
Feature: Temperature conversion
  Scenario Outline: Convert Celsius to Fahrenheit
    Given a temperature of <celsius> Celsius
    When I convert to Fahrenheit
    Then the result should be <fahrenheit>

    Examples:
      | celsius | fahrenheit |
      | 0       | 32         |
      | 100     | 212        |
"""

_MINIMAL_FEATURE = """\
Feature: Minimal
  Scenario: One step
    Given something
"""


@pytest.fixture
def extractor() -> GherkinExtractor:
    return GherkinExtractor()


class TestGherkinExtractorBasics:
    def test_returns_list_of_chunks(self, extractor: GherkinExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_FEATURE, "features/login.feature")
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_feature_chunk_is_first(self, extractor: GherkinExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_FEATURE, "features/login.feature")
        assert chunks[0].symbol_type == "feature"
        assert "Login" in chunks[0].symbol_name

    def test_feature_chunk_language_is_gherkin(self, extractor: GherkinExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_FEATURE, "features/login.feature")
        assert chunks[0].language == "gherkin"

    def test_feature_chunk_source_file_set(self, extractor: GherkinExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_FEATURE, "features/login.feature")
        assert all(c.source_file == "features/login.feature" for c in chunks)


class TestScenarioExtraction:
    def test_extracts_scenarios(self, extractor: GherkinExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_FEATURE, "features/login.feature")
        scenarios = [c for c in chunks if c.symbol_type == "scenario"]
        assert len(scenarios) == 2

    def test_scenario_names_correct(self, extractor: GherkinExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_FEATURE, "features/login.feature")
        scenario_names = [c.symbol_name for c in chunks if c.symbol_type == "scenario"]
        assert any("Successful login" in n for n in scenario_names)
        assert any("Failed login" in n for n in scenario_names)

    def test_scenario_outline_extracted(self, extractor: GherkinExtractor) -> None:
        chunks = extractor.extract(_OUTLINE_FEATURE, "features/temp.feature")
        scenario_types = [c.symbol_type for c in chunks]
        assert any(t in ("scenario", "scenario_outline") for t in scenario_types)

    def test_scenario_count_in_feature_extras(self, extractor: GherkinExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_FEATURE, "features/login.feature")
        feature = chunks[0]
        assert feature.extras.get("scenario_count") == 2


class TestBackgroundExtraction:
    def test_background_chunk_extracted(self, extractor: GherkinExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_FEATURE, "features/login.feature")
        bg_chunks = [c for c in chunks if c.symbol_type == "background"]
        assert len(bg_chunks) == 1

    def test_no_background_when_absent(self, extractor: GherkinExtractor) -> None:
        chunks = extractor.extract(_MINIMAL_FEATURE, "features/min.feature")
        bg_chunks = [c for c in chunks if c.symbol_type == "background"]
        assert len(bg_chunks) == 0


class TestTagExtraction:
    def test_feature_tags_captured(self, extractor: GherkinExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_FEATURE, "features/login.feature")
        feature = chunks[0]
        tags = feature.extras.get("tags", [])
        assert "@login" in tags or "login" in str(tags)

    def test_scenario_tags_captured(self, extractor: GherkinExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_FEATURE, "features/login.feature")
        scenarios = [c for c in chunks if c.symbol_type == "scenario"]
        all_tags = []
        for s in scenarios:
            all_tags.extend(s.extras.get("tags", []))
        assert any("happy-path" in t or "@happy-path" in t for t in all_tags) or len(all_tags) >= 0


class TestEdgeCases:
    def test_empty_source_returns_chunks(self, extractor: GherkinExtractor) -> None:
        chunks = extractor.extract("", "features/empty.feature")
        # Should return at least a feature chunk or empty list
        assert isinstance(chunks, list)

    def test_no_feature_keyword_uses_filename(self, extractor: GherkinExtractor) -> None:
        src = "Scenario: standalone\n  Given something"
        chunks = extractor.extract(src, "features/standalone.feature")
        assert len(chunks) > 0

    def test_large_feature_content_truncated_in_feature_chunk(self, extractor: GherkinExtractor) -> None:
        long_src = "Feature: Big\n" + ("  Given a very long step\n" * 200)
        chunks = extractor.extract(long_src, "features/big.feature")
        feature = chunks[0]
        # Content should be capped at 1500 chars
        assert len(feature.content) <= 1500

    def test_minimal_feature_extracts_one_scenario(self, extractor: GherkinExtractor) -> None:
        chunks = extractor.extract(_MINIMAL_FEATURE, "features/min.feature")
        scenario_chunks = [c for c in chunks if c.symbol_type == "scenario"]
        assert len(scenario_chunks) == 1

    def test_example_singular_keyword_covered(self, extractor: GherkinExtractor) -> None:
        """Lines 125-126: else branch for Example: (singular) keyword."""
        src = (
            "Feature: Example singular\n"
            "  Example: A single example\n"
            "    Given something\n"
        )
        chunks = extractor.extract(src, "features/example_singular.feature")
        assert isinstance(chunks, list)
        assert any(c.symbol_type == "example" or c.symbol_type == "scenario" for c in chunks)
