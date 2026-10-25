"""Tests for ProcessRegistry, QueryRefiner, BedrockStream,
CodeChunk/BaseExtractor helpers, JSONExtractor, and GherkinExtractor."""

import json

from mullet.core.tools.process_registry import ProcessRegistry
from mullet.core.knowledge.query_refinement import QueryRefiner
from mullet.core.providers.bedrock_stream import BedrockStreamEvent, BedrockStreamHandler
from mullet.knowledge.code.extractors.base import CodeChunk
from mullet.knowledge.code.extractors.json_extractor import JSONExtractor, _strip_jsonc
from mullet.knowledge.code.extractors.gherkin_extractor import GherkinExtractor


# ---------------------------------------------------------------------------
# ProcessRegistry
# ---------------------------------------------------------------------------

class TestProcessRegistry:
    def test_empty_list_active(self):
        reg = ProcessRegistry()
        assert reg.list_active() == []

    def test_register_adds_process(self):
        reg = ProcessRegistry()
        reg.register(1001, "worker")
        active = reg.list_active()
        assert (1001, "worker") in active

    def test_unregister_removes_process(self):
        reg = ProcessRegistry()
        reg.register(1001, "worker")
        reg.unregister(1001)
        assert reg.list_active() == []

    def test_unregister_nonexistent_no_crash(self):
        reg = ProcessRegistry()
        reg.unregister(9999)  # should not raise

    def test_register_multiple(self):
        reg = ProcessRegistry()
        reg.register(1, "a")
        reg.register(2, "b")
        assert len(reg.list_active()) == 2

    def test_cleanup_all_empties(self):
        reg = ProcessRegistry()
        reg.register(1, "a")
        reg.register(2, "b")
        reg.cleanup_all()
        assert reg.list_active() == []

    def test_list_active_returns_list(self):
        reg = ProcessRegistry()
        assert isinstance(reg.list_active(), list)

    def test_register_overwrites_same_pid(self):
        reg = ProcessRegistry()
        reg.register(1, "first")
        reg.register(1, "second")
        active = dict(reg.list_active())
        assert active[1] == "second"


# ---------------------------------------------------------------------------
# QueryRefiner
# ---------------------------------------------------------------------------

class TestQueryRefiner:
    def test_returns_string(self):
        qr = QueryRefiner()
        assert isinstance(qr.expand_query("search documents"), str)

    def test_known_word_expands(self):
        qr = QueryRefiner()
        result = qr.expand_query("search")
        # "search" has synonyms: find, lookup, discover, retrieve
        assert "find" in result or "lookup" in result

    def test_unknown_word_kept(self):
        qr = QueryRefiner()
        result = qr.expand_query("purple elephant")
        assert "purple" in result
        assert "elephant" in result

    def test_expand_query_case_insensitive(self):
        qr = QueryRefiner()
        result = qr.expand_query("SEARCH")
        # lowercased internally; synonyms added
        assert "find" in result or "search" in result

    def test_refine_returns_string(self):
        qr = QueryRefiner()
        assert isinstance(qr.refine("query document"), str)

    def test_refine_same_as_expand(self):
        qr = QueryRefiner()
        assert qr.refine("search code") == qr.expand_query("search code")

    def test_refine_with_context_no_crash(self):
        qr = QueryRefiner()
        result = qr.refine("search", context="some context here")
        assert isinstance(result, str)

    def test_code_synonym_added(self):
        qr = QueryRefiner()
        result = qr.expand_query("code")
        assert "program" in result or "script" in result

    def test_tool_synonym_added(self):
        qr = QueryRefiner()
        result = qr.expand_query("tool")
        assert "utility" in result or "instrument" in result

    def test_empty_query_returns_string(self):
        qr = QueryRefiner()
        result = qr.expand_query("")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# BedrockStreamEvent
# ---------------------------------------------------------------------------

class TestBedrockStreamEvent:
    def test_fields_stored(self):
        e = BedrockStreamEvent(type="content_block_delta", text="hello")
        assert e.type == "content_block_delta"
        assert e.text == "hello"

    def test_default_text_empty(self):
        e = BedrockStreamEvent(type="message_stop")
        assert e.text == ""

    def test_default_events_none(self):
        e = BedrockStreamEvent(type="content_block_start")
        assert e.start_event is None
        assert e.delta_event is None
        assert e.stop_event is None

    def test_explicit_start_event(self):
        payload = {"index": 0, "content_block": {"type": "text"}}
        e = BedrockStreamEvent(type="content_block_start", start_event=payload)
        assert e.start_event == payload


# ---------------------------------------------------------------------------
# BedrockStreamHandler
# ---------------------------------------------------------------------------

class TestBedrockStreamHandler:
    def _make_chunk_event(self, payload: dict) -> dict:
        return {"chunk": {"bytes": json.dumps(payload).encode("utf-8")}}

    def test_yields_content_block_start(self):
        handler = BedrockStreamHandler()
        stream = [self._make_chunk_event({"content_block_start": {"index": 0}})]
        events = list(handler.parse_stream(iter(stream)))
        assert len(events) == 1
        assert events[0].type == "content_block_start"

    def test_yields_content_block_delta(self):
        handler = BedrockStreamHandler()
        payload = {"content_block_delta": {"delta": {"text": "Hello world"}}}
        stream = [self._make_chunk_event(payload)]
        events = list(handler.parse_stream(iter(stream)))
        assert len(events) == 1
        assert events[0].type == "content_block_delta"
        assert events[0].text == "Hello world"

    def test_yields_message_stop(self):
        handler = BedrockStreamHandler()
        payload = {"message_stop": {"stop_reason": "end_turn"}}
        stream = [self._make_chunk_event(payload)]
        events = list(handler.parse_stream(iter(stream)))
        assert len(events) == 1
        assert events[0].type == "message_stop"

    def test_skips_invalid_json(self):
        handler = BedrockStreamHandler()
        bad_event = {"chunk": {"bytes": b"not valid json {"}}
        events = list(handler.parse_stream(iter([bad_event])))
        assert len(events) == 0

    def test_skips_event_without_chunk_key(self):
        handler = BedrockStreamHandler()
        events = list(handler.parse_stream(iter([{"other_key": "value"}])))
        assert len(events) == 0

    def test_empty_stream_yields_nothing(self):
        handler = BedrockStreamHandler()
        events = list(handler.parse_stream(iter([])))
        assert events == []

    def test_multiple_events_in_sequence(self):
        handler = BedrockStreamHandler()
        stream = [
            self._make_chunk_event({"content_block_start": {"index": 0}}),
            self._make_chunk_event({"content_block_delta": {"delta": {"text": "Hi"}}}),
            self._make_chunk_event({"message_stop": {}}),
        ]
        events = list(handler.parse_stream(iter(stream)))
        assert len(events) == 3


# ---------------------------------------------------------------------------
# CodeChunk (base extractor)
# ---------------------------------------------------------------------------

class TestCodeChunk:
    def _make(self, **overrides):
        defaults = dict(
            content="def foo(): pass",
            source_file="src/foo.py",
            language="python",
            symbol_type="function",
            symbol_name="foo",
            line_start=1,
            line_end=1,
        )
        defaults.update(overrides)
        return CodeChunk(**defaults)

    def test_fields_stored(self):
        c = self._make()
        assert c.content == "def foo(): pass"
        assert c.source_file == "src/foo.py"
        assert c.language == "python"
        assert c.symbol_type == "function"
        assert c.symbol_name == "foo"

    def test_to_dict_returns_dict(self):
        c = self._make()
        assert isinstance(c.to_dict(), dict)

    def test_to_dict_has_content(self):
        c = self._make()
        d = c.to_dict()
        assert d["content"] == "def foo(): pass"

    def test_to_dict_has_metadata(self):
        c = self._make()
        d = c.to_dict()
        assert "metadata" in d

    def test_to_dict_metadata_has_symbol_name(self):
        c = self._make()
        assert c.to_dict()["metadata"]["symbol_name"] == "foo"

    def test_optional_docstring_in_metadata(self):
        c = self._make(docstring="Do something")
        d = c.to_dict()
        assert d["metadata"]["docstring"] == "Do something"

    def test_optional_signature_absent_when_none(self):
        c = self._make()
        assert "signature" not in c.to_dict()["metadata"]

    def test_decorators_default_empty(self):
        c = self._make()
        assert c.decorators == []

    def test_visibility_default_public(self):
        c = self._make()
        assert c.visibility == "public"


# ---------------------------------------------------------------------------
# _strip_jsonc
# ---------------------------------------------------------------------------

class TestStripJsonc:
    def test_plain_json_unchanged(self):
        result = _strip_jsonc('{"key": "value"}')
        assert json.loads(result) == {"key": "value"}

    def test_single_line_comment_stripped(self):
        jsonc = '{"key": "value"} // this is a comment'
        result = _strip_jsonc(jsonc)
        data = json.loads(result)
        assert data["key"] == "value"

    def test_block_comment_stripped(self):
        jsonc = '{"key": /* block comment */ "value"}'
        result = _strip_jsonc(jsonc)
        data = json.loads(result)
        assert data["key"] == "value"

    def test_trailing_comma_removed(self):
        jsonc = '{"a": 1, "b": 2,}'
        result = _strip_jsonc(jsonc)
        data = json.loads(result)
        assert data["a"] == 1
        assert data["b"] == 2

    def test_comment_in_string_preserved(self):
        jsonc = '{"url": "http://example.com/page"}'
        result = _strip_jsonc(jsonc)
        data = json.loads(result)
        assert "http://example.com/page" in data["url"]

    def test_multiline_block_comment(self):
        jsonc = '{\n  /* multiline\n  comment */\n  "key": "value"\n}'
        result = _strip_jsonc(jsonc)
        data = json.loads(result)
        assert data["key"] == "value"

    def test_trailing_comma_in_array(self):
        jsonc = '{"arr": [1, 2, 3,]}'
        result = _strip_jsonc(jsonc)
        data = json.loads(result)
        assert data["arr"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# JSONExtractor._detect_json_type
# ---------------------------------------------------------------------------

class TestDetectJsonType:
    def test_composer_json_by_filename(self):
        ext = JSONExtractor()
        assert ext._detect_json_type({}, "path/to/composer.json") == "composer"

    def test_composer_lock_by_filename(self):
        ext = JSONExtractor()
        assert ext._detect_json_type({}, "composer.lock") == "composer-lock"

    def test_package_json_by_filename(self):
        ext = JSONExtractor()
        assert ext._detect_json_type({}, "package.json") == "package"

    def test_package_lock_json_by_filename(self):
        ext = JSONExtractor()
        assert ext._detect_json_type({}, "package-lock.json") == "package-lock"

    def test_tsconfig_by_filename(self):
        ext = JSONExtractor()
        assert ext._detect_json_type({}, "tsconfig.json") == "tsconfig"

    def test_tsconfig_by_content(self):
        ext = JSONExtractor()
        data = {"compilerOptions": {"target": "ES2020"}}
        assert ext._detect_json_type(data, "config.json") == "tsconfig"

    def test_generic_for_unknown(self):
        ext = JSONExtractor()
        data = {"some": "data"}
        assert ext._detect_json_type(data, "arbitrary.json") == "generic"

    def test_package_by_content(self):
        ext = JSONExtractor()
        data = {"dependencies": {"react": "18"}, "scripts": {"build": "webpack"}}
        assert ext._detect_json_type(data, "my-config.json") == "package"


# ---------------------------------------------------------------------------
# JSONExtractor.extract
# ---------------------------------------------------------------------------

class TestJSONExtractorExtract:
    def test_simple_json_returns_chunks(self):
        ext = JSONExtractor()
        src = '{"name": "test", "version": "1.0"}'
        chunks = ext.extract(src, "config.json")
        assert len(chunks) >= 1

    def test_chunks_are_code_chunks(self):
        ext = JSONExtractor()
        src = '{"key": "value"}'
        chunks = ext.extract(src, "data.json")
        for c in chunks:
            assert isinstance(c, CodeChunk)

    def test_array_json_returns_document_chunk(self):
        ext = JSONExtractor()
        src = '[1, 2, 3]'
        chunks = ext.extract(src, "list.json")
        assert len(chunks) == 1
        assert chunks[0].symbol_type == "document"

    def test_invalid_json_returns_document_chunk(self):
        ext = JSONExtractor()
        src = '{not valid json'
        chunks = ext.extract(src, "broken.json")
        assert len(chunks) == 1
        assert chunks[0].symbol_type == "document"

    def test_jsonc_with_comments_parsed(self):
        ext = JSONExtractor()
        src = '{\n  // a comment\n  "key": "value"\n}'
        chunks = ext.extract(src, "settings.json")
        assert len(chunks) >= 1

    def test_package_json_returns_project_chunk(self):
        ext = JSONExtractor()
        data = {"name": "my-app", "version": "1.0.0", "scripts": {"build": "webpack"}}
        src = json.dumps(data)
        chunks = ext.extract(src, "package.json")
        types = [c.symbol_type for c in chunks]
        assert "project" in types

    def test_composer_json_returns_project_chunk(self):
        ext = JSONExtractor()
        data = {"name": "vendor/package", "description": "A PHP lib"}
        src = json.dumps(data)
        chunks = ext.extract(src, "composer.json")
        types = [c.symbol_type for c in chunks]
        assert "project" in types

    def test_tsconfig_returns_config_chunk(self):
        ext = JSONExtractor()
        data = {"compilerOptions": {"target": "ES2020", "strict": True}}
        src = json.dumps(data)
        chunks = ext.extract(src, "tsconfig.json")
        assert chunks[0].symbol_type == "config"

    def test_chunk_has_source_file(self):
        ext = JSONExtractor()
        src = '{"key": "value"}'
        chunks = ext.extract(src, "my/file.json")
        assert all(c.source_file == "my/file.json" for c in chunks)

    def test_chunk_language_is_json(self):
        ext = JSONExtractor()
        src = '{"key": "value"}'
        chunks = ext.extract(src, "data.json")
        assert all(c.language == "json" for c in chunks)


# ---------------------------------------------------------------------------
# GherkinExtractor.extract
# ---------------------------------------------------------------------------

SIMPLE_FEATURE = """\
Feature: User login
  As a user I want to log in so that I can access the system.

  Scenario: Successful login
    Given I have valid credentials
    When I submit the login form
    Then I should see the dashboard

  Scenario: Failed login
    Given I have invalid credentials
    When I submit the login form
    Then I should see an error message
"""

FEATURE_WITH_BACKGROUND = """\
Feature: Shopping cart

  Background:
    Given I am a logged-in user
    And the cart is empty

  Scenario: Add item to cart
    When I add a product to the cart
    Then the cart should have one item
"""

FEATURE_WITH_TAGS = """\
@smoke @login
Feature: Tagged feature
  Description here.

  @fast
  Scenario: Quick test
    Given something
    Then something else
"""


class TestGherkinExtractorExtract:
    def test_returns_list(self):
        ext = GherkinExtractor()
        chunks = ext.extract(SIMPLE_FEATURE, "login.feature")
        assert isinstance(chunks, list)

    def test_contains_feature_chunk(self):
        ext = GherkinExtractor()
        chunks = ext.extract(SIMPLE_FEATURE, "login.feature")
        types = [c.symbol_type for c in chunks]
        assert "feature" in types

    def test_feature_chunk_has_name(self):
        ext = GherkinExtractor()
        chunks = ext.extract(SIMPLE_FEATURE, "login.feature")
        feature_chunks = [c for c in chunks if c.symbol_type == "feature"]
        assert feature_chunks[0].symbol_name == "User login"

    def test_scenarios_extracted(self):
        ext = GherkinExtractor()
        chunks = ext.extract(SIMPLE_FEATURE, "login.feature")
        scenario_chunks = [c for c in chunks if c.symbol_type == "scenario"]
        assert len(scenario_chunks) == 2

    def test_scenario_names_correct(self):
        ext = GherkinExtractor()
        chunks = ext.extract(SIMPLE_FEATURE, "login.feature")
        names = [c.symbol_name for c in chunks if c.symbol_type == "scenario"]
        assert "Successful login" in names
        assert "Failed login" in names

    def test_background_extracted(self):
        ext = GherkinExtractor()
        chunks = ext.extract(FEATURE_WITH_BACKGROUND, "cart.feature")
        types = [c.symbol_type for c in chunks]
        assert "background" in types

    def test_no_background_in_simple_feature(self):
        ext = GherkinExtractor()
        chunks = ext.extract(SIMPLE_FEATURE, "login.feature")
        types = [c.symbol_type for c in chunks]
        assert "background" not in types

    def test_chunk_has_source_file(self):
        ext = GherkinExtractor()
        chunks = ext.extract(SIMPLE_FEATURE, "path/login.feature")
        assert all(c.source_file == "path/login.feature" for c in chunks)

    def test_chunk_language_is_gherkin(self):
        ext = GherkinExtractor()
        chunks = ext.extract(SIMPLE_FEATURE, "login.feature")
        assert all(c.language == "gherkin" for c in chunks)

    def test_steps_extracted_in_scenario(self):
        ext = GherkinExtractor()
        chunks = ext.extract(SIMPLE_FEATURE, "login.feature")
        scenario = next(c for c in chunks if c.symbol_type == "scenario")
        assert len(scenario.extras.get("steps", [])) > 0

    def test_feature_tags_in_extras(self):
        ext = GherkinExtractor()
        chunks = ext.extract(FEATURE_WITH_TAGS, "tagged.feature")
        feature_chunk = next(c for c in chunks if c.symbol_type == "feature")
        tags = feature_chunk.extras.get("tags", [])
        assert "@smoke" in tags or "smoke" in [t.lstrip("@") for t in tags]

    def test_empty_feature_no_crash(self):
        ext = GherkinExtractor()
        chunks = ext.extract("", "empty.feature")
        assert isinstance(chunks, list)

    def test_scenario_count_in_feature_extras(self):
        ext = GherkinExtractor()
        chunks = ext.extract(SIMPLE_FEATURE, "login.feature")
        feature_chunk = next(c for c in chunks if c.symbol_type == "feature")
        assert feature_chunk.extras.get("scenario_count", 0) == 2
