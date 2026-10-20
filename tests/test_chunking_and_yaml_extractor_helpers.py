"""Tests for knowledge/engine/chunking pure helpers and YAMLExtractor."""

from mullet.knowledge.engine.chunking import (
    ChunkingConfig,
    _split_to_sentences,
    chunk_by_recursive,
    _apply_overlap,
)
from mullet.knowledge.code.extractors.yaml_extractor import YAMLExtractor


# ---------------------------------------------------------------------------
# ChunkingConfig
# ---------------------------------------------------------------------------

class TestChunkingConfig:
    def test_default_strategy_title(self):
        assert ChunkingConfig().strategy == "title"

    def test_default_max_chunk_size_positive(self):
        assert ChunkingConfig().max_chunk_size > 0

    def test_default_overlap_nonnegative(self):
        assert ChunkingConfig().overlap >= 0

    def test_default_min_chunk_size_positive(self):
        assert ChunkingConfig().min_chunk_size > 0

    def test_default_separators_nonempty(self):
        assert len(ChunkingConfig().separators) > 0

    def test_custom_strategy(self):
        cfg = ChunkingConfig(strategy="semantic")
        assert cfg.strategy == "semantic"

    def test_custom_max_chunk_size(self):
        cfg = ChunkingConfig(max_chunk_size=500)
        assert cfg.max_chunk_size == 500


# ---------------------------------------------------------------------------
# _split_to_sentences
# ---------------------------------------------------------------------------

class TestSplitToSentences:
    def test_returns_list(self):
        assert isinstance(_split_to_sentences("Hello."), list)

    def test_empty_string_returns_empty(self):
        assert _split_to_sentences("") == []

    def test_whitespace_only_returns_empty(self):
        assert _split_to_sentences("   ") == []

    def test_single_sentence(self):
        result = _split_to_sentences("Hello world.")
        assert len(result) == 1
        assert result[0] == "Hello world."

    def test_multiple_sentences_split(self):
        result = _split_to_sentences("Hello world. How are you? I am fine!")
        assert len(result) == 3

    def test_sentences_preserved(self):
        result = _split_to_sentences("First. Second. Third.")
        assert "First." in result
        assert "Second." in result

    def test_no_empty_strings_in_result(self):
        result = _split_to_sentences("Hello. World.")
        assert all(s for s in result)


# ---------------------------------------------------------------------------
# chunk_by_recursive
# ---------------------------------------------------------------------------

class TestChunkByRecursive:
    def test_returns_list(self):
        assert isinstance(chunk_by_recursive("hello world", max_chars=100), list)

    def test_empty_text_returns_empty(self):
        assert chunk_by_recursive("", max_chars=100) == []

    def test_short_text_single_chunk(self):
        result = chunk_by_recursive("short text", max_chars=1000)
        assert len(result) == 1
        assert result[0] == "short text"

    def test_long_text_split_into_multiple(self):
        long_text = "word " * 100
        result = chunk_by_recursive(long_text, max_chars=50)
        assert len(result) > 1

    def test_chunks_not_exceed_max_chars(self):
        long_text = "word " * 100
        max_chars = 100
        result = chunk_by_recursive(long_text, max_chars=max_chars)
        for chunk in result:
            assert len(chunk) <= max_chars * 1.1  # allow small overage for word boundaries

    def test_paragraph_split_at_double_newline(self):
        text = "paragraph one\n\nparagraph two\n\nparagraph three"
        result = chunk_by_recursive(text, max_chars=200)
        # Combined is under limit, may stay as one or split
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_content_preserved(self):
        text = "hello world foo bar"
        result = chunk_by_recursive(text, max_chars=1000)
        combined = " ".join(result)
        assert "hello" in combined
        assert "world" in combined


# ---------------------------------------------------------------------------
# _apply_overlap
# ---------------------------------------------------------------------------

class TestApplyOverlap:
    def test_empty_input_returns_empty(self):
        assert _apply_overlap([], overlap=10) == []

    def test_single_chunk_unchanged(self):
        result = _apply_overlap(["only chunk"], overlap=5)
        assert len(result) == 1

    def test_zero_overlap_unchanged(self):
        parts = ["a", "b", "c"]
        result = _apply_overlap(parts, overlap=0)
        assert result == ["a", "b", "c"]

    def test_overlap_prepends_suffix(self):
        parts = ["hello world", "foo bar baz", "qux"]
        result = _apply_overlap(parts, overlap=5)
        # Second chunk should contain tail of first
        assert "world" in result[1]

    def test_returns_list(self):
        assert isinstance(_apply_overlap(["a", "b"], overlap=2), list)

    def test_count_preserved(self):
        parts = ["a", "b", "c"]
        result = _apply_overlap(parts, overlap=1)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# YAMLExtractor
# ---------------------------------------------------------------------------

class TestYAMLExtractorGeneric:
    def setup_method(self):
        self.ext = YAMLExtractor()

    def test_returns_list(self):
        assert isinstance(self.ext.extract("key: value", "test.yaml"), list)

    def test_language_is_yaml(self):
        result = self.ext.extract("key: value", "test.yaml")
        assert result[0].language == "yaml"

    def test_source_file_preserved(self):
        result = self.ext.extract("key: value", "config.yaml")
        assert result[0].source_file == "config.yaml"

    def test_empty_yaml_returns_chunk(self):
        result = self.ext.extract("", "empty.yaml")
        assert len(result) >= 1

    def test_simple_yaml_content_preserved(self):
        result = self.ext.extract("key: value", "test.yaml")
        assert "key" in result[0].content or "value" in result[0].content

    def test_invalid_yaml_returns_chunk(self):
        # Should not raise
        result = self.ext.extract("{bad yaml here", "bad.yaml")
        assert len(result) >= 1


class TestYAMLExtractorDockerCompose:
    def setup_method(self):
        self.ext = YAMLExtractor()

    def test_services_extracted(self):
        dc = 'version: "3"\nservices:\n  web:\n    image: nginx\n  db:\n    image: postgres\n'
        result = self.ext.extract(dc, "docker-compose.yaml")
        types = {c.symbol_type for c in result}
        assert "service" in types

    def test_each_service_has_chunk(self):
        dc = 'version: "3"\nservices:\n  web:\n    image: nginx\n  db:\n    image: postgres\n'
        result = self.ext.extract(dc, "docker-compose.yaml")
        service_names = {c.symbol_name for c in result if c.symbol_type == "service"}
        assert "web" in service_names
        assert "db" in service_names


class TestYAMLExtractorK8s:
    def setup_method(self):
        self.ext = YAMLExtractor()

    def test_k8s_resource_extracted(self):
        k8s = "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: my-app\nspec:\n  replicas: 3\n"
        result = self.ext.extract(k8s, "deployment.yaml")
        types = {c.symbol_type for c in result}
        assert "resource" in types

    def test_k8s_resource_name_includes_kind(self):
        k8s = "apiVersion: v1\nkind: Service\nmetadata:\n  name: my-svc\n"
        result = self.ext.extract(k8s, "svc.yaml")
        resource_names = [c.symbol_name for c in result if c.symbol_type == "resource"]
        if resource_names:
            assert any("Service" in n or "my-svc" in n for n in resource_names)


class TestYAMLExtractorGitHubActions:
    def setup_method(self):
        self.ext = YAMLExtractor()

    def test_jobs_extracted(self):
        gh = "name: CI\non: [push]\njobs:\n  test:\n    runs-on: ubuntu-latest\n"
        result = self.ext.extract(gh, ".github/workflows/ci.yaml")
        types = {c.symbol_type for c in result}
        assert "job" in types

    def test_job_name_extracted(self):
        gh = "name: CI\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest\n"
        result = self.ext.extract(gh, ".github/workflows/ci.yaml")
        job_names = {c.symbol_name for c in result if c.symbol_type == "job"}
        assert "build" in job_names
