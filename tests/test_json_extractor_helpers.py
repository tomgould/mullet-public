"""Tests for mullet.knowledge.code.extractors.json_extractor — pure helpers."""

import json
from mullet.knowledge.code.extractors.json_extractor import (
    _strip_jsonc,
    JSONExtractor,
)


# ---------------------------------------------------------------------------
# _strip_jsonc — comment stripping
# ---------------------------------------------------------------------------

class TestStripJsonc:
    def test_plain_json_unchanged(self):
        text = '{"key": "value"}'
        result = _strip_jsonc(text)
        assert json.loads(result) == {"key": "value"}

    def test_single_line_comment_stripped(self):
        text = '{\n  // this is a comment\n  "key": "value"\n}'
        result = _strip_jsonc(text)
        data = json.loads(result)
        assert data["key"] == "value"

    def test_block_comment_stripped(self):
        text = '{\n  /* block comment */\n  "key": "value"\n}'
        result = _strip_jsonc(text)
        data = json.loads(result)
        assert data["key"] == "value"

    def test_trailing_comma_object_removed(self):
        text = '{"a": 1, "b": 2,}'
        result = _strip_jsonc(text)
        data = json.loads(result)
        assert data["a"] == 1

    def test_trailing_comma_array_removed(self):
        text = '[1, 2, 3,]'
        result = _strip_jsonc(text)
        assert json.loads(result) == [1, 2, 3]

    def test_url_in_string_preserved(self):
        text = '{"url": "http://example.com/path"}'
        result = _strip_jsonc(text)
        data = json.loads(result)
        assert data["url"] == "http://example.com/path"

    def test_comment_marker_inside_string_preserved(self):
        text = '{"note": "// not a comment"}'
        result = _strip_jsonc(text)
        data = json.loads(result)
        assert data["note"] == "// not a comment"

    def test_block_comment_marker_in_string_preserved(self):
        text = '{"note": "/* not a block comment */"}'
        result = _strip_jsonc(text)
        data = json.loads(result)
        assert data["note"] == "/* not a block comment */"

    def test_empty_string_returns_empty(self):
        assert _strip_jsonc("") == ""

    def test_multiple_comments_all_stripped(self):
        text = '{\n  // comment 1\n  "a": 1,\n  "b": 2\n}'
        result = _strip_jsonc(text)
        data = json.loads(result)
        assert data["a"] == 1
        assert data["b"] == 2

    def test_escape_in_string_preserved(self):
        text = r'{"key": "val\"ue"}'
        result = _strip_jsonc(text)
        data = json.loads(result)
        assert "val" in data["key"]


# ---------------------------------------------------------------------------
# JSONExtractor._detect_json_type
# ---------------------------------------------------------------------------

class TestDetectJsonType:
    def setup_method(self):
        self.extractor = JSONExtractor()

    def test_composer_json_by_filename(self):
        assert self.extractor._detect_json_type({}, "path/to/composer.json") == "composer"

    def test_composer_lock_by_filename(self):
        assert self.extractor._detect_json_type({}, "path/to/composer.lock") == "composer-lock"

    def test_package_json_by_filename(self):
        assert self.extractor._detect_json_type({}, "path/to/package.json") == "package"

    def test_package_lock_by_filename(self):
        assert self.extractor._detect_json_type({}, "path/to/package-lock.json") == "package-lock"

    def test_tsconfig_by_filename(self):
        assert self.extractor._detect_json_type({}, "path/to/tsconfig.json") == "tsconfig"

    def test_jsconfig_by_filename(self):
        assert self.extractor._detect_json_type({}, "path/to/jsconfig.json") == "tsconfig"

    def test_tsconfig_by_content(self):
        data = {"compilerOptions": {"target": "ES2020"}}
        assert self.extractor._detect_json_type(data, "config.json") == "tsconfig"

    def test_package_by_content(self):
        data = {"name": "myapp", "dependencies": {"react": "^18"}, "scripts": {"build": "tsc"}}
        assert self.extractor._detect_json_type(data, "config.json") == "package"

    def test_generic_fallback(self):
        data = {"arbitrary": "data", "no": "known fields"}
        assert self.extractor._detect_json_type(data, "unknown.json") == "generic"


# ---------------------------------------------------------------------------
# JSONExtractor.extract — integration
# ---------------------------------------------------------------------------

class TestJSONExtractorExtract:
    def setup_method(self):
        self.extractor = JSONExtractor()

    def test_returns_list(self):
        result = self.extractor.extract('{"a": 1}', "test.json")
        assert isinstance(result, list)

    def test_malformed_json_returns_single_chunk(self):
        result = self.extractor.extract("not valid json at all!!!", "bad.json")
        assert len(result) == 1

    def test_package_json_extracts_project_chunk(self):
        data = json.dumps({
            "name": "my-app",
            "version": "1.0.0",
            "description": "Test app",
            "scripts": {"build": "tsc"},
            "main": "index.js",
        })
        chunks = self.extractor.extract(data, "package.json")
        types = {c.symbol_type for c in chunks}
        assert "project" in types

    def test_package_json_with_deps_extracts_deps(self):
        data = json.dumps({
            "name": "app",
            "dependencies": {"express": "^4"},
            "scripts": {"start": "node index.js"},
            "main": "index.js",
        })
        chunks = self.extractor.extract(data, "package.json")
        types = {c.symbol_type for c in chunks}
        assert "dependencies" in types

    def test_tsconfig_returns_config_chunk(self):
        data = json.dumps({"compilerOptions": {"target": "ES2020", "strict": True}})
        chunks = self.extractor.extract(data, "tsconfig.json")
        assert any(c.symbol_type == "config" for c in chunks)

    def test_jsonc_with_comments_parsed(self):
        jsonc = '{\n  "name": "test-app",\n  // a comment\n  "scripts": {"build": "tsc"},\n  "main": "index.js"\n}'
        chunks = self.extractor.extract(jsonc, "package.json")
        assert len(chunks) > 0

    def test_array_json_returns_document_chunk(self):
        data = json.dumps([1, 2, 3])
        chunks = self.extractor.extract(data, "array.json")
        assert len(chunks) == 1
        assert chunks[0].symbol_type == "document"

    def test_chunks_have_source_file(self):
        data = json.dumps({"key": "value"})
        chunks = self.extractor.extract(data, "my/file.json")
        assert all(c.source_file == "my/file.json" for c in chunks)

    # -- non-dict JSON (list) ------------------------------------------------

    def test_list_json_returns_document_chunk(self):
        data = json.dumps(["a", "b"])
        chunks = self.extractor.extract(data, "data.json")
        assert len(chunks) == 1
        assert chunks[0].symbol_type == "document"

    # -- composer.json -------------------------------------------------------

    def test_composer_json_extracts_all_chunks(self):
        data = json.dumps({
            "name": "vendor/pkg",
            "description": "A PHP package",
            "require": {"php": ">=8.0", "guzzlehttp/guzzle": "^7.0"},
            "require-dev": {"phpunit/phpunit": "^9.0"},
            "type": "library",
            "autoload": {"psr-4": {"Pkg\\\\": "src/"}},
            "scripts": {"test": "phpunit"},
        })
        chunks = self.extractor.extract(data, "composer.json")
        types = {c.symbol_type for c in chunks}
        assert "project" in types
        assert "dependencies" in types
        assert "config" in types
        assert "scripts" in types
        # Verify project chunk specifics
        proj = [c for c in chunks if c.symbol_type == "project"][0]
        assert proj.symbol_name == "vendor/pkg"
        assert proj.extras["json_type"] == "composer"
        # Verify require chunk (should exclude php)
        req = [c for c in chunks if c.symbol_name == "require"][0]
        assert "guzzlehttp/guzzle" in req.content
        assert "php" not in req.content
        # Verify require-dev chunk
        req_dev = [c for c in chunks if c.symbol_name == "require-dev"][0]
        assert "phpunit/phpunit" in req_dev.content

    def test_composer_json_minimal(self):
        data = json.dumps({"name": "vendor/minimal"})
        chunks = self.extractor.extract(data, "composer.json")
        assert len(chunks) == 1
        assert chunks[0].symbol_type == "project"

    # -- composer.lock -------------------------------------------------------

    def test_composer_lock_extracts_lockfile_chunk(self):
        data = json.dumps({
            "_readme": [],
            "packages": [
                {"name": "vendor/dep", "version": "1.0.0"},
            ],
            "packages-dev": [],
        })
        chunks = self.extractor.extract(data, "composer.lock")
        assert len(chunks) == 1
        assert chunks[0].symbol_type == "lockfile"
        assert chunks[0].extras["json_type"] == "composer-lock"
        assert chunks[0].extras["package_count"] == 1
        assert chunks[0].extras["dev_package_count"] == 0

    def test_composer_lock_many_packages_truncates(self):
        packages = [{"name": f"vendor/pkg{i}", "version": "1.0.0"} for i in range(25)]
        dev_packages = [{"name": f"vendor/dev{i}", "version": "1.0.0"} for i in range(15)]
        data = json.dumps({
            "packages": packages,
            "packages-dev": dev_packages,
        })
        chunks = self.extractor.extract(data, "composer.lock")
        assert len(chunks) == 1
        assert "... and 5 more" in chunks[0].content
        assert "... and 5 more" in chunks[0].content.split("Dev Packages")[1]

    # -- package.json (devDependencies branch) -------------------------------

    def test_package_json_with_dev_deps(self):
        data = json.dumps({
            "name": "my-app",
            "version": "1.0.0",
            "dependencies": {"react": "^18.0"},
            "devDependencies": {"jest": "^29.0"},
            "scripts": {"build": "webpack"},
        })
        chunks = self.extractor.extract(data, "package.json")
        names = {c.symbol_name for c in chunks}
        assert "devDependencies" in names
        dev_chunk = [c for c in chunks if c.symbol_name == "devDependencies"][0]
        assert "jest" in dev_chunk.content

    # -- package-lock.json ---------------------------------------------------

    def test_package_lock_extracts_lockfile(self):
        data = json.dumps({
            "name": "my-app",
            "lockfileVersion": 2,
            "packages": {
                "": {"name": "my-app"},
                "node_modules/react": {"version": "18.0.0"},
                "node_modules/lodash": {"version": "4.17.21"},
            },
        })
        chunks = self.extractor.extract(data, "package-lock.json")
        assert len(chunks) == 1
        assert chunks[0].symbol_type == "lockfile"
        assert chunks[0].extras["json_type"] == "package-lock"
        assert chunks[0].extras["package_count"] == 2
        assert "Total packages: 2" in chunks[0].content

    # -- tsconfig.json (truncation path) -------------------------------------

    def test_tsconfig_large_truncates(self):
        # Create a tsconfig with >2000 chars of source_code
        compiler_options = {f"opt{i}": f"value{i}" for i in range(100)}
        data = json.dumps({
            "compilerOptions": compiler_options,
            "include": ["src"],
        }, indent=2)
        assert len(data) > 2000
        chunks = self.extractor.extract(data, "tsconfig.json")
        assert len(chunks) == 1
        assert chunks[0].symbol_type == "config"
        assert "// truncated..." in chunks[0].content

    # -- generic JSON (large file path) --------------------------------------

    def test_generic_json_large_extracts_keys(self):
        data = {f"key{i}": f"value{i}" for i in range(80)}
        source = json.dumps(data, indent=2)
        assert len(source) >= 1500
        chunks = self.extractor.extract(source, "config.json")
        assert len(chunks) > 1
        names = {c.symbol_name for c in chunks}
        assert "key0" in names
        assert "key79" in names
        assert all(c.symbol_type == "key" for c in chunks)

    def test_generic_json_large_value_truncates(self):
        big_value = "x" * 3000
        data = json.dumps({"small": "val", "huge": big_value}, indent=2)
        chunks = self.extractor.extract(data, "config.json")
        huge_chunk = [c for c in chunks if c.symbol_name == "huge"][0]
        assert "// truncated..." in huge_chunk.content

    def test_generic_json_small_returns_document(self):
        data = json.dumps({"key1": "value1", "key2": [1, 2, 3], "key3": {"nested": True}})
        chunks = self.extractor.extract(data, "config.json")
        assert len(chunks) == 1
        assert chunks[0].symbol_type == "document"

    # -- content detection ---------------------------------------------------

    def test_detect_composer_by_content(self):
        data = json.dumps({
            "name": "vendor/pkg",
            "require": {"php": ">=8.0"},
            "autoload": {"psr-4": {"App\\\\": "src/"}},
        })
        chunks = self.extractor.extract(data, "some-file.json")
        assert any(c.extras.get("json_type") == "composer" for c in chunks)

    def test_detect_package_by_content(self):
        data = json.dumps({
            "name": "my-app",
            "dependencies": {"react": "^18.0"},
            "scripts": {"build": "webpack"},
        })
        chunks = self.extractor.extract(data, "manifest.json")
        assert any(c.extras.get("json_type") == "package" for c in chunks)

    def test_detect_tsconfig_by_content(self):
        data = json.dumps({
            "compilerOptions": {"target": "ES2020", "strict": True},
            "include": ["src"],
        })
        chunks = self.extractor.extract(data, "config.json")
        assert any(c.extras.get("json_type") == "tsconfig" for c in chunks)

    def test_detect_composer_lock_by_content(self):
        data = {"packages": {"": {"_resolved": "https://repo.example/pkg.tgz"}}}
        assert self.extractor._detect_json_type(data, "lock.json") == "composer-lock"
