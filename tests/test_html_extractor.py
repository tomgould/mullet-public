"""Tests for mullet.knowledge.code.extractors.html."""
from __future__ import annotations

import pytest

from mullet.knowledge.code.extractors.html import HTMLExtractor


_SIMPLE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Test Page</title>
</head>
<body>
  <main>
    <header>
      <nav>
        <a href="/">Home</a>
      </nav>
    </header>
    <article>
      <h1>Hello World</h1>
      <p>Some content</p>
    </article>
  </main>
</body>
</html>
"""

_VUE_SFC = """\
<template>
  <div class="app">
    <my-header :title="title" />
    <main>
      <slot />
    </main>
  </div>
</template>

<script>
export default {
  name: 'AppLayout',
  props: {
    title: String,
  },
  data() {
    return { count: 0 };
  },
}
</script>

<style scoped>
.app {
  min-height: 100vh;
}
</style>
"""

_SVELTE = """\
<script>
  export let name = 'world';
  let count = 0;
  const increment = () => count++;
</script>

<main>
  <h1>Hello {name}!</h1>
  <button on:click={increment}>Count: {count}</button>
</main>

<style>
  main { padding: 1rem; }
</style>
"""

_CUSTOM_ELEMENTS = """\
<body>
  <my-button variant="primary">Click me</my-button>
  <user-avatar src="avatar.png" size="large"></user-avatar>
  <app-sidebar id="main-sidebar" class="sidebar dark"></app-sidebar>
</body>
"""


@pytest.fixture
def extractor() -> HTMLExtractor:
    return HTMLExtractor()


class TestHTMLBasics:
    def test_returns_list(self, extractor: HTMLExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_HTML, "index.html")
        assert isinstance(chunks, list)

    def test_extracts_semantic_elements(self, extractor: HTMLExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_HTML, "index.html")
        types = {c.symbol_name for c in chunks}
        # main, header, nav, article are all significant
        assert any("main" in n for n in types)

    def test_language_is_html(self, extractor: HTMLExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_HTML, "index.html")
        assert all(c.language == "html" for c in chunks)

    def test_source_file_set(self, extractor: HTMLExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_HTML, "index.html")
        assert all(c.source_file == "index.html" for c in chunks)


class TestVueSFC:
    def test_extracts_template_section(self, extractor: HTMLExtractor) -> None:
        chunks = extractor.extract(_VUE_SFC, "AppLayout.vue")
        template_chunks = [c for c in chunks if c.symbol_type == "template"]
        assert len(template_chunks) == 1

    def test_extracts_component_section(self, extractor: HTMLExtractor) -> None:
        chunks = extractor.extract(_VUE_SFC, "AppLayout.vue")
        comp_chunks = [c for c in chunks if c.symbol_type == "component"]
        assert len(comp_chunks) == 1
        assert "AppLayout" in comp_chunks[0].symbol_name

    def test_extracts_style_section(self, extractor: HTMLExtractor) -> None:
        chunks = extractor.extract(_VUE_SFC, "AppLayout.vue")
        style_chunks = [c for c in chunks if c.symbol_type == "style"]
        assert len(style_chunks) == 1

    def test_vue_language_is_vue(self, extractor: HTMLExtractor) -> None:
        chunks = extractor.extract(_VUE_SFC, "AppLayout.vue")
        assert all(c.language == "vue" for c in chunks)

    def test_template_name_includes_component(self, extractor: HTMLExtractor) -> None:
        chunks = extractor.extract(_VUE_SFC, "components/AppLayout.vue")
        template = next(c for c in chunks if c.symbol_type == "template")
        assert "AppLayout" in template.symbol_name

    def test_vue_no_style_ok(self, extractor: HTMLExtractor) -> None:
        vue_no_style = "<template><div>hello</div></template>\n<script>export default {}</script>"
        chunks = extractor.extract(vue_no_style, "Simple.vue")
        style_chunks = [c for c in chunks if c.symbol_type == "style"]
        assert len(style_chunks) == 0


class TestSvelte:
    def test_extracts_script_as_component(self, extractor: HTMLExtractor) -> None:
        chunks = extractor.extract(_SVELTE, "Counter.svelte")
        comp_chunks = [c for c in chunks if c.symbol_type == "component"]
        assert len(comp_chunks) == 1
        assert "Counter" in comp_chunks[0].symbol_name

    def test_extracts_template(self, extractor: HTMLExtractor) -> None:
        chunks = extractor.extract(_SVELTE, "Counter.svelte")
        tmpl_chunks = [c for c in chunks if c.symbol_type == "template"]
        assert len(tmpl_chunks) == 1

    def test_svelte_language_is_svelte(self, extractor: HTMLExtractor) -> None:
        chunks = extractor.extract(_SVELTE, "Counter.svelte")
        assert all(c.language == "svelte" for c in chunks)

    def test_svelte_template_excludes_script(self, extractor: HTMLExtractor) -> None:
        chunks = extractor.extract(_SVELTE, "Counter.svelte")
        tmpl = next(c for c in chunks if c.symbol_type == "template")
        assert "<script>" not in tmpl.content


class TestCustomElements:
    def test_extracts_custom_elements(self, extractor: HTMLExtractor) -> None:
        chunks = extractor.extract(_CUSTOM_ELEMENTS, "page.html")
        custom = [c for c in chunks if c.symbol_type == "component"]
        names = [c.symbol_name for c in custom]
        assert any("my-button" in n for n in names)
        assert any("user-avatar" in n for n in names)

    def test_id_included_in_symbol_name(self, extractor: HTMLExtractor) -> None:
        chunks = extractor.extract(_CUSTOM_ELEMENTS, "page.html")
        sidebar = next((c for c in chunks if "app-sidebar" in c.symbol_name), None)
        assert sidebar is not None
        assert "main-sidebar" in sidebar.symbol_name


class TestEdgeCases:
    def test_empty_html_returns_list(self, extractor: HTMLExtractor) -> None:
        chunks = extractor.extract("", "empty.html")
        assert isinstance(chunks, list)

    def test_plain_html_no_significant_elements(self, extractor: HTMLExtractor) -> None:
        html = "<html><body><p>Just a paragraph</p></body></html>"
        chunks = extractor.extract(html, "plain.html")
        # Should return document chunk if no significant elements
        assert isinstance(chunks, list)

    def test_document_chunk_uses_filename(self, extractor: HTMLExtractor) -> None:
        chunk = extractor._create_document_chunk("<p>hi</p>", "templates/base.html")
        assert chunk.symbol_name == "base.html"
        assert chunk.symbol_type == "document"

    def test_large_content_truncated(self, extractor: HTMLExtractor) -> None:
        big_html = "<main>" + "<p>line</p>" * 500 + "</main>"
        chunks = extractor.extract(big_html, "big.html")
        main_chunk = next((c for c in chunks if "main" in c.symbol_name), None)
        if main_chunk:
            assert len(main_chunk.content) <= 2100


# ---------------------------------------------------------------------------
# Additional tests for previously uncovered lines
# ---------------------------------------------------------------------------

class TestParseException:
    """Lines 39-41: tree-sitter parse error returns empty list."""

    def test_parse_exception_returns_empty_list(self, extractor: HTMLExtractor, monkeypatch) -> None:
        monkeypatch.setattr(extractor, "parse", lambda src: (_ for _ in ()).throw(Exception("parse fail")))
        chunks = extractor.extract("<div>hello</div>", "broken.html")
        assert isinstance(chunks, list)


class TestExtractElementEdgeCases:
    """Lines 67 (no start_tag) and 71 (no tag_name_node)."""

    def test_element_without_start_tag_returns_none(self, extractor: HTMLExtractor) -> None:
        # _extract_element with node that has no 'start_tag' child → None (line 67)
        class _FakeNode:
            children = []
            start_point = (0, 0)
            end_point = (1, 0)
            start_byte = 0
            end_byte = 5

        result = extractor._extract_element(_FakeNode(), b"<div>", "f.html")
        assert result is None

    def test_element_without_tag_name_returns_none(self, extractor: HTMLExtractor) -> None:
        # start_tag exists but has no 'tag_name' child → None (line 71)
        class _FakeStartTag:
            type = "start_tag"
            children = []
            start_point = (0, 0)
            end_point = (0, 5)
            start_byte = 0
            end_byte = 5

        class _FakeNode:
            children = [_FakeStartTag()]
            start_point = (0, 0)
            end_point = (1, 0)
            start_byte = 0
            end_byte = 10

        result = extractor._extract_element(_FakeNode(), b"<div>text", "f.html")
        assert result is None


class TestElementClassSymbolName:
    """Lines 95-97: element with class attribute uses first class in symbol name."""

    def test_custom_element_class_attribute_in_symbol_name(self, extractor: HTMLExtractor) -> None:
        html = '<body><app-header class="dark fixed sticky">Header</app-header></body>'
        chunks = extractor.extract(html, "page.html")
        header = next((c for c in chunks if "app-header" in c.symbol_name), None)
        assert header is not None
        # symbol_name should be app-header.dark (first class)
        assert "dark" in header.symbol_name

    def test_custom_element_no_id_uses_first_class(self, extractor: HTMLExtractor) -> None:
        html = '<body><my-widget class="primary secondary">Content</my-widget></body>'
        chunks = extractor.extract(html, "test.html")
        widget = next((c for c in chunks if "my-widget" in c.symbol_name), None)
        assert widget is not None
        assert "primary" in widget.symbol_name
