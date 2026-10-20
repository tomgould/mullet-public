"""Smoke tests for the Rust code extractor."""

from mullet.knowledge.code.extractors.rust import RustExtractor


class TestRustExtractorBasics:
    """Basic import and extraction tests."""

    def test_import_works(self):
        """Importing RustExtractor should succeed."""
        assert RustExtractor is not None

    def test_extract_small_snippet(self):
        """A small Rust snippet yields the expected chunks."""
        code = """\
/// summary
pub fn greet(name: &str) -> String {
    format!("Hello, {}", name)
}

struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn distance(&self, other: &Point) -> f64 {
        0.0
    }
}

trait Drawable {
    fn draw(&self);
}

enum Color {
    Red,
    Green,
    Blue,
}

const MAX_SIZE: usize = 1024;

static COUNTER: i32 = 0;

type IntResult = Result<i32, String>;

mod submod {
    fn helper() {}
}

macro_rules! say_hello {
    () => { println!("hello"); };
}
"""
        extractor = RustExtractor()
        chunks = extractor.extract(code, "test.rs")

        # Expected symbols:
        # greet (function), Point (struct), distance (method),
        # Drawable (trait), Color (enum), MAX_SIZE (constant),
        # COUNTER (constant), IntResult (type_alias),
        # submod (module), helper (function inside submod),
        # say_hello (macro)
        names = {c.symbol_name: c.symbol_type for c in chunks}
        assert names["greet"] == "function"
        assert names["Point"] == "struct"
        assert names["distance"] == "method"
        assert names["Drawable"] == "trait"
        assert names["Color"] == "enum"
        assert names["MAX_SIZE"] == "constant"
        assert names["COUNTER"] == "constant"
        assert names["IntResult"] == "type_alias"
        assert names["submod"] == "module"
        assert names["helper"] == "function"
        assert names["say_hello"] == "macro"

    def test_pub_fn_is_public(self):
        """A pub fn should have visibility=public."""
        code = "pub fn public_fn() {}"
        extractor = RustExtractor()
        chunks = extractor.extract(code, "test.rs")
        assert len(chunks) == 1
        assert chunks[0].visibility == "public"

    def test_non_pub_fn_is_private(self):
        """A non-pub fn should have visibility=private."""
        code = "fn private_fn() {}"
        extractor = RustExtractor()
        chunks = extractor.extract(code, "test.rs")
        assert len(chunks) == 1
        assert chunks[0].visibility == "private"

    def test_method_parent_class(self):
        """A method inside impl Foo should have parent_class='Foo'."""
        code = """\
impl Foo {
    fn bar(&self) {}
}
"""
        extractor = RustExtractor()
        chunks = extractor.extract(code, "test.rs")
        method = next(c for c in chunks if c.symbol_name == "bar")
        assert method.symbol_type == "method"
        assert method.parent_class == "Foo"

    def test_doc_comment_captured(self):
        """A /// doc comment should be captured as docstring."""
        code = """\
/// summary
pub fn hello() {}
"""
        extractor = RustExtractor()
        chunks = extractor.extract(code, "test.rs")
        assert len(chunks) == 1
        assert chunks[0].docstring == "summary"

    def test_impl_trait_for_type(self):
        """impl Trait for Foo should record impl_trait in extras."""
        code = """\
impl Clone for Foo {
    fn clone(&self) -> Self {
        Foo
    }
}
"""
        extractor = RustExtractor()
        chunks = extractor.extract(code, "test.rs")
        method = next(c for c in chunks if c.symbol_name == "clone")
        assert method.parent_class == "Foo"
        assert method.extras.get("impl_trait") == "Clone"
