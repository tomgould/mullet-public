"""Tests for the EXTRACTOR_MAP registry — verify extension → extractor routing."""



# --- EXTRACTOR_MAP sanity --------------------------------------------------


def test_extractor_map_is_dict():
    from mullet.knowledge.code.extractors import EXTRACTOR_MAP
    assert isinstance(EXTRACTOR_MAP, dict)
    assert len(EXTRACTOR_MAP) > 0


def test_extractor_map_python_family():
    from mullet.knowledge.code.extractors import EXTRACTOR_MAP, PythonExtractor
    assert EXTRACTOR_MAP[".py"] is PythonExtractor


def test_extractor_map_php_family_drupal():
    """PHP + Drupal-specific extensions all route to PHPExtractor."""
    from mullet.knowledge.code.extractors import EXTRACTOR_MAP, PHPExtractor
    for ext in (".php", ".module", ".install", ".theme", ".inc", ".profile"):
        assert EXTRACTOR_MAP[ext] is PHPExtractor, f"{ext} should route to PHPExtractor"


def test_extractor_map_javascript_family():
    from mullet.knowledge.code.extractors import EXTRACTOR_MAP, JavaScriptExtractor
    for ext in (".js", ".jsx", ".mjs", ".cjs"):
        assert EXTRACTOR_MAP[ext] is JavaScriptExtractor


def test_extractor_map_typescript_family():
    from mullet.knowledge.code.extractors import EXTRACTOR_MAP, TypeScriptExtractor
    for ext in (".ts", ".tsx"):
        assert EXTRACTOR_MAP[ext] is TypeScriptExtractor


def test_extractor_map_html_family_includes_components():
    """HTML extractor covers .html + component frameworks (vue, svelte, mdx)."""
    from mullet.knowledge.code.extractors import EXTRACTOR_MAP, HTMLExtractor
    for ext in (".html", ".htm", ".vue", ".svelte", ".mdx"):
        assert EXTRACTOR_MAP[ext] is HTMLExtractor, f"{ext} → HTMLExtractor"


def test_extractor_map_css_family():
    from mullet.knowledge.code.extractors import EXTRACTOR_MAP, CSSExtractor
    for ext in (".css", ".scss", ".sass", ".less"):
        assert EXTRACTOR_MAP[ext] is CSSExtractor


def test_extractor_map_yaml_family():
    from mullet.knowledge.code.extractors import EXTRACTOR_MAP, YAMLExtractor
    for ext in (".yml", ".yaml"):
        assert EXTRACTOR_MAP[ext] is YAMLExtractor


def test_extractor_map_json():
    from mullet.knowledge.code.extractors import EXTRACTOR_MAP, JSONExtractor
    assert EXTRACTOR_MAP[".json"] is JSONExtractor


def test_extractor_map_gherkin():
    from mullet.knowledge.code.extractors import EXTRACTOR_MAP, GherkinExtractor
    assert EXTRACTOR_MAP[".feature"] is GherkinExtractor


def test_extractor_map_dart():
    from mullet.knowledge.code.extractors import EXTRACTOR_MAP, DartExtractor
    assert EXTRACTOR_MAP[".dart"] is DartExtractor


def test_extractor_map_no_uppercase_keys():
    """All extension keys must be lowercase to match Path.suffix.lower() callers."""
    from mullet.knowledge.code.extractors import EXTRACTOR_MAP
    for ext in EXTRACTOR_MAP:
        assert ext == ext.lower(), f"{ext!r} should be lowercase"


def test_extractor_map_all_extensions_start_with_dot():
    """Every key must start with '.' — convention for file extension lookup."""
    from mullet.knowledge.code.extractors import EXTRACTOR_MAP
    for ext in EXTRACTOR_MAP:
        assert ext.startswith("."), f"{ext!r} should start with '.'"


def test_all_extractors_subclass_base():
    """Every extractor in the map must subclass BaseExtractor."""
    from mullet.knowledge.code.extractors import EXTRACTOR_MAP, BaseExtractor
    for ext, cls in EXTRACTOR_MAP.items():
        assert issubclass(cls, BaseExtractor), f"{ext} → {cls} must subclass BaseExtractor"


def test_all_listed_exports_are_importable():
    """Everything in __all__ should be importable from the package."""
    from mullet.knowledge.code import extractors
    for name in extractors.__all__:
        assert hasattr(extractors, name), f"extractors.{name} missing"
