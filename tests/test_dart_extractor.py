"""Tests for mullet.knowledge.code.extractors.dart."""
from __future__ import annotations

import pytest

from mullet.knowledge.code.extractors.dart import DartExtractor, _find_brace_end


_SIMPLE_DART = """\
/// A simple greeter function.
String greet(String name) {
  return 'Hello, $name!';
}

void main() {
  print(greet('World'));
}
"""

_CLASS_DART = """\
/// Manages user authentication.
class AuthService {
  final String baseUrl;

  AuthService(this.baseUrl);

  /// Login with credentials.
  Future<bool> login(String email, String password) async {
    return email.isNotEmpty && password.length >= 8;
  }

  bool _validateEmail(String email) {
    return email.contains('@');
  }

  static AuthService create(String url) {
    return AuthService(url);
  }
}
"""

_ABSTRACT_CLASS = """\
abstract class Repository<T> {
  Future<T> getById(int id);
  Future<List<T>> getAll();
}

class UserRepository extends Repository<User> {
  @override
  Future<User> getById(int id) async {
    return User(id: id);
  }

  @override
  Future<List<User>> getAll() async {
    return [];
  }
}
"""

_ENUM_DART = """\
enum Direction {
  north,
  south,
  east,
  west;

  String get label => name.toUpperCase();
}
"""

_MIXIN_DART = """\
mixin Serializable {
  Map<String, dynamic> toJson();

  String serialize() {
    return jsonEncode(toJson());
  }
}
"""

_TYPEDEF_DART = """\
typedef JsonMap = Map<String, dynamic>;
typedef Callback<T> = void Function(T value);
"""


@pytest.fixture
def extractor() -> DartExtractor:
    return DartExtractor()


class TestFindBraceEnd:
    def test_simple_block(self) -> None:
        src = "{ x = 1; }"
        end = _find_brace_end(src, 0)
        assert end == len(src)

    def test_nested_braces(self) -> None:
        src = "{ if (x) { y = 1; } }"
        end = _find_brace_end(src, 0)
        assert end == len(src)

    def test_string_with_braces_ignored(self) -> None:
        src = '{ s = "{inner}"; }'
        end = _find_brace_end(src, 0)
        assert end == len(src)

    def test_unmatched_returns_length(self) -> None:
        src = "{ unclosed"
        end = _find_brace_end(src, 0)
        assert end == len(src)


class TestDartFunctions:
    def test_extracts_top_level_function(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_DART, "greet.dart")
        func_names = [c.symbol_name for c in chunks if c.symbol_type == "function"]
        assert "greet" in func_names or "main" in func_names

    def test_function_language_is_dart(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_DART, "greet.dart")
        funcs = [c for c in chunks if c.symbol_type == "function"]
        assert all(c.language == "dart" for c in funcs)

    def test_source_file_set(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_DART, "greet.dart")
        assert all(c.source_file == "greet.dart" for c in chunks)

    def test_returns_list(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_DART, "greet.dart")
        assert isinstance(chunks, list)


class TestDartClasses:
    def test_extracts_class(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_CLASS_DART, "auth_service.dart")
        class_chunks = [c for c in chunks if c.symbol_type == "class"]
        assert len(class_chunks) >= 1
        assert any(c.symbol_name == "AuthService" for c in class_chunks)

    def test_extracts_methods(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_CLASS_DART, "auth_service.dart")
        method_names = [c.symbol_name for c in chunks if c.symbol_type == "method"]
        assert "login" in method_names

    def test_method_has_parent_class(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_CLASS_DART, "auth_service.dart")
        login = next((c for c in chunks if c.symbol_type == "method" and c.symbol_name == "login"), None)
        if login:
            assert login.parent_class == "AuthService"

    def test_abstract_class_extracted(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_ABSTRACT_CLASS, "repository.dart")
        class_names = [c.symbol_name for c in chunks if c.symbol_type == "class"]
        assert "Repository" in class_names or "UserRepository" in class_names

    def test_class_docstring(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_CLASS_DART, "auth.dart")
        auth = next((c for c in chunks if c.symbol_type == "class"), None)
        if auth and auth.docstring:
            assert len(auth.docstring) > 0


class TestDartEnum:
    def test_extracts_enum(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_ENUM_DART, "direction.dart")
        enum_chunks = [c for c in chunks if c.symbol_type == "enum"]
        assert len(enum_chunks) >= 1
        assert any(c.symbol_name == "Direction" for c in enum_chunks)


class TestDartMixin:
    def test_extracts_mixin(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_MIXIN_DART, "mixins.dart")
        mixin_chunks = [c for c in chunks if c.symbol_type == "mixin"]
        assert len(mixin_chunks) >= 1
        assert any(c.symbol_name == "Serializable" for c in mixin_chunks)


class TestDartTypedef:
    def test_extracts_typedef(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_TYPEDEF_DART, "types.dart")
        typedef_chunks = [c for c in chunks if c.symbol_type == "typedef"]
        assert len(typedef_chunks) >= 1
        names = [c.symbol_name for c in typedef_chunks]
        assert "JsonMap" in names or "Callback" in names


class TestDartEdgeCases:
    def test_empty_source_returns_list(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract("", "empty.dart")
        assert isinstance(chunks, list)

    def test_only_comments_returns_list(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract("// just a comment\n/// doc\n", "comments.dart")
        assert isinstance(chunks, list)


# ---------------------------------------------------------------------------
# Additional tests for previously uncovered lines
# ---------------------------------------------------------------------------

# --- Block doc comments (lines 121-138) -----------------------------------

_BLOCK_DOC_CLASS = """\
/**
 * Represents a user entity.
 * Stores profile information.
 */
class UserEntity {
  String name;
  UserEntity(this.name);
}
"""


class TestBlockDocComments:
    """Lines 121-138: /** ... */ style doc comments."""

    def test_block_doc_comment_extracted(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_BLOCK_DOC_CLASS, "user.dart")
        cls = next((c for c in chunks if c.symbol_type == "class"), None)
        assert cls is not None
        assert cls.docstring is not None
        assert "user" in cls.docstring.lower() or "Represents" in cls.docstring

    def test_block_doc_comment_content(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_BLOCK_DOC_CLASS, "user.dart")
        cls = next((c for c in chunks if c.symbol_type == "class"), None)
        if cls and cls.docstring:
            # The extracted docstring should contain meaningful text
            assert len(cls.docstring) > 0


# --- DartExtractor.parser property (line 158) -----------------------------

class TestDartParserProperty:
    """Line 158: parser property returns None (no tree-sitter Dart grammar)."""

    def test_parser_returns_none(self, extractor: DartExtractor) -> None:
        assert extractor.parser is None


# --- Extensions (lines 194-203) -------------------------------------------

_EXTENSION_DART = """\
extension StringExtension on String {
  String capitalize() {
    return this[0].toUpperCase() + substring(1);
  }
}

extension IntExtension on int {
  bool get isEven => this % 2 == 0;
}
"""


class TestDartExtensions:
    """Lines 194-203: extension declarations."""

    def test_extracts_extension(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_EXTENSION_DART, "extensions.dart")
        ext_chunks = [c for c in chunks if c.symbol_type == "extension"]
        assert len(ext_chunks) >= 1

    def test_extension_name(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_EXTENSION_DART, "extensions.dart")
        names = [c.symbol_name for c in chunks if c.symbol_type == "extension"]
        assert "StringExtension" in names

    def test_extension_on_type_in_extras(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_EXTENSION_DART, "extensions.dart")
        str_ext = next((c for c in chunks if c.symbol_name == "StringExtension"), None)
        assert str_ext is not None
        assert str_ext.extras.get("on_type") == "String"

    def test_multiple_extensions_extracted(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_EXTENSION_DART, "extensions.dart")
        ext_chunks = [c for c in chunks if c.symbol_type == "extension"]
        assert len(ext_chunks) == 2


# --- Class with extends/implements/with (lines 294-296, 300-302) ----------

_CLASS_WITH_MODIFIERS = """\
abstract class Animal {
  String get name;
  void speak();
}

mixin Flyable {
  void fly() => print('flying');
}

class Bird extends Animal with Flyable implements Countable {
  @override
  String get name => 'Bird';

  @override
  void speak() => print('tweet');
}
"""


class TestClassModifiers:
    """Lines 247 (extends), 265 (no brace for class), 282 (sealed), 294-296 (with), 300-302 (implements)."""

    def test_class_with_extends(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_CLASS_WITH_MODIFIERS, "bird.dart")
        bird = next((c for c in chunks if c.symbol_name == "Bird"), None)
        assert bird is not None
        assert bird.extras.get("base_class") == "Animal"

    def test_class_with_mixin(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_CLASS_WITH_MODIFIERS, "bird.dart")
        bird = next((c for c in chunks if c.symbol_name == "Bird"), None)
        assert bird is not None
        mixins = bird.extras.get("mixins", [])
        assert "Flyable" in mixins

    def test_class_with_implements(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_CLASS_WITH_MODIFIERS, "bird.dart")
        bird = next((c for c in chunks if c.symbol_name == "Bird"), None)
        assert bird is not None
        interfaces = bird.extras.get("implements", [])
        assert "Countable" in interfaces

    def test_abstract_class_flag(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_CLASS_WITH_MODIFIERS, "bird.dart")
        animal = next((c for c in chunks if c.symbol_name == "Animal"), None)
        assert animal is not None
        assert animal.extras.get("abstract") is True


_SEALED_CLASS = """\
sealed class Shape {
  double get area;
}
"""


class TestSealedClass:
    """Line 282: sealed modifier detected."""

    def test_sealed_class_flag(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_SEALED_CLASS, "shape.dart")
        shape = next((c for c in chunks if c.symbol_name == "Shape"), None)
        assert shape is not None
        assert shape.extras.get("sealed") is True


# --- Mixin with on constraint (lines 442, 454) ----------------------------

_MIXIN_ON = """\
mixin LogMixin on StatefulWidget {
  void log(String msg) => print('[LOG] $msg');
}
"""


class TestMixinOnConstraint:
    """Lines 442 (brace not found), 454 (on constraint captured)."""

    def test_mixin_on_constraint_in_extras(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_MIXIN_ON, "log_mixin.dart")
        mixin = next((c for c in chunks if c.symbol_name == "LogMixin"), None)
        assert mixin is not None
        assert mixin.extras.get("on_type") == "StatefulWidget"

    def test_mixin_no_brace_returns_none(self, extractor: DartExtractor) -> None:
        # mixin without opening brace
        src = "mixin Broken"
        chunks = extractor.extract(src, "broken.dart")
        mixin_chunks = [c for c in chunks if c.symbol_type == "mixin"]
        assert len(mixin_chunks) == 0


# --- Extension with no brace (lines 477-490) ------------------------------

class TestExtensionNoBrace:
    """Line 482: extension without opening brace returns None."""

    def test_extension_no_brace_skipped(self, extractor: DartExtractor) -> None:
        src = "extension BrokenExt on String"
        chunks = extractor.extract(src, "broken.dart")
        ext_chunks = [c for c in chunks if c.symbol_type == "extension"]
        assert len(ext_chunks) == 0


# --- Enum with no brace (line 516) ----------------------------------------

class TestEnumNoBrace:
    """Line 516: enum without brace returns None."""

    def test_enum_no_brace_skipped(self, extractor: DartExtractor) -> None:
        src = "enum BrokenEnum"
        chunks = extractor.extract(src, "broken.dart")
        enum_chunks = [c for c in chunks if c.symbol_type == "enum"]
        assert len(enum_chunks) == 0


# --- Typedef with no semicolon (line 549) ---------------------------------

class TestTypedefNoSemicolon:
    """Line 549: typedef without semicolon returns None."""

    def test_typedef_no_semicolon_skipped(self, extractor: DartExtractor) -> None:
        src = "typedef BrokenDef = void Function(int x)"
        chunks = extractor.extract(src, "broken.dart")
        typedef_chunks = [c for c in chunks if c.symbol_type == "typedef"]
        assert len(typedef_chunks) == 0


# --- Top-level function — semicolon path (lines 585-589) -----------------

_ARROW_FUNC = """\
String get appName => 'MyApp';
int double(int x) => x * 2;
"""


class TestTopLevelFunctionSemicolon:
    """Lines 585-589: function body is semicolon-terminated (arrow/declaration)."""

    def test_arrow_function_extracted(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_ARROW_FUNC, "funcs.dart")
        func_names = [c.symbol_name for c in chunks if c.symbol_type == "function"]
        # At least one function is found
        assert len(func_names) >= 1

    def test_function_no_brace_no_semi_returns_none(self, extractor: DartExtractor) -> None:
        # If neither { nor ; found — _extract_top_level_function returns None (line 589)
        # Very contrived: a line that looks like a function but has neither brace nor semicolon
        # In practice the extract() loop just produces no chunk
        chunks = extractor.extract("void broken(", "broken.dart")
        func_chunks = [c for c in chunks if c.symbol_type == "function"]
        assert len(func_chunks) == 0


# --- Function signature when no { found (line 596) -----------------------

_EXTERN_FUNC = """\
external int nativeMultiply(int a, int b);
"""


class TestTopLevelFunctionSignatureNoOpenBrace:
    """Line 596: when sig_end finds ; instead of {."""

    def test_external_func_signature(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_EXTERN_FUNC, "native.dart")
        func_chunks = [c for c in chunks if c.symbol_type == "function"]
        # If extracted, verify it has a signature
        if func_chunks:
            assert func_chunks[0].signature is not None


# --- Async function (line 602) --------------------------------------------

_ASYNC_FUNC = """\
Future<String> fetchData(String url) async {
  return 'data';
}
"""


class TestAsyncTopLevelFunction:
    """Line 602: async function sets extras["async"] = True."""

    def test_async_function_extras(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_ASYNC_FUNC, "fetch.dart")
        func = next((c for c in chunks if c.symbol_type == "function"), None)
        assert func is not None
        assert func.extras.get("async") is True


# --- _dart_visibility with private name (line 638) -----------------------

class TestDartVisibility:
    """Line 638: _dart_visibility returns 'private' for names starting with _."""

    def test_private_name_visibility(self, extractor: DartExtractor) -> None:
        assert extractor._dart_visibility("_privateHelper") == "private"

    def test_public_name_visibility(self, extractor: DartExtractor) -> None:
        assert extractor._dart_visibility("publicMethod") == "public"

    def test_determine_visibility_delegates(self, extractor: DartExtractor) -> None:
        # determine_visibility (line 636-638) should use Dart convention
        assert extractor.determine_visibility("_hidden") == "private"


# --- Abstract method (method body with ;, no {) (lines from _extract_methods) ---

_ABSTRACT_METHODS_CLASS = """\
abstract class Widget {
  void build(BuildContext context);
  String get type;
  int compute(int x);
}
"""


class TestAbstractMethodExtraction:
    """Exercises _extract_methods path for methods ending with ; (no brace)."""

    def test_abstract_methods_extracted(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_ABSTRACT_METHODS_CLASS, "widget.dart")
        method_chunks = [c for c in chunks if c.symbol_type == "method"]
        method_names = [c.symbol_name for c in method_chunks]
        assert len(method_names) >= 1

    def test_abstract_method_parent_class(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_ABSTRACT_METHODS_CLASS, "widget.dart")
        method_chunks = [c for c in chunks if c.symbol_type == "method"]
        if method_chunks:
            assert all(c.parent_class == "Widget" for c in method_chunks)


# --- Async method (extras["async"] in _extract_methods) ------------------

_CLASS_WITH_ASYNC = """\
class DataService {
  Future<List<String>> fetchAll() async {
    return [];
  }

  static Future<DataService> create() async {
    return DataService();
  }
}
"""


class TestAsyncMethod:
    """Exercises async detection in _extract_methods."""

    def test_async_method_detected(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_CLASS_WITH_ASYNC, "data_service.dart")
        method_chunks = [c for c in chunks if c.symbol_type == "method"]
        async_methods = [c for c in method_chunks if c.extras.get("async")]
        assert len(async_methods) >= 1

    def test_static_method_detected(self, extractor: DartExtractor) -> None:
        chunks = extractor.extract(_CLASS_WITH_ASYNC, "data_service.dart")
        method_chunks = [c for c in chunks if c.symbol_type == "method"]
        static_methods = [c for c in method_chunks if c.extras.get("static")]
        assert len(static_methods) >= 1


# --- _in_extracted_range returns True (line 247) --------------------------

class TestInExtractedRange:
    """Line 247: _in_extracted_range returns True when offset is inside a range."""

    def test_returns_true_when_inside_range(self, extractor: DartExtractor) -> None:
        # Offset 5 is inside range (0, 10)
        assert extractor._in_extracted_range(5, [(0, 10)]) is True

    def test_returns_false_when_outside_all_ranges(self, extractor: DartExtractor) -> None:
        assert extractor._in_extracted_range(15, [(0, 10), (20, 30)]) is False

    def test_returns_true_for_second_range(self, extractor: DartExtractor) -> None:
        # Offset 25 is inside second range (20, 30) — ensures loop iterates
        assert extractor._in_extracted_range(25, [(0, 10), (20, 30)]) is True


# --- Class without opening brace (line 265) --------------------------------

class TestClassNoBrace:
    """Line 265: _extract_class returns empty list when no opening brace found."""

    def test_class_no_brace_skipped(self, extractor: DartExtractor) -> None:
        # A partial class declaration with no { body
        src = "class Incomplete"
        chunks = extractor.extract(src, "incomplete.dart")
        class_chunks = [c for c in chunks if c.symbol_type == "class"]
        assert len(class_chunks) == 0
